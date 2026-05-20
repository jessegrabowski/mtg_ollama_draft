import pytest
from pydantic import ValidationError

from mtg_drafting.config import DraftConfig
from mtg_drafting.seat import Seat
from mtg_drafting.strategist import (
    PoolClassification,
    StrategyDirection,
    StrategyState,
    build_strategy_messages,
    update_strategy,
)
from tests.conftest import StubLLM


def _direction_json(colors: tuple[str, ...], role: str, weight: int) -> str:
    colors_json = "[" + ", ".join(f'"{c}"' for c in colors) + "]"
    return (
        f'{{"colors": {colors_json}, "role": "{role}", '
        f'"weight": {weight}, "rationale": "supported by pool"}}'
    )


def _valid_state_json(
    directions: tuple[tuple[tuple[str, ...], str, int], ...] = ((("W", "U"), "midrange", 8),),
    maindeck: tuple[str, ...] = (),
) -> str:
    directions_json = "[" + ", ".join(_direction_json(*d) for d in directions) + "]"
    maindeck_json = "[" + ", ".join(f'"{c}"' for c in maindeck) + "]"
    return (
        f'{{"directions": {directions_json}, "color_commitment": "leaning", '
        f'"biggest_needs": ["evasion"], "watching_for": "any flyer", '
        f'"pool_classification": '
        f'{{"maindeck": {maindeck_json}, "sideboard": [], "chaff": []}}}}'
    )


def test_strategy_state_parses_full_schema():
    state = StrategyState.model_validate_json(_valid_state_json(maindeck=("Card 001",)))
    assert state.color_commitment == "leaning"
    assert len(state.directions) == 1
    assert state.directions[0].colors == ["W", "U"]
    assert state.directions[0].role == "midrange"
    assert state.directions[0].weight == 8
    assert state.biggest_needs == ["evasion"]
    assert state.pool_classification.maindeck == ["Card 001"]


def test_strategy_state_supports_multiple_directions():
    json_payload = _valid_state_json(
        directions=((("W", "U"), "control", 7), (("U", "R"), "beatdown", 4))
    )
    state = StrategyState.model_validate_json(json_payload)
    assert len(state.directions) == 2
    assert state.primary.colors == ["W", "U"]
    assert state.primary.role == "control"
    assert state.primary.weight == 7


def test_strategy_state_primary_picks_highest_weight_regardless_of_order():
    # Even if the LLM returns them out of order, .primary picks the highest weight.
    json_payload = _valid_state_json(
        directions=((("R",), "beatdown", 3), (("U", "R"), "midrange", 9))
    )
    state = StrategyState.model_validate_json(json_payload)
    assert state.primary.colors == ["U", "R"]


@pytest.mark.parametrize(
    "old,new",
    [
        ('"leaning"', '"locked-in"'),  # color_commitment not in Literal
        ('"role": "midrange"', '"role": "aristocrats"'),  # role not in Literal
        ('"weight": 8', '"weight": 11'),  # weight outside [1, 10]
    ],
    ids=["unknown_commitment", "unknown_role", "weight_out_of_range"],
)
def test_strategy_state_rejects_invalid_field(old, new):
    bad = _valid_state_json().replace(old, new)
    with pytest.raises(ValidationError):
        StrategyState.model_validate_json(bad)


def test_strategy_state_accepts_open_role_with_empty_colors():
    # The 'open' role is the uncommitted state the strategist returns early in the
    # draft. It carries no colors and is the expected primary direction when the
    # strategist would otherwise have to weakly commit to a single archetype.
    payload = _valid_state_json(directions=(((), "open", 9),))
    state = StrategyState.model_validate_json(payload)
    assert state.primary.role == "open"
    assert state.primary.colors == []
    assert state.primary.weight == 9


def test_strategy_state_rejects_empty_directions_list():
    # Structurally different from the other rejection tests: we substitute the
    # entire directions list rather than tweaking one field.
    bad = _valid_state_json().replace(
        '"directions": [' + _direction_json(("W", "U"), "midrange", 8) + "]",
        '"directions": []',
    )
    with pytest.raises(ValidationError):
        StrategyState.model_validate_json(bad)


def test_update_strategy_returns_parsed_state():
    seat = Seat(index=0, history_maxlen=5)
    llm = StubLLM([_valid_state_json(directions=((("R",), "beatdown", 9),))])
    state = update_strategy(seat, 0, llm, DraftConfig(), deck_profile="")
    assert state is not None
    assert state.primary.role == "beatdown"
    assert state.primary.colors == ["R"]


def test_update_strategy_returns_none_after_two_parse_failures():
    seat = Seat(index=0, history_maxlen=5)
    llm = StubLLM(["not json", "still not json"])
    assert update_strategy(seat, 0, llm, DraftConfig(), deck_profile="") is None


def test_update_strategy_succeeds_on_second_attempt(cube):
    # The retry loop exists for transient malformed replies. Proves it actually
    # runs a second attempt rather than returning early on the first failure.
    seat = Seat(index=0, history_maxlen=5)
    llm = StubLLM(["not json", _valid_state_json()])
    state = update_strategy(seat, 0, llm, DraftConfig(), deck_profile="")
    assert state is not None
    assert state.primary.role == "midrange"


def test_build_strategy_messages_includes_round_and_previous_plan():
    seat = Seat(index=2, history_maxlen=5)
    seat.strategy_state = StrategyState(
        directions=[
            StrategyDirection(
                colors=["U", "R"], role="midrange", weight=8, rationale="from pool"
            )
        ],
        color_commitment="committed",
        biggest_needs=["topend"],
        watching_for="any bomb",
        pool_classification=PoolClassification(maindeck=[], sideboard=[], chaff=[]),
    )
    messages = build_strategy_messages(seat, round_no=1, config=DraftConfig(), deck_profile="X")
    user = messages[-1]["content"]
    assert "Starting pack 2" in user
    assert "midrange" in user
    assert "committed" in user


def test_build_strategy_messages_handles_first_pack_no_plan():
    seat = Seat(index=0, history_maxlen=5)
    messages = build_strategy_messages(seat, round_no=0, config=DraftConfig(), deck_profile="X")
    user = messages[-1]["content"]
    assert "no previous plan" in user
    assert "Starting pack 1" in user


def test_build_strategy_messages_includes_memory_block():
    # Memory notes the picker, strategist, and wheel detector have appended should
    # be visible to the next strategist call so it can reason about prior signals.
    seat = Seat(index=0, history_maxlen=5)
    seat.memory.append("[wheel P1.1->P1.9] cards back: Card 002, Card 003")
    seat.memory.append("[P1.5] saw Goblin King passed at pick 5; R looks open")
    messages = build_strategy_messages(seat, round_no=1, config=DraftConfig(), deck_profile="X")
    user = messages[-1]["content"]
    assert "[wheel P1.1->P1.9]" in user
    assert "R looks open" in user

from mtg_drafting.cards import Card
from mtg_drafting.seat import Seat
from mtg_drafting.strategist import (
    PoolClassification,
    StrategyDirection,
    StrategyState,
)


def _card(name: str) -> Card:
    return Card(name=name, mana_cost="{1}", cmc=1.0, type_line="Creature — Test")


def _state(*directions: StrategyDirection, commitment: str = "leaning") -> StrategyState:
    return StrategyState(
        directions=list(directions),
        color_commitment=commitment,
        biggest_needs=["raw power"],
        watching_for="bombs",
        pool_classification=PoolClassification(maindeck=[], sideboard=[], chaff=[]),
    )


def test_maindeck_equals_pool_when_nothing_sidelined():
    seat = Seat(index=0, history_maxlen=5)
    seat.pool = [_card("A"), _card("B"), _card("C")]
    assert [c.name for c in seat.maindeck] == ["A", "B", "C"]
    assert seat.sideboard == []


def test_sidelined_names_move_to_sideboard():
    seat = Seat(index=0, history_maxlen=5)
    seat.pool = [_card("A"), _card("B"), _card("C")]
    seat.sidelined = {"B"}
    assert [c.name for c in seat.maindeck] == ["A", "C"]
    assert [c.name for c in seat.sideboard] == ["B"]


def test_sidelined_name_not_in_pool_is_ignored():
    seat = Seat(index=0, history_maxlen=5)
    seat.pool = [_card("A"), _card("B")]
    seat.sidelined = {"Phantom"}
    # Phantom is not in pool; maindeck stays full, sideboard stays empty.
    assert [c.name for c in seat.maindeck] == ["A", "B"]
    assert seat.sideboard == []


def test_duplicates_in_pool_split_by_name():
    seat = Seat(index=0, history_maxlen=5)
    seat.pool = [_card("A"), _card("A"), _card("B")]
    seat.sidelined = {"A"}
    # All instances of a sidelined name go to sideboard.
    assert [c.name for c in seat.maindeck] == ["B"]
    assert [c.name for c in seat.sideboard] == ["A", "A"]


def test_strategy_summary_when_state_is_none():
    seat = Seat(index=0, history_maxlen=5)
    assert seat.strategy_summary == "(no plan yet)"


def test_strategy_summary_collapses_open_role_to_uncommitted():
    seat = Seat(index=0, history_maxlen=5)
    seat.strategy_state = _state(
        StrategyDirection(colors=[], role="open", weight=9, rationale="too early"),
        commitment="open",
    )
    # An 'open' primary collapses to a single word; avoids the redundant
    # "open open" of commitment + role.
    assert seat.strategy_summary == "uncommitted"


def test_strategy_summary_appends_alt_count_for_multi_direction_plan():
    seat = Seat(index=0, history_maxlen=5)
    seat.strategy_state = _state(
        StrategyDirection(colors=["W", "U"], role="midrange", weight=7, rationale="ok"),
        StrategyDirection(colors=["U", "R"], role="beatdown", weight=4, rationale="alt"),
        StrategyDirection(colors=["G"], role="midrange", weight=2, rationale="alt2"),
    )
    assert seat.strategy_summary == "leaning W/U midrange (+2 alt)"

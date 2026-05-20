import pytest

from mtg_drafting.agent import DraftAgent
from mtg_drafting.config import DraftConfig
from mtg_drafting.seat import PickRecord, Seat
from tests.conftest import StubLLM


def _pick_reply(
    pick: str,
    reasoning: str = "ok",
    note: str | None = None,
    intent: str = "maindeck",
    hoping_to_wheel: str | None = None,
) -> str:
    note_field = "null" if note is None else f'"{note}"'
    wheel_field = "null" if hoping_to_wheel is None else f'"{hoping_to_wheel}"'
    return (
        f'{{"pick": "{pick}", "reasoning": "{reasoning}", '
        f'"intent": "{intent}", "note": {note_field}, '
        f'"hoping_to_wheel": {wheel_field}}}'
    )


def _strategist_reply(role: str = "midrange", colors: tuple[str, ...] = ("W", "U")) -> str:
    colors_json = "[" + ", ".join(f'"{c}"' for c in colors) + "]"
    return (
        f'{{"directions": [{{"colors": {colors_json}, "role": "{role}", '
        f'"weight": 8, "rationale": "supported by pool"}}], '
        f'"color_commitment": "leaning", "biggest_needs": ["2-mana evasion"], '
        f'"watching_for": "cards that break one-land-per-turn", '
        f'"pool_classification": {{"maindeck": [], "sideboard": [], "chaff": []}}}}'
    )


def test_match_exact(cube):
    pack = cube[:5]
    assert DraftAgent._match("Card 002", pack) is pack[2]


def test_match_case_insensitive(cube):
    pack = cube[:5]
    assert DraftAgent._match("  card 003 ", pack) is pack[3]


def test_match_close_name(cube):
    pack = cube[:5]
    # A small typo still resolves to the intended card.
    assert DraftAgent._match("Card 04", pack) is pack[4]


def test_match_rejects_unrelated_name(cube):
    pack = cube[:5]
    assert DraftAgent._match("Lightning Bolt", pack) is None


def test_parse_valid_json():
    raw = '{"pick": "Sol Ring", "reasoning": "Fast mana."}'
    parsed = DraftAgent._parse(raw)
    assert parsed is not None
    assert parsed.pick == "Sol Ring"
    assert parsed.reasoning == "Fast mana."


def test_parse_salvages_truncated_reply():
    # JSON cut off mid-reasoning by the token limit; the leading pick is intact.
    raw = '{"pick": "Lightning Bolt", "reasoning": "Efficient removal that also'
    parsed = DraftAgent._parse(raw)
    assert parsed is not None
    assert parsed.pick == "Lightning Bolt"


def test_parse_returns_none_when_pick_absent():
    assert DraftAgent._parse('{"reasoning": "incomplete reply with no pick yet') is None


def test_pick_returns_chosen_card_on_non_refresh_pick(cube):
    # A pick at a non-refresh point skips the strategist; only the picker call happens.
    pack = cube[:5]
    seat = Seat(index=0, history_maxlen=5)
    agent = DraftAgent(StubLLM([_pick_reply("Card 003")]), DraftConfig())

    card, reasoning, _ = agent.pick(seat, pack, round_no=0, pick_no=7)
    assert card is pack[3]
    assert reasoning == "ok"


def test_pick_retries_after_an_unusable_reply(cube):
    pack = cube[:5]
    seat = Seat(index=0, history_maxlen=5)
    agent = DraftAgent(
        StubLLM(['{"reasoning": "oops"', _pick_reply("Card 001")]), DraftConfig()
    )

    card, _, _ = agent.pick(seat, pack, round_no=0, pick_no=7)
    assert card is pack[1]


def test_pick_falls_back_to_first_card_when_every_reply_fails(cube):
    pack = cube[:5]
    seat = Seat(index=0, history_maxlen=5)
    # DraftConfig default is one retry, so the agent makes two attempts.
    agent = DraftAgent(StubLLM([_pick_reply("Nonexistent Card")] * 2), DraftConfig())

    card, _, wheel_hope = agent.pick(seat, pack, round_no=0, pick_no=7)
    # Fallback path: every reply named a card outside the pack, so the agent
    # returns pack[0] rather than stalling the draft. wheel_hope is dropped, and
    # a FALLBACK memory note makes the silent default visible to the next
    # strategist call and to the post-draft records.
    assert card is pack[0]
    assert wheel_hope is None
    assert any("[FALLBACK P1.8]" in note for note in seat.memory)


def test_pick_skips_the_model_for_a_single_card_pack(cube):
    pack = cube[:1]
    seat = Seat(index=0, history_maxlen=5)
    # An empty StubLLM raises IndexError if .chat is called at all.
    agent = DraftAgent(StubLLM([]), DraftConfig())

    card, reasoning, _ = agent.pick(seat, pack, 0, 14)
    assert card is pack[0]
    assert "only card" in reasoning


@pytest.mark.parametrize(
    "round_no,pick_no",
    [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 9), (1, 0), (2, 0)],
    ids=["P1.2", "P1.3", "P1.4", "P1.5", "P1.6", "P1.10", "P2.1", "P3.1"],
)
def test_pick_runs_strategist_at_refresh_points(cube, round_no, pick_no):
    pack = cube[:5]
    seat = Seat(index=0, history_maxlen=5)
    agent = DraftAgent(
        StubLLM([_strategist_reply(role="beatdown", colors=("R",)),
                 _pick_reply("Card 002")]),
        DraftConfig(),
    )
    card, _, _ = agent.pick(seat, pack, round_no=round_no, pick_no=pick_no)
    assert card is pack[2]
    assert seat.strategy_state is not None
    assert seat.strategy_state.primary.role == "beatdown"


@pytest.mark.parametrize(
    "round_no,pick_no",
    [(0, 0), (0, 6), (0, 8), (0, 14), (1, 4), (2, 7)],
    ids=["P1.1-empty-pool", "P1.7", "P1.9", "P1.15", "P2.5-mid", "P3.8-mid"],
)
def test_pick_skips_strategist_at_non_refresh_points(cube, round_no, pick_no):
    # StubLLM has only the picker reply queued; if the agent tried to call the
    # strategist here, the pop would raise IndexError and fail the test.
    pack = cube[:5]
    seat = Seat(index=0, history_maxlen=5)
    agent = DraftAgent(StubLLM([_pick_reply("Card 002")]), DraftConfig())
    card, _, _ = agent.pick(seat, pack, round_no=round_no, pick_no=pick_no)
    assert card is pack[2]
    assert seat.strategy_state is None


def test_strategist_first_call_does_not_log_sideline_moves(cube):
    # Initial classification is not a 'move'. The strategist seeing pool for the
    # first time and sidelining 1 card should NOT add a "moved to sideboard"
    # memory note - that would be misleading noise.
    seat = Seat(index=0, history_maxlen=5)
    seat.pool = [cube[0], cube[1], cube[2]]
    strategist_json = (
        '{"directions": [{"colors": ["U"], "role": "midrange", "weight": 8, '
        '"rationale": "ok"}], "color_commitment": "leaning", '
        '"biggest_needs": ["removal"], "watching_for": "free mana", '
        '"pool_classification": {"maindeck": ["Card 000", "Card 001"], '
        '"sideboard": ["Card 002"], "chaff": []}}'
    )
    pack = cube[:5]
    agent = DraftAgent(StubLLM([strategist_json, _pick_reply("Card 002")]), DraftConfig())
    agent.pick(seat, pack, round_no=0, pick_no=1)
    assert not any("moved to" in note for note in seat.memory)


def test_strategist_subsequent_call_logs_sideline_moves(cube):
    # Second strategist call moves a card OUT of sidelined - logged as
    # "moved to maindeck" so a card silently flipping is traceable.
    seat = Seat(index=0, history_maxlen=5)
    seat.pool = [cube[0], cube[1], cube[2]]
    seat.sidelined = {"Card 002"}  # already classified by an earlier call
    strategist_json = (
        '{"directions": [{"colors": ["U"], "role": "midrange", "weight": 8, '
        '"rationale": "ok"}], "color_commitment": "leaning", '
        '"biggest_needs": ["removal"], "watching_for": "free mana", '
        '"pool_classification": {"maindeck": ["Card 000", "Card 001", "Card 002"], '
        '"sideboard": [], "chaff": []}}'
    )
    pack = cube[:5]
    agent = DraftAgent(StubLLM([strategist_json, _pick_reply("Card 003")]), DraftConfig())
    agent.pick(seat, pack, round_no=0, pick_no=1)
    assert any(
        "moved to maindeck: Card 002" in note for note in seat.memory
    )


def test_strategist_call_populates_sidelined_from_pool_classification(cube):
    # Strategist returns Card 003 in sideboard, Card 001 in chaff. Both names appear
    # in seat.pool, so both end up in seat.sidelined.
    pool_cards = [cube[0], cube[1], cube[2], cube[3]]  # Card 000..003
    seat = Seat(index=0, history_maxlen=5)
    seat.pool = list(pool_cards)
    strategist_json = (
        '{"directions": [{"colors": ["U"], "role": "midrange", "weight": 8, '
        '"rationale": "ok"}], "color_commitment": "leaning", '
        '"biggest_needs": ["removal"], "watching_for": "free mana", '
        '"pool_classification": {"maindeck": ["Card 002"], '
        '"sideboard": ["Card 003"], "chaff": ["Card 001", "Hallucinated Card"]}}'
    )
    pack = cube[:5]
    agent = DraftAgent(StubLLM([strategist_json, _pick_reply("Card 002")]), DraftConfig())
    agent.pick(seat, pack, round_no=0, pick_no=1)
    # Hallucinated names are filtered out; the two real pool names stick.
    assert seat.sidelined == {"Card 003", "Card 001"}


def test_picker_note_is_appended_to_memory(cube):
    pack = cube[:5]
    seat = Seat(index=0, history_maxlen=5)
    agent = DraftAgent(
        StubLLM([_pick_reply("Card 002", note="hoping Bolt wheels")]), DraftConfig()
    )
    agent.pick(seat, pack, round_no=0, pick_no=7)
    assert len(seat.memory) == 1
    assert "hoping Bolt wheels" in seat.memory[0]
    assert seat.memory[0].startswith("[P1.8]")


def test_picker_skips_memory_when_note_is_null(cube):
    pack = cube[:5]
    seat = Seat(index=0, history_maxlen=5)
    agent = DraftAgent(StubLLM([_pick_reply("Card 002", note=None)]), DraftConfig())
    agent.pick(seat, pack, round_no=0, pick_no=7)
    assert len(seat.memory) == 0


def test_wheel_detection_silent_without_hopes(cube):
    # Seat picks from a 15-card pack at P1.1 with no hoping_to_wheel set, then sees
    # the same pack returning at P1.9. A wheel without a prediction to evaluate
    # carries no signal worth remembering, so the detector must stay silent.
    seat = Seat(index=0, history_maxlen=5)
    full_pack = cube[:15]
    seat.add_pick(
        full_pack[0],
        PickRecord(
            round_no=0, pick_no=0, pack=[c.name for c in full_pack], chosen="Card 000",
            reasoning="early",
        ),
    )
    wheeled_pack = cube[1:8]
    agent = DraftAgent(StubLLM([_pick_reply("Card 002")]), DraftConfig())
    agent.pick(seat, wheeled_pack, round_no=0, pick_no=8)
    assert not any("[wheel" in note for note in seat.memory)


def test_wheel_detection_cross_references_hoping_to_wheel(cube):
    # Seat earlier hoped Card 002 and Card 050 would wheel. Card 002 IS in the
    # returning pack; Card 050 is NOT. The wheel note should call out both.
    seat = Seat(index=0, history_maxlen=5)
    full_pack = cube[:15]
    seat.add_pick(
        full_pack[0],
        PickRecord(
            round_no=0, pick_no=0, pack=[c.name for c in full_pack],
            chosen="Card 000", reasoning="early", hoping_to_wheel="Card 002",
        ),
    )
    seat.add_pick(
        cube[20],
        PickRecord(
            round_no=0, pick_no=1, pack=[c.name for c in cube[10:25]],
            chosen="Card 020", reasoning="next", hoping_to_wheel="Card 050",
        ),
    )
    wheeled_pack = cube[1:8]  # subset of pick 0's pack, contains Card 002
    agent = DraftAgent(StubLLM([_pick_reply("Card 003")]), DraftConfig())
    agent.pick(seat, wheeled_pack, round_no=0, pick_no=8)
    wheel_note = next(n for n in seat.memory if "[wheel" in n)
    assert "wheeled: Card 002" in wheel_note
    assert "did NOT wheel: Card 050" in wheel_note


def test_wheel_detection_skips_when_pack_is_not_a_subset(cube):
    # Current pack overlaps with an earlier seen pack but contains a card the
    # earlier pack did not have - not a wheel, no note. Guards against a
    # regression that mistakes overlap for subset.
    seat = Seat(index=0, history_maxlen=5)
    earlier_pack = cube[:5]
    seat.add_pick(
        earlier_pack[0],
        PickRecord(
            round_no=0,
            pick_no=0,
            pack=[c.name for c in earlier_pack],
            chosen="Card 000",
            reasoning="early",
        ),
    )
    not_a_wheel = [cube[1], cube[2], cube[10]]  # Card 010 was not in earlier_pack
    agent = DraftAgent(StubLLM([_pick_reply("Card 002")]), DraftConfig())
    agent.pick(seat, not_a_wheel, round_no=0, pick_no=7)
    assert not any("[wheel" in note for note in seat.memory)


def test_wheel_detection_skips_across_rounds(cube):
    # A pack 2 pick can't be a wheel of a pack 1 pack even if contents overlap.
    seat = Seat(index=0, history_maxlen=5)
    full_pack = cube[:15]
    seat.add_pick(
        full_pack[0],
        PickRecord(
            round_no=0, pick_no=0, pack=[c.name for c in full_pack], chosen="Card 000",
            reasoning="early",
        ),
    )
    pack = cube[1:8]
    agent = DraftAgent(StubLLM([_pick_reply("Card 002")]), DraftConfig())
    # round_no=1 (pack 2) - even though contents are a subset of an earlier record,
    # cross-round detection is suppressed.
    agent.pick(seat, pack, round_no=1, pick_no=3)
    assert not any("[wheel" in note for note in seat.memory)


@pytest.mark.parametrize(
    "raw_name,expected",
    [
        ("Card 004", "Card 004"),  # exact name in the pack
        ("card 04", "Card 004"),  # fuzzy match via difflib
        ("Nonexistent", None),  # unmatched name drops to None
        (None, None),  # picker omitted the field
    ],
    ids=["exact", "fuzzy", "unmatched", "omitted"],
)
def test_picker_hoping_to_wheel_resolves(cube, raw_name, expected):
    pack = cube[:5]
    seat = Seat(index=0, history_maxlen=5)
    agent = DraftAgent(
        StubLLM([_pick_reply("Card 002", hoping_to_wheel=raw_name)]), DraftConfig()
    )
    _, _, wheel_hope = agent.pick(seat, pack, round_no=0, pick_no=7)
    assert wheel_hope == expected


def test_picker_intent_sideboard_adds_card_to_sidelined(cube):
    pack = cube[:5]
    seat = Seat(index=0, history_maxlen=5)
    agent = DraftAgent(
        StubLLM([_pick_reply("Card 002", intent="sideboard")]), DraftConfig()
    )
    card, _, _ = agent.pick(seat, pack, round_no=0, pick_no=7)
    assert card is pack[2]
    assert "Card 002" in seat.sidelined


def test_picker_intent_maindeck_leaves_sidelined_alone(cube):
    pack = cube[:5]
    seat = Seat(index=0, history_maxlen=5)
    agent = DraftAgent(
        StubLLM([_pick_reply("Card 002", intent="maindeck")]), DraftConfig()
    )
    agent.pick(seat, pack, round_no=0, pick_no=7)
    assert seat.sidelined == set()


def test_strategist_notes_to_add_appended_to_memory(cube):
    seat = Seat(index=0, history_maxlen=5)
    strategist_json = (
        '{"directions": [{"colors": [], "role": "open", "weight": 9, '
        '"rationale": "early"}], "color_commitment": "open", '
        '"biggest_needs": ["raw power"], "watching_for": "bombs", '
        '"pool_classification": {"maindeck": [], "sideboard": [], "chaff": []}, '
        '"notes_to_add": ["R looks open", "watch UW pivot at pack 2"]}'
    )
    pack = cube[:5]
    agent = DraftAgent(StubLLM([strategist_json, _pick_reply("Card 002")]), DraftConfig())
    agent.pick(seat, pack, round_no=0, pick_no=1)
    notes_text = " | ".join(seat.memory)
    assert "R looks open" in notes_text
    assert "watch UW pivot" in notes_text
    assert all(n.startswith("[strat P1]") for n in seat.memory)


def test_pick_keeps_previous_strategy_when_strategist_reply_fails(cube):
    pack = cube[:5]
    seat = Seat(index=0, history_maxlen=5)
    # The first reply is the strategist call - we feed it garbage twice (parse + retry),
    # then a valid picker reply. The seat keeps its previous (None) plan.
    agent = DraftAgent(
        StubLLM(["not json", "still not json", _pick_reply("Card 001")]),
        DraftConfig(),
    )

    card, _, _ = agent.pick(seat, pack, round_no=0, pick_no=1)
    assert card is pack[1]
    assert seat.strategy_state is None

from mtg_drafting.agent import DraftAgent
from mtg_drafting.config import DraftConfig
from mtg_drafting.seat import Seat


class StubLLM:
    """Stand-in for LLMClient that returns canned replies in order, no network."""

    def __init__(self, replies: list[str]):
        self._replies = list(replies)

    def chat(self, messages, schema) -> str:
        return self._replies.pop(0)


def _reply(pick: str, reasoning: str = "ok", strategy: str = "be aggressive") -> str:
    return f'{{"pick": "{pick}", "reasoning": "{reasoning}", "strategy": "{strategy}"}}'


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
    raw = '{"pick": "Sol Ring", "reasoning": "Fast mana.", "strategy": "Ramp."}'
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
    assert parsed.strategy == ""


def test_parse_returns_none_when_pick_absent():
    assert DraftAgent._parse('{"reasoning": "incomplete reply with no pick yet') is None


def test_pick_returns_chosen_card_and_updates_strategy(cube):
    pack = cube[:5]
    seat = Seat(index=0, history_maxlen=5)
    agent = DraftAgent(StubLLM([_reply("Card 003")]), DraftConfig())

    card, reasoning = agent.pick(seat, pack, 0, 0)
    assert card is pack[3]
    assert reasoning == "ok"
    assert seat.strategy == "be aggressive"


def test_pick_retries_after_an_unusable_reply(cube):
    pack = cube[:5]
    seat = Seat(index=0, history_maxlen=5)
    agent = DraftAgent(StubLLM(['{"reasoning": "oops"', _reply("Card 001")]), DraftConfig())

    card, _ = agent.pick(seat, pack, 0, 0)
    assert card is pack[1]


def test_pick_falls_back_to_first_card_when_every_reply_fails(cube):
    pack = cube[:5]
    seat = Seat(index=0, history_maxlen=5)
    # DraftConfig default is one retry, so the agent makes two attempts.
    agent = DraftAgent(StubLLM([_reply("Nonexistent Card")] * 2), DraftConfig())

    card, reasoning = agent.pick(seat, pack, 0, 0)
    assert card is pack[0]
    assert "fallback" in reasoning


def test_pick_keeps_existing_strategy_when_reply_omits_one(cube):
    pack = cube[:5]
    seat = Seat(index=0, history_maxlen=5)
    seat.strategy = "existing plan"
    agent = DraftAgent(StubLLM([_reply("Card 002", strategy="")]), DraftConfig())

    agent.pick(seat, pack, 0, 0)
    assert seat.strategy == "existing plan"

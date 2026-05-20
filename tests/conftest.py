import pytest

from mtg_drafting.cards import Card

_COLOR_CYCLE = [["W"], ["U"], ["B"], ["R"], ["G"]]


class StubLLM:
    """Stand-in for ``LLMClient`` that returns canned replies in order.

    Each ``chat`` call pops the next reply from the queue; if the queue is empty
    the underlying ``list.pop`` raises ``IndexError``, which tests rely on as a
    structural assertion that "the agent did not call the LLM more times than I
    queued replies for"."""

    def __init__(self, replies: list[str]):
        self._replies = list(replies)

    def chat(self, messages, schema, *, num_predict=None) -> str:
        return self._replies.pop(0)


def make_card(i: int) -> Card:
    """Build a distinct synthetic card for tests, with no network access."""
    colors = _COLOR_CYCLE[i % 5]
    return Card(
        name=f"Card {i:03d}",
        mana_cost="{1}{C}",
        cmc=float(i % 7),
        type_line="Creature — Test",
        oracle_text="",
        colors=colors,
        color_identity=colors,
        power="2",
        toughness="2",
        rarity="common",
    )


@pytest.fixture
def cube() -> list[Card]:
    """A 400-card synthetic cube, large enough for a default 8-seat draft."""
    return [make_card(i) for i in range(400)]

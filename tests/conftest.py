import pytest

from mtg_drafting.cards import Card

_COLOR_CYCLE = [["W"], ["U"], ["B"], ["R"], ["G"]]


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

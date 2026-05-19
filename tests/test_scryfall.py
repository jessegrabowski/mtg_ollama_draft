from mtg_drafting.cards import Card
from mtg_drafting.scryfall import ScryfallIndex


def test_get_is_case_insensitive():
    index = ScryfallIndex([Card(name="Sol Ring")])
    assert index.get("sol ring").name == "Sol Ring"
    assert index.get("SOL RING").name == "Sol Ring"


def test_get_normalizes_curly_apostrophes():
    # The snapshot stores a straight apostrophe; a pasted cube list may use a curly one.
    index = ScryfallIndex([Card(name="Jace's Erasure")])
    assert index.get("Jace’s Erasure").name == "Jace's Erasure"


def test_get_resolves_double_faced_card_by_front_name():
    index = ScryfallIndex([Card(name="Fire // Ice")])
    assert index.get("Fire // Ice").name == "Fire // Ice"
    assert index.get("Fire").name == "Fire // Ice"


def test_get_returns_none_for_unknown_card():
    index = ScryfallIndex([Card(name="Sol Ring")])
    assert index.get("Black Lotus") is None

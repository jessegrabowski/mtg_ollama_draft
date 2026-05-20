import pytest

from mtg_drafting.cards import Card
from mtg_drafting.scryfall import ScryfallIndex


def test_get_is_case_insensitive():
    index = ScryfallIndex([Card(name="Sol Ring", type_line="Artifact")])
    assert index.get("sol ring").name == "Sol Ring"
    assert index.get("SOL RING").name == "Sol Ring"


def test_get_normalizes_curly_apostrophes():
    # The snapshot stores a straight apostrophe; a pasted cube list may use a curly one.
    index = ScryfallIndex([Card(name="Jace's Erasure", type_line="Sorcery")])
    assert index.get("Jace’s Erasure").name == "Jace's Erasure"


def test_get_normalizes_multiple_apostrophes_in_one_name():
    # Names like Yawgmoth's have just one apostrophe, but a curly-paste of
    # something like "Saint's Bell's" hits two replacements. Verifies the
    # normalize step handles repeats, not just the first.
    index = ScryfallIndex([Card(name="A's B's", type_line="Artifact")])
    assert index.get("A’s B’s").name == "A's B's"


def test_get_resolves_double_faced_card_by_front_name():
    index = ScryfallIndex([Card(name="Fire // Ice", type_line="Instant // Instant")])
    assert index.get("Fire // Ice").name == "Fire // Ice"
    assert index.get("Fire").name == "Fire // Ice"


def test_get_returns_none_for_unknown_card():
    index = ScryfallIndex([Card(name="Sol Ring", type_line="Artifact")])
    assert index.get("Black Lotus") is None


@pytest.mark.parametrize(
    "junk_first",
    [True, False],
    ids=["junk_then_real", "real_then_junk"],
)
def test_real_card_wins_over_placeholder_regardless_of_order(junk_first):
    # A junk entry (Mystery Booster playtest card or similar) with type_line "Card"
    # must not win the name slot - whether it loads before or after the real card.
    junk = Card(name="Inferno", type_line="Card")
    real = Card(name="Inferno", type_line="Instant", oracle_text="Deals 6 damage")
    cards = [junk, real] if junk_first else [real, junk]
    resolved = ScryfallIndex(cards).get("Inferno")
    assert resolved is not None
    assert resolved.type_line == "Instant"


def test_index_skips_empty_type_line():
    junk = Card(name="Empty", type_line="")
    index = ScryfallIndex([junk])
    assert index.get("Empty") is None


def test_index_skips_token_type_line():
    # Wilds of Eldraine Role enchantment tokens slip past the layout filter but
    # are typed "Token Enchantment ..." - the type_line filter catches them.
    token = Card(name="Royal // Young Hero", type_line="Token Enchantment // Token Enchantment")
    index = ScryfallIndex([token])
    assert index.get("Royal // Young Hero") is None
    assert index.get("Royal") is None


def test_real_standalone_beats_split_card_front_face_shortcut():
    # A standalone "Smelt" (Instant) competing with the front-face shortcut from a
    # split "Smelt // Herd // Saw". Two-pass registration guarantees the standalone
    # wins the "smelt" key regardless of which card loaded first.
    split = Card(
        name="Smelt // Herd // Saw",
        type_line="Instant // Sorcery // Sorcery",
        oracle_text="three pieces",
    )
    standalone = Card(name="Smelt", type_line="Instant", oracle_text="real Smelt")
    for order in ([split, standalone], [standalone, split]):
        index = ScryfallIndex(order)
        assert index.get("Smelt").type_line == "Instant"
        # The split is still reachable by its full name.
        assert index.get("Smelt // Herd // Saw") is not None

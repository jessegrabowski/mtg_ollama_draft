from collections import Counter

from mtg_drafting.cards import Card
from mtg_drafting.deckbuilder import (
    DECK_SIZE,
    MAX_LAND_COUNT,
    MIN_LAND_COUNT,
    allocate_basics,
    build_deck,
    count_pips,
)


def _spell(name: str, mana_cost: str, cmc: float) -> Card:
    return Card(
        name=name,
        mana_cost=mana_cost,
        cmc=cmc,
        type_line="Instant",
        colors=[c for c in "WUBRG" if c in mana_cost],
    )


def _nonbasic_land(name: str, produces: list[str]) -> Card:
    return Card(
        name=name,
        type_line="Land",
        color_identity=produces,
        produced_mana=produces,
    )


def test_count_pips_simple_mana_costs():
    spells = [_spell("UU", "{1}{U}{U}", 3), _spell("R", "{R}", 1)]
    assert count_pips(spells) == {"W": 0, "U": 2, "B": 0, "R": 1, "G": 0}


def test_count_pips_handles_hybrid_and_phyrexian():
    # {W/U} counts the first color (we keep the first half of the hybrid for the pip
    # tally); {U/P} counts U because phyrexian is the colored half.
    spells = [_spell("WU hybrid", "{W/U}", 1), _spell("phyrexian", "{U/P}", 1)]
    assert count_pips(spells) == {"W": 1, "U": 1, "B": 0, "R": 0, "G": 0}


def test_allocate_basics_proportional_two_colors():
    pips = {"W": 6, "U": 6, "B": 0, "R": 0, "G": 0}
    assert allocate_basics(pips, 16) == {"W": 8, "U": 8, "B": 0, "R": 0, "G": 0}


def test_allocate_basics_rounds_total_correctly():
    # 1 pip per color, 7 basics: each color gets 7/5 = 1.4 → 1 floor, 2 leftover
    # distributed by largest remainder. All ties → WUBRG order takes the first two.
    pips = {"W": 1, "U": 1, "B": 1, "R": 1, "G": 1}
    alloc = allocate_basics(pips, 7)
    assert sum(alloc.values()) == 7
    assert alloc == {"W": 2, "U": 2, "B": 1, "R": 1, "G": 1}


def test_allocate_basics_zero_pips_spreads_evenly():
    alloc = allocate_basics({c: 0 for c in "WUBRG"}, 7)
    assert sum(alloc.values()) == 7
    # 7 across 5: two colors get 2, three get 1.
    assert sorted(alloc.values()) == [1, 1, 1, 2, 2]


def _build_pool(n_spells: int = 25) -> list[Card]:
    """A pool of red spells (cheap to deep) plus a dual land."""
    spells = [_spell(f"Bolt {i}", "{R}", 1) for i in range(n_spells)]
    spells.append(_nonbasic_land("Steam Vents", ["U", "R"]))
    return spells


def test_build_deck_returns_exactly_40_cards():
    deck = build_deck(_build_pool(), [f"Bolt {i}" for i in range(23)], ["Steam Vents"], 17)
    assert len(deck.cards) == DECK_SIZE


def test_build_deck_clamps_low_land_count():
    deck = build_deck(_build_pool(), [f"Bolt {i}" for i in range(23)], [], 5)
    assert len(deck.basics) + len(deck.nonbasic_lands) == MIN_LAND_COUNT
    assert any("clamped" in note for note in deck.notes)


def test_build_deck_clamps_high_land_count():
    deck = build_deck(_build_pool(40), [f"Bolt {i}" for i in range(30)], [], 25)
    assert len(deck.basics) + len(deck.nonbasic_lands) == MAX_LAND_COUNT
    assert any("clamped" in note for note in deck.notes)


def test_build_deck_trims_over_budget_spells():
    # Asks for 30 spells but only 23 slots fit at land count 17.
    deck = build_deck(_build_pool(35), [f"Bolt {i}" for i in range(30)], [], 17)
    assert len(deck.spells) == DECK_SIZE - 17
    assert any("trimmed" in note for note in deck.notes)


def test_build_deck_pads_under_budget_spells():
    # Asks for only 5 spells; budget is 23, so the builder pads 18 more from the pool.
    deck = build_deck(_build_pool(30), [f"Bolt {i}" for i in range(5)], [], 17)
    assert len(deck.spells) == 23
    assert any("padded" in note for note in deck.notes)


def test_build_deck_resolves_fuzzy_names():
    # Real typo: "bolt0" (missing space) bypasses the lowercase exact lookup and forces
    # the difflib fuzzy fallback.
    deck = build_deck(_build_pool(), ["bolt0"], [], 17)
    assert any(c.name == "Bolt 0" for c in deck.spells)


def test_build_deck_drops_unresolved_names():
    deck = build_deck(_build_pool(), ["Nonexistent Spell"], [], 17)
    assert any("unresolved" in n and "Nonexistent Spell" in n for n in deck.notes)


def test_build_deck_basic_distribution_matches_pip_share():
    # Mono-red deck: every basic should be a Mountain.
    deck = build_deck(_build_pool(), [f"Bolt {i}" for i in range(23)], [], 17)
    assert all(c.name == "Mountain" for c in deck.basics)


def test_build_deck_basic_split_reflects_two_color_pip_share():
    # 5 white pips : 20 red pips → 1:4 share over 15 basics = 3 Plains, 12 Mountain.
    # End-to-end check that pip counting actually drives the basic allocation in
    # build_deck (not just in allocate_basics in isolation).
    pool = [_spell(f"W{i}", "{W}", 1) for i in range(5)] + [
        _spell(f"R{i}", "{R}", 1) for i in range(20)
    ]
    deck = build_deck(pool, [c.name for c in pool], [], 15)
    assert Counter(c.name for c in deck.basics) == {"Plains": 3, "Mountain": 12}


def test_build_deck_includes_chosen_nonbasic_lands():
    deck = build_deck(_build_pool(), [f"Bolt {i}" for i in range(23)], ["Steam Vents"], 17)
    assert "Steam Vents" in {c.name for c in deck.nonbasic_lands}
    assert "Steam Vents" in {c.name for c in deck.cards}


def test_build_deck_padding_prefers_higher_cmc():
    # 30 spells with distinct CMCs 1..30. Asking for 5 low-CMC spells leaves 18 slots
    # for padding; the contract is "strongest by descending CMC".
    pool = [_spell(f"Spell{i:02d}", "{R}", float(i)) for i in range(1, 31)]
    chosen = [f"Spell{i:02d}" for i in range(1, 6)]
    deck = build_deck(pool, chosen, [], 17)
    pad_cmcs = [c.cmc for c in deck.spells[5:]]
    assert pad_cmcs == sorted(pad_cmcs, reverse=True)
    # The 18 padded spells should be exactly CMCs 13..30.
    assert set(pad_cmcs) == set(range(13, 31))


def test_build_deck_trims_nonbasics_over_budget():
    # 18 dual lands offered but land count is 15 → trim down to 15 nonbasics.
    duals = [_nonbasic_land(f"Dual {i}", ["U", "R"]) for i in range(18)]
    pool = duals + [_spell(f"Bolt {i}", "{R}", 1) for i in range(25)]
    deck = build_deck(pool, [f"Bolt {i}" for i in range(25)], [f"Dual {i}" for i in range(18)], 15)
    assert len(deck.nonbasic_lands) == 15
    assert len(deck.basics) == 0

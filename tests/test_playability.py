import random

from mtg_drafting.cards import Card
from mtg_drafting.deckbuilder import BASIC_LANDS
from mtg_drafting.playability import (
    _opening_hand,
    can_pay,
    generate_sample_hands,
    parse_cost,
    peak_color_requirement,
    simulate_deck,
)


def _spell(name: str, mana_cost: str, cmc: float) -> Card:
    return Card(
        name=name,
        mana_cost=mana_cost,
        cmc=cmc,
        type_line="Instant",
        colors=[c for c in "WUBRG" if c in mana_cost],
    )


def test_parse_cost_generic_and_colored():
    colored, generic = parse_cost("{2}{U}{U}")
    assert generic == 2
    assert colored == [frozenset({"U"}), frozenset({"U"})]


def test_parse_cost_hybrid_symbol_keeps_both_colors():
    colored, _ = parse_cost("{W/U}")
    assert colored == [frozenset({"W", "U"})]


def test_parse_cost_phyrexian_counts_as_the_colored_half():
    colored, _ = parse_cost("{U/P}")
    assert colored == [frozenset({"U"})]


def test_parse_cost_X_contributes_zero():
    colored, generic = parse_cost("{X}{R}")
    assert generic == 0
    assert colored == [frozenset({"R"})]


def test_parse_cost_unknown_symbol_falls_back_to_generic():
    # {C} colorless and {S} snow are read pessimistically as generic.
    colored, generic = parse_cost("{C}{S}")
    assert generic == 2
    assert colored == []


def test_can_pay_single_source_single_symbol():
    assert can_pay([frozenset({"R"})], [frozenset({"R"})], 0)
    assert not can_pay([frozenset({"R"})], [frozenset({"U"})], 0)


def test_can_pay_multicolor_land_can_pay_either_color():
    # A U/R dual can pay {U} or {R} (but only one per turn).
    dual = frozenset({"U", "R"})
    assert can_pay([dual], [frozenset({"U"})], 0)
    assert can_pay([dual], [frozenset({"R"})], 0)


def test_can_pay_one_land_cannot_pay_two_colored_symbols():
    # {U}{R} from a single U/R dual is impossible - one land taps once per turn.
    dual = frozenset({"U", "R"})
    assert not can_pay([dual], [frozenset({"U"}), frozenset({"R"})], 0)


def test_can_pay_insufficient_total_mana():
    # Need 3 mana but only 2 lands.
    assert not can_pay([frozenset({"R"}), frozenset({"R"})], [frozenset({"R"})], 2)


def test_can_pay_hybrid_lets_either_source_pay():
    # {W/U} can be paid by an Island OR a Plains.
    sym = frozenset({"W", "U"})
    assert can_pay([frozenset({"W"})], [sym], 0)
    assert can_pay([frozenset({"U"})], [sym], 0)


def test_can_pay_hybrid_assignment_with_mixed_color_sources():
    # Multi-symbol hybrid: the backtracker has to assign each {W/U} to a distinct land
    # whose colors overlap. A single color land covers at most one symbol.
    sym = frozenset({"W", "U"})
    assert can_pay([frozenset({"U"}), frozenset({"U"})], [sym, sym], 0)
    assert can_pay([frozenset({"W"}), frozenset({"U"})], [sym, sym], 0)
    assert not can_pay([frozenset({"R"})], [sym], 0)
    # Plains + Mountain: Plains covers one {W/U}, Mountain covers neither.
    assert not can_pay([frozenset({"W"}), frozenset({"R"})], [sym, sym], 0)


def _build_deck(spells: list[Card], basic_color: str, n_basics: int = 17) -> list[Card]:
    return spells + [BASIC_LANDS[basic_color]] * n_basics


def test_mono_color_deck_casts_one_drops_reliably():
    spells = [_spell(f"Bolt {i}", "{R}", 1) for i in range(23)]
    deck = _build_deck(spells, "R")
    report = simulate_deck(deck, sims=200, seed=42)
    on_curve = [c.castable_on_curve for c in report.per_card]
    # With 17/40 Mountains, every 1-drop is castable by T2 essentially always.
    assert min(on_curve) > 0.95


def test_wrong_color_deck_never_casts():
    # 17 Plains + 23 {R} 1-drops - Plains cannot produce red, so nothing is castable.
    spells = [_spell(f"Bolt {i}", "{R}", 1) for i in range(23)]
    deck = _build_deck(spells, "W")
    report = simulate_deck(deck, sims=200, seed=42)
    assert all(c.castable_by_horizon == 0 for c in report.per_card)
    assert set(report.stranded) == {s.name for s in spells}


def test_two_color_deck_handles_double_pips():
    # {U}{U} double-blue 2-drops cast far less reliably from a 9/8 U/W split base than
    # from a pure Island base. Comparing the two configurations is a stronger signal
    # than asserting an absolute range, and it doesn't depend on the seed.
    spells = [_spell(f"Counter {i}", "{U}{U}", 2) for i in range(23)]
    pure = spells + [BASIC_LANDS["U"]] * 17
    split = spells + [BASIC_LANDS["U"]] * 9 + [BASIC_LANDS["W"]] * 8

    pure_report = simulate_deck(pure, sims=300, seed=42)
    split_report = simulate_deck(split, sims=300, seed=42)

    pure_avg = sum(c.castable_on_curve for c in pure_report.per_card) / len(pure_report.per_card)
    split_avg = sum(c.castable_on_curve for c in split_report.per_card) / len(
        split_report.per_card
    )
    assert pure_avg > 0.95
    assert pure_avg - split_avg > 0.15


def test_simulate_records_per_turn_curve_hits():
    spells = [_spell(f"Bolt {i}", "{R}", 1) for i in range(23)]
    deck = _build_deck(spells, "R")
    report = simulate_deck(deck, sims=200, seed=42)
    # Deck only has 1-drops, so curve_hit only spans T1.
    assert set(report.curve_hit) == {1}
    assert report.curve_hit[1] > 0.95


def test_simulate_skips_curves_without_spells_at_that_cmc():
    # Deck has only 2-drops, so per-turn curve hit is reported for T2 only.
    spells = [_spell(f"Two {i}", "{1}{R}", 2) for i in range(23)]
    deck = _build_deck(spells, "R")
    report = simulate_deck(deck, sims=200, seed=42)
    assert set(report.curve_hit) == {2}


def test_opening_hand_mulligans_when_opener_is_outside_land_range():
    # All-Mountain deck: any 7-card opener is out of the [2, 5] land range and must
    # mulligan; the implementation does a single mulligan to 6.
    all_lands = [BASIC_LANDS["R"]] * 40
    hand, library = _opening_hand(all_lands, random.Random(0))
    assert len(hand) == 6
    assert len(hand) + len(library) == 40


def test_opening_hand_keeps_in_range_opener():
    # A normal 17-land deck has a comfortable land distribution; almost every shuffle
    # produces an opener with 2-5 lands, so the keep should be 7.
    deck = [_spell(f"X {i}", "{R}", 1) for i in range(23)] + [BASIC_LANDS["R"]] * 17
    sizes = [len(_opening_hand(deck, random.Random(seed))[0]) for seed in range(50)]
    # Some seeds will roll an extreme opener and mulligan; the vast majority will not.
    assert sum(s == 7 for s in sizes) > 40


def _interaction_spell(name: str) -> Card:
    return Card(
        name=name,
        mana_cost="{R}",
        cmc=1.0,
        type_line="Instant",
        colors=["R"],
        oracle_text="Deals 3 damage to any target.",
    )


def _flyer(name: str, power: str = "2") -> Card:
    return Card(
        name=name,
        mana_cost="{1}{R}",
        cmc=2.0,
        type_line="Creature — Bird",
        colors=["R"],
        power=power,
        toughness="1",
        oracle_text="Flying",
    )


def test_simulate_tracks_interaction_by_turn():
    deck = [_interaction_spell(f"Bolt {i}") for i in range(23)] + [BASIC_LANDS["R"]] * 17
    report = simulate_deck(deck, sims=300, seed=42)
    # Interaction availability grows with each draw step; T8 must exceed T1.
    assert report.interaction_by_turn[8] > report.interaction_by_turn[1]
    # By T8 the average drawn-and-castable count should be meaningful (~8 bolts).
    assert report.interaction_by_turn[8] > 4


def test_simulate_tracks_creature_power_by_turn():
    creatures = [_flyer(f"Bird {i}", power="3") for i in range(23)]
    deck = creatures + [BASIC_LANDS["R"]] * 17
    report = simulate_deck(deck, sims=300, seed=42)
    # Sum of power scales linearly with drawn count: T8 has more board than T1.
    assert report.creature_power_by_turn[8] > report.creature_power_by_turn[1]
    # Each flyer is 3 power; by T8 average should be > 10 (a few castable flyers).
    assert report.creature_power_by_turn[8] > 10


def test_simulate_tracks_evasion_by_turn():
    flyers = [_flyer(f"Bird {i}") for i in range(23)]
    deck = flyers + [BASIC_LANDS["R"]] * 17
    report = simulate_deck(deck, sims=300, seed=42)
    assert report.evasion_by_turn[8] > 3


def test_simulate_reports_mulligan_rate_for_all_land_deck():
    # All-land deck mulligans every game because the opener has 7 lands, well above
    # the [2, 5] keep window. Each game mulligans to 6, so the rate is 1.0 and the
    # average opening size is 6.
    deck = [BASIC_LANDS["R"]] * 40
    report = simulate_deck(deck, sims=100, seed=42)
    assert report.mulligan_rate == 1.0
    assert report.average_opening_hand_size == 6.0


def test_simulate_normal_deck_rarely_mulligans():
    spells = [_spell(f"X {i}", "{R}", 1) for i in range(23)]
    deck = spells + [BASIC_LANDS["R"]] * 17
    report = simulate_deck(deck, sims=300, seed=42)
    # Hypergeometric for 17 lands in 40 gives an opener of 0-1 or 6-7 lands about
    # 12% of the time; mulligan rate should land near that, not anywhere near the
    # all-land deck's 100%.
    assert report.mulligan_rate < 0.20
    assert report.average_opening_hand_size > 6.8


def test_castable_hand_fraction_lower_for_too_expensive_deck():
    # 23 five-drops + 17 Mountains: nothing in hand is castable until T5, so the
    # fraction is near zero on early turns. Compare against an all-1-drop deck whose
    # hand stays mostly castable from T1 onwards.
    expensive = [_spell(f"Big {i}", "{4}{R}", 5) for i in range(23)]
    cheap = [_spell(f"Bolt {i}", "{R}", 1) for i in range(23)]
    expensive_deck = expensive + [BASIC_LANDS["R"]] * 17
    cheap_deck = cheap + [BASIC_LANDS["R"]] * 17

    expensive_report = simulate_deck(expensive_deck, sims=300, seed=42)
    cheap_report = simulate_deck(cheap_deck, sims=300, seed=42)

    # Cheap deck on T2 has many castable cards in hand; expensive deck has none.
    assert cheap_report.castable_hand_fraction_by_turn[2] > 0.3
    assert expensive_report.castable_hand_fraction_by_turn[2] < 0.05


def test_generate_sample_hands_is_deterministic():
    deck = [_spell(f"X {i}", "{R}", 1) for i in range(23)] + [BASIC_LANDS["R"]] * 17
    first = generate_sample_hands(deck, n=2, seed=42)
    second = generate_sample_hands(deck, n=2, seed=42)
    assert [h.opening for h in first] == [h.opening for h in second]
    assert [h.upcoming_draws for h in first] == [h.upcoming_draws for h in second]


def test_generate_sample_hands_returns_requested_count_and_draws():
    deck = [_spell(f"X {i}", "{R}", 1) for i in range(23)] + [BASIC_LANDS["R"]] * 17
    hands = generate_sample_hands(deck, n=3, draws_per_hand=4, seed=0)
    assert len(hands) == 3
    for h in hands:
        assert len(h.opening) in (6, 7)
        assert len(h.upcoming_draws) == 4
        assert h.mulliganed == (len(h.opening) == 6)


def test_generate_sample_hands_mulligans_on_all_land_deck():
    deck = [BASIC_LANDS["R"]] * 40
    hands = generate_sample_hands(deck, n=2, seed=0)
    assert all(h.mulliganed and len(h.opening) == 6 for h in hands)


def test_peak_color_requirement_picks_heaviest_single_color_pip():
    deck = [
        Card(name="Sengir Vampire", mana_cost="{3}{B}{B}{B}", cmc=6, type_line="Creature"),
        Card(name="Counterspell", mana_cost="{U}{U}", cmc=2, type_line="Instant"),
        Card(name="Bant Charm", mana_cost="{G}{W}{U}", cmc=3, type_line="Instant"),
    ]
    assert peak_color_requirement(deck) == ("B", 3, "Sengir Vampire")


def test_peak_color_requirement_none_for_colorless_deck():
    deck = [Card(name="Sol Ring", mana_cost="{1}", cmc=1, type_line="Artifact")] * 5
    assert peak_color_requirement(deck) is None


def test_simulate_lands_and_spells_in_hand():
    # Balanced 17-land mono-red deck. After the first land drop on T1, the average
    # opener of ~3 lands + ~4 spells becomes ~2 lands + ~4 spells (one land moves to
    # the battlefield).
    deck = [_spell(f"X {i}", "{R}", 1) for i in range(23)] + [BASIC_LANDS["R"]] * 17
    report = simulate_deck(deck, sims=300, seed=42)
    # Lands in hand should be well below spells in hand at T1.
    assert report.lands_in_hand_by_turn[1] < report.nonland_in_hand_by_turn[1]
    # Total in hand at T1 ≈ opening size - 1 land drop, around 5.5-6.
    total_t1 = report.lands_in_hand_by_turn[1] + report.nonland_in_hand_by_turn[1]
    assert 5.0 < total_t1 < 6.5


def test_simulate_peak_color_hit_reaches_target():
    # Mono-black deck with Sengir Vampire ({3}{B}{B}{B}): peak target is 3 Swamps.
    sengir = Card(name="Sengir", mana_cost="{3}{B}{B}{B}", cmc=6, type_line="Creature")
    spells = [
        Card(name=f"Fil {i}", mana_cost="{1}{B}", cmc=2, type_line="Instant")
        for i in range(22)
    ]
    deck = [sengir, *spells, *[BASIC_LANDS["B"]] * 17]
    report = simulate_deck(deck, sims=300, seed=42)
    assert report.peak_color_requirement == ("B", 3, "Sengir")
    # 3 Swamps by T3 is impossible (only 3 lands max), and unreliable; by T8 it
    # should be nearly automatic with 17 Swamps in the deck.
    assert report.peak_color_hit_by_turn[3] < report.peak_color_hit_by_turn[8]
    assert report.peak_color_hit_by_turn[8] > 0.8


def test_simulate_peak_color_hit_low_for_wrong_color_base():
    # Sengir in a deck of Plains - the peak target is 3 Swamps but there are zero.
    sengir = Card(name="Sengir", mana_cost="{3}{B}{B}{B}", cmc=6, type_line="Creature")
    spells = [Card(name=f"X {i}", mana_cost="{1}", cmc=2, type_line="Instant") for i in range(22)]
    deck = [sengir, *spells, *[BASIC_LANDS["W"]] * 17]
    report = simulate_deck(deck, sims=200, seed=42)
    assert all(rate == 0 for rate in report.peak_color_hit_by_turn.values())


def test_simulate_zero_metrics_for_deck_without_those_categories():
    # Plain creatures, no evasion or interaction; both series should be flat at zero.
    plain = [
        Card(
            name=f"Bear {i}",
            mana_cost="{R}",
            cmc=1.0,
            type_line="Creature — Bear",
            colors=["R"],
            power="2",
            toughness="2",
        )
        for i in range(23)
    ]
    deck = plain + [BASIC_LANDS["R"]] * 17
    report = simulate_deck(deck, sims=200, seed=42)
    assert all(v == 0 for v in report.interaction_by_turn.values())
    assert all(v == 0 for v in report.evasion_by_turn.values())
    # Power should be non-zero (these are 2/2 creatures).
    assert report.creature_power_by_turn[8] > 0

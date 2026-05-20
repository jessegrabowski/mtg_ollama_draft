from mtg_drafting.cards import Card
from mtg_drafting.deck_stats import SPLASH_THRESHOLD, compute_stats, render_stats
from mtg_drafting.deckbuilder import build_deck


def _spell(name: str, mana_cost: str, cmc: float, type_line: str = "Instant") -> Card:
    return Card(
        name=name,
        mana_cost=mana_cost,
        cmc=cmc,
        type_line=type_line,
        colors=[c for c in "WUBRG" if c in mana_cost],
    )


def _creature(name: str, mana_cost: str, cmc: float, power: str, oracle: str = "") -> Card:
    return Card(
        name=name,
        mana_cost=mana_cost,
        cmc=cmc,
        type_line="Creature — Test",
        colors=[c for c in "WUBRG" if c in mana_cost],
        power=power,
        toughness="1",
        oracle_text=oracle,
    )


def _nonbasic_land(name: str, produces: list[str]) -> Card:
    return Card(
        name=name, type_line="Land", color_identity=produces, produced_mana=produces
    )


def test_compute_stats_basic_totals():
    pool = [_spell(f"R{i}", "{R}", 1) for i in range(25)]
    deck = build_deck(pool, [c.name for c in pool[:23]], [], 17)
    stats = compute_stats(deck)
    assert stats.total_cards == 40
    assert stats.cmc_distribution[1] == 23
    assert stats.average_cmc == 1.0


def test_compute_stats_mana_sources_from_basics_and_duals():
    # 23 red spells + 1 U/R dual + 16 basic Mountains, 17 lands total. Sources: R=17,
    # U=1 (from the dual). U is below the 4-source splash threshold.
    pool = [_spell(f"R{i}", "{R}", 1) for i in range(25)] + [
        _nonbasic_land("Steam Vents", ["U", "R"])
    ]
    deck = build_deck(pool, [c.name for c in pool[:23]], ["Steam Vents"], 17)
    stats = compute_stats(deck)
    assert stats.mana_sources["R"] == 17
    assert stats.mana_sources["U"] == 1
    assert "U" in stats.splash_colors


def test_compute_stats_splash_empty_for_solid_mono_color():
    pool = [_spell(f"R{i}", "{R}", 1) for i in range(25)]
    deck = build_deck(pool, [c.name for c in pool[:23]], [], 17)
    stats = compute_stats(deck)
    assert stats.splash_colors == []
    assert stats.mana_sources["R"] >= SPLASH_THRESHOLD


def test_compute_stats_type_counts_split_by_type_line():
    pool = (
        [_creature(f"C{i}", "{R}", 1, "1") for i in range(15)]
        + [_spell(f"I{i}", "{R}", 1, type_line="Instant") for i in range(5)]
        + [_spell(f"S{i}", "{R}", 1, type_line="Sorcery") for i in range(3)]
    )
    deck = build_deck(pool, [c.name for c in pool[:23]], [], 17)
    stats = compute_stats(deck)
    assert stats.type_counts["Creature"] == 15
    assert stats.type_counts["Instant"] == 5
    assert stats.type_counts["Sorcery"] == 3
    assert stats.type_counts["Land"] == 17


def test_compute_stats_creature_power_total():
    pool = [_creature(f"C{i}", "{R}", 1, "3") for i in range(25)]
    deck = build_deck(pool, [c.name for c in pool[:23]], [], 17)
    stats = compute_stats(deck)
    assert stats.creature_power_total == 23 * 3


def test_compute_stats_counts_tagged_categories():
    interaction = [
        Card(
            name=f"Bolt{i}",
            mana_cost="{R}",
            cmc=1.0,
            type_line="Instant",
            colors=["R"],
            oracle_text="Deals 3 damage to any target.",
        )
        for i in range(5)
    ]
    creatures = [_creature(f"Bear{i}", "{R}", 1, "2") for i in range(18)]
    pool = interaction + creatures
    deck = build_deck(pool, [c.name for c in pool], [], 17)
    stats = compute_stats(deck)
    assert stats.interaction_count == 5
    # No ramp or evasion was added.
    assert stats.ramp_count == 0
    assert stats.evasion_count == 0


def test_render_stats_is_a_single_block_of_text():
    pool = [_spell(f"R{i}", "{R}", 1) for i in range(25)]
    deck = build_deck(pool, [c.name for c in pool[:23]], [], 17)
    rendered = render_stats(compute_stats(deck))
    # The rendering is consumed by both humans and the LLM prompt, so it should at
    # least mention the core stat sections.
    for section in ("Mana curve", "Average CMC", "Types", "Mana sources", "Splash"):
        assert section in rendered

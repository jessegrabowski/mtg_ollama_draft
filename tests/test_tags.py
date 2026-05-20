from mtg_drafting.cards import Card
from mtg_drafting.tags import effective_power, is_evasive, is_interaction, is_ramp, tag


def _creature(name: str, power: str | None, oracle_text: str = "") -> Card:
    return Card(
        name=name,
        type_line="Creature — Test",
        mana_cost="{1}",
        cmc=1.0,
        power=power,
        toughness="1" if power is not None else None,
        oracle_text=oracle_text,
    )


def _spell(name: str, type_line: str, oracle_text: str) -> Card:
    return Card(name=name, type_line=type_line, mana_cost="{R}", cmc=1.0, oracle_text=oracle_text)


def test_is_interaction_targeted_removal():
    bolt = _spell("Lightning Bolt", "Instant", "Lightning Bolt deals 3 damage to any target.")
    assert is_interaction(bolt)


def test_is_interaction_counterspell():
    cs = _spell("Counterspell", "Instant", "Counter target spell.")
    assert is_interaction(cs)


def test_is_interaction_sweeper_without_target_keyword():
    wrath = _spell("Wrath of God", "Sorcery", "Destroy all creatures. They can't be regenerated.")
    assert is_interaction(wrath)


def test_is_interaction_bounce():
    unsummon = _spell("Unsummon", "Instant", "Return target creature to its owner's hand.")
    assert is_interaction(unsummon)


def test_is_interaction_skips_lands():
    land = Card(name="Plains", type_line="Basic Land — Plains", oracle_text="({T}: Add {W}.)")
    assert not is_interaction(land)


def test_is_interaction_skips_vanilla_creature():
    crow = _creature("Storm Crow", "1", "Flying")
    assert not is_interaction(crow)


def test_is_interaction_skips_target_draw_clause():
    # "target opponent draws a card" - has the word "target" but no removal verb.
    weird = _spell("Helpful Cantrip", "Instant", "Target opponent draws a card.")
    assert not is_interaction(weird)


def test_is_evasive_flying():
    assert is_evasive(_creature("Storm Crow", "1", "Flying"))


def test_is_evasive_menace():
    assert is_evasive(_creature("Goblin Bruiser", "2", "Menace"))


def test_is_evasive_trample():
    assert is_evasive(_creature("Wurm", "5", "Trample"))


def test_is_evasive_cant_be_blocked():
    text = "Slither Blade can't be blocked except by two or more creatures."
    assert is_evasive(_creature("Slither Blade", "1", text))


def test_is_evasive_skips_grounded_creature():
    assert not is_evasive(_creature("Grizzly Bears", "2", ""))


def test_is_evasive_skips_non_creature():
    spell = _spell("Lightning Bolt", "Instant", "Deals 3 damage to any target with flying.")
    # "Flying" appears in the oracle but the card isn't a creature - it's not evasion.
    assert not is_evasive(spell)


def test_is_ramp_mana_dork():
    dork = _creature("Llanowar Elves", "1", "{T}: Add {G}.")
    assert is_ramp(dork)


def test_is_ramp_mana_rock():
    rock = _spell("Sol Ring", "Artifact", "{T}: Add {C}{C}.")
    # {C} isn't WUBRG but the "Add {C}" pattern still matches the ramp regex.
    assert is_ramp(rock)


def test_is_ramp_land_fetch():
    cultivate = _spell(
        "Cultivate",
        "Sorcery",
        "Search your library for up to two basic land cards, reveal them, "
        "put one onto the battlefield tapped and the other into your hand.",
    )
    assert is_ramp(cultivate)


def test_is_ramp_skips_non_ramp_spell():
    bolt = _spell("Lightning Bolt", "Instant", "Lightning Bolt deals 3 damage to any target.")
    assert not is_ramp(bolt)


def test_is_ramp_skips_basics():
    plains = Card(name="Plains", type_line="Basic Land — Plains", oracle_text="({T}: Add {W}.)")
    assert not is_ramp(plains)


def test_effective_power_creature():
    assert effective_power(_creature("Bear", "3", "")) == 3


def test_effective_power_non_creature_is_zero():
    assert effective_power(_spell("Bolt", "Instant", "")) == 0


def test_effective_power_star_is_zero():
    # "*" creatures (Tarmogoyf, Mortivore) have non-numeric power.
    assert effective_power(_creature("Goyf", "*", "")) == 0


def test_tag_bundles_all_classifications():
    crow = _creature("Storm Crow", "1", "Flying")
    tags = tag(crow)
    assert tags.name == "Storm Crow"
    assert tags.is_evasive
    assert not tags.is_interaction
    assert not tags.is_ramp
    assert tags.effective_power == 1

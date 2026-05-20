from mtg_drafting.evaluate import CutOrKeep, DeckVerdict, TurnPlay, verdict_validation_failures


def _verdict(
    *,
    rating_justification: str = "Tier 6-7 solid; Sengir Vampire is the closer.",
    win_condition: str = "Sengir Vampire on T6.",
    best_card: str = "Sengir Vampire wins games on its own.",
    worst_card: str = "Goblin Filler is a 1/1 vanilla.",
    cut_or_keep_cards: tuple[str, ...] = ("Sengir Vampire", "Counterspell", "Goblin Filler"),
) -> DeckVerdict:
    return DeckVerdict(
        rating=7,
        rating_justification=rating_justification,
        win_condition=win_condition,
        best_card=best_card,
        worst_card=worst_card,
        cut_or_keep=[
            CutOrKeep(card=name, decision="keep", reason="example")
            for name in cut_or_keep_cards
        ],
        hardest_game_state="aggro on T3 outraces Sengir Vampire.",
        sample_hand_plan=[
            TurnPlay(turn=i, plays=f"Land {i}", rationale="example") for i in range(1, 5)
        ],
        comparable_archetype="mono-black midrange",
    )


_DECK_CARDS = {"Sengir Vampire", "Counterspell", "Goblin Filler"}


def test_validator_passes_when_every_field_names_a_deck_card():
    assert verdict_validation_failures(_verdict(), _DECK_CARDS) == []


def test_validator_flags_field_without_card_reference():
    bad = _verdict(rating_justification="Solid deck, plays out fine, reasonable curve.")
    failures = verdict_validation_failures(bad, _DECK_CARDS)
    assert any("rating_justification" in f for f in failures)


def test_validator_flags_cut_or_keep_with_unknown_card():
    bad = _verdict(cut_or_keep_cards=("Sengir Vampire", "Lightning Bolt", "Counterspell"))
    failures = verdict_validation_failures(bad, _DECK_CARDS)
    assert any("Lightning Bolt" in f for f in failures)


def test_validator_is_case_insensitive():
    bad = _verdict(best_card="my best card is sengir vampire honestly.")
    assert verdict_validation_failures(bad, _DECK_CARDS) == []


def test_validator_allows_win_condition_without_named_card():
    # win_condition is excluded from the strict card-citation check so the model can
    # honestly say there is no real closer.
    bad = _verdict(win_condition="no real finisher; plans to grind out with cheap removal.")
    assert verdict_validation_failures(bad, _DECK_CARDS) == []


def test_validator_collects_all_failures():
    bad = _verdict(
        rating_justification="A coherent deck.",
        best_card="The removal pile.",
        worst_card="Filler.",
    )
    failures = verdict_validation_failures(bad, _DECK_CARDS)
    # Three card-required fields (rating_justification, best_card, worst_card) flag.
    assert len(failures) == 3

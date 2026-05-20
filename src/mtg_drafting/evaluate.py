import json
from dataclasses import asdict
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError
from rich.console import Console

from mtg_drafting.cards import Card
from mtg_drafting.config import LLMConfig, Paths
from mtg_drafting.deck_stats import DeckStats, compute_stats, render_stats
from mtg_drafting.deckbuilder import BuiltDeck, build_deck
from mtg_drafting.llm import LLMClient
from mtg_drafting.playability import (
    PlayabilityReport,
    SampleHand,
    generate_sample_hands,
    simulate_deck,
)
from mtg_drafting.records import load_draft
from mtg_drafting.scryfall import load_index
from mtg_drafting.tags import is_evasive, is_interaction, is_ramp

# Lower temperature than the draft default - we want a sharp opinion, not a smoothed
# polite one. RLHF-trained models otherwise cluster ratings around 7/10 regardless of
# input ("the central tendency bias"); a tighter sampler plus an explicit rubric plus a
# required justification spread the distribution.
_EVAL_TEMPERATURE = 0.3

# The evaluator schemas are large: the build proposal lists ~23 spell names (~300
# tokens), the verdict has nine fields including 3 cut-or-keep entries and 4 sample-
# hand-plan entries (~500-600 tokens of JSON). The 512-token default for picks would
# truncate either reply mid-JSON. Evaluation runs once per seat so the extra tokens
# are cheap.
_EVAL_NUM_PREDICT = 2048

_BUILD_SYSTEM = (
    "You are an expert Magic: The Gathering cube analyst building a deck from a player's "
    "draft pool. Identify the strongest archetype and choose the spells and any nonbasic "
    "lands you want to play. Basic lands are filled in automatically from your spell "
    "color pips - you only need to suggest the total land count (15-19).\n\n"
    "Pick ~21-25 spells. Card names must match the pool verbatim. Do not include basic "
    "lands; they are added by the deckbuilder."
)

_VERDICT_SYSTEM = (
    "You are an expert Magic: The Gathering cube analyst rating a finished 40-card deck. "
    "You will see the player's pool, your own chosen build (each card annotated with its "
    "role tags and on-curve castability), deterministic deck statistics, a Monte Carlo "
    "playability report from 1000 simulated games, and one concrete sample opening hand "
    "to walk through. Use those objective measurements to anchor a candid rating.\n\n"
    "RATING RUBRIC (cite the tier in your justification):\n"
    "  1-2  Unplayable. No archetype, no win plan, the mana base actually does not "
    "function (curve-hit below 30% across most turns).\n"
    "  3-4  Severely flawed. Missing two or more key roles (no removal AND no closers, "
    "or wrong colors), or multiple stranded cards (>= 3).\n"
    "  5-6  Average cube deck. Coherent archetype, normal mana, plays a real game.\n"
    "  7-8  Strong. Multiple bombs, real synergy, premium removal, well-fixed mana.\n"
    "  9-10 Pristine. Every slot earns its place, perfect mana, top-tier threats.\n\n"
    "In a real cube draft most decks land 5-7. Rate below 5 only with a concrete "
    "structural flaw, not a mildly weak measurement.\n\n"
    "STRICT OUTPUT RULES:\n"
    "- Every answer must name specific cards from the deck. Generic claims about "
    "'the curve', 'the colors', or 'the strategy' are wrong.\n"
    "- Banned words: 'solid', 'reasonable', 'coherent', 'plays through', 'plays out', "
    "'playable archetype', 'good mix', 'well-rounded'. Strike them from your reply.\n"
    "- Use the role tags shown next to each card in BUILT DECK (creature, evasion, "
    "interaction, ramp). Do not invent your own descriptors like 'high-toughness "
    "threat' or 'big beater' - say 'evasive creature' if its tag is evasion.\n"
    "- Cite numbers verbatim from DECK STATISTICS or PLAYABILITY SIMULATION. Do not "
    "invent source counts, percentages, or pip totals. If a claim needs a number, the "
    "number must appear somewhere in this prompt.\n"
    "- rating_justification must cite (a) the rubric tier by name and (b) one specific "
    "measurement or named card from the prompt.\n"
    "- sample_hand_plan walks through the actual hand shown in the prompt, naming the "
    "land you would play and the cards you would cast on each of turns 1-4."
)


class BuildProposal(BaseModel):
    """Pass-1 schema: the LLM commits to a build before it sees any sim data."""

    archetype: str = Field(description="Best colors and archetype name for this pool.")
    spells: list[str] = Field(
        description="21-25 non-land cards from the pool to play in the maindeck."
    )
    nonbasic_lands: list[str] = Field(
        description="Non-basic lands from the pool to include. Empty list if none."
    )
    suggested_land_count: int = Field(
        ge=15, le=19, description="Total land count, 15-19. Basics are filled in automatically."
    )


class CutOrKeep(BaseModel):
    """One cut-or-keep recommendation for a card in the built deck."""

    card: str = Field(description="Exact card name as it appears in the built deck.")
    decision: str = Field(description="'keep' or 'cut' - lowercase, no other values.")
    reason: str = Field(description="One short sentence citing what the card does or fails to do.")


class TurnPlay(BaseModel):
    """One turn's planned play for the sample hand walkthrough."""

    turn: int = Field(ge=1, le=4, description="Turn number, 1 through 4.")
    plays: str = Field(
        description="Cards played this turn (land + any spells cast), naming each one. "
        "Use 'no play - stuck on X' if there is nothing castable."
    )
    rationale: str = Field(
        description="One short sentence explaining the decision (why this card, why this "
        "order, what threat is being beaten or set up)."
    )


class DeckVerdict(BaseModel):
    """Pass-2 schema: a battery of specific questions, each requiring named cards.

    Every field asks for a concrete claim grounded in a card the deck actually
    contains. There are no free-form prose fields, so the model cannot fall back on
    stat-restating filler."""

    rating: int = Field(ge=1, le=10, description="Deck power on the 1-10 rubric.")
    rating_justification: str = Field(
        description="One sentence: name the rubric tier AND cite one specific card or "
        "measurement that anchors the rating."
    )
    win_condition: str = Field(
        description="The card or two cards that close the game and the turn you expect "
        "them to win on. If the deck has no real closer, say so explicitly: e.g. 'no "
        "true finisher; plans to grind out with cheap removal and 2/2 creatures' or "
        "'no closer; wins by burn from Lightning Bolt and Shock'. Inventing a finisher "
        "out of a vanilla creature is worse than saying there isn't one."
    )
    best_card: str = Field(
        description="The most impactful card in the maindeck. Name it and what it does."
    )
    worst_card: str = Field(
        description="The weakest card in the maindeck. Name it and why it is the weakest."
    )
    cut_or_keep: list[CutOrKeep] = Field(
        min_length=3,
        max_length=3,
        description="Exactly three cut-or-keep recommendations for cards in the deck.",
    )
    hardest_game_state: str = Field(
        description="A specific board state or matchup where this deck struggles, naming "
        "the card or play pattern that punishes it (e.g. 'an opponent on a 3/3 with reach "
        "by T3 kills my plan because my flyers are all 2/1')."
    )
    sample_hand_plan: list[TurnPlay] = Field(
        min_length=4,
        max_length=4,
        description="Exactly four turns of play for the sample hand shown in the prompt.",
    )
    comparable_archetype: str = Field(
        description="One real MTG archetype this deck resembles (e.g. 'mono-red sligh', "
        "'Esper Spirits', 'green stompy ramp')."
    )


def _render_pool(pool: list[Card]) -> str:
    return "\n".join(f"  - {card.render()}" for card in pool)


def _spell_role_tags(card: Card) -> str:
    """Comma-joined role tags for a non-land card, drawn from :mod:`tags`."""
    tags = []
    if card.is_creature:
        tags.append("creature")
    if is_interaction(card):
        tags.append("interaction")
    if is_ramp(card):
        tags.append("ramp")
    if is_evasive(card):
        tags.append("evasion")
    if tags:
        return ", ".join(tags)
    for category in ("Instant", "Sorcery", "Enchantment", "Artifact", "Planeswalker"):
        if category in card.type_line:
            return category.lower()
    return "spell"


def _card_shape(card: Card) -> str:
    """Mana cost plus P/T if a creature, joined for compact inline display."""
    parts = [card.mana_cost] if card.mana_cost else []
    if card.is_creature and card.power is not None:
        parts.append(f"{card.power}/{card.toughness}")
    return " ".join(parts)


def _render_built_deck(deck: BuiltDeck, report: PlayabilityReport) -> str:
    """Render the built 40-card deck with role tags and on-curve castability inline.

    The annotations come from data the verdict prompt already contains - tags from
    :mod:`tags`, castability from the per-card playability report - but putting them
    at the point of use keeps the model from inventing its own descriptors or
    fabricating percentages from a section three blocks away."""
    castability = {c.name: c for c in report.per_card}
    stranded = set(report.stranded)

    def render_spell(card: Card) -> str:
        shape = _card_shape(card)
        tags = _spell_role_tags(card)
        stat = castability.get(card.name)
        if stat is None:
            cast_part = ""
        else:
            pct = int(stat.castable_on_curve * 100)
            marker = " STRANDED" if card.name in stranded else ""
            cast_part = f" | on-curve {pct}%{marker}"
        shape_part = f" {shape}" if shape else ""
        return f"  - {card.name}{shape_part} | {tags}{cast_part}"

    def render_land(card: Card) -> str:
        produces = "/".join(card.producible_colors) if card.producible_colors else "colorless"
        return f"  - {card.name} | land, produces {produces}"

    basics: dict[str, int] = {}
    for card in deck.basics:
        basics[card.name] = basics.get(card.name, 0) + 1

    lines = [f"SPELLS ({len(deck.spells)}):"]
    lines.extend(render_spell(c) for c in deck.spells)
    if deck.nonbasic_lands:
        lines.append(f"NONBASIC LANDS ({len(deck.nonbasic_lands)}):")
        lines.extend(render_land(c) for c in deck.nonbasic_lands)
    basics_str = ", ".join(f"{n} {c}" for n, c in basics.items())
    lines.append(f"BASICS ({len(deck.basics)}): {basics_str}")
    return "\n".join(lines)


def _by_turn_line(series: dict[int, float]) -> str:
    """Render a per-turn metric series as ``T1 0.3 · T2 0.8 · ...``, skipping zeros."""
    points = [(t, v) for t, v in sorted(series.items()) if v > 0]
    if not points:
        return "none"
    return " · ".join(f"T{t} {v:.2f}" for t, v in points)


def _per_turn_table(report: PlayabilityReport) -> str:
    """Render the per-turn breakdown as a fixed-width table.

    Columns: turn · Hit% (curve hit, made on-curve play) · Cast% (fraction of hand
    castable now) · Lands · Spells (avg hand composition) · PeakClr% (probability
    the peak color requirement is met by this turn). ``-`` for Hit% means the deck
    has no spell of that CMC."""
    header = (
        f"  {'Turn':<6}{'Hit%':>6}{'Cast%':>8}{'Lands':>8}"
        f"{'Spells':>8}{'PeakClr%':>10}"
    )
    rows = [header]
    for turn in range(1, report.horizon + 1):
        hit = report.curve_hit.get(turn)
        hit_cell = f"{int(hit * 100)}%" if hit is not None else "-"
        cast_pct = int(report.castable_hand_fraction_by_turn[turn] * 100)
        lands = report.lands_in_hand_by_turn[turn]
        spells = report.nonland_in_hand_by_turn[turn]
        peak_pct = int(report.peak_color_hit_by_turn[turn] * 100)
        rows.append(
            f"  T{turn:<5}{hit_cell:>6}{cast_pct:>7}%"
            f"{lands:>8.1f}{spells:>8.1f}{peak_pct:>9}%"
        )
    return "\n".join(rows)


def _render_report(report: PlayabilityReport) -> str:
    """Render the playability report for the verdict prompt."""
    per_card_lines = "\n".join(
        f"  - {c.name} ({c.cmc} CMC): on-curve {int(c.castable_on_curve * 100)}%, "
        f"by-T{report.horizon} {int(c.castable_by_horizon * 100)}%, median T{c.median_turn}"
        for c in report.per_card
    )
    stranded = ", ".join(report.stranded) if report.stranded else "none"
    interaction = _by_turn_line(report.interaction_by_turn)
    power = _by_turn_line(report.creature_power_by_turn)
    evasion = _by_turn_line(report.evasion_by_turn)

    if report.peak_color_requirement is None:
        peak_line = "Peak color requirement: none (no colored spells)."
    else:
        color, count, spell_name = report.peak_color_requirement
        peak_line = (
            f"Peak color requirement: {count} {color} source(s) "
            f"(forced by {spell_name})."
        )

    peak_count = report.peak_color_requirement[1] if report.peak_color_requirement else 0
    peak_tautology = (
        f"  NOTE: PeakClr% can only be > 0 from turn {peak_count} onward (the peak "
        f"requires {peak_count} sources, and you can play at most one land per turn). "
        f"T1 through T{peak_count - 1} sitting at 0% is arithmetic, not a deck flaw."
        if peak_count >= 2
        else ""
    )
    return (
        f"Simulated {report.sims} games to T{report.horizon}.\n"
        f"Mulligan rate: {int(report.mulligan_rate * 100)}% "
        f"(average opening hand size {report.average_opening_hand_size:.2f}).\n"
        f"{peak_line}\n\n"
        f"Per-turn breakdown:\n"
        f"  Hit%   = % of games that made an on-curve play (deck has CMC=turn spell + mana).\n"
        f"  Cast%  = % of current hand that is castable right now (curve+mana smoothness).\n"
        f"  Lands/Spells = average hand composition after the land drop.\n"
        f"  PeakClr% = % of games meeting the peak color requirement.\n"
        f"{peak_tautology}\n"
        f"{_per_turn_table(report)}\n\n"
        f"Available interaction by turn (drawn + castable): {interaction}\n"
        f"Available creature power by turn: {power}\n"
        f"Available evasion by turn: {evasion}\n"
        f"Stranded cards (cast-by-horizon < 50%): {stranded}\n"
        f"Per-card castability:\n{per_card_lines}"
    )


def _build_messages(pool: list[Card], seat_index: int) -> list[dict[str, str]]:
    user = (
        f"Seat {seat_index + 1} drafted this {len(pool)}-card pool:\n\n"
        f"{_render_pool(pool)}\n\n"
        "Build the strongest 40-card deck: name the archetype, list the spells and any "
        "nonbasic lands to play, and suggest a total land count between 15 and 19."
    )
    return [{"role": "system", "content": _BUILD_SYSTEM}, {"role": "user", "content": user}]


def _render_sample_hand(hand: SampleHand) -> str:
    """Render one deterministic sample hand: the opener + the next few draws.

    The LLM is asked to walk through this hand turn by turn, which forces it to do
    real card-level reasoning rather than restating aggregate stats."""
    opener = ", ".join(c.name for c in hand.opening)
    draws = ", ".join(c.name for c in hand.upcoming_draws)
    mull_note = " (kept after one mulligan)" if hand.mulliganed else ""
    draws_section = (
        "\n  ".join(f"Draw turn {i + 2}: {c.name}" for i, c in enumerate(hand.upcoming_draws))
        if hand.upcoming_draws
        else "(no upcoming draws in horizon)"
    )
    return (
        f"Opening hand{mull_note}: {opener}\n  "
        f"Upcoming draws: {draws}\n  "
        f"{draws_section}"
    )


def _verdict_messages(
    pool: list[Card],
    proposal: BuildProposal,
    deck: BuiltDeck,
    stats: DeckStats,
    report: PlayabilityReport,
    sample_hand: SampleHand,
    retry_feedback: str = "",
) -> list[dict[str, str]]:
    user = (
        f"DRAFTED POOL ({len(pool)} cards):\n{_render_pool(pool)}\n\n"
        f"PROPOSED ARCHETYPE: {proposal.archetype}\n\n"
        f"BUILT DECK:\n{_render_built_deck(deck, report)}\n\n"
        f"DECK STATISTICS:\n{render_stats(stats)}\n\n"
        f"PLAYABILITY SIMULATION:\n{_render_report(report)}\n\n"
        f"SAMPLE OPENING HAND (use this exact hand for sample_hand_plan):\n  "
        f"{_render_sample_hand(sample_hand)}\n\n"
        "Fill in every field of the schema. Name specific cards from the built deck "
        "in every answer. Do not restate the statistics; explain what they mean for "
        "how this deck plays."
    )
    if retry_feedback:
        user += f"\n\nREVISION NEEDED: {retry_feedback}"
    return [{"role": "system", "content": _VERDICT_SYSTEM}, {"role": "user", "content": user}]


def _log_parse_failure(
    console: Console, label: str, raw: str, err: ValidationError
) -> None:
    """Surface a parse failure with a head/tail snippet of the model reply.

    Silent failures are the worst kind of bug here; if the model's reply truncated
    mid-JSON because num_predict is too low, the snippet makes that obvious."""
    head = raw[:200].replace("\n", " ")
    tail = raw[-120:].replace("\n", " ") if len(raw) > 320 else ""
    console.print(
        f"    [red]parse failed[/red] ({label}): {len(raw)} chars, "
        f"{str(err).splitlines()[0]}"
    )
    console.print(f"      [dim]head:[/dim] {head}")
    if tail:
        console.print(f"      [dim]tail:[/dim] {tail}")


def _call_with_retry(
    llm: LLMClient,
    messages: list,
    schema: type[BaseModel],
    console: Console,
    label: str,
) -> BaseModel | None:
    """Try to parse a structured reply twice; return None when both attempts fail.

    Logs the raw reply on each failure so the operator can see truncated JSON, empty
    bodies (a sign of thinking-tokens eating the budget), or schema mismatches."""
    for attempt in range(2):
        raw = llm.chat(messages, schema)
        try:
            return schema.model_validate_json(raw)
        except ValidationError as err:
            _log_parse_failure(console, f"{label} attempt {attempt + 1}", raw, err)
    return None


def verdict_validation_failures(verdict: DeckVerdict, deck_card_names: set[str]) -> list[str]:
    """Return human-readable validation failures, empty list when the verdict is clean.

    Required-card-reference fields are ``rating_justification``, ``best_card``, and
    ``worst_card``; each must contain at least one card name from the built deck
    (case-insensitive substring match). ``win_condition`` is intentionally excluded:
    some decks legitimately have no closer, and the field invites honest 'no real
    finisher' answers that would otherwise fail this check. ``cut_or_keep`` entries
    must each name a card that is actually in the deck."""
    failures: list[str] = []
    deck_names_lower = {name.lower() for name in deck_card_names}

    def mentions_card(text: str) -> bool:
        text_lower = text.lower()
        return any(name in text_lower for name in deck_names_lower)

    for field_name in ("rating_justification", "best_card", "worst_card"):
        if not mentions_card(getattr(verdict, field_name)):
            failures.append(f"{field_name} must name a specific card from the built deck.")

    for entry in verdict.cut_or_keep:
        if entry.card.lower() not in deck_names_lower:
            failures.append(
                f"cut_or_keep entry '{entry.card}' is not a card in the built deck."
            )
    return failures


def _get_verdict(
    llm: LLMClient,
    pool: list[Card],
    proposal: BuildProposal,
    deck: BuiltDeck,
    stats: DeckStats,
    report: PlayabilityReport,
    sample_hand: SampleHand,
    console: Console,
    seat_index: int,
) -> DeckVerdict | None:
    """Get a parsed-and-validated verdict, retrying once with feedback when validation
    fails. Accepts the second reply with a warning when it still fails - better to
    surface a flawed verdict than to drop the seat entirely."""
    deck_card_names = {c.name for c in deck.cards}
    feedback = ""
    last_verdict: DeckVerdict | None = None

    for attempt in range(2):
        messages = _verdict_messages(pool, proposal, deck, stats, report, sample_hand, feedback)
        raw = llm.chat(messages, DeckVerdict)
        try:
            verdict = DeckVerdict.model_validate_json(raw)
        except ValidationError as err:
            label = f"seat {seat_index + 1} verdict attempt {attempt + 1}"
            _log_parse_failure(console, label, raw, err)
            continue
        last_verdict = verdict
        failures = verdict_validation_failures(verdict, deck_card_names)
        if not failures:
            return verdict
        feedback = (
            "Your previous reply failed: "
            + " ".join(failures)
            + " Cite cards by name as they appear in the BUILT DECK section."
        )

    if last_verdict is not None:
        console.print(
            f"  Seat {seat_index + 1}: [yellow]verdict accepted with validation "
            f"warnings (model would not cite specific cards)[/yellow]"
        )
        return last_verdict
    return None


def _basic_summary(deck: BuiltDeck) -> dict[str, int]:
    by_color: dict[str, int] = {}
    for card in deck.basics:
        by_color[card.name] = by_color.get(card.name, 0) + 1
    return by_color


def _curve_console(report: PlayabilityReport) -> str:
    if not report.curve_hit:
        return "no curve plays"
    return " · ".join(f"T{t} {int(rate * 100)}%" for t, rate in sorted(report.curve_hit.items()))


def _evaluate_seat(
    seat: dict, pool: list[Card], llm: LLMClient, console: Console
) -> dict | None:
    """Two-pass evaluation: LLM proposes a build, the simulator and stats are produced,
    then the LLM rates the finished deck with all of that in front of it."""
    proposal = _call_with_retry(
        llm,
        _build_messages(pool, seat["index"]),
        BuildProposal,
        console,
        f"seat {seat['index'] + 1} build",
    )
    if proposal is None:
        console.print(
            f"  Seat {seat['index'] + 1}: [red]build skipped (unparseable reply)[/red]"
        )
        return None

    deck = build_deck(
        pool=pool,
        spell_names=proposal.spells,
        nonbasic_land_names=proposal.nonbasic_lands,
        suggested_land_count=proposal.suggested_land_count,
    )
    report = simulate_deck(deck.cards)
    stats = compute_stats(deck)
    sample_hand = generate_sample_hands(deck.cards, n=1, draws_per_hand=3, seed=seat["index"])[0]

    verdict = _get_verdict(
        llm, pool, proposal, deck, stats, report, sample_hand, console, seat["index"]
    )
    if verdict is None:
        console.print(
            f"  Seat {seat['index'] + 1}: [red]verdict skipped (unparseable reply)[/red]"
        )
        return None

    record = {
        "index": seat["index"],
        "archetype": proposal.archetype,
        **verdict.model_dump(),
        "build": {
            "spells": [c.name for c in deck.spells],
            "nonbasic_lands": [c.name for c in deck.nonbasic_lands],
            "basics": _basic_summary(deck),
            "suggested_land_count": proposal.suggested_land_count,
            "pip_counts": deck.pip_counts,
            "notes": deck.notes,
        },
        "sample_hand": {
            "opening": [c.name for c in sample_hand.opening],
            "upcoming_draws": [c.name for c in sample_hand.upcoming_draws],
            "mulliganed": sample_hand.mulliganed,
        },
        "deck_stats": asdict(stats),
        "playability": {
            "sims": report.sims,
            "horizon": report.horizon,
            "mulligan_rate": report.mulligan_rate,
            "average_opening_hand_size": report.average_opening_hand_size,
            "peak_color_requirement": (
                {
                    "color": report.peak_color_requirement[0],
                    "count": report.peak_color_requirement[1],
                    "spell": report.peak_color_requirement[2],
                }
                if report.peak_color_requirement
                else None
            ),
            "curve_hit": {str(t): rate for t, rate in report.curve_hit.items()},
            "castable_hand_fraction_by_turn": {
                str(t): v for t, v in report.castable_hand_fraction_by_turn.items()
            },
            "lands_in_hand_by_turn": {
                str(t): v for t, v in report.lands_in_hand_by_turn.items()
            },
            "nonland_in_hand_by_turn": {
                str(t): v for t, v in report.nonland_in_hand_by_turn.items()
            },
            "peak_color_hit_by_turn": {
                str(t): v for t, v in report.peak_color_hit_by_turn.items()
            },
            "interaction_by_turn": {str(t): v for t, v in report.interaction_by_turn.items()},
            "creature_power_by_turn": {
                str(t): v for t, v in report.creature_power_by_turn.items()
            },
            "evasion_by_turn": {str(t): v for t, v in report.evasion_by_turn.items()},
            "stranded": report.stranded,
            "per_card": [asdict(c) for c in report.per_card],
        },
    }

    stranded_note = f", [red]{len(report.stranded)} stranded[/red]" if report.stranded else ""
    mulligan_note = (
        f", mulls {int(report.mulligan_rate * 100)}%"
        if report.mulligan_rate > 0
        else ""
    )
    console.print(
        f"  Seat {seat['index'] + 1}: [bold]{verdict.rating}/10[/bold] - {proposal.archetype}\n"
        f"    [dim]Mana base:[/dim] {_curve_console(report)}{stranded_note}{mulligan_note}\n"
        f"    [dim]Win:[/dim] {verdict.win_condition}\n"
        f"    [dim]Justification:[/dim] {verdict.rating_justification}"
    )
    return record


def evaluate_draft(results_dir: Path, llm_config: LLMConfig) -> Path:
    """Have the LLM evaluate every seat's pool in a finished draft.

    For each seat: ask the model for a build (pass 1), feed the resulting 40-card deck
    through the deckbuilder, run the playability simulator, compute deterministic deck
    statistics, then ask the model to rate the finished deck with the build + stats +
    sim report in front of it (pass 2). Writes ``evaluation.json`` alongside the draft.

    Parameters
    ----------
    results_dir : Path
        A directory produced by :func:`mtg_drafting.records.save_draft`.
    llm_config : LLMConfig
        Model settings for the evaluator. Internally overrides the temperature to a
        low value; evaluation benefits from sharper, less smoothed sampling.

    Returns
    -------
    Path
        Path of the written ``evaluation.json``.
    """
    console = Console()
    draft = load_draft(results_dir)
    index = load_index(Paths().cache_dir)

    eval_config = llm_config.model_copy(
        update={"temperature": _EVAL_TEMPERATURE, "num_predict": _EVAL_NUM_PREDICT}
    )
    llm = LLMClient(eval_config)
    llm.ensure_model()

    evaluations: list[dict] = []
    for seat in draft["seats"]:
        pool, missing = [], []
        for name in seat["pool"]:
            card = index.get(name)
            (pool if card is not None else missing).append(card or name)
        if missing:
            console.print(
                f"  Seat {seat['index'] + 1}: [yellow]{len(missing)} pool cards not "
                f"in current Scryfall snapshot: {', '.join(missing)}[/yellow]"
            )
        record = _evaluate_seat(seat, pool, llm, console)
        if record is not None:
            evaluations.append(record)

    out_path = results_dir / "evaluation.json"
    out_path.write_text(json.dumps(evaluations, indent=2))
    console.print(f"\n[green]Evaluation written:[/green] [bold]{out_path}[/bold]")
    return out_path

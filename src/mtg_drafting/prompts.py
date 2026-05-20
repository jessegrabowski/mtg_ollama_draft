from typing import Literal

from pydantic import BaseModel, Field

from mtg_drafting.cards import COLOR_NAMES, Card
from mtg_drafting.config import DraftConfig
from mtg_drafting.deckbuilder import count_pips
from mtg_drafting.seat import Seat, format_memory
from mtg_drafting.strategist import StrategyState
from mtg_drafting.tags import is_evasive, is_interaction, is_ramp

# Order pool buckets are presented in.
_CATEGORY_ORDER = [COLOR_NAMES[c] for c in ("W", "U", "B", "R", "G")] + [
    "Multicolor",
    "Colorless",
    "Land",
]

_SYSTEM_PROMPT = (
    "You are the picker for one seat in a Magic: The Gathering cube draft. The "
    "strategist has decided the seat's direction between packs; your job is to apply "
    "it.\n\n"
    "The plan lists 1-3 candidate directions, each as colors + role + weight. Role "
    "is one of 'open', 'beatdown', 'control', 'midrange'. The first (primary) is "
    "your main direction; the others are viable alternatives if signals shift.\n\n"
    "WHEN PRIMARY ROLE IS 'open': the strategist is telling you to stay flexible. "
    "Take the raw strongest card in the pack regardless of color. Format-defining "
    "bombs, premium removal, and free mana are the top of the list. biggest_needs "
    "is a soft hint, not a constraint; do not pass a clearly stronger card for an "
    "in-color playable just because the pool has a couple of cards in one color. "
    "The pool is still small.\n\n"
    "WHEN PRIMARY ROLE IS committed (beatdown/control/midrange): pick the card "
    "that best fits the primary plan and the biggest-needs list. A card that "
    "strongly fits a secondary direction can pull weight toward it.\n\n"
    "Do NOT reason in named modern archetypes ('Aristocrats', 'Skies', 'Reanimator', "
    "'Tempo', 'Tokens'). Reason in primitives: this card is a cheap evasive threat in "
    "my colors that fits my beatdown role; that card is removal that fills my biggest "
    "need; the other card is a high-CMC finisher that pushes my role toward control.\n\n"
    "RECOGNISE CARDS THAT BREAK FUNDAMENTAL VALUE EQUATIONS. These trump plan-fit "
    "even when off-color. Reason from principle:\n"
    "- Breaks the one-land-per-turn symmetry. Free or near-free extra mana sources "
    "(e.g. a Mox, Sol Ring, Dark Ritual) skip you ahead a turn. The card pays for "
    "itself; splashing it is free.\n"
    "- Generates card advantage. Trades one card for multiple effects - a sweeper "
    "killing several creatures, a draw-3 spell, a two-for-one removal piece.\n"
    "- Mana efficiency far above curve. Cheap effects that do work much larger than "
    "their cost (Lightning Bolt: 1 mana for 3 damage; Counterspell: 2 mana to stop "
    "any spell).\n"
    "- Inevitability. A card that wins the game on its own if unanswered (a "
    "planeswalker that ticks up to a game-winning ultimate; a self-sustaining "
    "engine).\n"
    "Splashing a card that fits one of these is better than passing it.\n\n"
    "You can also draft DIRECTLY into the sideboard. If you see a card whose value "
    "is narrow or matchup-specific - useful against a particular opponent but "
    "underpowered as a maindeck inclusion - take it and mark intent 'sideboard'. "
    "It goes into the pool but is set aside, not counted in the deck profile. "
    "Default intent is 'maindeck'; only mark 'sideboard' when the card is "
    "meaningfully better as a sideboard piece than as a maindeck inclusion.\n\n"
    "Always pick a card whose name appears verbatim in the pack. Reply ONLY with a "
    "compact JSON object weighing the two or three strongest options in 'reasoning' "
    "(two or three sentences) and naming the chosen card in 'pick'. Example shape:\n"
    '{"reasoning": "Brainstorm and Snapcaster Mage are the standouts here. Snapcaster '
    "wants a built spell-heavy deck; Brainstorm is color-flexible card draw that "
    'fills my biggest need and works with any blue plan. I take Brainstorm.", '
    '"pick": "Brainstorm", "intent": "maindeck"}'
)


class PickResponse(BaseModel):
    """Schema the picker model must return for each pick.

    ``reasoning`` comes first so the model weighs its options before committing to
    ``pick`` - chain-of-thought in plain sight, at structured-output speed.
    """

    reasoning: str = Field(
        description="Two or three sentences weighing the strongest cards, ending in a "
        "decision."
    )
    pick: str = Field(description="The exact name of one card from the offered pack.")
    intent: Literal["maindeck", "sideboard"] = Field(
        default="maindeck",
        description="Where this pick is meant to go. 'maindeck' is the default and "
        "what you'll use for nearly every pick. 'sideboard' is for cards whose "
        "value is narrow or matchup-specific - useful against a particular "
        "opponent but underpowered for the maindeck. The strategist may "
        "re-classify at the next pack boundary - this is a hint.",
    )
    hoping_to_wheel: str | None = Field(
        default=None,
        description="OPTIONAL: name of one card in the offered pack you hope will "
        "wheel back. A wheel is when the pack passes through 7 other seats and "
        "returns - so a wheel candidate is a card you'd want as a later pick AND "
        "that downstream seats probably have other priorities than. Naming a card "
        "here does NOT take it; it logs a prediction the wheel detector checks. "
        "If the card does wheel, that's a strong signal other players did not "
        "value it; if it does not, another seat is competing for your direction. "
        "You are not obligated to take a card you marked - by the time the pack "
        "returns your plan may have shifted, and passing it again is fine. "
        "Pack 1 picks 1-3 rarely have wheel candidates (everything is premium); "
        "mid- and late-pack picks often do.",
    )
    note: str | None = Field(
        default=None,
        max_length=160,
        description="OPTIONAL one-sentence observation worth carrying forward. "
        "Think inferences, not raw data. Examples of the shape: 'got Mox Ruby at "
        "P1.3, something stronger must have been in the pack', 'hoping "
        "Counterspell wheels in pack 2', 'the Counterspell I was tracking did "
        "not come back, I am not the only blue drafter', 'P1.7 had no blue "
        "cards, someone upstream is on blue'. Leave null when nothing happened "
        "this pick worth carrying forward. Most picks should leave it null - the "
        "bar is 'changes how I think about future picks', not 'I had thoughts'.",
    )


def _curve(pool: list[Card]) -> str:
    """Render a mana-value histogram for the non-land cards in a pool."""
    buckets = [0] * 8
    for card in pool:
        if card.is_land:
            continue
        buckets[min(int(card.cmc), 7)] += 1
    cells = [f"{i}:{n}" for i, n in enumerate(buckets[:7])]
    cells.append(f"7+:{buckets[7]}")
    return "  ".join(cells)


def _group_by_category(pool: list[Card]) -> list[tuple[str, list[Card]]]:
    """Bucket a pool by color category in display order, low CMC first within each."""
    grouped: dict[str, list[Card]] = {}
    for card in pool:
        grouped.setdefault(card.color_category, []).append(card)
    ordered = []
    for category in _CATEGORY_ORDER:
        cards = grouped.get(category)
        if cards:
            ordered.append((category, sorted(cards, key=lambda c: c.cmc)))
    return ordered


def summarize_pool(pool: list[Card]) -> str:
    """Group a seat's pool by color category as a compact name + CMC list.

    Used for the human-facing verbose draft display, where a one-line-per-category
    overview is more readable than full card text."""
    if not pool:
        return "(empty - this is your first pick)"

    lines = [f"Total {len(pool)} cards. Mana curve (non-land): {_curve(pool)}"]
    for category, cards in _group_by_category(pool):
        names = ", ".join(f"{c.name} ({int(c.cmc)})" for c in cards)
        lines.append(f"  {category} ({len(cards)}): {names}")
    return "\n".join(lines)


def render_pool(pool: list[Card]) -> str:
    """Render a seat's pool grouped by color category with full card detail.

    Used in the pick prompt: the model sees each owned card's cost, type, and rules
    text - the same detail as the offered pack - so it can reason about what its deck
    actually does, not just which names it holds."""
    if not pool:
        return "(empty - this is your first pick)"

    lines = [f"Total {len(pool)} cards. Mana curve (non-land): {_curve(pool)}"]
    for category, cards in _group_by_category(pool):
        lines.append(f"{category} ({len(cards)}):")
        lines.extend(f"  {card.render()}" for card in cards)
    return "\n".join(lines)


def _recent_picks(seat: Seat) -> str:
    """Render the seat's recent picks kept verbatim for context."""
    if not seat.recent:
        return "    (none yet)"
    return "\n".join(
        f"    P{r.pick_no + 1} (pack of {len(r.pack)}): took {r.chosen}" for r in seat.recent
    )


def render_deck_profile(cards: list[Card], seat: Seat) -> str:
    """Render a compact deck profile: pip counts, curve, role counts, recent picks.

    Parameters
    ----------
    cards : list of Card
        The cards to summarise. The picker passes ``seat.maindeck`` (excludes anything
        the strategist set aside); the strategist passes ``seat.pool`` so it can see
        everything before deciding what to sideboard.
    seat : Seat
        Source of the ``recent`` picks block - the recent-picks history is identical
        across the two callers and always shows the latest picks regardless of which
        sub-pool is being profiled.
    """
    if not cards:
        return "  (no cards yet)"

    pip_counts = count_pips(cards)
    pip_str = (
        " · ".join(f"{c} {n}" for c, n in pip_counts.items() if n > 0)
        or "(no colored pips)"
    )

    creatures = sum(1 for c in cards if c.is_creature)
    interaction = sum(1 for c in cards if is_interaction(c))
    ramp = sum(1 for c in cards if is_ramp(c))
    evasion = sum(1 for c in cards if is_evasive(c))

    return (
        f"  Colors (pips) : {pip_str}\n"
        f"  Curve (nonland): {_curve(cards)}\n"
        f"  Roles          : creatures {creatures} · interaction {interaction} · "
        f"ramp {ramp} · evasion {evasion}\n"
        f"  Recent picks   :\n{_recent_picks(seat)}"
    )


def _strategy_block(state: StrategyState | None) -> str:
    """Render the strategist's current plan for the picker's prompt."""
    if state is None:
        return (
            "  (strategist has not produced a plan yet - take the strongest, most "
            "flexible card. Read the pick number in the user message above to know "
            "where you are in the draft.)"
        )
    needs = "; ".join(state.biggest_needs) if state.biggest_needs else "(none listed)"
    direction_lines = "\n".join(
        f"    {i + 1}. [{d.weight}/10] {'/'.join(d.colors) or 'uncommitted'} "
        f"{d.role} - {d.rationale}"
        for i, d in enumerate(sorted(state.directions, key=lambda x: -x.weight))
    )
    return (
        f"  color_commitment : {state.color_commitment}\n"
        f"  directions (primary first):\n{direction_lines}\n"
        f"  biggest_needs    : {needs}\n"
        f"  watching_for     : {state.watching_for}"
    )


def build_pick_messages(
    seat: Seat, pack: list[Card], round_no: int, pick_no: int, config: DraftConfig
) -> list[dict[str, str]]:
    """Assemble the chat messages for one pick.

    The picker sees the strategist's structured plan plus the deck profile (concrete
    numbers - pip counts, curve, role counts) and picks the card in the pack that
    best fits the plan. The strategist's plan is refreshed at pack boundaries; within
    a pack the picker just applies it.

    Parameters
    ----------
    seat : Seat
        The seat currently on the clock.
    pack : list of Card
        Cards available to pick from.
    round_no : int
        Zero-based round index.
    pick_no : int
        Zero-based pick index within the round.
    config : DraftConfig
        Draft settings, used for the seat-count context line.

    Returns
    -------
    list of dict
        Ollama chat messages (a system message followed by one user message).
    """
    pack_listing = "\n".join(f"  {i + 1}. {card.render()}" for i, card in enumerate(pack))

    user = (
        f"You are seat {seat.index + 1} of {config.n_seats}. "
        f"Pack {round_no + 1}, pick {pick_no + 1}.\n\n"
        f"YOUR PLAN (from the strategist):\n{_strategy_block(seat.strategy_state)}\n\n"
        f"DECK PROFILE (main deck only; sideboarded cards are not counted):\n"
        f"{render_deck_profile(seat.maindeck, seat)}\n\n"
        f"DRAFT MEMORY (newest first):\n{format_memory(seat)}\n\n"
        f"PACK ({len(pack)} cards):\n{pack_listing}\n\n"
        "Pick the card that best fits the plan. Reply with two-or-three-sentence "
        "reasoning, the exact card name, and optionally a one-sentence note to "
        "remember across picks (most picks should leave note null)."
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]

from pydantic import BaseModel, Field

from mtg_drafting.cards import COLOR_NAMES, Card
from mtg_drafting.config import DraftConfig
from mtg_drafting.seat import Seat

# Order pool buckets are presented in.
_CATEGORY_ORDER = [COLOR_NAMES[c] for c in ("W", "U", "B", "R", "G")] + [
    "Multicolor",
    "Colorless",
    "Land",
]

_SYSTEM_PROMPT = (
    "You are an expert Magic: The Gathering cube drafter. You are drafting one seat at "
    "a multi-player table. Each pick you receive a pack of cards and must take exactly "
    "one, then the pack passes on.\n\n"
    "Principles:\n"
    "- Early picks: take the most powerful, flexible cards and stay open.\n"
    "- Mid draft: commit to two colours and an archetype that matches your pool.\n"
    "- Late draft: fill your curve, shore up weaknesses, and respect mana.\n"
    "- Aim for a coherent 40-card deck (about 23 spells + 17 lands).\n"
    "- Value removal, card advantage, evasion, and an efficient mana curve.\n\n"
    "Always pick a card whose name appears verbatim in the offered pack.\n\n"
    "Respond ONLY with a compact JSON object. In 'reasoning', weigh the two or three "
    "strongest cards in two or three sentences and then commit; keep 'strategy' to one "
    "sentence. Match the shape and length of this example:\n"
    '{"reasoning": "Brainstorm and Snapcaster Mage are the standouts here. Snapcaster '
    "is powerful but wants a built deck of spells, while Brainstorm is premium card "
    'advantage that keeps me colour-open. I take the more flexible card.", '
    '"pick": "Brainstorm", '
    '"strategy": "Open early, leaning blue control for card advantage and removal."}'
)


class PickResponse(BaseModel):
    """Schema the drafting model must return for each pick.

    ``reasoning`` comes first so the model weighs its options before committing to
    ``pick`` - chain-of-thought in plain sight, at structured-output speed.
    """

    reasoning: str = Field(
        description="Two or three sentences weighing the strongest cards, ending in a "
        "decision."
    )
    pick: str = Field(description="The exact name of one card from the offered pack.")
    strategy: str = Field(
        description="ONE short sentence naming this seat's colours, archetype, and "
        "biggest current need."
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


def summarize_pool(pool: list[Card]) -> str:
    """Group a seat's pool by colour category for compact display in a prompt."""
    if not pool:
        return "(empty - this is your first pick)"

    grouped: dict[str, list[Card]] = {}
    for card in pool:
        grouped.setdefault(card.color_category, []).append(card)

    lines = [f"Total {len(pool)} cards. Mana curve (non-land): {_curve(pool)}"]
    for category in _CATEGORY_ORDER:
        cards = grouped.get(category)
        if not cards:
            continue
        names = ", ".join(f"{c.name} ({int(c.cmc)})" for c in sorted(cards, key=lambda c: c.cmc))
        lines.append(f"  {category} ({len(cards)}): {names}")
    return "\n".join(lines)


def _recent_picks(seat: Seat) -> str:
    """Render the seat's recent picks kept verbatim for context."""
    if not seat.recent:
        return "(none yet)"
    return "\n".join(
        f"  P{r.pick_no + 1} (pack of {len(r.pack)}): took {r.chosen}" for r in seat.recent
    )


def build_pick_messages(
    seat: Seat, pack: list[Card], round_no: int, pick_no: int, config: DraftConfig
) -> list[dict[str, str]]:
    """Assemble the chat messages for one pick.

    The seat's context is hybrid: a running strategy summary and a grouped pool view
    give the long arc of the draft, while the recent-picks block keeps the last few
    decisions verbatim. Prompt size stays bounded because only ``history_maxlen`` recent
    picks are included.

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
    strategy = seat.strategy.strip() or "Not yet committed - still reading signals."

    user = (
        f"You are seat {seat.index + 1} of {config.n_seats}. "
        f"Round {round_no + 1}, pick {pick_no + 1}.\n\n"
        f"YOUR CURRENT STRATEGY:\n{strategy}\n\n"
        f"YOUR POOL SO FAR:\n{summarize_pool(seat.pool)}\n\n"
        f"YOUR RECENT PICKS:\n{_recent_picks(seat)}\n\n"
        f"PACK ({len(pack)} cards):\n{pack_listing}\n\n"
        "Take the single best card for this seat. Reply with one-sentence reasoning, "
        "the exact card name, and a one-sentence updated strategy."
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]

from dataclasses import dataclass

from mtg_drafting.cards import COLOR_NAMES, COLOR_ORDER, Card
from mtg_drafting.deckbuilder import BuiltDeck
from mtg_drafting.tags import effective_power, is_evasive, is_interaction, is_ramp

# A color below this source count is considered a splash rather than a real color.
SPLASH_THRESHOLD = 4

# Card type categories shown in the type-count summary, in the order they appear in
# stats output.
_TYPE_CATEGORIES = (
    "Creature",
    "Planeswalker",
    "Instant",
    "Sorcery",
    "Enchantment",
    "Artifact",
    "Land",
)


@dataclass
class DeckStats:
    """Static, deterministic summary of a 40-card deck.

    Computed from the built deck alone - no simulation - so it is cheap to surface
    next to the playability report when prompting the LLM evaluator.

    Parameters
    ----------
    total_cards : int
        Card count (should always be 40 for a built deck).
    cmc_distribution : dict mapping int to int
        Non-land mana-value histogram (CMC 0-7+, with 7 acting as a 7+ bucket).
    average_cmc : float
        Mean mana value over non-land cards.
    type_counts : dict mapping str to int
        Counts per card-type category in the order shown to humans.
    creature_power_total : int
        Sum of numeric power across creatures in the deck.
    mana_sources : dict mapping str to int
        Producible WUBRG sources from every land in the deck. A dual that taps for U
        or R contributes one each to U and R.
    splash_colors : list of str
        Color letters whose source count is below :data:`SPLASH_THRESHOLD`. Empty
        for a mono- or two-color deck with a solid mana base.
    interaction_count : int
        Removal/counter/bounce/sweeper spells, per :func:`tags.is_interaction`.
    ramp_count : int
        Mana-producing or land-fetching cards, per :func:`tags.is_ramp`.
    evasion_count : int
        Creatures with an evasion keyword, per :func:`tags.is_evasive`.
    """

    total_cards: int
    cmc_distribution: dict[int, int]
    average_cmc: float
    type_counts: dict[str, int]
    creature_power_total: int
    mana_sources: dict[str, int]
    splash_colors: list[str]
    interaction_count: int
    ramp_count: int
    evasion_count: int


def _type_category(card: Card) -> str:
    """The first matching category from :data:`_TYPE_CATEGORIES`. Multi-type cards
    pick their primary type by category order - Artifact Creatures count as Creatures,
    Land Creatures (rare) count as Creatures."""
    for category in _TYPE_CATEGORIES:
        if category in card.type_line:
            return category
    return "Other"


def _cmc_bucket(card: Card) -> int:
    """Bucket a card's mana value into 0-7 for the histogram (7 catches 7+)."""
    return min(int(card.cmc), 7)


def compute_stats(deck: BuiltDeck) -> DeckStats:
    """Compute :class:`DeckStats` for a built 40-card deck."""
    cards = deck.cards
    nonland = [c for c in cards if not c.is_land]

    cmc_dist = {b: 0 for b in range(8)}
    for card in nonland:
        cmc_dist[_cmc_bucket(card)] += 1
    avg_cmc = sum(c.cmc for c in nonland) / len(nonland) if nonland else 0.0

    type_counts = dict.fromkeys(_TYPE_CATEGORIES, 0)
    for card in cards:
        type_counts[_type_category(card)] += 1

    mana_sources = dict.fromkeys(COLOR_ORDER, 0)
    for card in cards:
        if not card.is_land:
            continue
        for color in card.producible_colors:
            mana_sources[color] += 1

    splash = [c for c in COLOR_ORDER if 0 < mana_sources[c] < SPLASH_THRESHOLD]

    return DeckStats(
        total_cards=len(cards),
        cmc_distribution=cmc_dist,
        average_cmc=round(avg_cmc, 2),
        type_counts=type_counts,
        creature_power_total=sum(effective_power(c) for c in cards),
        mana_sources=mana_sources,
        splash_colors=splash,
        interaction_count=sum(1 for c in cards if is_interaction(c)),
        ramp_count=sum(1 for c in cards if is_ramp(c)),
        evasion_count=sum(1 for c in cards if is_evasive(c)),
    )


def render_stats(stats: DeckStats) -> str:
    """Render :class:`DeckStats` as a compact human/LLM-readable block.

    Used inside the evaluator's pass-2 prompt and in the verbose console summary."""
    cmc_line = "  ".join(
        f"{b}{'+' if b == 7 else ''}:{stats.cmc_distribution[b]}" for b in range(8)
    )
    types = ", ".join(
        f"{name} {count}" for name, count in stats.type_counts.items() if count
    )
    sources = ", ".join(
        f"{COLOR_NAMES[c]} {n}" for c, n in stats.mana_sources.items() if n
    )
    splashes = (
        ", ".join(f"{COLOR_NAMES[c]} ({stats.mana_sources[c]})" for c in stats.splash_colors)
        or "none"
    )
    return (
        f"Total cards: {stats.total_cards}\n"
        f"Mana curve (non-land): {cmc_line}\n"
        f"Average CMC: {stats.average_cmc}\n"
        f"Types: {types}\n"
        f"Total creature power: {stats.creature_power_total}\n"
        f"Mana sources: {sources or 'none'}\n"
        f"Splash colors (<{SPLASH_THRESHOLD} sources): {splashes}\n"
        f"Interaction spells: {stats.interaction_count}\n"
        f"Ramp spells: {stats.ramp_count}\n"
        f"Evasive creatures: {stats.evasion_count}"
    )

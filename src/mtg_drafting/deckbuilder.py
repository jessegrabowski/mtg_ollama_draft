import difflib
import re
from dataclasses import dataclass, field

from mtg_drafting.cards import COLOR_ORDER, Card

# Five basic lands as Card objects. Hard-coded so the deckbuilder does not depend on
# which printing of a basic the Scryfall snapshot happens to return.
BASIC_LANDS: dict[str, Card] = {
    "W": Card(
        name="Plains",
        type_line="Basic Land — Plains",
        color_identity=["W"],
        produced_mana=["W"],
    ),
    "U": Card(
        name="Island",
        type_line="Basic Land — Island",
        color_identity=["U"],
        produced_mana=["U"],
    ),
    "B": Card(
        name="Swamp",
        type_line="Basic Land — Swamp",
        color_identity=["B"],
        produced_mana=["B"],
    ),
    "R": Card(
        name="Mountain",
        type_line="Basic Land — Mountain",
        color_identity=["R"],
        produced_mana=["R"],
    ),
    "G": Card(
        name="Forest",
        type_line="Basic Land — Forest",
        color_identity=["G"],
        produced_mana=["G"],
    ),
}

DECK_SIZE = 40
MIN_LAND_COUNT = 15
MAX_LAND_COUNT = 19

# Colored mana symbol in a cost string. Hybrid {W/U} and phyrexian {W/P} are matched
# because the first character of either symbol is a single color letter.
_PIP_RE = re.compile(r"\{([WUBRG])(?:/[WUBRGP])?\}")


@dataclass
class BuiltDeck:
    """A finished 40-card deck assembled from a draft pool.

    Parameters
    ----------
    spells : list of Card
        Non-land cards taken from the pool.
    nonbasic_lands : list of Card
        Non-basic lands taken from the pool.
    basics : list of Card
        Basic lands allocated by the deckbuilder to fill the mana base.
    pip_counts : dict mapping str to int
        WUBRG pip counts found in ``spells`` - the basis for basic allocation.
    notes : list of str
        Adjustments the builder made (unmatched names, count clamps, padding).
    """

    spells: list[Card]
    nonbasic_lands: list[Card]
    basics: list[Card]
    pip_counts: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def cards(self) -> list[Card]:
        """All 40 cards in deck order: spells, then nonbasic lands, then basics."""
        return [*self.spells, *self.nonbasic_lands, *self.basics]


def count_pips(spells: list[Card]) -> dict[str, int]:
    """Count WUBRG mana symbols across the spells' mana costs.

    Hybrid (``{W/U}``) and phyrexian (``{W/P}``) symbols contribute one to the first
    color of the symbol — a conservative read that biases the mana base toward the
    symbols actually printed rather than guessing which half a hybrid will be paid for.
    """
    counts = dict.fromkeys(COLOR_ORDER, 0)
    for card in spells:
        for color in _PIP_RE.findall(card.mana_cost):
            counts[color] += 1
    return counts


def allocate_basics(pip_counts: dict[str, int], total: int) -> dict[str, int]:
    """Distribute ``total`` basics across colors by pip-share using largest remainders.

    A color with zero pips gets zero basics. If every color is zero (a colorless
    pool) the basics fall to Wastes-less default: split as evenly as possible across
    WUBRG, which is harmless because no spell needs them.
    """
    if total <= 0:
        return dict.fromkeys(COLOR_ORDER, 0)

    pip_total = sum(pip_counts.values())
    if pip_total == 0:
        base, rem = divmod(total, 5)
        return {c: base + (1 if i < rem else 0) for i, c in enumerate(COLOR_ORDER)}

    raw = {c: total * pip_counts[c] / pip_total for c in COLOR_ORDER}
    floors = {c: int(raw[c]) for c in COLOR_ORDER}
    remainder = total - sum(floors.values())
    # Largest-remainder method: hand out the leftover basics to the colors whose
    # fractional share is biggest, breaking ties by WUBRG order.
    leftovers = sorted(COLOR_ORDER, key=lambda c: (-(raw[c] - floors[c]), COLOR_ORDER.index(c)))
    for c in leftovers[:remainder]:
        floors[c] += 1
    return floors


def _resolve_names(names: list[str], pool: list[Card]) -> tuple[list[Card], list[str]]:
    """Match each name to a card in the remaining pool, exact first then fuzzy.

    Each match consumes its pool card so duplicate names take distinct copies. Returns
    the resolved cards in input order plus the names that could not be matched.

    Duplicates in the pool stack into a buffer keyed by lowercase name; each match
    pops one buffered instance. Built once instead of per-name so the loop is O(names
    + pool) rather than O(names * pool).
    """
    by_name: dict[str, list[Card]] = {}
    for card in pool:
        by_name.setdefault(card.name.lower(), []).append(card)

    resolved: list[Card] = []
    missed: list[str] = []
    for name in names:
        wanted = name.strip().lower()
        if not wanted:
            continue
        bucket = by_name.get(wanted)
        if bucket is None:
            close = difflib.get_close_matches(
                wanted, [k for k, v in by_name.items() if v], n=1, cutoff=0.85
            )
            bucket = by_name[close[0]] if close else None
        if bucket:
            resolved.append(bucket.pop(0))
        else:
            missed.append(name)
    return resolved, missed


def build_deck(
    pool: list[Card],
    spell_names: list[str],
    nonbasic_land_names: list[str],
    suggested_land_count: int,
) -> BuiltDeck:
    """Assemble a 40-card deck from a draft pool plus the LLM's selections.

    The land count is clamped to :data:`MIN_LAND_COUNT`-:data:`MAX_LAND_COUNT` and
    determines the spell budget (``40 - land_count``). Nonbasic lands are taken first;
    the remaining land slots become basics, allocated by colored-pip share of the
    spells. Spell selections that exceed the budget are trimmed in input order; if
    they fall short, the builder pads from the strongest remaining pool cards
    (highest CMC first, lands excluded).

    Parameters
    ----------
    pool : list of Card
        The seat's full drafted pool.
    spell_names : list of str
        Non-land cards the LLM chose for the maindeck.
    nonbasic_land_names : list of str
        Non-basic lands the LLM wants from the pool.
    suggested_land_count : int
        The LLM's target total land count. Clamped to the valid range.

    Returns
    -------
    BuiltDeck
        The 40-card deck plus pip counts and adjustment notes.
    """
    notes: list[str] = []

    land_count = max(MIN_LAND_COUNT, min(MAX_LAND_COUNT, suggested_land_count))
    if land_count != suggested_land_count:
        notes.append(
            f"land count {suggested_land_count} clamped to {land_count} "
            f"(allowed range {MIN_LAND_COUNT}-{MAX_LAND_COUNT})"
        )

    pool_lands = [c for c in pool if c.is_land]
    pool_spells = [c for c in pool if not c.is_land]

    nonbasics, missed_lands = _resolve_names(nonbasic_land_names, pool_lands)
    if missed_lands:
        notes.append(f"unresolved nonbasic land names: {', '.join(missed_lands)}")

    # If the LLM asked for more nonbasics than the land budget, keep the first few that
    # fit so the basic allocation has room to balance the mana base.
    if len(nonbasics) > land_count:
        notes.append(f"trimmed {len(nonbasics) - land_count} nonbasic land(s) over land budget")
        nonbasics = nonbasics[:land_count]

    spell_budget = DECK_SIZE - land_count
    spells, missed_spells = _resolve_names(spell_names, pool_spells)
    if missed_spells:
        notes.append(f"unresolved spell names: {', '.join(missed_spells)}")

    if len(spells) > spell_budget:
        notes.append(f"trimmed {len(spells) - spell_budget} spell(s) over spell budget")
        spells = spells[:spell_budget]
    elif len(spells) < spell_budget:
        # Drop one pool instance per already-chosen spell so duplicates are still
        # available for padding when the pool has them.
        unused = list(pool_spells)
        for card in spells:
            unused.remove(card)
        # Fall back to the strongest unused spells by descending CMC, then name for
        # determinism. Crude, but the LLM should rarely under-pick this far.
        filler = sorted(unused, key=lambda c: (-c.cmc, c.name))
        needed = spell_budget - len(spells)
        spells.extend(filler[:needed])
        notes.append(f"padded {needed} spell(s) from leftover pool to reach {spell_budget}")

    pip_counts = count_pips(spells)
    basic_alloc = allocate_basics(pip_counts, land_count - len(nonbasics))
    basics: list[Card] = []
    for color, count in basic_alloc.items():
        basics.extend([BASIC_LANDS[color]] * count)

    return BuiltDeck(
        spells=spells, nonbasic_lands=nonbasics, basics=basics, pip_counts=pip_counts, notes=notes
    )

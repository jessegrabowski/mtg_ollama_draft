import re
from dataclasses import dataclass

from mtg_drafting.cards import Card

# A removal-flavoured verb anywhere in the rules text. Order with "target" matters
# (Lightning Bolt has "deals ... target", Murder has "destroy target") so we check for
# presence of the verb and the word "target" independently rather than requiring an
# ordering.
_REMOVAL_VERB_RE = re.compile(
    r"\b(?:deals?|destroy|counters?|exiles?|returns?|sacrifices?)\b",
    re.IGNORECASE,
)
_NEG_PT_RE = re.compile(r"-\d+/-\d+")
# Sweeper pattern: "destroy all", "exile all", "destroy each", etc. Catches Wrath
# variants without needing a "target" keyword.
_SWEEPER_RE = re.compile(
    r"\b(?:destroy|exile)\s+(?:all|each)\b|deal damage to each",
    re.IGNORECASE,
)

# Evasion keywords as plain substrings against the lower-cased oracle text. Reach is
# defensive, not evasion, and stays off this list.
_EVASION_KEYWORDS = (
    "flying",
    "menace",
    "trample",
    "shadow",
    "horsemanship",
    "landwalk",
    "skulk",
    "fear",
    "intimidate",
    "can't be blocked",
    "unblockable",
)

# Ramp: produces extra mana, fetches a land out of the library, or "puts a land onto
# the battlefield". Excludes lands themselves (a tapland is not "ramp").
_ADD_MANA_RE = re.compile(r"\badd\s+(?:\{[^}]+\}|one mana|two mana)", re.IGNORECASE)
_FETCH_LAND_RE = re.compile(
    r"search\s+(?:your|target player's)\s+library\s+for[^.\n]*\bland",
    re.IGNORECASE,
)
_PUT_LAND_RE = re.compile(
    r"put[^.\n]*\bland[^.\n]*onto\s+the\s+battlefield",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CardTags:
    """Pre-computed classification for one card.

    Cached per simulation run so per-turn metric loops do not re-run the regex on
    every card on every turn.

    Parameters
    ----------
    name : str
        Card name, used as the key in tag lookups.
    is_interaction : bool
        True for targeted removal/counters/bounce and for sweepers.
    is_evasive : bool
        True for creatures with an evasion keyword in their oracle text.
    is_ramp : bool
        True for cards that produce extra mana or fetch a land.
    effective_power : int
        Numeric creature power, or 0 for non-creatures and ``*`` printings.
    """

    name: str
    is_interaction: bool
    is_evasive: bool
    is_ramp: bool
    effective_power: int


def is_interaction(card: Card) -> bool:
    """True if the card removes, counters, bounces, or sweeps the opponent's stuff.

    Heuristic: a sweeper phrase always counts; otherwise the card needs both a
    targeted reference (``"target"`` somewhere in the text) and a removal-flavoured
    verb or a negative-power-toughness pattern. This filters out cards that mention
    ``"target"`` but only affect draw / pumping / search (``"target opponent draws a
    card"`` does not count).
    """
    if card.is_land or not card.oracle_text:
        return False
    text = card.oracle_text
    if _SWEEPER_RE.search(text):
        return True
    has_target = "target" in text.lower()
    has_removal = bool(_REMOVAL_VERB_RE.search(text) or _NEG_PT_RE.search(text))
    return has_target and has_removal


def is_evasive(card: Card) -> bool:
    """True if the card is a creature with an evasion keyword in its rules text."""
    if not card.is_creature or not card.oracle_text:
        return False
    text = card.oracle_text.lower()
    return any(kw in text for kw in _EVASION_KEYWORDS)


def is_ramp(card: Card) -> bool:
    """True if the card produces extra mana or fetches a land. Lands themselves are
    excluded - a basic or a tapland is mana, not ramp."""
    if card.is_land or not card.oracle_text:
        return False
    text = card.oracle_text
    return bool(
        _ADD_MANA_RE.search(text) or _FETCH_LAND_RE.search(text) or _PUT_LAND_RE.search(text)
    )


def effective_power(card: Card) -> int:
    """Numeric creature power, or 0 for non-creatures and unparseable printings."""
    if not card.is_creature or card.power is None:
        return 0
    try:
        return int(card.power)
    except ValueError:
        return 0


def tag(card: Card) -> CardTags:
    """Bundle all classifications for one card into a :class:`CardTags`."""
    return CardTags(
        name=card.name,
        is_interaction=is_interaction(card),
        is_evasive=is_evasive(card),
        is_ramp=is_ramp(card),
        effective_power=effective_power(card),
    )

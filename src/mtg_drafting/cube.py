import re
from pathlib import Path

from mtg_drafting.cards import Card
from mtg_drafting.scryfall import ScryfallIndex

# A leading quantity ("1 ", "4x ") and a trailing set tag ("(MH2) 123") as exported by
# Arena / MTGO / CubeCobra.
_QUANTITY_RE = re.compile(r"^\s*(\d+)\s*x?\s+", re.IGNORECASE)
_SET_TAG_RE = re.compile(r"\s*\([A-Za-z0-9]{2,6}\)\s*\d*\s*$")


def _parse_line(line: str) -> tuple[int, str]:
    """Parse one cube-list line into a ``(quantity, card name)`` pair.

    Drops a trailing ``#`` comment and a trailing ``(SET) num`` tag. A leading integer
    is read as a copy count, so ``"3 Sol Ring"`` yields three copies; without one the
    quantity is 1. A blank or comment-only line returns ``(0, "")``.
    """
    line = line.split("#", 1)[0].strip()
    if not line:
        return 0, ""
    line = _SET_TAG_RE.sub("", line).strip()
    match = _QUANTITY_RE.match(line)
    if match:
        return int(match.group(1)), line[match.end() :].strip()
    return 1, line


def load_cube(path: Path, index: ScryfallIndex) -> tuple[list[Card], list[str]]:
    """Parse a cube list and resolve every entry against the Scryfall index.

    The file is plain text: one card per line, ``#`` starts a comment, blank lines are
    ignored. A line may carry a copy count (``"4 Lightning Bolt"``) and a trailing
    ``(SET) num`` tag. Duplicates are kept — both a copy count and the same name
    repeated on several lines add multiple copies to the cube.

    Parameters
    ----------
    path : Path
        Cube list file.
    index : ScryfallIndex
        Card lookup built from a Scryfall snapshot.

    Returns
    -------
    cards : list of Card
        Resolved cards, in file order, including duplicates.
    unresolved : list of str
        Names that could not be matched, in file order.
    """
    cards: list[Card] = []
    unresolved: list[str] = []

    for raw in path.read_text().splitlines():
        quantity, name = _parse_line(raw)
        if not name:
            continue
        card = index.get(name)
        if card is None:
            unresolved.append(name)
            continue
        cards.extend([card] * quantity)

    return cards, unresolved

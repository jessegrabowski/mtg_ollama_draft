from collections import deque
from dataclasses import dataclass

from mtg_drafting.cards import Card


@dataclass
class PickRecord:
    """One completed pick at the table.

    Parameters
    ----------
    round_no : int
        Zero-based round (pack) index.
    pick_no : int
        Zero-based pick index within the round.
    pack : list of str
        Names of every card in the pack as it was offered to the seat.
    chosen : str
        Name of the card taken.
    reasoning : str
        The model's stated reason for the pick.
    """

    round_no: int
    pick_no: int
    pack: list[str]
    chosen: str
    reasoning: str


class Seat:
    """A draft seat: a card pool plus the rolling context the LLM sees as this seat.

    The LLM plays every seat, so each seat owns the state that makes its picks
    coherent — the cards taken so far, a running strategy summary the model rewrites
    each pick, and a bounded window of recent picks kept verbatim in the prompt.

    Parameters
    ----------
    index : int
        Seat position at the table, ``0`` to ``n_seats - 1``.
    history_maxlen : int
        Recent picks kept verbatim for prompt context. ``picks`` always keeps the full
        history regardless of this bound.
    """

    def __init__(self, index: int, history_maxlen: int):
        self.index = index
        self.pool: list[Card] = []
        self.strategy: str = ""
        self.picks: list[PickRecord] = []
        self.recent: deque[PickRecord] = deque(maxlen=history_maxlen)

    def add_pick(self, card: Card, record: PickRecord) -> None:
        """Record a taken card and append the pick to both history views."""
        self.pool.append(card)
        self.picks.append(record)
        self.recent.append(record)

    def __repr__(self) -> str:
        return f"Seat(index={self.index}, picks={len(self.pool)})"

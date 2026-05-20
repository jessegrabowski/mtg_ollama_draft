from collections import deque
from dataclasses import dataclass

from mtg_drafting.cards import Card
from mtg_drafting.strategist import StrategyState


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
    hoping_to_wheel : str, optional
        Name of a card in ``pack`` the picker hopes will wheel back. Defaults to
        None when the picker did not name a wheel candidate. Default None.
    """

    round_no: int
    pick_no: int
    pack: list[str]
    chosen: str
    reasoning: str
    hoping_to_wheel: str | None = None


class Seat:
    """A draft seat: a card pool plus the rolling context the LLM sees as this seat.

    The LLM plays every seat. Each seat owns the cards taken so far, a structured
    strategy plan produced by the strategist at pack boundaries, the full pick history,
    and a bounded window of recent picks kept verbatim in the prompt.

    Parameters
    ----------
    index : int
        Seat position at the table, ``0`` to ``n_seats - 1``.
    history_maxlen : int
        Recent picks kept verbatim for prompt context. ``picks`` always keeps the full
        history regardless of this bound.
    """

    def __init__(self, index: int, history_maxlen: int, memory_maxlen: int = 10):
        self.index = index
        self.pool: list[Card] = []
        self.strategy_state: StrategyState | None = None
        self.sidelined: set[str] = set()
        self.picks: list[PickRecord] = []
        self.recent: deque[PickRecord] = deque(maxlen=history_maxlen)
        # Free-form notes the picker, strategist, and code-side wheel detector can
        # append across the draft. Oldest notes drop off when the deque fills.
        self.memory: deque[str] = deque(maxlen=memory_maxlen)

    def add_pick(self, card: Card, record: PickRecord) -> None:
        """Record a taken card and append the pick to both history views."""
        self.pool.append(card)
        self.picks.append(record)
        self.recent.append(record)

    @property
    def maindeck(self) -> list[Card]:
        """Cards in the proposed main deck: pool minus anything the strategist has
        set aside in its latest classification. New picks default to the maindeck
        until the next strategist call has the chance to move them."""
        return [c for c in self.pool if c.name not in self.sidelined]

    @property
    def sideboard(self) -> list[Card]:
        """Cards the strategist has classified as sideboard or chaff. The picker does
        not see these when reasoning about the deck's curve, colors, or roles."""
        return [c for c in self.pool if c.name in self.sidelined]

    @property
    def strategy_summary(self) -> str:
        """One-line human-readable summary of the strategist's current plan, or a
        placeholder when the strategist has not yet produced a plan.

        Uses the primary (highest-weight) direction; appends a count of alternatives
        when the strategist returned more than one candidate. An ``open``-role
        primary collapses to just ``"uncommitted"`` rather than the redundant
        ``"open  open"`` of commitment + role."""
        s = self.strategy_state
        if s is None:
            return "(no plan yet)"
        primary = s.primary
        extra = f" (+{len(s.directions) - 1} alt)" if len(s.directions) > 1 else ""
        if primary.role == "open":
            return f"uncommitted{extra}"
        colors = "/".join(primary.colors) if primary.colors else "open"
        return f"{s.color_commitment} {colors} {primary.role}{extra}"

    def __repr__(self) -> str:
        return f"Seat(index={self.index}, picks={len(self.pool)})"


def format_memory(seat: Seat) -> str:
    """Render a seat's draft memory deque newest-first as a numbered text block.

    Shared by the picker and strategist prompts so the two see identical formatting.
    Returns a placeholder line when the deque is empty so the prompt block is never
    blank."""
    if not seat.memory:
        return "  (no notes yet)"
    return "\n".join(
        f"  {i + 1}. {note}" for i, note in enumerate(reversed(seat.memory))
    )

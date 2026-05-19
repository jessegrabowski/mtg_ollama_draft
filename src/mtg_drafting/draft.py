from collections.abc import Callable

from mtg_drafting.cards import Card
from mtg_drafting.config import DraftConfig
from mtg_drafting.packs import Rounds
from mtg_drafting.seat import PickRecord, Seat

# Chooses a card from a pack. Receives the seat, the pack, and the round / pick
# indices; returns the taken card and a short reason for the pick.
PickFn = Callable[[Seat, list[Card], int, int], tuple[Card, str]]

# Callback fired after each completed pick, for progress display or streaming output.
PickHook = Callable[[Seat, PickRecord], None]


def pass_direction(round_no: int) -> int:
    """Return the pass direction for a round: ``+1`` passes to the next-higher seat
    index, ``-1`` to the next-lower. Direction alternates each round, as in a real
    draft (left, right, left)."""
    return 1 if round_no % 2 == 0 else -1


def rotate_packs(packs: list[list[Card]], direction: int) -> list[list[Card]]:
    """Pass every pack one seat along. With ``direction == +1`` the pack at seat ``i``
    moves to seat ``i + 1``."""
    n = len(packs)
    return [packs[(j - direction) % n] for j in range(n)]


class DraftState:
    """Drives the mechanics of a draft: dealing rounds, calling for picks, and passing
    packs around the table.

    The state machine holds no LLM knowledge — picks are delegated to an injected
    ``PickFn`` — so it can be exercised with a deterministic picker in tests.

    Parameters
    ----------
    config : DraftConfig
        Seat count, round count, pack size, and history bound.
    rounds : list of list of list of Card
        Pre-dealt packs from :func:`mtg_drafting.packs.generate_packs`.
    """

    def __init__(self, config: DraftConfig, rounds: Rounds):
        self.config = config
        self.rounds = rounds
        self.seats = [Seat(i, config.history_maxlen) for i in range(config.n_seats)]
        self.packs_in_flight: list[list[Card]] = []
        self.round_no = -1

    def run(self, pick_fn: PickFn, on_pick: PickHook | None = None) -> None:
        """Run the draft to completion, calling ``pick_fn`` for every pick.

        Within a round all seats pick from the pack in front of them, then every pack
        passes one seat along; this repeats until the packs are empty, then the next
        round is opened.

        Parameters
        ----------
        pick_fn : callable
            Chooses a card for a seat from its current pack.
        on_pick : callable, optional
            Invoked with ``(seat, record)`` after each pick. Default None.
        """
        for round_no in range(self.config.packs_per_round):
            self.round_no = round_no
            self.packs_in_flight = [list(pack) for pack in self.rounds[round_no]]
            direction = pass_direction(round_no)

            for pick_no in range(self.config.cards_per_pack):
                for seat in self.seats:
                    pack = self.packs_in_flight[seat.index]
                    pack_snapshot = [card.name for card in pack]
                    card, reasoning = pick_fn(seat, pack, round_no, pick_no)
                    pack.remove(card)
                    record = PickRecord(
                        round_no=round_no,
                        pick_no=pick_no,
                        pack=pack_snapshot,
                        chosen=card.name,
                        reasoning=reasoning,
                    )
                    seat.add_pick(card, record)
                    if on_pick is not None:
                        on_pick(seat, record)

                self.packs_in_flight = rotate_packs(self.packs_in_flight, direction)

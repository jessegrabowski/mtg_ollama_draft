import random

from mtg_drafting.cards import Card
from mtg_drafting.config import DraftConfig

# A draft's packs: rounds[r][s] is the pack seat s opens in round r.
Rounds = list[list[list[Card]]]


def generate_packs(cube: list[Card], config: DraftConfig) -> tuple[Rounds, int]:
    """Shuffle a cube and deal it into packs for every round and seat.

    Parameters
    ----------
    cube : list of Card
        The cube to draft. Must hold at least ``config.min_cube_size`` cards.
    config : DraftConfig
        Supplies seat count, rounds, pack size, and the shuffle seed.

    Returns
    -------
    rounds : list of list of list of Card
        ``rounds[r][s]`` is the pack seat ``s`` opens in round ``r``.
    seed : int
        The seed actually used — the configured one, or a freshly drawn random seed
        when ``config.seed`` is None. Recorded so the run can be reproduced.
    """
    if len(cube) < config.min_cube_size:
        raise ValueError(
            f"Cube has {len(cube)} cards but {config.min_cube_size} are needed for "
            f"{config.n_seats} seats x {config.packs_per_round} packs x "
            f"{config.cards_per_pack} cards."
        )

    seed = config.seed if config.seed is not None else random.randrange(2**32)
    pool = list(cube)
    random.Random(seed).shuffle(pool)

    size = config.cards_per_pack
    rounds: Rounds = []
    cursor = 0
    for _ in range(config.packs_per_round):
        round_packs: list[list[Card]] = []
        for _ in range(config.n_seats):
            round_packs.append(pool[cursor : cursor + size])
            cursor += size
        rounds.append(round_packs)

    return rounds, seed

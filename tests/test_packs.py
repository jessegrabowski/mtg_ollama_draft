import pytest

from mtg_drafting.config import DraftConfig
from mtg_drafting.packs import generate_packs


@pytest.mark.parametrize(
    "config",
    [DraftConfig(), DraftConfig(n_seats=4, packs_per_round=2, cards_per_pack=10)],
    ids=["default", "four-seat"],
)
def test_pack_shapes(cube, config):
    rounds, _ = generate_packs(cube, config)

    assert len(rounds) == config.packs_per_round
    for round_packs in rounds:
        assert len(round_packs) == config.n_seats
        for pack in round_packs:
            assert len(pack) == config.cards_per_pack


def test_all_dealt_cards_distinct_and_from_cube(cube):
    config = DraftConfig()
    rounds, _ = generate_packs(cube, config)

    dealt = [card for round_packs in rounds for pack in round_packs for card in pack]
    names = {card.name for card in dealt}
    assert len(names) == len(dealt) == config.min_cube_size
    assert names <= {card.name for card in cube}


def test_small_cube_rejected(cube):
    config = DraftConfig()
    with pytest.raises(ValueError, match="needed"):
        generate_packs(cube[: config.min_cube_size - 1], config)


def test_cube_of_exactly_min_size_is_accepted(cube):
    config = DraftConfig()
    rounds, _ = generate_packs(cube[: config.min_cube_size], config)

    dealt = [card for round_packs in rounds for pack in round_packs for card in pack]
    assert len(dealt) == config.min_cube_size


def test_seed_is_reproducible(cube):
    config = DraftConfig(seed=1234)
    rounds_a, seed_a = generate_packs(cube, config)
    rounds_b, seed_b = generate_packs(cube, config)

    assert seed_a == seed_b == 1234
    names_a = [c.name for rp in rounds_a for p in rp for c in p]
    names_b = [c.name for rp in rounds_b for p in rp for c in p]
    assert names_a == names_b


def test_random_seed_recorded(cube):
    rounds, seed = generate_packs(cube, DraftConfig(seed=None))
    assert isinstance(seed, int)
    # The recorded seed reproduces the same deal.
    replay, _ = generate_packs(cube, DraftConfig(seed=seed))
    assert [c.name for rp in rounds for p in rp for c in p] == [
        c.name for rp in replay for p in rp for c in p
    ]

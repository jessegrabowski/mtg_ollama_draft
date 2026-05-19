from mtg_drafting.config import DraftConfig
from mtg_drafting.draft import DraftState, pass_direction, rotate_packs
from mtg_drafting.packs import generate_packs


def pick_first(seat, pack, round_no, pick_no):
    """Deterministic picker: always take the first card in the pack."""
    return pack[0], "first card"


def test_pass_direction_alternates():
    assert pass_direction(0) == 1
    assert pass_direction(1) == -1
    assert pass_direction(2) == 1


def test_rotate_packs_left():
    packs = [["a"], ["b"], ["c"], ["d"]]
    rotated = rotate_packs(packs, direction=1)
    # Seat 0's pack moves to seat 1, etc.; seat 0 receives seat 3's pack.
    assert rotated == [["d"], ["a"], ["b"], ["c"]]


def test_rotate_packs_right():
    packs = [["a"], ["b"], ["c"], ["d"]]
    rotated = rotate_packs(packs, direction=-1)
    assert rotated == [["b"], ["c"], ["d"], ["a"]]


def test_every_seat_fills_its_pool(cube):
    config = DraftConfig()
    rounds, _ = generate_packs(cube, config)
    state = DraftState(config, rounds)
    state.run(pick_first)

    for seat in state.seats:
        assert len(seat.pool) == config.picks_per_seat
        assert len(seat.picks) == config.picks_per_seat


def test_each_card_drafted_exactly_once(cube):
    config = DraftConfig()
    rounds, _ = generate_packs(cube, config)
    state = DraftState(config, rounds)
    state.run(pick_first)

    drafted = [card.name for seat in state.seats for card in seat.pool]
    assert len(drafted) == config.min_cube_size
    assert len(set(drafted)) == config.min_cube_size


def test_on_pick_hook_fires_for_every_pick(cube):
    config = DraftConfig()
    rounds, _ = generate_packs(cube, config)
    state = DraftState(config, rounds)

    seen = []
    state.run(pick_first, on_pick=lambda seat, record: seen.append((seat.index, record)))
    assert len(seen) == config.min_cube_size


def test_pack_snapshot_shrinks_through_a_round(cube):
    config = DraftConfig()
    rounds, _ = generate_packs(cube, config)
    state = DraftState(config, rounds)

    seat0_packs = []
    state.run(
        pick_first,
        on_pick=lambda seat, record: seat0_packs.append(len(record.pack))
        if seat.index == 0 and record.round_no == 0
        else None,
    )
    # Seat 0 sees a 15-card pack, then 14, ... as picks deplete the round's packs.
    assert seat0_packs == list(range(config.cards_per_pack, 0, -1))

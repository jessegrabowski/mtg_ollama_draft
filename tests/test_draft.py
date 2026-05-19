from mtg_drafting.cards import Card
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

    seat0_pack_sizes = []

    def record_seat0_round0(seat, record):
        if seat.index == 0 and record.round_no == 0:
            seat0_pack_sizes.append(len(record.pack))

    state.run(pick_first, on_pick=record_seat0_round0)
    # Seat 0 sees a 15-card pack, then 14, ... as picks deplete the round's packs.
    assert seat0_pack_sizes == list(range(config.cards_per_pack, 0, -1))


def test_recent_picks_are_bounded_while_full_history_is_kept(cube):
    config = DraftConfig(history_maxlen=5)
    rounds, _ = generate_packs(cube, config)
    state = DraftState(config, rounds)
    state.run(pick_first)

    for seat in state.seats:
        assert len(seat.picks) == config.picks_per_seat
        assert len(seat.recent) == config.history_maxlen


def test_packs_pass_in_alternating_directions():
    config = DraftConfig(n_seats=3, packs_per_round=2, cards_per_pack=3)

    def pack(round_no, seat_no):
        return [Card(name=f"r{round_no}s{seat_no}c{c}") for c in range(config.cards_per_pack)]

    rounds = [[pack(r, s) for s in range(config.n_seats)] for r in range(config.packs_per_round)]
    state = DraftState(config, rounds)

    seat0_pack_origins: dict[int, list[str]] = {0: [], 1: []}

    def record_seat0_origins(seat, record):
        if seat.index == 0:
            seat0_pack_origins[record.round_no].append(record.pack[0][:4])

    state.run(pick_first, on_pick=record_seat0_origins)

    # Round 0 passes toward higher seats: seat 0 receives packs opened at 0, 2, 1.
    assert seat0_pack_origins[0] == ["r0s0", "r0s2", "r0s1"]
    # Round 1 reverses: seat 0 receives packs opened at 0, 1, 2.
    assert seat0_pack_origins[1] == ["r1s0", "r1s1", "r1s2"]

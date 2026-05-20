import random
import re
from dataclasses import dataclass, field
from statistics import median

from mtg_drafting.cards import COLOR_ORDER, Card
from mtg_drafting.tags import CardTags, tag

# Every {...} mana symbol in a cost string.
_SYMBOL_RE = re.compile(r"\{([^}]+)\}")
# Single-color pip used by the land-play heuristic to read pip demand from cost strings.
_SOLO_PIP_RE = re.compile(r"\{([WUBRG])(?:/[WUBRGP])?\}")

DEFAULT_TURN_HORIZON = 8
DEFAULT_SIMS = 1000
MULLIGAN_LAND_RANGE = (2, 5)


def parse_cost(mana_cost: str) -> tuple[list[frozenset[str]], int]:
    """Parse a mana-cost string into ``(colored, generic)``.

    ``colored`` is a list with one entry per colored mana symbol; each entry is the
    set of WUBRG colors that symbol can be paid with - a singleton ``{"W"}`` for
    ``{W}``, a pair ``{"W","U"}`` for hybrid ``{W/U}``, and a singleton again for
    phyrexian ``{W/P}`` (life payment is ignored, biasing the simulator toward mana
    payment). ``generic`` is the total of all numeric pips. ``{X}`` contributes
    zero. Unrecognised symbols (``{C}``, ``{S}``, twobrid ``{2/W}``) fall back to
    one generic - pessimistic rather than optimistic.

    The ``(colored, generic)`` order matches :func:`can_pay`'s positional argument
    order so callers can spread the tuple directly.
    """
    generic = 0
    colored: list[frozenset[str]] = []
    for sym in _SYMBOL_RE.findall(mana_cost):
        sym = sym.upper()
        if sym.isdigit():
            generic += int(sym)
        elif sym == "X":
            continue
        elif sym in {"W", "U", "B", "R", "G"}:
            colored.append(frozenset({sym}))
        elif "/" in sym:
            parts = sym.split("/")
            colors = frozenset(p for p in parts if p in {"W", "U", "B", "R", "G"})
            # Treat anything without a recognisable color half (and any twobrid form)
            # as generic to stay on the conservative side of castability.
            if colors and "P" in parts:
                colored.append(colors)
            elif colors and not any(p.isdigit() for p in parts):
                colored.append(colors)
            else:
                generic += 1
        else:
            generic += 1
    return colored, generic


def can_pay(sources: list[frozenset[str]], colored: list[frozenset[str]], generic: int) -> bool:
    """Can these mana sources pay a cost?

    Each colored symbol must be assigned to a distinct source whose producible colors
    include one of the symbol's allowed colors; any unused source contributes to the
    generic cost. Solves the assignment by recursive backtracking, which is fine for
    the small sizes seen here (≤10 lands, ≤6 colored symbols)."""
    if len(sources) < len(colored) + generic:
        return False
    if not colored:
        return True
    sym = colored[0]
    rest = colored[1:]
    for i, src in enumerate(sources):
        if src & sym:
            if can_pay(sources[:i] + sources[i + 1 :], rest, generic):
                return True
    return False


def _land_sources(land: Card) -> frozenset[str]:
    """Mana colors this land can produce, as a frozenset for set-operations during
    castability checks. Backed by :attr:`Card.producible_colors`."""
    return frozenset(land.producible_colors)


def _pick_best_land(in_hand: list[Card], hand: list[Card], battlefield: list[Card]) -> Card:
    """Greedy land choice: cover the color the hand still needs most.

    Counts colored pips across non-land cards in hand and subtracts what the battle-
    field already supplies (one count per source per color it produces). Picks the
    land whose color menu reduces the largest remaining shortage. Ties break by WUBRG
    order, which keeps the choice deterministic at fixed seed."""
    if len(in_hand) == 1:
        return in_hand[0]

    need = dict.fromkeys(COLOR_ORDER, 0)
    for card in hand:
        if card.is_land:
            continue
        for color in _SOLO_PIP_RE.findall(card.mana_cost):
            need[color] += 1

    have = dict.fromkeys(COLOR_ORDER, 0)
    for land in battlefield:
        for color in _land_sources(land):
            have[color] += 1

    shortage = {c: max(0, need[c] - have[c]) for c in COLOR_ORDER}

    def score(land: Card) -> tuple[int, int]:
        produces = _land_sources(land)
        gain = sum(shortage[c] for c in produces)
        # Break ties by WUBRG order: prefer the land whose first producible color comes
        # earliest. Deterministic at fixed seed.
        order = min((COLOR_ORDER.index(c) for c in produces), default=99)
        return gain, -order

    return max(in_hand, key=score)


@dataclass
class SampleHand:
    """A deterministic draw used to show the LLM evaluator a concrete opening.

    Parameters
    ----------
    opening : list of Card
        Cards kept after the mulligan rule fires (6 or 7).
    upcoming_draws : list of Card
        The next cards that would be drawn on turns 2, 3, … in order.
    mulliganed : bool
        True if the opener required a mulligan; the player would be on 6 cards.
    """

    opening: list[Card]
    upcoming_draws: list[Card]
    mulliganed: bool


def generate_sample_hands(
    deck: list[Card],
    n: int = 1,
    draws_per_hand: int = 3,
    seed: int = 0,
) -> list[SampleHand]:
    """Generate reproducible opening hands plus the next few draws each.

    Each hand uses an independent RNG seeded from ``seed`` so the same call always
    returns the same hands - useful for showing the LLM a concrete example that does
    not drift between runs.

    Parameters
    ----------
    deck : list of Card
        The 40-card built deck to draw from.
    n : int, optional
        Number of hands to generate. Default 1.
    draws_per_hand : int, optional
        Cards drawn after the opener (one per turn from turn 2 onward). Default 3.
    seed : int, optional
        Base RNG seed; each hand's RNG uses ``seed + i``. Default 0.

    Returns
    -------
    list of SampleHand
        ``n`` sample hands in generation order.
    """
    hands: list[SampleHand] = []
    for i in range(n):
        rng = random.Random(seed + i)
        opening, library = _opening_hand(deck, rng)
        hands.append(
            SampleHand(
                opening=opening,
                upcoming_draws=library[:draws_per_hand],
                mulliganed=len(opening) < 7,
            )
        )
    return hands


def _opening_hand(deck: list[Card], rng: random.Random) -> tuple[list[Card], list[Card]]:
    """Draw an opening hand with a single London-style mulligan if the land count is
    outside :data:`MULLIGAN_LAND_RANGE`. The mulligan goes 7 → 6 only; deeper digging
    is rare enough that the extra simulation cost is not worth the precision."""
    low, high = MULLIGAN_LAND_RANGE
    for size in (7, 6):
        cards = list(deck)
        rng.shuffle(cards)
        hand = cards[:size]
        if low <= sum(1 for c in hand if c.is_land) <= high or size == 6:
            return hand, cards[size:]
    raise AssertionError("unreachable")  # pragma: no cover


def _simulate_game(
    deck: list[Card],
    rng: random.Random,
    n_turns: int,
    tags_by_name: dict[str, CardTags],
    costs_by_name: dict[str, tuple[list[frozenset[str]], int]],
    peak_requirement: tuple[str, int, str] | None,
) -> dict:
    """Play out one game: draw, mulligan once, play lands greedily, and record the
    metrics aggregated by :func:`simulate_deck`.

    Three metric families come out of each game: (1) per-card earliest castable turn
    (mana-base only, ignores draw), (2) per-turn 'on-curve spell cast from hand', and
    (3) per-turn counts/sums of drawn-and-castable cards in each tag category. (3) is
    computed against the *drawn* pile rather than the hand, so spells consumed by the
    curve-hit check still contribute to the by-turn availability snapshot - this gives
    an upper bound on what the deck could deploy by each turn."""
    hand, library = _opening_hand(deck, rng)
    opening_size = len(hand)
    drawn: list[Card] = list(hand)
    battlefield: list[Card] = []
    cast_turn: dict[str, int] = {}
    curve_hit: dict[int, bool] = {}
    interaction_by_turn: dict[int, int] = {}
    creature_power_by_turn: dict[int, int] = {}
    evasion_by_turn: dict[int, int] = {}
    castable_hand_fraction: dict[int, float] = {}
    lands_in_hand_by_turn: dict[int, int] = {}
    nonland_in_hand_by_turn: dict[int, int] = {}
    peak_color_hit_by_turn: dict[int, bool] = {}

    for turn in range(1, n_turns + 1):
        if turn > 1 and library:
            new_card = library.pop(0)
            hand.append(new_card)
            drawn.append(new_card)

        lands_in_hand = [c for c in hand if c.is_land]
        if lands_in_hand:
            chosen = _pick_best_land(lands_in_hand, hand, battlefield)
            hand.remove(chosen)
            battlefield.append(chosen)

        sources = [_land_sources(land) for land in battlefield]

        # Hand composition: lands vs spells, captured after the land drop. Lets the
        # report tell flooding apart from screwing - same low cast fraction, opposite
        # land/spell ratios.
        lands_in_hand_by_turn[turn] = sum(1 for c in hand if c.is_land)
        nonland_in_hand_by_turn[turn] = sum(1 for c in hand if not c.is_land)

        # Peak color requirement: do the lands in play satisfy the deck's heaviest
        # single-color pip count? Tracks just the color requirement, not total mana,
        # so {3}{B}{B}{B} flips True as soon as the player has 3 Swamps even though
        # the spell needs T7 to actually cast.
        if peak_requirement is None:
            peak_color_hit_by_turn[turn] = True
        else:
            target_color, target_count, _ = peak_requirement
            sources_of_color = sum(1 for src in sources if target_color in src)
            peak_color_hit_by_turn[turn] = sources_of_color >= target_count

        # Per-card metric: first turn the mana base could pay for this card. Asks "if
        # I had drawn it, could I cast it on curve?" - decoupled from draw luck.
        for card in deck:
            if card.is_land or card.name in cast_turn:
                continue
            colored, generic = costs_by_name[card.name]
            if can_pay(sources, colored, generic):
                cast_turn[card.name] = turn

        # Castable hand fraction: snapshot at start of main phase, before any curve
        # cast consumes a spell. Counts non-land cards in hand whose cost is paid by
        # the current mana sources. Denominator is the full hand size so lands count
        # against the ratio - a flooded hand drops the fraction just as a screwed one
        # does, which is what makes this metric a joint curve-and-mana-base measure.
        if hand:
            castable_now = sum(
                1 for c in hand
                if (not c.is_land) and can_pay(sources, *costs_by_name[c.name])
            )
            castable_hand_fraction[turn] = castable_now / len(hand)
        else:
            castable_hand_fraction[turn] = 0.0

        # Per-turn curve metric: try to actually cast a CMC = turn spell from hand and
        # remove it. Casting a spell consumes it; otherwise an early bomb left in hand
        # would falsely satisfy every later turn's curve check.
        castable_curve = next(
            (
                c
                for c in hand
                if (not c.is_land)
                and int(c.cmc) == turn
                and can_pay(sources, *costs_by_name[c.name])
            ),
            None,
        )
        if castable_curve is not None:
            hand.remove(castable_curve)
            curve_hit[turn] = True
        else:
            curve_hit[turn] = False

        # Tag-availability metrics: walk the drawn pile, count cards whose mana base
        # is already satisfied. Interaction is a count, power is a sum, evasion is a
        # count of evasive creatures.
        interaction = 0
        power = 0
        evasion = 0
        for card in drawn:
            if card.is_land:
                continue
            colored, generic = costs_by_name[card.name]
            if not can_pay(sources, colored, generic):
                continue
            t = tags_by_name[card.name]
            if t.is_interaction:
                interaction += 1
            power += t.effective_power
            if t.is_evasive:
                evasion += 1
        interaction_by_turn[turn] = interaction
        creature_power_by_turn[turn] = power
        evasion_by_turn[turn] = evasion

    return {
        "cast_turn": cast_turn,
        "curve_hit": curve_hit,
        "interaction_by_turn": interaction_by_turn,
        "creature_power_by_turn": creature_power_by_turn,
        "evasion_by_turn": evasion_by_turn,
        "castable_hand_fraction": castable_hand_fraction,
        "lands_in_hand": lands_in_hand_by_turn,
        "nonland_in_hand": nonland_in_hand_by_turn,
        "peak_color_hit": peak_color_hit_by_turn,
        "opening_size": opening_size,
    }


@dataclass
class CardCastability:
    """Per-card playability summary across N simulated games.

    Parameters
    ----------
    name : str
        Card name.
    cmc : int
        Card mana value, used to compute the on-curve target turn.
    castable_on_curve : float
        Fraction of games where the mana base could cast the card by turn ``cmc + 1``.
    castable_by_horizon : float
        Fraction of games where the mana base could cast the card by the simulation
        horizon. ``< 1`` means some games never produced the right colors.
    median_turn : int
        Median earliest turn (across games that ever cast it) the mana base supported
        the card. Useful for spotting cards that come online late even when they do.
    """

    name: str
    cmc: int
    castable_on_curve: float
    castable_by_horizon: float
    median_turn: int


def peak_color_requirement(deck: list[Card]) -> tuple[str, int, str] | None:
    """The deck's heaviest single-color pip count and the spell that forces it.

    Returns ``(color, count, spell_name)`` for the worst offender, or None when the
    deck has no colored spells. Ties break by WUBRG order then by alphabetic spell
    name so the result is reproducible."""
    best: tuple[str, int, str] | None = None
    for card in deck:
        if card.is_land or not card.mana_cost:
            continue
        counts = dict.fromkeys(COLOR_ORDER, 0)
        for color in _SOLO_PIP_RE.findall(card.mana_cost):
            counts[color] += 1
        for color in COLOR_ORDER:
            count = counts[color]
            if count == 0:
                continue
            candidate = (color, count, card.name)
            if best is None or candidate[1] > best[1]:
                best = candidate
    return best


@dataclass
class PlayabilityReport:
    """Aggregated playability simulation results for a built deck.

    Parameters
    ----------
    sims : int
        Number of games simulated.
    horizon : int
        Last turn each game ran out to.
    per_card : list of CardCastability
        One entry per unique non-land card name in the deck, sorted by descending CMC
        then ascending on-curve rate so the most stranded curve-toppers float to the
        top.
    curve_hit : dict mapping int to float
        Turn -> fraction of games where the deck had a CMC-equal spell in hand AND the
        mana to cast it. Turns with no spell of that CMC in the deck are absent.
    interaction_by_turn : dict mapping int to float
        Turn -> expected count of drawn-and-castable interaction cards (targeted
        removal / counters / bounce / sweepers). Upper bound: "how often could you
        answer something on this turn."
    creature_power_by_turn : dict mapping int to float
        Turn -> expected sum of power across drawn-and-castable creatures. Upper
        bound: "how big could your board be if you cast everything on curve."
    evasion_by_turn : dict mapping int to float
        Turn -> expected count of drawn-and-castable evasive creatures (flying, menace,
        trample, etc.).
    castable_hand_fraction_by_turn : dict mapping int to float
        Turn -> average fraction of hand-at-main-phase that is non-land and castable
        with the current mana sources. A joint curve + mana-base smoothness metric.
    lands_in_hand_by_turn : dict mapping int to float
        Turn -> average number of lands left in hand after the land drop. Pair with
        ``nonland_in_hand_by_turn`` to spot flooding vs screwing.
    nonland_in_hand_by_turn : dict mapping int to float
        Turn -> average number of non-land cards in hand.
    peak_color_requirement : tuple of (str, int, str) or None
        ``(color, count, spell_name)`` of the deck's heaviest single-color pip
        demand. None for a colorless deck.
    peak_color_hit_by_turn : dict mapping int to float
        Turn -> fraction of games where the battlefield held ≥ ``count`` lands
        producing the peak color. Tracks the color requirement only, not the total
        mana cost.
    mulligan_rate : float
        Fraction of games that took a mulligan (opening hand kept at 6 instead of 7).
    average_opening_hand_size : float
        Mean opening hand size across sims. A direct readout of how often the mulligan
        rule fires (``7 - mulligan_rate``).
    stranded : list of str
        Names whose ``castable_by_horizon`` is below 0.5 - cards the mana base cannot
        be relied on to support at all.
    """

    sims: int
    horizon: int
    per_card: list[CardCastability]
    curve_hit: dict[int, float]
    interaction_by_turn: dict[int, float]
    creature_power_by_turn: dict[int, float]
    evasion_by_turn: dict[int, float]
    castable_hand_fraction_by_turn: dict[int, float]
    lands_in_hand_by_turn: dict[int, float]
    nonland_in_hand_by_turn: dict[int, float]
    peak_color_hit_by_turn: dict[int, float]
    mulligan_rate: float
    average_opening_hand_size: float
    peak_color_requirement: tuple[str, int, str] | None = None
    stranded: list[str] = field(default_factory=list)


def simulate_deck(
    deck: list[Card],
    sims: int = DEFAULT_SIMS,
    horizon: int = DEFAULT_TURN_HORIZON,
    seed: int | None = None,
) -> PlayabilityReport:
    """Run ``sims`` games of ``deck`` and aggregate playability metrics.

    Per game: draw seven (one mulligan to six if lands are outside [2, 5]), play one
    land per turn greedily, and on each turn record (1) the earliest turn the mana
    base could pay each non-land card and (2) whether the deck had an on-curve spell
    of CMC = turn in hand that could be cast that turn.

    Parameters
    ----------
    deck : list of Card
        The 40-card built deck.
    sims : int, optional
        Number of games to simulate. Default 1000.
    horizon : int, optional
        Final turn to run each game to. Default 8.
    seed : int, optional
        RNG seed for reproducibility. None draws a fresh one. Default None.

    Returns
    -------
    PlayabilityReport
        Aggregated per-card and per-turn metrics.
    """
    rng = random.Random(seed)
    nonland_cards = {c.name: c for c in deck if not c.is_land}
    # Cache tags and parsed costs outside the inner loop: both are read for every
    # card on every turn of every sim. With 1000 sims * 8 turns * 40 cards, parsing
    # mana costs from scratch each pass was the bulk of the runtime.
    tags_by_name = {c.name: tag(c) for c in deck}
    costs_by_name = {c.name: parse_cost(c.mana_cost) for c in deck if not c.is_land}
    peak_req = peak_color_requirement(deck)

    cast_turns: dict[str, list[int]] = {name: [] for name in nonland_cards}
    cmcs_in_deck = {int(c.cmc) for c in nonland_cards.values()}
    curve_hits: dict[int, list[bool]] = {
        turn: [] for turn in range(1, horizon + 1) if turn in cmcs_in_deck
    }
    turn_range = range(1, horizon + 1)
    interaction_totals = {t: 0 for t in turn_range}
    power_totals = {t: 0 for t in turn_range}
    evasion_totals = {t: 0 for t in turn_range}
    castable_fraction_totals = {t: 0.0 for t in turn_range}
    lands_hand_totals = {t: 0 for t in turn_range}
    nonland_hand_totals = {t: 0 for t in turn_range}
    peak_hit_totals = {t: 0 for t in turn_range}
    opening_sizes: list[int] = []

    for _ in range(sims):
        game = _simulate_game(deck, rng, horizon, tags_by_name, costs_by_name, peak_req)
        opening_sizes.append(game["opening_size"])
        for name in nonland_cards:
            if name in game["cast_turn"]:
                cast_turns[name].append(game["cast_turn"][name])
        for turn in curve_hits:
            curve_hits[turn].append(game["curve_hit"].get(turn, False))
        for turn in turn_range:
            interaction_totals[turn] += game["interaction_by_turn"][turn]
            power_totals[turn] += game["creature_power_by_turn"][turn]
            evasion_totals[turn] += game["evasion_by_turn"][turn]
            castable_fraction_totals[turn] += game["castable_hand_fraction"][turn]
            lands_hand_totals[turn] += game["lands_in_hand"][turn]
            nonland_hand_totals[turn] += game["nonland_in_hand"][turn]
            if game["peak_color_hit"][turn]:
                peak_hit_totals[turn] += 1

    per_card: list[CardCastability] = []
    for name, card in nonland_cards.items():
        turns = cast_turns[name]
        cmc = int(card.cmc)
        on_curve = sum(1 for t in turns if t <= cmc + 1) / sims
        by_horizon = len(turns) / sims
        med = int(median(turns)) if turns else horizon + 1
        per_card.append(
            CardCastability(
                name=name,
                cmc=cmc,
                castable_on_curve=on_curve,
                castable_by_horizon=by_horizon,
                median_turn=med,
            )
        )
    per_card.sort(key=lambda c: (-c.cmc, c.castable_on_curve))

    curve_hit_rates = {t: sum(hits) / sims for t, hits in curve_hits.items()}
    stranded = [c.name for c in per_card if c.castable_by_horizon < 0.5]

    return PlayabilityReport(
        sims=sims,
        horizon=horizon,
        per_card=per_card,
        curve_hit=curve_hit_rates,
        interaction_by_turn={t: interaction_totals[t] / sims for t in turn_range},
        creature_power_by_turn={t: power_totals[t] / sims for t in turn_range},
        evasion_by_turn={t: evasion_totals[t] / sims for t in turn_range},
        castable_hand_fraction_by_turn={
            t: castable_fraction_totals[t] / sims for t in turn_range
        },
        lands_in_hand_by_turn={t: lands_hand_totals[t] / sims for t in turn_range},
        nonland_in_hand_by_turn={t: nonland_hand_totals[t] / sims for t in turn_range},
        peak_color_hit_by_turn={t: peak_hit_totals[t] / sims for t in turn_range},
        peak_color_requirement=peak_req,
        mulligan_rate=sum(1 for s in opening_sizes if s < 7) / sims,
        average_opening_hand_size=sum(opening_sizes) / sims,
        stranded=stranded,
    )

import logging
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, ValidationError

from mtg_drafting.cards import Card
from mtg_drafting.config import DraftConfig

if TYPE_CHECKING:
    from mtg_drafting.llm import LLMClient
    from mtg_drafting.seat import Seat

_log = logging.getLogger(__name__)

# Strategist reply scales with pool size: at pack 3 the model classifies ~30 cards
# across maindeck/sideboard/chaff plus the directions block, easily 500+ tokens of
# JSON. The picker's 512-token default truncates mid-string here, so the strategist
# overrides with a higher cap.
_STRATEGIST_NUM_PREDICT = 2048

_STRATEGIST_SYSTEM = (
    "You are the strategist for one seat in a Magic: The Gathering cube draft. The "
    "picker handles individual card picks; your job is to step back between packs and "
    "decide the seat's direction. You produce a structured plan the picker will follow "
    "for the next pack.\n\n"
    "Reason in MTG fundamentals, NOT in named modern archetypes. Forbidden labels: "
    "'Aristocrats', 'Skies', 'Reanimator', 'Tempo', 'Tokens', 'Storm', 'Pox', "
    "'Sneak-and-Show', 'Sligh' - anyone who plays the format will recognise these as "
    "post-hoc names for decks that emerged from a pool. You are building the pool; the "
    "label will follow. Describe directions in primitives:\n"
    "  - COLORS the pool supports (one or two main colors; a third only as a splash).\n"
    "  - ROLE: who is the beatdown? Valid values: 'open' = uncommitted, take raw "
    "power; 'beatdown' = faster than the average opponent, races with cheap "
    "threats; 'control' = slower, outlasts with removal and a finisher; "
    "'midrange' = solid threats plus answers, wins on card quality.\n"
    "  - NEEDS: what fundamental functions are short - cheap interaction, evasive "
    "threats, top-end finisher, card draw, mana acceleration, sweepers.\n\n"
    "Give the picker between 1 and 3 candidate directions, each with a weight from 1 "
    "to 10 reflecting how strongly the pool supports it.\n\n"
    "DEFAULT TO 'open' WHEN UNCERTAIN. With a tiny pool (the first few picks of "
    "pack 1) or genuinely ambiguous signals, return a single direction with role "
    "'open', empty colors, and a HIGH weight (8-10) - that is the strategist "
    "telling the picker 'stay flexible, take raw power'. Do NOT return a weakly-"
    "weighted committed direction (e.g. '[3/10] B midrange') when the real answer "
    "is 'we have no plan yet'; a weak committed weight reads to the picker as a "
    "committed plan and over-anchors the picks. Move off 'open' to a real role "
    "only when the pool actually points one way - typically around pack 1 pick 6 "
    "or later, or once you have several premium cards in two compatible colors.\n\n"
    "A confident committed plan returns ONE direction at weight 8-10 with a real "
    "role. An ambiguous-but-committed pool returns 2-3 with weights reflecting "
    "their relative likelihoods. Sort by weight descending; the picker treats the "
    "first entry as primary.\n\n"
    "Be honest about commitment level. 'open' means you have not committed colors; "
    "'leaning' means signals point one way but you would still drop them for a bomb; "
    "'committed' means you are locked in and a splash needs to clear a high bar; "
    "'splashing' means committed to two colors with a third for premium cards only.\n\n"
    "Classify the pool. Maindeck is cards that would make the 23-spell deck right now. "
    "Sideboard is playable but narrow or matchup-specific - cards you'd bring "
    "in against a particular opponent but don't want in the maindeck. Chaff is "
    "filler, uncastable, off-plan, or dead. "
    "For the very first pack (pool < 16 cards) leave all three classification lists "
    "empty; there is not enough material to classify yet.\n\n"
    "biggest_needs is an ordered list - the picker takes the first item as the highest "
    "priority pickup, the second as the next, and so on. Phrase needs as functions, "
    "not as named archetype payoffs. Good: 'cheap removal', 'evasive 3-drop', 'card "
    "advantage', '5-mana finisher'. Bad: 'aristocrats payoff', 'reanimator target', "
    "'token generator for sacrifice value'.\n\n"
    "watching_for calls out universal pickups by PRINCIPLE, not by name. Two principles "
    "drive most premium cards in Magic: they break a symmetry of the game (free extra "
    "mana skips the one-land-per-turn rule; instants act outside the turn structure), "
    "or they generate card advantage (one card destroying many; one card drawing "
    "several). Phrase watching_for in structural terms - describe the shape of the "
    "effect, not the card name or archetype - so the picker can recognise specific "
    "cards that fit. Vary your phrasing pack to pack; do not copy the same description "
    "every call. Also mention signal cards: what kinds of picks would shift the color "
    "or role weights.\n\n"
    "Anchor on bombs and build-arounds. A bomb in the pool is a card that wins "
    "games on its own when cast; the plan should shape itself around being able "
    "to cast and protect it (colors, curve, interaction to clear the way). A "
    "build-around is a card whose ceiling is much higher than its baseline - it "
    "needs specific surrounding pieces to do real work. Early in the draft you "
    "have time to commit to a build-around and pick up its supporting pieces in "
    "later packs; later in the draft you usually do not. When you see either "
    "kind of card in the pool, push the plan toward what it wants and use "
    "biggest_needs and watching_for to direct the picker at the supporting "
    "pieces.\n\n"
    "Cite cards by exact name as they appear in the pool. Card names you invent are "
    "an error."
)


class PoolClassification(BaseModel):
    """Maindeck / sideboard / chaff split of the seat's current pool."""

    maindeck: list[str] = Field(
        description="Cards in the pool that would make the 23-spell deck right now. "
        "Empty list for pack 1."
    )
    sideboard: list[str] = Field(
        description="Cards in the pool that are playable but narrow or matchup-"
        "specific - bring in against a particular opponent rather than maindeck. "
        "Empty for pack 1."
    )
    chaff: list[str] = Field(
        description="Cards in the pool that are filler, uncastable, off-plan, or "
        "dead. Chaff sits in the sideboard pile alongside the real sideboard "
        "cards but, unlike them, you would not bring it in for any matchup. "
        "Empty for pack 1."
    )


class StrategyDirection(BaseModel):
    """One candidate direction the strategist is considering for this seat.

    A direction is colors plus role plus weight - the MTG primitives - not a named
    modern archetype. The label for what the deck ends up being will emerge from the
    picks; the strategist does not pick a label in advance.
    """

    colors: list[str] = Field(
        description="WUBRG letters of the colors this direction commits to. Empty "
        "list when role is 'open'."
    )
    role: Literal["open", "beatdown", "control", "midrange"] = Field(
        description="'open' = no commitment yet, picker should take raw power; "
        "'beatdown' = faster than the average opponent; 'control' = slower, "
        "outlasts with removal and a finisher; 'midrange' = solid threats plus "
        "answers (most cube decks once committed)."
    )
    weight: int = Field(
        ge=1,
        le=10,
        description="Relative weight 1-10. 8-10 = confident, 5-7 = plausible, "
        "1-4 = backup option. The picker ranks directions by weight.",
    )
    rationale: str = Field(
        description="One sentence on why the pool supports this direction, citing "
        "specific cards already in hand. Do NOT name a modern archetype."
    )


class StrategyState(BaseModel):
    """The strategist's plan for one seat, refreshed at every pack boundary.

    Carries 1-3 candidate directions (colors + role + weight) so the picker sees the
    seat's options, not just one locked label. Pool classification and biggest needs
    are shared across all candidates."""

    directions: list[StrategyDirection] = Field(
        min_length=1,
        max_length=3,
        description="1-3 candidate directions sorted by weight (primary first).",
    )
    color_commitment: Literal["open", "leaning", "committed", "splashing"] = Field(
        description="How locked-in the color choice is across the candidate "
        "directions. Be honest."
    )
    biggest_needs: list[str] = Field(
        description="Ordered list of what the deck is short on, phrased as functions "
        "('cheap removal', 'evasive 3-drop'), not as named archetype payoffs."
    )
    watching_for: str = Field(
        description="Universal-quality pickups (any Mox, premium removal, bombs) and "
        "signal cards that would shift the color or role weights."
    )
    pool_classification: PoolClassification = Field(
        description="Maindeck / sideboard / chaff split of the current pool."
    )
    notes_to_add: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="0-3 observational notes to append to the seat's draft memory. "
        "Think inferences, not raw data. Examples of the shape: 'cuts in green at "
        "pack 1 - someone upstream is on green', 'no premium white cards arrived "
        "in pack 1 - white was open and we missed it'. Skip with an empty list when "
        "no new inferences came out of this pack.",
    )

    @property
    def primary(self) -> StrategyDirection:
        """The highest-weight candidate direction; the picker's main plan."""
        return max(self.directions, key=lambda d: d.weight)


def _previous_state_block(state: StrategyState | None) -> str:
    """Render the previous strategist state for the next strategist call, or a placeholder
    when this is the seat's first strategist call."""
    if state is None:
        return "(no previous plan - this is the start of the draft)"
    direction_lines = "\n".join(
        f"    [{d.weight}/10] {'/'.join(d.colors) or 'uncommitted'} {d.role}"
        for d in sorted(state.directions, key=lambda x: -x.weight)
    )
    return (
        f"  color_commitment : {state.color_commitment}\n"
        f"  directions       :\n{direction_lines}\n"
        f"  biggest_needs    : {', '.join(state.biggest_needs) or '(none)'}\n"
        f"  watching_for     : {state.watching_for}\n"
        f"  maindeck         : {len(state.pool_classification.maindeck)} cards\n"
        f"  sideboard        : {len(state.pool_classification.sideboard)} cards\n"
        f"  chaff            : {len(state.pool_classification.chaff)} cards"
    )


def _pool_listing(pool: list[Card]) -> str:
    """Render the pool as a numbered list with full card detail so the strategist can
    classify accurately."""
    if not pool:
        return "(empty - first pack just opened)"
    return "\n".join(f"  {i + 1}. {card.render()}" for i, card in enumerate(pool))


def build_strategy_messages(
    seat: "Seat",
    round_no: int,
    config: DraftConfig,
    deck_profile: str,
) -> list[dict[str, str]]:
    """Assemble the chat messages for one strategist update.

    Parameters
    ----------
    seat : Seat
        The seat currently being strategised over.
    round_no : int
        Zero-based pack index (0, 1, or 2 in a standard 3-pack draft).
    config : DraftConfig
        Draft configuration, used for the seat-context line.
    deck_profile : str
        The deck-profile block (pip counts, curve, role counts, recent picks). Built by
        ``prompts.render_deck_profile`` so both the strategist and the picker see the
        same numbers.

    Returns
    -------
    list of dict
        Ollama chat messages.
    """
    # Deferred import: seat.py imports StrategyState from this module at load time,
    # so we cannot pull format_memory at module scope without circling back.
    from mtg_drafting.seat import format_memory

    user = (
        f"You are seat {seat.index + 1} of {config.n_seats}. "
        f"Starting pack {round_no + 1} of {config.packs_per_round}.\n\n"
        f"DECK PROFILE:\n{deck_profile}\n\n"
        f"PREVIOUS PLAN:\n{_previous_state_block(seat.strategy_state)}\n\n"
        f"DRAFT MEMORY (newest first):\n{format_memory(seat)}\n\n"
        f"YOUR POOL ({len(seat.pool)} cards):\n{_pool_listing(seat.pool)}\n\n"
        "Update the plan for the next pack. If signals have shifted, change colors / "
        "role / weights. If the previous plan still fits, keep it and refine the needs "
        "list. Use notes_to_add to record 0-3 things worth remembering across the next "
        "pack (signal updates, neighbour guesses); leave the list empty when nothing "
        "is worth saving."
    )
    return [
        {"role": "system", "content": _STRATEGIST_SYSTEM},
        {"role": "user", "content": user},
    ]


def update_strategy(
    seat: "Seat",
    round_no: int,
    llm: "LLMClient",
    config: DraftConfig,
    deck_profile: str,
) -> StrategyState | None:
    """Run the strategist for one seat and return the parsed plan.

    Try up to two attempts; return None if both fail to parse. Callers should treat
    None as 'no plan change' and keep the previous state. Logs a warning on the
    final failure so silent strategist drop-outs surface in normal runs.
    """
    messages = build_strategy_messages(seat, round_no, config, deck_profile)
    last_error: ValidationError | None = None
    for _ in range(2):
        raw = llm.chat(messages, StrategyState, num_predict=_STRATEGIST_NUM_PREDICT)
        try:
            return StrategyState.model_validate_json(raw)
        except ValidationError as exc:
            last_error = exc
    _log.warning(
        "strategist parse failed twice for seat %d, pack %d: %s",
        seat.index,
        round_no + 1,
        last_error,
    )
    return None

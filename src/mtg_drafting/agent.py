import difflib
import re

from pydantic import ValidationError

from mtg_drafting.cards import Card
from mtg_drafting.config import DraftConfig
from mtg_drafting.llm import LLMClient
from mtg_drafting.prompts import PickResponse, build_pick_messages, render_deck_profile
from mtg_drafting.seat import Seat
from mtg_drafting.strategist import update_strategy

# Pulls the value of the "pick" field out of a malformed or truncated JSON reply.
_PICK_RE = re.compile(r'"pick"\s*:\s*"((?:[^"\\]|\\.)*)"')


def _record_wheel_if_any(seat: Seat, pack: list[Card], round_no: int, pick_no: int) -> None:
    """Append a memory note when a wheeled pack carries a hit-or-miss verdict on an
    earlier ``hoping_to_wheel`` prediction this round.

    A wheel is detected by content subset: the current pack's card names form a
    subset of some earlier ``PickRecord.pack`` from the same round, and the earlier
    pack was strictly larger. A wheel without any active prediction to evaluate
    carries no signal worth remembering - just a mechanical fact of how the draft
    rotates - so the note is suppressed in that case. With at least one prediction,
    the note reports which hopes hit ("X wheeled") and which missed ("Y did not
    wheel"), both of which are real reads about the table."""
    current_names = {card.name for card in pack}
    if not current_names:
        return
    for record in seat.picks:
        if record.round_no != round_no:
            continue
        earlier_names = set(record.pack)
        if len(earlier_names) > len(current_names) and current_names <= earlier_names:
            hopes = [r.hoping_to_wheel for r in seat.picks
                     if r.round_no == round_no and r.hoping_to_wheel]
            if not hopes:
                return
            hit = [h for h in hopes if h in current_names]
            miss = [h for h in hopes if h not in current_names]
            parts = []
            if hit:
                parts.append(f"wheeled: {', '.join(hit)}")
            if miss:
                parts.append(f"did NOT wheel: {', '.join(miss)}")
            seat.memory.append(
                f"[wheel P{round_no + 1}.{pick_no + 1}] " + "; ".join(parts)
            )
            return


def _record_pick_note(seat: Seat, round_no: int, pick_no: int, note: str | None) -> None:
    """Append the picker's optional one-sentence note to the seat's memory."""
    if note is None:
        return
    trimmed = note.strip()
    if trimmed:
        seat.memory.append(f"[P{round_no + 1}.{pick_no + 1}] {trimmed}")


def _describe_sideline_changes(old: set[str], new: set[str]) -> str | None:
    """Render the diff between two sidelined sets as a one-line memory note.

    Returns None on the first classification (``old`` empty) - everything ending
    up sidelined then is initial classification, not a 'move'. Also None when
    nothing changed. Otherwise lists cards that moved to maindeck and to
    sideboard so a card silently flipping between sets is traceable."""
    if not old or old == new:
        return None
    moved_to_maindeck = sorted(old - new)
    moved_to_sideboard = sorted(new - old)
    parts = []
    if moved_to_maindeck:
        parts.append("moved to maindeck: " + ", ".join(moved_to_maindeck))
    if moved_to_sideboard:
        parts.append("moved to sideboard: " + ", ".join(moved_to_sideboard))
    return "; ".join(parts) if parts else None

# Pack 1 strategist refresh points, 0-indexed: every pick from 2 through 6, then
# a single late check at pick 10. Pack 1 pick 1 (pick_no=0) is skipped because the
# pool is empty. Pack 2 and pack 3 always refresh at their own pick 1. Dense early
# coverage helps the strategist commit as signals arrive instead of leaving stale
# "open" plans persisting through gaps between calls.
_PACK_1_REFRESH_PICKS: frozenset[int] = frozenset({1, 2, 3, 4, 5, 9})


def _should_refresh_strategy(round_no: int, pick_no: int) -> bool:
    """Whether the strategist should run before this pick.

    Pack 1 refreshes at every pick from 2 through 6 and once more at pick 10 - the
    early pack is where colour and role signals firm up fastest, so the strategist
    needs frequent updates to commit instead of staying stale-open. Later packs
    refresh only at their own pick 1."""
    if round_no == 0:
        return pick_no in _PACK_1_REFRESH_PICKS
    return pick_no == 0


class DraftAgent:
    """Turns an :class:`LLMClient` into a pick function for :class:`DraftState`.

    Two-tier per seat: at the start of every pack the strategist updates the seat's
    structured plan; for every pick within the pack the picker chooses a card that
    fits the plan. The strategist owns color commitment, archetype, and pool
    classification; the picker just applies them.

    Parameters
    ----------
    llm : LLMClient
        The model used for both strategist and picker calls.
    config : DraftConfig
        Draft settings; supplies the retry budget and prompt context.
    """

    def __init__(self, llm: LLMClient, config: DraftConfig):
        self.llm = llm
        self.config = config

    def pick(
        self, seat: Seat, pack: list[Card], round_no: int, pick_no: int
    ) -> tuple[Card, str, str | None]:
        """Choose a card for ``seat`` from ``pack``.

        A single-card pack is taken immediately without querying the model - there is
        no choice to make. The strategist runs first at the refresh points defined by
        :func:`_should_refresh_strategy`. Wheel detection runs next: if the pack's
        contents are a strict subset of a pack this seat saw earlier in the same
        round, a code-side note goes into the seat's memory. The picker then chooses
        a card; any non-null ``note`` it returns is appended to memory too. On
        repeated unusable replies the picker retries up to ``llm.max_retries`` times,
        then falls back to ``pack[0]`` (a logged failure note) so the draft never
        stalls.

        Returns
        -------
        card : Card
            The card taken (always a member of ``pack``).
        reasoning : str
            The model's stated reason, or a note when the pick was forced or fell back.
        hoping_to_wheel : str or None
            Name of a card in ``pack`` the picker hopes will wheel back, fuzzy-matched
            against the pack so a mistyped name still resolves. None when the picker
            did not name a wheel candidate or named a card that's not in the pack.
        """
        if len(pack) == 1:
            return pack[0], "only card left in the pack", None

        if _should_refresh_strategy(round_no, pick_no):
            self._refresh_strategy(seat, round_no)

        _record_wheel_if_any(seat, pack, round_no, pick_no)

        messages = build_pick_messages(seat, pack, round_no, pick_no, self.config)

        for _ in range(1 + self.config.llm.max_retries):
            response = self._parse(self.llm.chat(messages, PickResponse))
            if response is None:
                continue
            card = self._match(response.pick, pack)
            if card is None:
                continue
            _record_pick_note(seat, round_no, pick_no, response.note)
            if response.intent == "sideboard":
                seat.sidelined.add(card.name)
            wheel_card = (
                self._match(response.hoping_to_wheel, pack)
                if response.hoping_to_wheel
                else None
            )
            wheel_hope = wheel_card.name if wheel_card is not None else None
            return card, response.reasoning.strip(), wheel_hope

        fallback = pack[0]
        seat.memory.append(
            f"[FALLBACK P{round_no + 1}.{pick_no + 1}] picker returned no usable "
            f"card; defaulted to {fallback.name}"
        )
        return fallback, "fallback pick: the model did not return a valid card name", None

    def _refresh_strategy(self, seat: Seat, round_no: int) -> None:
        """Run the strategist for this seat and update its plan + sidelined set.

        Strategist sees the full pool so it can move cards between maindeck and
        sideboard each call. The sidelined set is rebuilt from the new
        ``pool_classification.sideboard`` and ``chaff`` lists, intersected with
        actual pool names (hallucinated names are silently dropped). Net moves
        between maindeck and sideboard (vs. the previous classification) are
        logged to memory so a card silently flipping between sets is traceable.
        Any ``notes_to_add`` are appended to ``seat.memory``. Keeps the previous
        plan on parse failure so a transient strategist failure does not leave
        the picker plan-less."""
        profile = render_deck_profile(seat.pool, seat)
        new_state = update_strategy(seat, round_no, self.llm, self.config, profile)
        if new_state is None:
            return
        seat.strategy_state = new_state
        pool_names = {c.name for c in seat.pool}
        sideboard = set(new_state.pool_classification.sideboard)
        chaff = set(new_state.pool_classification.chaff)
        new_sidelined = (sideboard | chaff) & pool_names
        moves = _describe_sideline_changes(seat.sidelined, new_sidelined)
        seat.sidelined = new_sidelined
        # Prefer the strategist's own rationale ("we're in B aggro: added X,
        # sidelined Y because ..."). Fall back to the delta string only when the
        # strategist left the rationale blank but cards moved anyway, so a
        # silent reclassification still leaves a paper trail.
        rationale = new_state.pool_classification.rationale.strip()
        if rationale:
            seat.memory.append(f"[strat P{round_no + 1}] {rationale}")
        elif moves:
            seat.memory.append(f"[strat P{round_no + 1}] {moves}")
        for note in new_state.notes_to_add:
            if note.strip():
                seat.memory.append(f"[strat P{round_no + 1}] {note.strip()}")

    @staticmethod
    def _parse(content: str) -> PickResponse | None:
        """Parse a pick reply, salvaging the pick from a malformed JSON object.

        A reply that fails strict schema validation may still contain a usable ``pick``
        field; recover it by regex and leave the free-text fields empty. Return None
        when the pick cannot be found at all."""
        try:
            return PickResponse.model_validate_json(content)
        except ValidationError:
            match = _PICK_RE.search(content)
            if match is None:
                return None
            return PickResponse(
                pick=match.group(1), reasoning="(recovered from a malformed reply)"
            )

    @staticmethod
    def _match(name: str, pack: list[Card]) -> Card | None:
        """Resolve a model-supplied name to a card in the pack, exact match first then
        closest name. Return None if nothing is close enough."""
        wanted = name.strip().lower()
        by_name = {card.name.lower(): card for card in pack}
        if wanted in by_name:
            return by_name[wanted]
        close = difflib.get_close_matches(wanted, by_name.keys(), n=1, cutoff=0.85)
        return by_name[close[0]] if close else None

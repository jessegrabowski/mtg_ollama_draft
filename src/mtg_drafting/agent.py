import difflib
import re

from pydantic import ValidationError

from mtg_drafting.cards import Card
from mtg_drafting.config import DraftConfig
from mtg_drafting.llm import LLMClient
from mtg_drafting.prompts import PickResponse, build_pick_messages
from mtg_drafting.seat import Seat

# Pulls the value of the "pick" field out of a malformed or truncated JSON reply.
_PICK_RE = re.compile(r'"pick"\s*:\s*"((?:[^"\\]|\\.)*)"')


class DraftAgent:
    """Turns an :class:`LLMClient` into a pick function for :class:`DraftState`.

    For each pick the agent builds the seat's prompt, queries the model, validates that
    the chosen card is actually in the pack, and updates the seat's running strategy.

    Parameters
    ----------
    llm : LLMClient
        The model used to make picks.
    config : DraftConfig
        Draft settings; supplies the retry budget and prompt context.
    """

    def __init__(self, llm: LLMClient, config: DraftConfig):
        self.llm = llm
        self.config = config

    def pick(
        self, seat: Seat, pack: list[Card], round_no: int, pick_no: int
    ) -> tuple[Card, str]:
        """Choose a card for ``seat`` from ``pack``.

        Matches the model's chosen name to a pack card exactly first, then by closest
        name. On repeated unusable replies it retries up to ``llm.max_retries`` times,
        then falls back to the first card so the draft never stalls.

        Returns
        -------
        card : Card
            The card taken (always a member of ``pack``).
        reasoning : str
            The model's stated reason, or a note when a fallback was used.
        """
        messages = build_pick_messages(seat, pack, round_no, pick_no, self.config)

        for _ in range(1 + self.config.llm.max_retries):
            response = self._parse(self.llm.chat(messages, PickResponse))
            if response is None:
                continue
            card = self._match(response.pick, pack)
            if card is not None:
                if response.strategy.strip():
                    seat.strategy = response.strategy.strip()
                return card, response.reasoning.strip()

        return pack[0], "fallback pick: the model did not return a valid card name"

    @staticmethod
    def _parse(content: str) -> PickResponse | None:
        """Parse a pick reply, salvaging the pick from a malformed JSON object.

        A reply that fails strict schema validation may still contain a usable ``pick``
        field; it is recovered by regex while the free-text fields are left empty.
        Returns None when the pick cannot be found at all."""
        try:
            return PickResponse.model_validate_json(content)
        except ValidationError:
            match = _PICK_RE.search(content)
            if match is None:
                return None
            return PickResponse(
                pick=match.group(1), reasoning="(recovered from a malformed reply)", strategy=""
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

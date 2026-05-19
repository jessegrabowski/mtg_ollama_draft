import json
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError
from rich.console import Console

from mtg_drafting.cards import Card
from mtg_drafting.config import LLMConfig, Paths
from mtg_drafting.llm import LLMClient
from mtg_drafting.records import load_draft
from mtg_drafting.scryfall import load_index

_EVAL_SYSTEM = (
    "You are an expert Magic: The Gathering cube analyst. Given the full card pool a "
    "player drafted, identify the deck they should build, rate it, and explain its "
    "strengths and weaknesses. Be candid - a weak pool should score low."
)


class DeckEvaluation(BaseModel):
    """Schema for the LLM's evaluation of one drafted pool."""

    archetype: str = Field(description="The pool's best colours and archetype.")
    rating: int = Field(ge=1, le=10, description="Overall deck power: 1 weak, 10 excellent.")
    strengths: str = Field(description="What the deck does well.")
    weaknesses: str = Field(description="Where the deck is lacking.")
    maindeck: list[str] = Field(
        description="About 23 non-land cards from the pool to play in the 40-card deck."
    )


def _eval_messages(pool: list[Card], seat_index: int) -> list[dict[str, str]]:
    """Build the chat messages asking the model to evaluate one seat's pool."""
    listing = "\n".join(f"  - {card.render()}" for card in pool)
    user = (
        f"Seat {seat_index + 1} drafted this {len(pool)}-card pool:\n\n{listing}\n\n"
        "Identify the strongest archetype, choose roughly 23 non-land cards for the "
        "deck, rate the deck from 1 to 10, and describe its strengths and weaknesses."
    )
    return [
        {"role": "system", "content": _EVAL_SYSTEM},
        {"role": "user", "content": user},
    ]


def evaluate_draft(results_dir: Path, llm_config: LLMConfig) -> Path:
    """Have the LLM evaluate every seat's pool in a finished draft.

    Reads ``draft.json`` from ``results_dir``, re-resolves each pool against the
    Scryfall snapshot, and asks the model for an archetype, a maindeck, a 1-10 rating,
    and strengths/weaknesses. Writes ``evaluation.json`` alongside the draft.

    Parameters
    ----------
    results_dir : Path
        A directory produced by :func:`mtg_drafting.records.save_draft`.
    llm_config : LLMConfig
        Model settings for the evaluator.

    Returns
    -------
    Path
        Path of the written ``evaluation.json``.
    """
    console = Console()
    draft = load_draft(results_dir)
    index = load_index(Paths().cache_dir)

    llm = LLMClient(llm_config)
    llm.ensure_model()

    evaluations = []
    for seat in draft["seats"]:
        pool = [card for name in seat["pool"] if (card := index.get(name)) is not None]
        messages = _eval_messages(pool, seat["index"])

        evaluation = None
        for _ in range(2):
            try:
                evaluation = DeckEvaluation.model_validate_json(llm.chat(messages, DeckEvaluation))
                break
            except ValidationError:
                continue
        if evaluation is None:
            console.print(
                f"  Seat {seat['index'] + 1}: [red]evaluation skipped (unparseable reply)[/red]"
            )
            continue

        evaluations.append({"index": seat["index"], **evaluation.model_dump()})
        console.print(
            f"  Seat {seat['index'] + 1}: [bold]{evaluation.rating}/10[/bold] "
            f"- {evaluation.archetype}"
        )

    out_path = results_dir / "evaluation.json"
    out_path.write_text(json.dumps(evaluations, indent=2))
    console.print(f"\n[green]Evaluation written:[/green] [bold]{out_path}[/bold]")
    return out_path

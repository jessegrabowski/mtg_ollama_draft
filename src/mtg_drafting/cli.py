from pathlib import Path
from typing import Annotated

import typer

from mtg_drafting.config import DEFAULT_MODEL, DraftConfig, LLMConfig

app = typer.Typer(
    add_completion=False,
    help="Simulate Magic: The Gathering cube drafts with a local Ollama LLM.",
)


@app.command()
def draft(
    cube: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="Cube list file."),
    ],
    seats: Annotated[int, typer.Option(min=2, help="Number of seats at the table.")] = 8,
    packs: Annotated[int, typer.Option(min=1, help="Packs each seat opens.")] = 3,
    pack_size: Annotated[int, typer.Option(min=2, help="Cards per pack.")] = 15,
    model: Annotated[str, typer.Option(help="Ollama model tag for the drafter.")] = DEFAULT_MODEL,
    seed: Annotated[
        int | None, typer.Option(help="Shuffle seed; omit for a random one.")
    ] = None,
    history: Annotated[
        int, typer.Option(min=0, help="Recent picks kept in each seat's context.")
    ] = 5,
    think: Annotated[
        bool,
        typer.Option(help="Enable model thinking (much slower, unreliable structured output)."),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Print every pick: pack, choice, and deck so far."),
    ] = False,
    refresh_cards: Annotated[
        bool, typer.Option(help="Re-download the Scryfall snapshot.")
    ] = False,
) -> None:
    """Run a draft of CUBE with the LLM playing every seat."""
    from mtg_drafting.runner import run_draft

    config = DraftConfig(
        n_seats=seats,
        packs_per_round=packs,
        cards_per_pack=pack_size,
        seed=seed,
        history_maxlen=history,
        llm=LLMConfig(model=model, think=think),
    )
    run_draft(cube, config, refresh_cards=refresh_cards, verbose=verbose)


@app.command()
def evaluate(
    results_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, help="A results directory from a draft."),
    ],
    model: Annotated[
        str, typer.Option(help="Ollama model tag for the evaluator.")
    ] = DEFAULT_MODEL,
) -> None:
    """Rate every seat's drafted pool in RESULTS_DIR."""
    from mtg_drafting.evaluate import evaluate_draft

    evaluate_draft(results_dir, LLMConfig(model=model))


if __name__ == "__main__":
    app()

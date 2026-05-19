from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from mtg_drafting.agent import DraftAgent
from mtg_drafting.config import DraftConfig
from mtg_drafting.cube import load_cube
from mtg_drafting.draft import DraftState
from mtg_drafting.llm import LLMClient
from mtg_drafting.packs import generate_packs
from mtg_drafting.prompts import summarize_pool
from mtg_drafting.records import save_draft
from mtg_drafting.scryfall import load_index
from mtg_drafting.seat import PickRecord, Seat


def _print_pick(console: Console, seat: Seat, record: PickRecord) -> None:
    """Print one completed pick: the pack offered, the card taken, and the seat's
    deck so far."""
    console.rule(
        f"[dim]Round {record.round_no + 1} · Pick {record.pick_no + 1}[/dim]", style="dim"
    )
    pack = " · ".join(
        f"[bold green]{escape(name)}[/bold green]" if name == record.chosen else escape(name)
        for name in record.pack
    )
    console.print(f"[bold cyan]Seat {seat.index + 1}[/bold cyan] picks from {len(record.pack)}:")
    console.print(f"  [dim]Pack:[/dim] {pack}")
    console.print(f"  [bold green]→ {escape(record.chosen)}[/bold green]")
    console.print(f"    [dim]{escape(record.reasoning)}[/dim]")
    console.print("  [dim]Deck:[/dim]")
    deck = escape(summarize_pool(seat.pool))
    console.print("\n".join("    " + line for line in deck.splitlines()))


def run_draft(
    cube_path: Path, config: DraftConfig, refresh_cards: bool = False, verbose: bool = False
) -> Path:
    """Run a full draft of ``cube_path`` and write the results.

    Loads and enriches the cube, deals packs, then drives :class:`DraftState` with an
    LLM-backed :class:`DraftAgent`. The seed actually used is folded back into the
    config before results are saved, so the run is reproducible.

    Parameters
    ----------
    cube_path : Path
        Cube list file to draft.
    config : DraftConfig
        Draft and model settings.
    refresh_cards : bool, optional
        Force a re-download of the Scryfall snapshot. Default False.
    verbose : bool, optional
        Print every pick - pack, chosen card, reasoning, and the seat's deck - instead
        of a single progress bar. Default False.

    Returns
    -------
    Path
        Directory the draft results were written to.
    """
    console = Console()

    index = load_index(config.paths.cache_dir, refresh=refresh_cards)
    cards, unresolved = load_cube(cube_path, index)
    if unresolved:
        console.print(f"[yellow]Skipped {len(unresolved)} unresolved card name(s):[/yellow]")
        for name in unresolved:
            console.print(f"  [yellow]- {name}[/yellow]")
    if len(cards) < config.min_cube_size:
        raise SystemExit(
            f"Cube '{cube_path.name}' resolved to {len(cards)} cards; "
            f"{config.min_cube_size} are needed for this draft."
        )

    rounds, seed = generate_packs(cards, config)
    config = config.model_copy(update={"seed": seed})
    console.print(
        f"Drafting [bold]{cube_path.stem}[/bold] - {config.n_seats} seats, "
        f"{config.packs_per_round} packs of {config.cards_per_pack} (seed {seed})"
    )

    llm = LLMClient(config.llm)
    llm.ensure_model()
    agent = DraftAgent(llm, config)
    state = DraftState(config, rounds)

    if verbose:
        state.run(agent.pick, on_pick=lambda seat, record: _print_pick(console, seat, record))
    else:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Picks", total=config.min_cube_size)
            state.run(agent.pick, on_pick=lambda seat, record: progress.advance(task))

    out_dir = save_draft(state, config, cube_path.stem)
    console.print(f"\n[green]Draft complete.[/green] Results: [bold]{out_dir}[/bold]\n")
    for seat in state.seats:
        console.print(f"  Seat {seat.index + 1}: {seat.strategy or '(no strategy recorded)'}")
    return out_dir

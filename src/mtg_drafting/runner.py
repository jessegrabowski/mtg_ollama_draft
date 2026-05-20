import math
import re
from pathlib import Path

from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from mtg_drafting.agent import DraftAgent
from mtg_drafting.cards import COLOR_ORDER, Card
from mtg_drafting.config import DraftConfig
from mtg_drafting.cube import load_cube
from mtg_drafting.deckbuilder import count_pips
from mtg_drafting.draft import DraftState
from mtg_drafting.llm import LLMClient
from mtg_drafting.packs import generate_packs
from mtg_drafting.records import save_draft
from mtg_drafting.scryfall import load_index
from mtg_drafting.seat import PickRecord, Seat

# Per-seat accent colors, cycled by seat index. Keeps each seat visually distinct in
# the verbose draft display without depending on the seat's drafted colors.
_SEAT_COLORS = ("cyan", "magenta", "green", "yellow", "blue", "red", "white", "bright_cyan")
# Each ``{...}`` mana symbol; used to drop the braces from rendered costs.
_COST_SYMBOL_RE = re.compile(r"\{([^}]*)\}")
# Cap on the curve histogram's printed row count. Beyond this each row represents
# multiple cards and the y-axis labels scale to match.
_MAX_HISTOGRAM_HEIGHT = 5
# Terminal color per MTG mana symbol. Black uses magenta because true black is
# invisible on dark terminals; everything else maps to its natural color. Generic
# digits, colorless ``C``, ``X``, hybrid slashes, and unrecognised symbols are left
# unstyled - they render in the terminal's default foreground, which reads cleanly
# next to the colored pips without disappearing into the background like
# ``bright_black`` does on darker themes.
_MANA_STYLES: dict[str, str] = {
    "W": "bright_yellow",
    "U": "bright_blue",
    "B": "magenta",
    "R": "bright_red",
    "G": "bright_green",
    "S": "bright_white",
}
# Decklist row ordering by card type. First match wins for multi-type cards (Artifact
# Creature counts as Creature). "Other" catches anything that hits none of these.
_TYPE_ORDER = (
    "Creature",
    "Planeswalker",
    "Instant",
    "Sorcery",
    "Enchantment",
    "Artifact",
    "Land",
)


def _curve_histogram(pool: list[Card]) -> str:
    """Multi-row vertical ASCII histogram of non-land mana values.

    Capped at :data:`_MAX_HISTOGRAM_HEIGHT` printed rows. When the deck's tallest
    bucket fits inside the cap each row represents one card and the y-axis labels
    count up by one; when it exceeds the cap, each row represents ``cards_per_row``
    cards (``ceil(max_count / max_height)``) and the labels reflect the scaled
    thresholds. Bottom row labels CMC 0 through 7+ (the catch-all for CMC ≥ 7)."""
    buckets = [0] * 8
    for card in pool:
        if card.is_land:
            continue
        buckets[min(int(card.cmc), 7)] += 1
    if not any(buckets):
        return "(no spells yet)"
    actual_max = max(buckets)
    cards_per_row = max(1, math.ceil(actual_max / _MAX_HISTOGRAM_HEIGHT))
    height = math.ceil(actual_max / cards_per_row)
    rows: list[str] = []
    for level in range(height, 0, -1):
        floor_for_row = (level - 1) * cards_per_row + 1
        bar = " ".join("█" if n >= floor_for_row else " " for n in buckets)
        label = level * cards_per_row
        rows.append(f"{label:>2} │ {bar}")
    rows.append("   └" + "─" * 16)
    rows.append("     0 1 2 3 4 5 6 7+")
    return "\n".join(rows)


def _card_type_bucket(card: Card) -> str:
    """First matching type from :data:`_TYPE_ORDER`. Multi-type cards pick by category
    order (Artifact Creature → Creature). Falls back to 'Other' when no type matches."""
    for type_name in _TYPE_ORDER:
        if type_name in card.type_line:
            return type_name
    return "Other"


def _categorise_by_type(cards: list[Card]) -> dict[str, list[Card]]:
    grouped: dict[str, list[Card]] = {}
    for card in cards:
        grouped.setdefault(_card_type_bucket(card), []).append(card)
    return grouped


def _bare_cost(mana_cost: str) -> str:
    """Strip the curly braces from a mana cost: ``{1}{W}{W}`` → ``1WW``. Used as the
    plain-text version for computing left-pad width before styling."""
    return "".join(_COST_SYMBOL_RE.findall(mana_cost))


def _styled_cost(mana_cost: str) -> Text:
    """Render a brace-stripped mana cost with each color symbol painted its MTG
    color. Generic digits, hybrid slashes, and unrecognised symbols come through
    in the terminal's default foreground color."""
    text = Text()
    for symbol in _COST_SYMBOL_RE.findall(mana_cost):
        for char in symbol:
            text.append(char, style=_MANA_STYLES.get(char.upper()))
    return text


def _decklist_lines(cards: list[Card]) -> Text:
    """Render a card list as a decklist: type-grouped sections, each unique name on
    its own line as ``cost  Name`` (with ``(Nx)`` suffix for duplicates) and costs
    left-padded so every name in this list starts at the same column. Sorted by CMC
    then name within each type. The type header still counts physical cards, so a
    deck with two Kird Apes shows ``Creature (2)`` above one line."""
    text = Text()
    if not cards:
        text.append("(empty)", style="dim italic")
        return text
    bare_widths = {c.name: len(_bare_cost(c.mana_cost)) for c in cards}
    pad = max(bare_widths.values(), default=0)
    grouped = _categorise_by_type(cards)
    for type_name in (*_TYPE_ORDER, "Other"):
        bucket = grouped.get(type_name)
        if not bucket:
            continue
        text.append(f"{type_name} ({len(bucket)})\n", style="bold")
        # Collapse duplicates within this type: one row per name, count suffix when > 1.
        counts: dict[str, int] = {}
        unique: list[Card] = []
        for card in bucket:
            if card.name not in counts:
                unique.append(card)
            counts[card.name] = counts.get(card.name, 0) + 1
        for card in sorted(unique, key=lambda c: (c.cmc, c.name)):
            cost_text = _styled_cost(card.mana_cost)
            cost_text.append(" " * (pad - bare_widths[card.name]))
            text.append("  ")
            text.append(cost_text)
            n = counts[card.name]
            suffix = f" ({n}x)" if n > 1 else ""
            text.append(f"  {card.name}{suffix}")
            # When a card falls into the "Other" bucket, surface its raw type_line so
            # the cause is visible at a glance - the most common culprit is an empty
            # or weirdly-formed type_line from the Scryfall index.
            if type_name == "Other":
                shown = card.type_line or "(empty)"
                text.append(f"  [type: {shown!r}]", style="dim italic")
            text.append("\n")
    return text


def _main_block(seat: Seat) -> Text:
    """Left column of the pick panel: pip header, mana-curve histogram, decklist."""
    cards = seat.maindeck
    text = Text()
    if not cards:
        text.append("MAIN (0): ", style="bold")
        text.append("(no cards yet)", style="dim italic")
        return text

    pip_counts = count_pips(cards)
    text.append(f"MAIN ({len(cards)}): ", style="bold")
    nonzero = [(c, pip_counts[c]) for c in COLOR_ORDER if pip_counts[c] > 0]
    if not nonzero:
        text.append("colorless")
    else:
        for i, (color, count) in enumerate(nonzero):
            if i:
                text.append(" · ")
            text.append(color, style=_MANA_STYLES.get(color))
            text.append(f" {count}")
    text.append("\n")
    text.append(_curve_histogram(cards))
    text.append("\n\n")
    text.append(_decklist_lines(cards))
    return text


def _side_block(seat: Seat) -> Text:
    """Right column of the pick panel: sideboarded cards as a decklist."""
    cards = seat.sideboard
    text = Text()
    text.append(f"SIDE ({len(cards)})\n", style="bold")
    text.append(_decklist_lines(cards))
    return text


def _memory_block(seat: Seat) -> Text:
    """Render the seat's draft memory deque, newest at the top. Returns an empty Text
    when there are no notes so the panel suppresses an empty section."""
    text = Text()
    if not seat.memory:
        return text
    text.append("Memory ", style="bold yellow")
    text.append(f"({len(seat.memory)})", style="dim yellow")
    for i, note in enumerate(reversed(seat.memory), start=1):
        text.append(f"\n  {i}. {note}")
    return text


def _plan_block(seat: Seat) -> Text:
    """Full-width plan / needs / watching block above the main+side columns."""
    text = Text()
    state = seat.strategy_state
    text.append("Plan ", style="bold yellow")
    if state is None:
        text.append("(no plan yet)", style="dim italic")
        return text

    text.append(f"({state.color_commitment})  ", style="dim yellow")
    directions = " · ".join(
        f"[{d.weight}/10] {'/'.join(d.colors) or 'open'} {d.role}"
        for d in sorted(state.directions, key=lambda d: -d.weight)
    )
    text.append(directions)
    if state.biggest_needs:
        text.append("\nNeeds: ", style="bold")
        text.append(" · ".join(state.biggest_needs))
    if state.watching_for:
        text.append("\nWatching: ", style="bold")
        text.append(state.watching_for, style="italic")
    return text


def _pack_line(record: PickRecord) -> Text:
    """The 'pack: ...' line with the chosen card bolded green and the wheel-hope
    card (if any) in subtle italic dim cyan. Only the first occurrence of each
    marker fires so cubes with duplicates only mark one copy."""
    text = Text()
    text.append("Pack: ", style="dim")
    highlighted = False
    wheel_marked = False
    for i, name in enumerate(record.pack):
        if i:
            text.append(" · ", style="dim")
        if not highlighted and name == record.chosen:
            text.append(name, style="bold green")
            highlighted = True
        elif (
            not wheel_marked
            and record.hoping_to_wheel is not None
            and name == record.hoping_to_wheel
        ):
            text.append(name, style="italic dim cyan")
            wheel_marked = True
        else:
            text.append(name)
    return text


def _pick_line(record: PickRecord) -> Text:
    """The '→ chosen · reasoning' line."""
    text = Text()
    text.append("→ ", style="bold green")
    text.append(record.chosen, style="bold green")
    text.append("  ", style="dim")
    text.append(record.reasoning, style="italic")
    return text


def _print_pick(console: Console, seat: Seat, record: PickRecord) -> None:
    """Render one pick as a single bordered panel.

    Layout from top: pack + pick + reasoning (full width), plan + needs + watching
    (full width), then a two-column grid with the maindeck (with histogram and a
    type-grouped decklist) on the left and the sideboard on the right."""
    seat_color = _SEAT_COLORS[seat.index % len(_SEAT_COLORS)]
    grid = Table.grid(padding=(0, 2), expand=True)
    grid.add_column(ratio=3)
    grid.add_column(ratio=2)
    grid.add_row(_main_block(seat), _side_block(seat))
    sections = [
        _pack_line(record),
        Text(""),
        _pick_line(record),
        Text(""),
        _plan_block(seat),
    ]
    memory = _memory_block(seat)
    if len(memory):
        sections.extend([Text(""), memory])
    sections.extend([Text(""), grid])
    body = Group(*sections)
    console.print(
        Panel(
            body,
            title=(
                f"[bold {seat_color}]Seat {seat.index + 1}[/bold {seat_color}] "
                f"[dim]· Pack {record.round_no + 1}, Pick "
                f"{record.pick_no + 1} (from {len(record.pack)})[/dim]"
            ),
            title_align="left",
            border_style=seat_color,
            padding=(0, 1),
        )
    )


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
        Print every pick - pack, chosen card, reasoning, the seat's running strategy,
        and its deck - instead of a single progress bar. Default False.

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
        console.print(f"  Seat {seat.index + 1}: {seat.strategy_summary}")
    return out_dir

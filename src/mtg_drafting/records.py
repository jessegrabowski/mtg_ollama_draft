import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from mtg_drafting.cards import Card
from mtg_drafting.config import DraftConfig
from mtg_drafting.draft import DraftState

_CATEGORY_ORDER = ["White", "Blue", "Black", "Red", "Green", "Multicolor", "Colorless", "Land"]


def _decklist(pool: list[Card]) -> str:
    """Render a seat's pool as a human-readable, color-grouped card list."""
    grouped: dict[str, list[Card]] = {}
    for card in pool:
        grouped.setdefault(card.color_category, []).append(card)

    blocks = []
    for category in _CATEGORY_ORDER:
        cards = grouped.get(category)
        if not cards:
            continue
        lines = [f"# {category} ({len(cards)})"]
        lines += [card.name for card in sorted(cards, key=lambda c: (c.cmc, c.name))]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


def save_draft(state: DraftState, config: DraftConfig, cube_name: str) -> Path:
    """Write a finished draft to a timestamped directory under ``results/``.

    Produces ``draft.json`` (full configuration, seed, and every seat's picks with
    reasoning) plus a ``seat_N.txt`` decklist per seat.

    Parameters
    ----------
    state : DraftState
        The completed draft.
    config : DraftConfig
        The configuration used, with ``seed`` set to the seed actually drawn.
    cube_name : str
        Cube identifier, used in the directory name.

    Returns
    -------
    Path
        The directory the results were written to.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = config.paths.results_dir / f"{timestamp}_{cube_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "cube": cube_name,
        "seed": config.seed,
        "config": config.model_dump(mode="json"),
        "seats": [
            {
                "index": seat.index,
                "strategy_state": (
                    seat.strategy_state.model_dump() if seat.strategy_state else None
                ),
                "strategy_summary": seat.strategy_summary,
                "memory": list(seat.memory),
                "sidelined": sorted(seat.sidelined),
                "pool": [card.name for card in seat.pool],
                "picks": [asdict(pick) for pick in seat.picks],
            }
            for seat in state.seats
        ],
    }
    (out_dir / "draft.json").write_text(json.dumps(payload, indent=2))

    for seat in state.seats:
        (out_dir / f"seat_{seat.index + 1}.txt").write_text(_decklist(seat.pool))

    return out_dir


def load_draft(results_dir: Path) -> dict:
    """Load a previously saved ``draft.json`` payload from a results directory."""
    return json.loads((results_dir / "draft.json").read_text())

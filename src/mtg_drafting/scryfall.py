import json
from pathlib import Path

import requests
from tqdm import tqdm

from mtg_drafting.cards import Card

BULK_DATA_URL = "https://api.scryfall.com/bulk-data"
USER_AGENT = "mtg-drafting/0.1 (cube draft simulator)"

# Layouts that are not real, draftable cards.
_SKIP_LAYOUTS = {
    "token",
    "double_faced_token",
    "emblem",
    "art_series",
    "scheme",
    "planar",
    "vanguard",
}


def _normalize(name: str) -> str:
    """Fold a card name to a lookup key: lowercase, straight apostrophes, trimmed."""
    return name.strip().lower().replace("’", "'")


def _card_from_scryfall(obj: dict) -> Card:
    """Build a :class:`Card` from one Scryfall bulk-data object, merging multi-face
    cards into a single record."""
    faces = obj.get("card_faces")
    if faces:
        mana_cost = " // ".join(f.get("mana_cost", "") for f in faces).strip(" /")
        oracle_text = "\n//\n".join(
            f"{f.get('name', '')}: {f.get('oracle_text', '')}".strip(": ") for f in faces
        )
        colors: list[str] = []
        for f in faces:
            for c in f.get("colors", []):
                if c not in colors:
                    colors.append(c)
        power = next((f.get("power") for f in faces if f.get("power") is not None), None)
        toughness = next(
            (f.get("toughness") for f in faces if f.get("toughness") is not None), None
        )
    else:
        mana_cost = obj.get("mana_cost", "")
        oracle_text = obj.get("oracle_text", "")
        colors = obj.get("colors", [])
        power = obj.get("power")
        toughness = obj.get("toughness")

    return Card(
        name=obj["name"],
        mana_cost=mana_cost,
        cmc=obj.get("cmc", 0.0),
        type_line=obj.get("type_line", ""),
        oracle_text=oracle_text,
        colors=colors,
        color_identity=obj.get("color_identity", []),
        power=power,
        toughness=toughness,
        rarity=obj.get("rarity", ""),
    )


class ScryfallIndex:
    """Offline lookup of cards by name, backed by a Scryfall bulk-data snapshot.

    Names are matched case-insensitively. Multi-face cards are reachable by their full
    name (``"Fire // Ice"``) and by their front-face name (``"Fire"``).
    """

    def __init__(self, cards: list[Card], updated_at: str = ""):
        self.updated_at = updated_at
        self._by_name: dict[str, Card] = {}
        for card in cards:
            self._by_name.setdefault(_normalize(card.name), card)
            if " // " in card.name:
                front = card.name.split(" // ", 1)[0]
                self._by_name.setdefault(_normalize(front), card)

    def __len__(self) -> int:
        return len(self._by_name)

    def get(self, name: str) -> Card | None:
        """Return the card for ``name``, or None if it is not in the snapshot."""
        return self._by_name.get(_normalize(name))


def _oracle_bulk_meta() -> dict:
    """Fetch Scryfall bulk-data metadata and return the ``oracle_cards`` entry."""
    resp = requests.get(
        BULK_DATA_URL, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}, timeout=30
    )
    resp.raise_for_status()
    for entry in resp.json()["data"]:
        if entry["type"] == "oracle_cards":
            return entry
    raise RuntimeError("Scryfall bulk-data feed has no 'oracle_cards' entry.")


def load_index(cache_dir: Path, refresh: bool = False) -> ScryfallIndex:
    """Load the card index, downloading and parsing the Scryfall snapshot if needed.

    The first call (or any call with ``refresh=True`` when a newer snapshot exists)
    downloads the ~150 MB ``oracle_cards`` bulk file and parses it into a slim index
    cached as ``card_index.json``. Subsequent calls read only that small cache.

    Parameters
    ----------
    cache_dir : Path
        Directory for the cached snapshot and slim index.
    refresh : bool, optional
        Re-download when Scryfall reports a newer snapshot than the cache. Default
        False.

    Returns
    -------
    ScryfallIndex
        Name-keyed lookup over every draftable card in the snapshot.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / "card_index.json"

    if index_path.exists() and not refresh:
        payload = json.loads(index_path.read_text())
        cards = [Card.model_validate(c) for c in payload["cards"]]
        return ScryfallIndex(cards, payload.get("updated_at", ""))

    meta = _oracle_bulk_meta()
    if index_path.exists():
        cached_at = json.loads(index_path.read_text()).get("updated_at", "")
        if cached_at == meta["updated_at"]:
            payload = json.loads(index_path.read_text())
            cards = [Card.model_validate(c) for c in payload["cards"]]
            return ScryfallIndex(cards, cached_at)

    raw_path = cache_dir / "oracle_cards.json"
    _download(meta["download_uri"], raw_path)

    raw = json.loads(raw_path.read_text())
    cards = [
        _card_from_scryfall(obj)
        for obj in tqdm(raw, desc="Parsing cards", unit="card")
        if obj.get("layout") not in _SKIP_LAYOUTS
    ]
    index_path.write_text(
        json.dumps(
            {"updated_at": meta["updated_at"], "cards": [c.model_dump() for c in cards]}
        )
    )
    return ScryfallIndex(cards, meta["updated_at"])


def _download(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` with a progress bar."""
    with requests.get(url, headers={"User-Agent": USER_AGENT}, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0))
        with (
            dest.open("wb") as fh,
            tqdm(total=total, unit="B", unit_scale=True, desc="Downloading Scryfall data") as bar,
        ):
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                bar.update(len(chunk))

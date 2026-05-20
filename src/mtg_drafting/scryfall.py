import json
import logging
from pathlib import Path

import requests
from tqdm import tqdm

from mtg_drafting.cards import Card

_log = logging.getLogger(__name__)

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

# Type-line strings used by Scryfall as placeholders for entries that slip past the
# layout filter - most commonly Mystery Booster playtest cards typed only as
# ``"Card"``. Without this filter a placeholder entry can win the name slot in the
# index over a real card with the same name.
_PLACEHOLDER_TYPE_LINES = frozenset({"", "Card"})

# Scryfall set_type values whose cards are not part of standard cube drafting.
# Catches silver-bordered Un-sets ("funny": Unstable, Unhinged, Unglued, etc.) and
# Mystery Booster playtest cards / oversized memorabilia ("memorabilia").
_SKIP_SET_TYPES = frozenset({"funny", "memorabilia", "minigame"})


def _is_real_card_type(type_line: str) -> bool:
    """True when this entry has a real card type.

    False for placeholder type lines (``""``, ``"Card"``) and for token cards typed
    ``"Token ..."`` that slip past the layout filter (e.g. Wilds of Eldraine Role
    enchantment tokens carry layout ``"normal"`` but type ``"Token Enchantment"``)."""
    if type_line in _PLACEHOLDER_TYPE_LINES:
        return False
    if type_line.startswith("Token "):
        return False
    return True


def _is_draftable_obj(obj: dict) -> bool:
    """True when this Scryfall bulk-data entry is a real, draftable Magic card.

    Combines the layout filter, the set-type filter, and the type-line filter into
    one predicate. Each catches a different class of non-cards (tokens / emblems /
    schemes by layout; Un-sets and playtest cards by set_type; placeholder and
    token type lines by type_line) and together they leave the index with only the
    cards a cube might actually contain."""
    if obj.get("layout") in _SKIP_LAYOUTS:
        return False
    if obj.get("set_type") in _SKIP_SET_TYPES:
        return False
    return _is_real_card_type(obj.get("type_line", ""))


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
        # Two-pass registration so a real standalone always beats a split's
        # front-face shortcut. Without this, "Smelt // Herd // Saw" loading before
        # the standalone "Smelt" would steal the "smelt" key, and the standalone
        # would lose silently (or noisily, given the collision warning) for no
        # good reason.
        real = [c for c in cards if _is_real_card_type(c.type_line)]
        for card in real:
            self._register(_normalize(card.name), card)
        # Split-card front-face shortcuts run second: setdefault so any standalone
        # registered in pass 1 keeps the slot. Shortcut-vs-shortcut collisions
        # (multiple splits sharing a front-face word - "Start", "Fast", "Royal")
        # are expected ambiguity and stay silent.
        for card in real:
            if " // " in card.name:
                front_key = _normalize(card.name.split(" // ", 1)[0])
                self._by_name.setdefault(front_key, card)

    def _register(self, key: str, card: Card) -> None:
        """Insert ``card`` under ``key`` if free; otherwise keep the first registration
        and log the collision so silently-dropped real-vs-real name clashes (rare but
        possible with reprints under errata) leave a paper trail."""
        existing = self._by_name.get(key)
        if existing is None:
            self._by_name[key] = card
            return
        if existing.name != card.name or existing.type_line != card.type_line:
            _log.warning(
                "ScryfallIndex name collision on %r: kept %r (%s), dropped %r (%s)",
                key, existing.name, existing.type_line, card.name, card.type_line,
            )

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

    cached_payload = (
        json.loads(index_path.read_text()) if index_path.exists() else None
    )

    if cached_payload is not None and not refresh:
        return _index_from_payload(cached_payload)

    meta = _oracle_bulk_meta()
    if cached_payload is not None and cached_payload.get("updated_at") == meta["updated_at"]:
        return _index_from_payload(cached_payload)

    raw_path = cache_dir / "oracle_cards.json"
    _download(meta["download_uri"], raw_path)

    raw = json.loads(raw_path.read_text())
    cards = [
        _card_from_scryfall(obj)
        for obj in tqdm(raw, desc="Parsing cards", unit="card")
        if _is_draftable_obj(obj)
    ]
    index_path.write_text(
        json.dumps(
            {"updated_at": meta["updated_at"], "cards": [c.model_dump() for c in cards]}
        )
    )
    return ScryfallIndex(cards, meta["updated_at"])


def _index_from_payload(payload: dict) -> ScryfallIndex:
    """Build a :class:`ScryfallIndex` from a previously cached JSON payload."""
    cards = [Card.model_validate(c) for c in payload["cards"]]
    return ScryfallIndex(cards, payload.get("updated_at", ""))


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

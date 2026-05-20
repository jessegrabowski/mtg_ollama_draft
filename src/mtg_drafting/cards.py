from pydantic import BaseModel, ConfigDict

# Canonical WUBRG ordering used to sort and group cards by color.
COLOR_ORDER = ["W", "U", "B", "R", "G"]
COLOR_NAMES = {"W": "White", "U": "Blue", "B": "Black", "R": "Red", "G": "Green"}


class Card(BaseModel):
    """A single Magic card, slimmed down to the fields a drafter needs.

    Populated from a Scryfall bulk-data snapshot. Instances are immutable so they can be
    shared freely between packs, pools, and records.

    Parameters
    ----------
    name : str
        Primary (front-face) card name.
    mana_cost : str
        Mana cost string, e.g. ``"{1}{U}{U}"``. Empty for lands and other costless cards.
    cmc : float
        Converted mana cost / mana value.
    type_line : str
        Full type line, e.g. ``"Legendary Creature — Human Wizard"``.
    oracle_text : str
        Rules text. Empty if the card has none (e.g. a vanilla creature).
    colors : list of str
        Colors of the card itself, as WUBRG letters.
    color_identity : list of str
        Color identity, as WUBRG letters. Includes colors from mana symbols in rules
        text, which matters for build-around and fixing decisions.
    produced_mana : list of str
        Colors of mana this card can produce (WUBRG plus ``C`` for colorless). Empty
        for non-mana sources. Drives the playability simulator's land base.
    power : str, optional
        Creature power as printed (may be ``"*"``). None for non-creatures.
    toughness : str, optional
        Creature toughness as printed. None for non-creatures.
    rarity : str
        Printing rarity (``common``, ``uncommon``, ``rare``, ``mythic``, ...).
    """

    model_config = ConfigDict(frozen=True)

    name: str
    mana_cost: str = ""
    cmc: float = 0.0
    type_line: str = ""
    oracle_text: str = ""
    colors: list[str] = []
    color_identity: list[str] = []
    produced_mana: list[str] = []
    power: str | None = None
    toughness: str | None = None
    rarity: str = ""

    @property
    def is_land(self) -> bool:
        return "Land" in self.type_line

    @property
    def is_creature(self) -> bool:
        return "Creature" in self.type_line

    @property
    def producible_colors(self) -> list[str]:
        """WUBRG colors of mana this card can produce.

        Falls back to ``color_identity`` when ``produced_mana`` is empty; identity is
        a sound approximation for most lands."""
        source = self.produced_mana or self.color_identity
        return [c for c in source if c in COLOR_ORDER]

    @property
    def color_category(self) -> str:
        """Bucket name used to group a pool: a single color, ``Multicolor``,
        ``Colorless``, or ``Land``."""
        if self.is_land:
            return "Land"
        if not self.colors:
            return "Colorless"
        if len(self.colors) > 1:
            return "Multicolor"
        return COLOR_NAMES[self.colors[0]]

    def render(self) -> str:
        """One-line description shown to the LLM inside a pack listing."""
        parts = [self.name]
        if self.mana_cost:
            parts.append(self.mana_cost)
        parts.append(f"— {self.type_line}" if self.type_line else "")
        if self.is_creature and self.power is not None:
            parts.append(f"({self.power}/{self.toughness})")
        head = " ".join(p for p in parts if p)
        if self.oracle_text:
            body = self.oracle_text.replace("\n", " / ")
            return f"{head} | {body}"
        return head

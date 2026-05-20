import os
from pathlib import Path

from pydantic import BaseModel, Field

# Default model: taken from the MTG_DRAFT_MODEL environment variable (set it in the
# project .env file), falling back to an arbitrary tag. Any chat model the Ollama
# server can run works; override per-run with --model.
DEFAULT_MODEL = os.environ.get("MTG_DRAFT_MODEL", "gemma4:e4b")


class Paths(BaseModel):
    """Filesystem locations used by the simulator.

    Defaults are resolved relative to ``root``, which is the current working directory
    when unset — the pixi tasks run from the project root, so that is the cube/results
    directory.

    Parameters
    ----------
    root : Path, optional
        Project root. Default the current working directory.
    """

    root: Path = Field(default_factory=Path.cwd)

    @property
    def cubes_dir(self) -> Path:
        return self.root / "data" / "cubes"

    @property
    def cache_dir(self) -> Path:
        return self.root / "data" / "cache"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"


class LLMConfig(BaseModel):
    """Settings for the Ollama-backed drafting model.

    Parameters
    ----------
    model : str, optional
        Tag of any chat model the Ollama server can run. Default ``"gemma4:e4b"``.
    host : str, optional
        Ollama server URL. None lets the client fall back to ``OLLAMA_HOST`` or
        ``http://localhost:11434``. Default None.
    think : bool, optional
        Enable the model's hidden reasoning pass. When on, ``num_predict`` is
        bypassed so reasoning is never truncated; thinking is much slower per pick
        and the structured answer is reliable without it. Default False.
    temperature : float, optional
        Sampling temperature. Default 0.7.
    num_ctx : int, optional
        Context-window size requested from the model. Default 8192.
    num_predict : int, optional
        Ceiling on tokens generated per response — a runaway guard set high enough not
        to truncate a normal reply. Bypassed entirely when ``think`` is on, since
        reasoning needs an uncapped budget. Default 512.
    max_retries : int, optional
        Extra attempts when the model returns an invalid pick. Default 1.
    transport_retries : int, optional
        Retries on transient connection/transport failures, with backoff. Default 3.
    timeout : float, optional
        Per-request timeout in seconds. Default 120.0.
    """

    model: str = DEFAULT_MODEL
    host: str | None = None
    think: bool = False
    temperature: float = 0.7
    num_ctx: int = 8192
    num_predict: int = 512
    max_retries: int = 1
    transport_retries: int = 3
    timeout: float = 120.0


class DraftConfig(BaseModel):
    """Top-level configuration for a draft run.

    Parameters
    ----------
    n_seats : int, optional
        Number of seats at the table. Default 8.
    packs_per_round : int, optional
        Packs each seat opens, i.e. number of rounds. Default 3.
    cards_per_pack : int, optional
        Cards in each freshly opened pack. Default 15.
    seed : int, optional
        Seed for pack shuffling. None draws a random seed, which is then recorded.
        Default None.
    history_maxlen : int, optional
        Recent pick exchanges kept verbatim in each seat's prompt context. Default 5.
    llm : LLMConfig, optional
        Model settings. Default an ``LLMConfig`` with library defaults.
    paths : Paths, optional
        Filesystem locations. Default a ``Paths`` rooted at the working directory.
    """

    n_seats: int = Field(default=8, ge=2)
    packs_per_round: int = Field(default=3, ge=1)
    cards_per_pack: int = Field(default=15, ge=2)
    seed: int | None = None
    history_maxlen: int = Field(default=5, ge=0)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    paths: Paths = Field(default_factory=Paths)

    @property
    def min_cube_size(self) -> int:
        """Smallest cube that can supply every pack."""
        return self.n_seats * self.packs_per_round * self.cards_per_pack

    @property
    def picks_per_seat(self) -> int:
        """Total cards each seat ends the draft with."""
        return self.packs_per_round * self.cards_per_pack

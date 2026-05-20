# mtg-drafting

Simulate Magic: The Gathering **cube drafts** with a local LLM (via [Ollama](https://ollama.com))
playing every seat at the table. The goal is cube *tuning*: run a draft, read the
final pools and an LLM self-evaluation, and see which cards and archetypes are over- or
under-powered.

## How it works

1. A cube list is loaded and enriched with card data from a local Scryfall snapshot.
2. Packs are generated and dealt to `n_seats` seats.
3. For every pick, the LLM is prompted *as the seat on the clock* — it sees that seat's
   pool, a running strategy summary, its recent picks, and the current pack — and
   returns a structured choice.
4. Packs pass around the table (direction alternates each round) until empty.
5. Final pools, pick reasoning, and a per-seat LLM rating are written to `results/`.

The LLM plays all seats, so each seat keeps its own **hybrid context**: a running
strategy summary plus the last few pick exchanges (older turns are dropped to bound the
prompt size).

## Setup

This project needs two tools on your machine: **pixi** (environment manager) and
**Ollama** (local LLM runtime).

### 1. Install pixi

```sh
curl -fsSL https://pixi.sh/install.sh | sh
```

On macOS you can instead run `brew install pixi`. See the [pixi docs](https://pixi.sh)
for other platforms. Restart your shell afterwards so `pixi` is on your `PATH`.

### 2. Install Ollama

Download the installer from [ollama.com/download](https://ollama.com/download), or on
macOS run `brew install ollama`. Make sure the server is running — launch the Ollama
app, or run `ollama serve` — then pull a model to draft with. Any Ollama chat model
works; `gemma4:26b` is the default because it strikes a good balance of speed and
quality on the test system (an M4 MacBook Pro):

```sh
ollama pull gemma4:26b
```

Use any other model by passing `--model <tag>` (see Usage), or change the default for
every run by setting `MTG_DRAFT_MODEL` in the project's `.env` file:

```sh
MTG_DRAFT_MODEL=llama3.1:8b
```

Larger models generally draft better; smaller ones are faster.

### 3. Install the project environment

```sh
pixi install
```

## Usage

```sh
# Run a draft (downloads the Scryfall bulk snapshot on first run).
pixi run draft data/cubes/my_cube.txt

# Watch every pick: pack offered, card taken with reasoning, deck so far.
pixi run draft data/cubes/my_cube.txt -v

# Evaluate a finished draft.
pixi run evaluate results/<run-dir>/
```

### Pick speed and `--think`

By default the model reasons briefly *inside* the structured reply (the `reasoning`
field), which is fast (~3s per pick) and records the deliberation in `draft.json`.

Passing `--think` enables the model's full hidden reasoning pass. It picks more
carefully but is far slower (~10-15s per pick), and the token budget for the answer is
uncapped so the structured reply still completes. Hidden thinking models *must* run
uncapped — a token limit truncates the reasoning before the answer is ever produced.

### Cube list format

Plain text, one card per line. `#` starts a comment; blank lines are ignored. A line
may carry a copy count (`3 Sol Ring` or `3x Sol Ring`) and a trailing `(SET) num` tag.
Duplicates are kept — a copy count and the same name repeated on several lines both
add multiple copies to the cube.

## Development

```sh
pixi run test    # unit tests
pixi run lint    # ruff
```

## Layout

| Module        | Responsibility                                           |
|---------------|----------------------------------------------------------|
| `config.py`   | Draft / LLM / path configuration                         |
| `cards.py`    | `Card` data model                                        |
| `scryfall.py` | Bulk-data download, parse, offline name index            |
| `cube.py`     | Parse a cube list into `Card`s                           |
| `packs.py`    | Seeded pack generation                                   |
| `seat.py`     | Per-seat pool, strategy summary, pick history            |
| `draft.py`    | Draft state machine (passing, rounds)                    |
| `llm.py`      | Ollama client wrapper with structured output             |
| `prompts.py`  | Prompt construction                                      |
| `agent.py`    | Make one pick for one seat                               |
| `runner.py`   | Full draft loop                                          |
| `records.py`  | Serialize results                                        |
| `evaluate.py` | Post-draft deck build + LLM rating                       |
| `cli.py`      | Command-line entrypoint                                  |

## License

MIT — see [LICENSE](LICENSE).

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

# Watch every pick as a Rich panel (see below).
pixi run draft data/cubes/my_cube.txt -v

# Evaluate a finished draft.
pixi run evaluate results/<run-dir>/
```

### Verbose output

With `-v`, each pick renders as a single panel showing the pack, the picker's
reasoning, the strategist's current plan, the seat's draft memory, and a running
maindeck / sideboard view with a mana-curve histogram and decklist:

```
╭─ Seat 5 · Pack 1, Pick 2 (from 14) ───────────────────────────────────────────────────────╮
│ Pack: Bad Moon · Phantom Monster · Goblin King · Scryb Sprites · Pit Trap · Earthquake ·  │
│ Erg Raiders · Shambling Strider · Orcish Lumberjack · Sedge Troll · Seasinger · Black     │
│ Knight · Mind Twist · River Merfolk                                                       │
│                                                                                           │
│ → Mind Twist  The standout here is Mind Twist, a powerful card advantage tool that fits   │
│ both the control and open directions. It's a strong discard spell that can help control   │
│ the game's tempo while also being raw power in an uncommitted pool. I take Mind Twist.    │
│                                                                                           │
│ Plan (committed)  [10/10] U control · [7/10] open open · [3/10] U midrange                │
│ Needs: cheap removal · evasive 3-drop · card advantage                                    │
│ Watching: free mana sources, recurring card draw effects                                  │
│                                                                                           │
│ Memory (1)                                                                                │
│   1. [strat P1] Monitor for efficient removal and evasive creatures to support Ancestral  │
│ Recall in a control or midrange strategy.                                                 │
│                                                                                           │
│ MAIN (2): U 1 · B 1                                   SIDE (0)                            │
│  2 │   █                                              (empty)                             │
│  1 │   █                                                                                  │
│    └────────────────                                                                      │
│      0 1 2 3 4 5 6 7+                                                                     │
│                                                                                           │
│ Instant (1)                                                                               │
│   U   Ancestral Recall                                                                    │
│ Sorcery (1)                                                                               │
│   XB  Mind Twist                                                                          │
│                                                                                           │
╰───────────────────────────────────────────────────────────────────────────────────────────╯
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

## License

MIT — see [LICENSE](LICENSE).

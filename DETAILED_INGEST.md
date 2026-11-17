# Detailed graph building on a stronger machine

`INGEST_MODE=detailed` builds the graph in three passes per document instead of one call per chunk:

1. **Read the whole document** and extract entities and facts. Every fact carries a quote copied from the text.
2. **"What did we miss?"** The model sees what was found and adds what is missing.
3. **Check every fact.** Its quote must be in the document (checked in code), and the model must confirm the
   quote states that fact. Anything else is dropped.

Each kept fact is linked to the chunk holding its quote, so answers cite the passage it came from.
Code: [src/deep_extract.py](src/deep_extract.py). The rebuild script: [src/rebuild_graph.py](src/rebuild_graph.py).

On one Northbridge CV (with qwen2.5:7b), the current graph had 7 entities and 6 relations; the detailed run
kept 18 entities and 15 facts, and dropped 1 fact whose quote wasn't in the text.

## The setup: the PC runs the model, the Mac runs everything else

The documents, Neo4j, Postgres and the embeddings stay on the Mac. Only the extraction model's calls go to the PC
(`EXTRACT_OLLAMA_URL`), so questions on the Mac never wait behind the rebuild.

### On the PC (the evening before)

1. Install Ollama and download the model (about 17 GB):
   ```
   ollama pull gemma4:26b
   ```
2. Let the Mac reach it: set the environment variable `OLLAMA_HOST=0.0.0.0` (Windows: *Edit the system
   environment variables* → Environment Variables → New), quit Ollama from the tray and start it again.
3. Allow port **11434** for private networks in the firewall (Windows: the first time Ollama listens, accept the
   prompt for *Private networks*).
4. Note the PC's address: `ipconfig` → *IPv4 Address*, e.g. `192.168.1.50`.
5. Turn sleep off (Settings → Power → Screen and sleep → *Never*).

### On the Mac

Neo4j and Postgres (Docker) and Ollama (for the bge-m3 embeddings) must be running. Then, from this folder:

```
cd ~/Desktop/KGRAG-detailed
export EXTRACT_OLLAMA_URL=http://192.168.1.50:11434 EXTRACT_MODEL=gemma4:26b
```

**1. Check (writes nothing).** It confirms the PC answers, runs one document through all three passes, shows
what it found, and prints the estimated finish time for a start at 07:00:

```
uv run --no-project --python 3.11 --with-requirements requirements.txt python -m src.rebuild_graph --all --dry-run
```

**2. Run.** `caffeinate -i` keeps the Mac awake for the whole run:

```
caffeinate -i uv run --no-project --python 3.11 --with-requirements requirements.txt python -m src.rebuild_graph --all
```

Each document prints a line with its time, entities, facts, how many were dropped, and the ETA. A document that
fails keeps its old graph. After 3 failures in a row the run stops and prints the command to continue.

**3. Continue after a stop** (Ctrl-C, the PC slept, a crash): the report remembers what's done.

```
caffeinate -i uv run --no-project --python 3.11 --with-requirements requirements.txt python -m src.rebuild_graph --all --resume data/rebuilds/rebuild-YYYYMMDD-HHMM.json
```

Other options: `--corpus c_large_northbridge` (repeatable) instead of `--all`, `--limit 5`, `--doc DOC-…`,
`--mode fast` (the old one-call-per-chunk extraction, on the PC's model).

## How long it takes

`--all` is 132 documents (125,791 characters). One document takes three calls that read about 3,000 tokens and
write about 1,600 (the quotes make the replies long). Estimates for the PC (RTX 4060 8 GB, 32 GB RAM); the dry
run prints the real figure:

| Model | Per document | 132 documents | Started 07:00, done around |
|---|---|---|---|
| `gemma4:26b` | ~100–160 s | ~3.7–6 h | 10:45–13:00 |
| `gemma4:12b` | ~55–85 s | ~2–3 h | 09:00–10:15 |

## The report

`data/rebuilds/rebuild-*.json`, saved after every document: the settings, each corpus's entity and relation
counts before and after, and per document its time, model calls, tokens, entities and facts, and what each pass
found or dropped.

## New uploads in detailed mode

Set `INGEST_MODE=detailed` (with `EXTRACT_OLLAMA_URL` and `EXTRACT_MODEL`) in `.env` and restart the server: files
uploaded in the app are then built the detailed way too.

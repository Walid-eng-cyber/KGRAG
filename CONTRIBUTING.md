# Contributing to KGRAG

## Branches

| Branch | Role | Who writes to it |
|---|---|---|
| `main` | Production. Every commit is a release and gets a version tag. | Only PRs from `dev` (release) or `hotfix/*` |
| `dev` | Integration. All features and fixes land here first. Default branch. | Only PRs from feature/fix branches |
| `feat/*`, `fix/*`, `docs/*`, `refactor/*`, `chore/*` | Short-lived work branches, cut from `dev` | You |
| `hotfix/*` | Urgent production fix, cut from `main` | You |

Nobody pushes directly to `main` or `dev`; both are protected and need a PR with green CI.

```
main ──●───────────────●──────── tags: v1.0.0, v1.1.0 …
        \             ↗
dev  ────●──●──●──●──●──────────
          \  ↗  \  ↗
       feat/x   fix/y
```

## Everyday work: a feature or a fix

```bash
git switch dev && git pull
git switch -c feat/hybrid-reranking      # or fix/…, docs/…, refactor/…, chore/…
# … work, commit …
git push -u origin feat/hybrid-reranking
gh pr create --base dev --fill
```

Squash-merge the PR into `dev` once CI is green, then delete the branch.

## Releasing: `dev` → `main`

```bash
gh pr create --base main --head dev --title "Release v1.1.0"
# after merging:
git switch main && git pull
git tag -a v1.1.0 -m "v1.1.0" && git push origin v1.1.0
gh release create v1.1.0 --generate-notes
```

Use a **merge commit** (not squash) for release PRs so `dev` and `main` keep a shared history. Release often; if `dev` is more than a week or two ahead of `main`, cut a release.

Versioning follows [SemVer](https://semver.org): `MAJOR` for breaking changes (schema, config, CLI), `MINOR` for features, `PATCH` for fixes.

## Hotfix: urgent bug in `main`

```bash
git switch main && git pull
git switch -c hotfix/citation-crash
# … fix, commit, push …
gh pr create --base main --fill            # merge, then tag a patch: v1.1.1
# then bring the fix back so dev doesn't lose it:
gh pr create --base dev --head main --title "Back-merge v1.1.1 into dev"
```

The back-merge is the step people forget. Do it right after tagging.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org): `feat: add hybrid reranker`, `fix: drop invalid citations before retry`, `docs: explain router threshold`, `chore: bump neo4j driver`.

## CI

Every PR to `dev` or `main` runs:

- **lint** — ruff for syntax errors and undefined names, plus byte-compiling every file.
- **imports** — installs `requirements.txt` and imports `config` and every `src` module. No Neo4j, Postgres or Ollama needed.
- **source-branch** (PRs into `main` only) — rejects anything that isn't `dev` or `hotfix/*`.

Before opening a PR that touches retrieval, routing or answering, also run the relevant harness locally (`python -m src.eval_router`, `python -m src.eval_retrieval`, or `python -m src.benchmark`) and mention the result in the PR.

## Dependency updates

Dependabot opens at most two grouped PRs into `dev`: one weekly PR with all Python updates and one monthly PR with all GitHub Actions updates. CI only proves every module still imports, so for major version bumps (e.g. `neo4j`, `pandas`, `pgvector`), run the pipeline locally before merging.

## Never commit

`.env`, API keys, database passwords, or downloaded filings under `data/`.

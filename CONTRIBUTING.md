# Contributing

## Setup

```bash
git clone https://github.com/fodorad/StillFace
cd StillFace
make dev
```

## Workflow

[GitHub Flow](https://docs.github.com/en/get-started/using-github/github-flow): `main` is the
only long-lived branch. Work on a short-lived `feat/*`, `fix/*`, `docs/*` or `ci/*` branch and open
a pull request against `main`. Pull requests are squash-merged once CI is green. Releases are
cut by merging the `chore(main): release X.Y.Z` pull request opened by release-please.

## Test-driven development

Write the test first. Tests are written with `unittest` and should exercise real, useful behavior — no
trivial "does it not crash" placeholders. Target ≥90% coverage
(`make test`). Fixtures for real-data tests live in `tests/fixtures/`
and are only added when the underlying data is actually available.

`tests/` mirrors the `stillface/` source layout file-for-file. Tests use `unittest`, real
(small, synthetic) media generated with ffmpeg, and no mocks.

## Commit messages

[Conventional Commits](https://www.conventionalcommits.org/) — `feat:`,
`fix:`, `docs:`, `refactor:`, `test:`, `chore:`, `ci:`, `perf:`,
`build:`. These drive the semantic-versioned changelog via
release-please, so accuracy matters more than brevity.

## Docstrings

Google-style docstrings, including on module-level attributes and
constants (Sphinx autodoc otherwise renders an empty table for them).

## Before opening a PR

```bash
make check   # lint + type-check + test + docs
```

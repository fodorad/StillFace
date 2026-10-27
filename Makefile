.PHONY: install dev upgrade fix lint type-check test docs docs-serve diagrams check clean help

help:
	@echo "Setup:    install | dev | upgrade"
	@echo "Fix:      fix"
	@echo "Checks:   lint | type-check | test | docs | check"
	@echo "Docs:     docs-serve | diagrams"
	@echo "Cleanup:  clean"

# ── Setup ──────────────────────────────────────────────────────────────────────

install:
	uv venv --python 3.13
	uv pip install -e .

dev:
	uv venv --python 3.13
	uv pip install -e ".[dev,docs]"
	.venv/bin/pre-commit install

upgrade:
	uv pip install --upgrade -e ".[dev,docs]"

# ── Dev helpers (modify files) ─────────────────────────────────────────────────

fix:
	uv run ruff format .
	uv run ruff check --fix .

# ── Checks (read-only, mirror GitHub CI) ───────────────────────────────────────

lint:
	uv run ruff check .
	uv run ruff format --check .

type-check:
	uv run ty check stillface

test:
	uv run coverage run -m unittest discover -s tests
	uv run coverage report
	uv run coverage html
	uv run coverage xml -o coverage.xml

docs:
	uv run sphinx-build -W -b html docs/ site/

check: lint type-check test docs

# ── Docs ───────────────────────────────────────────────────────────────────────

docs-serve:
	uv run sphinx-autobuild docs/ site/

# Render docs/mermaid/*.mmd to JPG (white background) for the README; needs Node.js. If the
# browser bundled with the Mermaid CLI fails, point PUPPETEER_EXECUTABLE_PATH to a local
# Chromium-based browser (Chrome, Brave).
MERMAID_CLI := @mermaid-js/mermaid-cli@11.4.2

diagrams:
	for f in docs/mermaid/*.mmd; do \
		npx -y $(MERMAID_CLI) -i $$f -o $${f%.mmd}.png -b white -w 1600 -s 2 && \
		ffmpeg -loglevel error -y -i $${f%.mmd}.png -q:v 2 $${f%.mmd}.jpg && \
		rm $${f%.mmd}.png; \
	done

# ── Misc ───────────────────────────────────────────────────────────────────────

clean:
	rm -rf coverage_html dist/ site/ tmp/ coverage.xml .coverage
	find . -type d \( -name "__pycache__" -o -name ".ruff_cache" \) -not -path "./.venv/*" -exec rm -rf {} +

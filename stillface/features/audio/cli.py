"""``stillface audio …`` subcommands."""

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

from stillface.config import DatasetPaths
from stillface.data_preparation.metadata import load_metadata
from stillface.features.audio.config import DEFAULT_CONFIG_PATH, load_audio_config
from stillface.features.audio.inventory import build_inventory, read_inventory, write_inventory
from stillface.features.audio.prepare import prepare_all
from stillface.features.io import write_atomic

logger = logging.getLogger(__name__)

INVENTORY_FILE: str = "inventory.csv"
"""Name of the inventory table in the dataset-level audio folder."""


def add_audio_parser(groups: argparse._SubParsersAction) -> None:
    """Register the ``audio`` command group.

    Args:
        groups: Sub-parser collection of the top-level parser.
    """
    audio = groups.add_parser("audio", help="Audio pipeline").add_subparsers(
        dest="command", required=True
    )
    inventory = audio.add_parser("inventory", help="List the audio source of every session")
    prepare = audio.add_parser("prepare", help="Write the 16 kHz mono WAV of every phase")
    prepare.add_argument("--session-id", nargs="+", help="Sessions; all in scope if omitted")
    prepare.add_argument("--overwrite", action="store_true", help="Recompute existing outputs")
    for step in (inventory, prepare):
        step.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Audio config")


def run_audio(args: argparse.Namespace, paths: DatasetPaths) -> int:
    """Run an ``audio`` subcommand.

    Args:
        args: Parsed arguments.
        paths: Dataset locations.

    Returns:
        Process exit code.
    """
    config = load_audio_config(args.config)
    if args.command == "inventory":
        rows = build_inventory(paths, config, load_metadata(paths.metadata_path))
        out = paths.audio_dir / INVENTORY_FILE
        write_inventory(rows, out)
        logger.info("Wrote %d inventory rows to %s", len(rows), out)
    elif args.command == "prepare":
        inventory_path = paths.audio_dir / INVENTORY_FILE
        if not inventory_path.exists():
            logger.error(
                "No inventory at %s: run `stillface audio inventory` first", inventory_path
            )
            return 1
        sessions = set(args.session_id) if args.session_id else None
        report = prepare_all(
            paths, read_inventory(inventory_path), config, sessions, args.overwrite
        )
        text = json.dumps(asdict(report), indent=2) + "\n"
        write_atomic(paths.audio_dir / "prepare_report.json", lambda tmp: tmp.write_text(text))
        return 1 if report.failed else 0
    return 0

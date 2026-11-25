"""``stillface audio …`` subcommands."""

import argparse
import logging
from pathlib import Path

from stillface.config import DatasetPaths
from stillface.data_preparation.metadata import load_metadata
from stillface.features.audio.config import DEFAULT_CONFIG_PATH, load_audio_config
from stillface.features.audio.inventory import build_inventory, write_inventory

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
    inventory.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Audio config")


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
    return 0

"""Command line interface: ``stillface prepare <sync|cut|thumbnails|check-missing>``."""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from stillface.config import DatasetPaths
from stillface.data_preparation.cut import cut, cut_all
from stillface.data_preparation.metadata import load_metadata, missing_recordings, phase_windows
from stillface.data_preparation.sync import sync, sync_all
from stillface.data_preparation.thumbnails import generate_thumbnails

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(prog="stillface", description="StillFace analysis toolkit")
    parser.add_argument("--raw-dir", type=Path, help="Original dataset (default: data/DB/...)")
    parser.add_argument(
        "--processed-dir", type=Path, help="Derived data (default: data/DB_PROCESSED/...)"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    groups = parser.add_subparsers(dest="group", required=True)

    prepare = groups.add_parser("prepare", help="Data preparation").add_subparsers(
        dest="command", required=True
    )
    for name, help_text in (("sync", "Synchronize cameras"), ("cut", "Cut phases")):
        step = prepare.add_parser(name, help=help_text)
        step.add_argument("--session-id", help="Single session; all pending sessions if omitted")
        step.add_argument("--visualize", action="store_true", help="Write stacked overview videos")
    thumbs = prepare.add_parser("thumbnails", help="Generate session thumbnails")
    thumbs.add_argument("--session-id", help="Single session; all sessions if omitted")
    prepare.add_parser("check-missing", help="Report recordings flagged as missing")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line interface.

    Args:
        argv: Arguments, ``sys.argv[1:]`` if None.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    defaults = DatasetPaths.default()
    paths = DatasetPaths(
        args.raw_dir or defaults.raw_dir, args.processed_dir or defaults.processed_dir
    )

    if args.command == "sync":
        if args.session_id:
            sync(paths, args.session_id, args.visualize)
        else:
            sync_all(paths, args.visualize)
    elif args.command == "cut":
        if args.session_id:
            windows = phase_windows(load_metadata(paths.metadata_path), args.session_id)
            cut(paths, args.session_id, windows, args.visualize)
        else:
            cut_all(paths, args.visualize)
    elif args.command == "thumbnails":
        generate_thumbnails(paths, [args.session_id] if args.session_id else None)
    elif args.command == "check-missing":
        report = missing_recordings(load_metadata(paths.metadata_path))
        out = paths.processed_dir / "missing_files_report.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(out, index=False)
        logger.info("Wrote %d missing recordings to %s", len(report), out)
    return 0

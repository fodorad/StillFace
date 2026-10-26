"""Command line interface.

``stillface prepare <sync|cut|thumbnails|check-missing>`` runs the data preparation,
``stillface features heads`` the head and role pilot of the gaze extraction and
``stillface features gaze`` the gaze extraction itself.
"""

import argparse
import logging
from collections.abc import Callable, Sequence
from pathlib import Path

from stillface.config import DatasetPaths
from stillface.constants import CAMERAS, PHASES
from stillface.data_preparation.cut import cut, cut_all
from stillface.data_preparation.metadata import load_metadata, missing_recordings, phase_windows
from stillface.data_preparation.sync import sync, sync_all
from stillface.data_preparation.thumbnails import generate_thumbnails
from stillface.features.gaze.config import DEFAULT_CONFIG_PATH, load_gaze_config
from stillface.features.gaze.estimator import GazeEstimator
from stillface.features.gaze.heads import HeadTracker
from stillface.features.gaze.pilot import run_heads
from stillface.features.gaze.pipeline import run_gaze

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

    features = groups.add_parser("features", help="Feature extraction").add_subparsers(
        dest="command", required=True
    )
    for name, help_text in (
        ("heads", "Head detection, roles and overlays (no gaze model)"),
        ("gaze", "Gaze extraction"),
    ):
        step = features.add_parser(name, help=help_text)
        step.add_argument("--session-id", nargs="+", required=True, help="Sessions to process")
        step.add_argument("--camera", nargs="+", choices=CAMERAS, default=list(CAMERAS))
        step.add_argument("--phase", nargs="+", choices=PHASES, default=list(PHASES))
        step.add_argument("--no-overlay", action="store_true", help="Skip the overlay videos")
        step.add_argument("--overwrite", action="store_true", help="Recompute existing outputs")
        step.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Gaze config")
    return parser


def _exordium_head_tracker() -> HeadTracker:
    """Build the head tracker used on real data.

    Close cameras use the face detector. Side cameras combine OWLv2, prompted with the
    baby's head (reclined, in profile or half hidden, where face detectors fail), with the
    same face detector for the mother.
    """
    from stillface.features.gaze.combined import CombinedHeadTracker, PerCameraTracker
    from stillface.features.gaze.exordium_heads import ExordiumHeadTracker
    from stillface.features.gaze.owl_heads import OwlHeadTracker

    faces = ExordiumHeadTracker(device_id=0)
    return PerCameraTracker(side=CombinedHeadTracker(OwlHeadTracker(), faces), close=faces)


def _gaze_estimator(variant: str) -> GazeEstimator:
    """Build the Gaze-LLE estimator of a model variant."""
    raise RuntimeError(
        "The Gaze-LLE adapter is not available yet: it needs the exordium Gaze-LLE wrapper"
    )


def main(
    argv: Sequence[str] | None = None,
    head_tracker_factory: Callable[[], HeadTracker] = _exordium_head_tracker,
    gaze_estimator_factory: Callable[[str], GazeEstimator] = _gaze_estimator,
) -> int:
    """Run the command line interface.

    Args:
        argv: Arguments, ``sys.argv[1:]`` if None.
        head_tracker_factory: Builds the head tracker for ``features heads``; replaceable so
            the command can run without exordium models.
        gaze_estimator_factory: Builds the gaze estimator for a model variant, for
            ``features gaze``; replaceable for the same reason.

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

    if args.group == "features":
        config = load_gaze_config(args.config)
        tracker = head_tracker_factory()
        estimator = gaze_estimator_factory(config.model_variant) if args.command == "gaze" else None
        for session_id in args.session_id:
            options = dict(overlay=not args.no_overlay, overwrite=args.overwrite)
            if estimator is None:
                run_heads(paths, session_id, tracker, config, args.camera, args.phase, **options)
            else:
                run_gaze(
                    paths,
                    session_id,
                    tracker,
                    estimator,
                    config,
                    args.camera,
                    args.phase,
                    **options,
                )
    elif args.command == "sync":
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

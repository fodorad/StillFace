<!--
Kept near-identical to README.md minus GitHub-specific chrome (badges).
Update both in the same change.
-->

# StillFace

StillFace is a toolkit for the analysis of mother-baby interaction recorded with the **Still Face
paradigm**. A session consists of several synchronized camera views, heart rate variability
recordings and annotations, and is divided into the *baseline*, *play*, *stillface* and *reunion*
phases. The package prepares the raw recordings and provides the building blocks for extracting
behavioral features such as touch, affect and gaze between mother and baby.

## Features

| Stage | Functionality | Status |
|---|---|---|
| Data preparation | Camera synchronization (audio cross-correlation, via [avst](https://pypi.org/project/avst/)) | available |
| Data preparation | Phase cutting, stacked overview videos, session thumbnails | available |
| Data preparation | Metadata checks (missing recordings) | available |
| Feature extraction | Touch, affect, gaze | planned |
| Visualization | Overlays and export to [Annie](https://github.com/fodorad/Annie) | planned |

## Data layout

The original recordings are never modified. The `data/` folder of a checkout holds two soft links:

```text
data/
├── DB            -> <original dataset>/ELTE-PPK_StillFace      (read-only)
└── DB_PROCESSED  -> <processed dataset>/ELTE-PPK_StillFace     (generated)
```

All generated files are written below `DB_PROCESSED`:

```text
Sessions/<id>/normalized/   sources converted to mp4 at 60 fps
Sessions/<id>/synced/       time-aligned camera videos
Sessions/<id>/cut/          per-phase clips
Sessions/<id>/visualize/    stacked overview videos
Sessions/<id>/thumbnail.png
Thumbnails/                 one thumbnail per session
logs/                       synced / failed session bookkeeping
```

The locations can be overridden with the `STILLFACE_RAW_DIR` and `STILLFACE_PROCESSED_DIR`
environment variables or the `--raw-dir` / `--processed-dir` options.

## Installation

[ffmpeg](https://ffmpeg.org) has to be available on the `PATH`.

```bash
git clone https://github.com/fodorad/StillFace
cd StillFace
make dev        # virtual environment, editable install, pre-commit hooks
```

## Usage

```bash
stillface prepare check-missing                       # report recordings flagged as missing
stillface prepare sync --visualize                    # synchronize all pending sessions
stillface prepare sync --session-id 575613            # a single session
stillface prepare cut --visualize                     # cut phases and render overviews
stillface prepare thumbnails                          # session thumbnails
stillface features heads --session-id 605317          # mother and baby head boxes, overlays
```

Sessions are selected from `metadata_database.xlsx`: automatic processing is skipped for rows
with `Auto = n`, and synchronization for rows with a manually entered mother-baby offset.

## Head detection

Gaze, affect and touch features are computed per person, so every camera clip first gets a
head box for the mother and the baby on each analysed frame. The same pipeline runs on all
four cameras and all phases: a text-prompted head detector (Grounding DINO, prompt
"a human head.") and a face detector (YOLO11s) are run, their tracks are fused into one
track per head, and simple rules decide which track is the mother and which is the baby:
on the close-up cameras the owner is the head whose face is detected and lies near the image
centre, on the side cameras the two strongest heads are told apart by the fixed seating of
each phase. Frames without a reliable head box are marked as not codable with the reason, and
every row records whether the face was visible (the frame-wise usability mask).

<img src="https://raw.githubusercontent.com/fodorad/StillFace/main/docs/mermaid/head_pipeline.jpg" alt="Head detection pipeline" width="100%"/>

<img src="https://raw.githubusercontent.com/fodorad/StillFace/main/docs/mermaid/role_assignment.jpg" alt="Role assignment" width="100%"/>

The diagram sources are in `docs/mermaid/`; `make diagrams` renders them.

## Package structure

```text
stillface/
├── config.py            dataset locations
├── constants.py         cameras, phases, target frame rate
├── cli.py               command line interface
└── data_preparation/    sync, cut, visualize, thumbnails, metadata
```

Feature extraction submodules are added next to `data_preparation`, one subpackage per
behavior.

## Development

```bash
make check      # lint + type check + tests with coverage + docs (same as CI)
make fix        # format and auto-fix lint issues
```

Development follows test-driven practice: `tests/` mirrors the package layout, uses `unittest`
with small synthetic media and no mocks, and keeps coverage above 90%. Commits follow
[Conventional Commits](https://www.conventionalcommits.org); versions and the changelog are
managed by [release-please](https://github.com/googleapis/release-please). See
[CONTRIBUTING.md](https://github.com/fodorad/StillFace/blob/main/CONTRIBUTING.md).

## License

Released under the [MIT License](https://github.com/fodorad/StillFace/blob/main/LICENSE).


```{toctree}
:hidden:

autoapi/index
```

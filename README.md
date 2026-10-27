<div align="center">

<img src="https://raw.githubusercontent.com/fodorad/StillFace/main/docs/assets/logo.svg" alt="StillFace" width="440"/>

<br/>

**Synchronization and behavioral feature extraction for Still Face mother-baby recordings**

[![GitHub Release](https://img.shields.io/github/v/release/fodorad/StillFace?color=purple)](https://github.com/fodorad/StillFace/releases)
[![CI](https://github.com/fodorad/StillFace/actions/workflows/ci.yml/badge.svg)](https://github.com/fodorad/StillFace/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/fodorad/StillFace/graph/badge.svg)](https://codecov.io/gh/fodorad/StillFace)
[![Docs](https://img.shields.io/badge/docs-online-blue?logo=githubpages)](https://fodorad.github.io/StillFace/)
<br/>
[![Python](https://img.shields.io/badge/python-3.13%2B-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

</div>

---

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

## Dataset statistics

Computed from `metadata_database.xlsx` and the processed phase clips (`Sessions/<id>/processed/`),
which are the only video inputs of the analysis. All processed clips are 1920x1080 at 60 fps.

- **Sessions:** 78 recorded, 70 marked for analysis (`Analysis = y`).
- **Setup versions (analysed):** v4: 57, v3: 5, v2: 8.
- **Excluded from analysis:** 646022 (v4), 566015 (v4), 635928 (v2), 555623 (v2), 656417 (v2), 747417 (v1), 606117 (v1), 697115 (v1).

**Camera availability** (analysed sessions; a camera counts if it has at least one processed phase clip):

| Cameras | Sessions | Session ids |
|---|---|---|
| mother + baby + window + door | 49 | all other analysed sessions |
| mother + baby + window | 14 | 124731, 210118, 595525, 606021, 615027, 616018, 626527, 635916, 636211, 674821, 686912, 696807, 888726, 920320 |
| mother + baby | 3 | 114754, 656531, 717115 |
| baby + window | 2 | 585803, 595514 |
| baby + window + door | 1 | 717729 |
| mother + baby + door | 1 | 726030 |

**Incomplete phase sets** (camera present but fewer than 4 phase clips): 706922 window (3/4), 596613 window (3/4), 596613 door (3/4), 596025 baby (3/4), 646012 door (2/4), 505629 baby (1/4), 585803 baby (3/4), 114754 mother (3/4), 114754 baby (3/4).

**Native camcorder frame rate** (from the original camcorder files, where available; processed clips are all 60 fps):

| Camera | 50 fps progressive | 25 fps interlaced |
|---|---|---|
| window | 50 | 15: 525222, 555820, 575716, 594801, 595813, 596613, 616205, 646302, 657006, 666921, 686527, 696707, 697426, 717729, 757314 |
| door | 33 | 18: 484603, 535124, 586228, 596426, 605317, 616022, 616211, 625527, 625710, 636023, 646012, 656024, 656520, 666712, 674703, 677022, 736920, 819104 |

The phone cameras (mother, baby) record at about 60 fps with a variable frame rate.

**Play-phase clip lengths** differ between the cameras of a session by a median of 9 frames; larger differences (frames): 606021 (32), 595514 (46), 706922 (53), 646721 (154).

**Probing set** (play phase, setup v4) used to choose the analysis tools:

| Session | Cameras | Case |
|---|---|---|
| 605317 | mother, baby, window, door | dev, door 25i |
| 626021 | mother, baby, window, door | dev |
| 819104 | mother, baby, window, door | dev, door 25i |
| 594801 | mother, baby, window, door | window 25i, door occlusion |
| 636211 | mother, baby, window | no door camera |
| 656531 | mother, baby | close cameras only |

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
[CONTRIBUTING.md](CONTRIBUTING.md).

## License

Released under the [MIT License](LICENSE).

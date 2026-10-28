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

## Probing results

The candidate tools were compared on the play phase of the probing set, using pretrained
models only and label-free quality checks with visual review. Heavy models ran on a 2 fps
grid (about 124 frames per camera). The outputs, review mosaics and one multi-view video per
session are stored under `DB_PROCESSED/ELTE-PPK_StillFace/probes/`.

**Chosen tools** for the next round:

| Block | Tool |
|---|---|
| Persons | Grounding DINO (prompts "a baby." and "a person.") + YOLOX-x persons |
| Roles | seating rules; on the side cameras baby boxes larger than 20% of the image are ignored, plus a session-level check of the baby's position and head size |
| 2D pose | RTMW-x whole body (133 keypoints); ViTPose++-H body as a candidate for fusion |
| Self-calibration | VGGT-1B initial cameras + bundle adjustment on face and arm keypoints |
| 3D bodies | confidence-weighted triangulation; views with more than 25 px error are rejected |
| Touch | 3D hand-to-body distance in baby head widths, OR a pose hand on the baby, OR a Grounding DINO mother hand on the baby (baby camera) |
| Gaze | Gaze-LLE ViT-L ChildPlay (in/out) on skeleton-anchored head boxes |
| Baby face | PyAFAR infant action units (AU6, AU12) for smiles, Qwen3-VL-4B label for smiles only |
| Mother face | LibreFace 2.0 |
| Voice type | VTC 2.2 (BabyHuBERT): baby and mother voice, overlaps kept |
| Vocal affect | AST AudioSet cry/sob/whimper and laughter probabilities, segment peak with an absolute threshold of 0.2 |
| Mother prosody | Parselmouth F0 per speech segment |

These candidates were rejected: the FiDIP infant pose model (not better than RTMW),
OWLv2 heads, radial distortion and multi-frame VGGT for calibration, and CLAP zero-shot
for fuss vs coo.

**Self-calibration and 3D** (RTMW keypoints):
- *Inlier px* is the median reprojection error of the inlier observations per camera.
- *Arm bone CV* is the median coefficient of variation of the four arm bone lengths over
  time (lower is more stable).
- *Baby 3D coverage* is the share of frames with a triangulated face, arms, legs and feet.

| Session | Quality level | Inlier px (mother, baby, window, door) | 3D median reprojection px | Arm bone CV mother / baby | Baby 3D coverage face / arms / legs / feet |
|---|---|---|---|---|---|
| 605317 | multi-view | 9, 10, 10, 11 | 8.1 | 0.05 / 0.30 | 92 / 65 / 21 / 5% |
| 626021 | multi-view | 16, 24, 14, 24 | 8.2 | 0.05 / 0.22 | 80 / 72 / 42 / 21% |
| 819104 | multi-view | 9, 16, 14, 18 | 11.0 | 0.06 / 0.13 | 93 / 77 / 52 / 19% |
| 594801 | multi-view | 8, 5, 4, 11 | 5.5 | 0.06 / 0.18 | 90 / 90 / 49 / 11% |
| 636211 | reduced-view | 6, 5, 3, – | 6.4 | 0.05 / 0.10 | 53 / 32 / 3 / 0% |
| 656531 | close-only | no calibration | – | – | – |

**Touch** (frames with touch at 2 fps, from each evidence source and fused):

| Session | 3D proximity | Pose hand on the baby | Detected mother hand on the baby | Fused |
|---|---|---|---|---|
| 605317 | 118 | 83 | 85 | 121 / 124 |
| 626021 | 63 | 52 | 68 | 98 / 124 |
| 819104 | 109 | 120 | 117 | 123 / 126 |
| 594801 | 116 | 56 | 117 | 120 / 124 |
| 636211 | 0 | 77 | 78 | 103 / 122 |
| 656531 | – | 17 | 66 | 66 / 124 |

**Gaze** on the side cameras: the share of frames with a head box, and of those, the share
with at least 30% of the gaze heatmap on the partner's head:

| Session | Window: mother head / on baby | Window: baby head / on mother | Door: mother head / on baby | Door: baby head / on mother |
|---|---|---|---|---|
| 605317 | 100 / 96% | 97 / 28% | 100 / 97% | 98 / 42% |
| 626021 | 100 / 87% | 87 / 82% | 100 / 95% | 99 / 95% |
| 819104 | 100 / 96% | 100 / 50% | 100 / 100% | 100 / 77% |
| 594801 | 100 / 100% | 100 / 67% | 100 / 2% | 4 / 40% |
| 636211 | 100 / 98% | 100 / 36% | – | – |

**Voice** (VTC seconds per phase, play vs still face, baby camera phone):

| Session | Baby voice, play | Baby voice, still face | Mother speech, play | Mother speech, still face |
|---|---|---|---|---|
| 605317 | 1.0 | 40.6 | 35.9 | 0.0 |
| 626021 | 10.8 | 21.1 | 50.1 | 0.0 |
| 819104 | 1.6 | 57.7 | 52.4 | 0.0 |
| 594801 | 1.3 | 2.7 | 31.0 | 0.0 |
| 636211 | 5.4 | 16.6 | 54.3 | 0.2 |
| 656531 | 10.3 | 35.3 | 47.1 | 0.0 |

The mother's speech vanishes in the still face, as the protocol requires, while the baby's
vocal time rises. The AST cry probability peaks at 0.83 (605317) and 0.72 (656531) in the
still face. It is higher on the frames the VLM labels crying (0.32 vs 0.14 and 0.28 vs
0.06). In 605317 cry above 0.2 covers 12% of the first half and 65% of the second half.

**Failures by cause:**

| Cause | Sessions | Effect | Handling |
|---|---|---|---|
| Occlusion | 594801 (door) | the mother hides the baby; the mother's skeleton was taken as the baby's | large baby boxes ignored, session-level baby check |
| Occlusion | 594801 (door) | the baby's head is visible on 4% of frames, so the mother's gaze has no partner box | gaze from the window camera |
| Occlusion | 605317 (mother camera) | the mother leans over the baby, her face is visible on 29 of 124 frames | face affect only on frames with a visible face |
| Pose | all | baby legs and feet rarely triangulated (feet 0–21%) | touch also uses 2D evidence in the baby camera |
| Pose | all (baby camera) | the mother's bare hands get no person box, so no skeleton | Grounding DINO hand detector |
| Framing | 605317 (baby camera) | a hand on the baby's belly not detected | open |
| Framing | 819104 (baby camera) | the mother's arm reaching over the baby from the top | open |
| Model | 626021 | calibration least accurate (inliers 14–24 px) | flagged; other initial frames and distortion did not help |
| Model | all | AU4, AU12 and AU20 fire on both cry faces and laughs | baby cry and fuss from audio only (AST) |
| Model | 605317, 656531 (still face) | the VLM labels fussing grimaces "smiling": a face veto would flip 14 of 25 correct cry segments | VLM not used to veto cry |
| Model | 656531 (play) | excited laughs reach an AST cry probability of 0.24–0.44: 2 of 27 cry segments are false | open |
| Model | all | CLAP labels most fuss segments "coo" | CLAP rejected for vocal affect |
| Camera set | 636211 | two views on the baby: no 3D touch | reduced-view level, touch from 2D evidence |
| Camera set | 656531 | close cameras only: no 3D | close-only level, touch from the hand detector |

No decoding failures occurred on the processed clips. Residual audio offsets between
the cameras are at most 23 ms, except 74 ms for the 636211 window camera.

**Compute** on a Mac mini M4 (seconds per camera frame):

| Model | s/frame |
|---|---|
| Grounding DINO persons (2 prompts) | 2.8 |
| Grounding DINO hands | 1.1 |
| RTMW-x | 0.25 |
| ViTPose++-H | 0.86 |
| Gaze-LLE ViT-L | 0.36–0.65 |
| Qwen3-VL-4B | 1.2–1.7 |

PyAFAR, LibreFace, VTC and AST take minutes per clip. A full run over 70 sessions and all
phases at 2 fps is about 136k camera frames, so Grounding DINO persons alone take about
110 hours. The full run therefore detects persons at 1 fps with tracking in between, and
estimates gaze at 5–10 Hz or on a GPU machine.

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

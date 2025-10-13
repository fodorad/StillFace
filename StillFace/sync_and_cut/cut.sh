#!/bin/bash

# ignore IDs, switch to manual cutting: 606021

ROOT_DIR=data/ELTE-PPK_StillFace
ID=555820
BASELINE_TIMESTAMP=0:34-1:31
PLAY_TIMESTAMP=2:06-3:08
STILLFACE_TIMESTAMP=3:08-4:40
REUNION_TIMESTAMP=4:40-5:42

python StillFace/cut.py \
    --synced_dir $ROOT_DIR/$ID/synced \
    --baseline_timestamps $BASELINE_TIMESTAMP \
    --play_timestamps $PLAY_TIMESTAMP \
    --stillface_timestamps $STILLFACE_TIMESTAMP \
    --reunion_timestamps $REUNION_TIMESTAMP \
    --out_dir $ROOT_DIR/$ID/processed

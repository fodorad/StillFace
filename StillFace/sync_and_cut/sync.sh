#!/bin/bash

ROOT_DIR=data/ELTE-PPK_StillFace
ID=606021

python StillFace/sync.py --db_dir $ROOT_DIR --session_id $ID

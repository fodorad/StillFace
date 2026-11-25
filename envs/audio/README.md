# Model environments of the audio pipeline

Each folder is a small uv project with its own lock file: one isolated environment per
group of models whose dependencies agree. The environments are created outside the
repository, below the folder named by `STILLFACE_AUDIO_ENVS`:

```bash
export STILLFACE_AUDIO_ENVS=/path/to/audio-envs
make audio-envs      # create or update every environment from its lock file
make audio-smoke     # run every installed model on a 10 s synthetic clip
```

The workers in `stillface/features/audio/workers/` run inside these environments with the
repository on `PYTHONPATH`; they import nothing from the package's own dependencies.

| Environment | Models |
|---|---|
| `vad` | Silero VAD, pyannote segmentation and speaker embeddings |
| `hf` | AST AudioSet, audEERING arousal / dominance / valence |
| `vtc` | VTC 2.2 voice types, addressee (KCDS / ADS / Other) |
| `e2v` | emotion2vec+ large (FunASR) |
| `littlebeats` | LittleBeats wav2vec2 speaker and vocalisation heads (Python 3.10, fairseq; disabled until verified) |

Gated Hugging Face models read the token from the `HF_TOKEN` environment variable.

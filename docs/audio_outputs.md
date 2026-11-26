# Audio model outputs

What every model of the audio pipeline returns, as checked by
`stillface.features.audio.contract` (the smoke test and the opt-in integration tests run each
model in its real environment and test its result against this table). Raw results are kept
per session and phase in `features/audio/raw/<model>/<phase>.npz`, at the model's own rate;
the fusion stage puts them on the common 10 Hz grid.

| Model | Group | Score names | Value range | Rate | Meaning |
|---|---|---|---|---|---|
| silero | frames/speech | speech | 0 .. 1 | 31.25 Hz | speech probability |
| pyannote_vad | frames/speech | speech | 0 .. 1 | 59 Hz | speech probability (max over local speakers, chunks aggregated) |
| vtc | frames/voice_type | KCHI, OCH, MAL, FEM | 0 .. 1 | 50 Hz | voice-type probabilities: key child, other child, male, female adult |
| vtc | segments/voice_type | mean_prob | 0 .. 1 | per segment | thresholded voice-type segments (label = class), mean probability |
| ast | frames/audioset | 527 names | 0 .. 1 | 4 Hz | AudioSet class probabilities, 2 s windows |
| prosody | frames/f0_adult | f0_hz | 100 .. 700 | 100 Hz | F0 in the adult range (Hz) |
| prosody | frames/f0_infant | f0_hz | 150 .. 1000 | 100 Hz | F0 in the infant range (Hz) |
| prosody | frames/intensity | intensity_db | -50 .. 130 | 100 Hz | intensity (dB) |
| prosody | frames/hnr | hnr_db | -200 .. 100 | 100 Hz | harmonics-to-noise ratio (dB) |
| egemaps | segments/egemaps | 88 names | - | per segment | eGeMAPS v02 functionals per role segment |
| littlebeats | frames/speaker | SIL, CHN, FAN, MAN, CXN | 0 .. 1 | 1 Hz | speaker: silence, key child, female adult, male adult, other child |
| littlebeats | frames/infant | CRY, FUS, BAB | 0 .. 1 | 1 Hz | infant vocalisation: cry, fuss, babble |
| littlebeats | frames/adult | CDS, ADS, LAU, SNG | 0 .. 1 | 1 Hz | adult vocalisation: child-directed, adult-directed, laugh, sing |
| addressee | segments/addressee | KCDS, ADS, Other | 0 .. 1 | per segment | addressee of each adult segment: key child, adult, other |
| pyannote_embedding | segments/embedding | duration_s | 0 .. 1e+06 | per segment | segment duration; L2-normalised speaker embedding per segment |
| audeering | segments/windows | arousal, dominance, valence | 0 .. 1 | per segment | dimensional affect per 3 s window inside role segments |
| emotion2vec | segments/windows | angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown | 0 .. 1 | per segment | emotion category probabilities per 3 s window (mother) |
| crisperwhisper | segments/verbatim_words | probability | 0 .. 1 | per segment | verbatim words (fillers, repetitions kept), word probability |
| crisperwhisper | segments/intended_words | probability | 0 .. 1 | per segment | intended words (cleaned), word probability |
| mfa | segments/words | duration_s | 0 .. 1e+06 | per segment | force-aligned words of the mother |
| mfa | segments/phones | duration_s | 0 .. 1e+06 | per segment | force-aligned phones (XPF) |

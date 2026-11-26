# AudioSet classes (AST)

AST scores **all 527** AudioSet classes in every 2 s window (0.25 s hop); nothing is filtered
before or during inference, and the raw scores of every class are kept in the raw output. The
priority tiers below only say which classes the analysis looks at first:

- **Tier 1, target vocalisations:** what the mother and the baby produce (cry, fuss, laugh,
  babble, singing, humming, speech by sex and age).
- **Tier 2, context:** the phase-change gong, bodily sounds, touch and play sounds, and
  background sounds that can mask the dyad.
- **Tier 0:** everything else (animals, vehicles, music genres ...).

Every tier 1-2 class gets its own probability lane on the 10 Hz fusion grid
(`ast.<label>.context`, set by `fusion.ast_lane_max_tier`). The table ships with the package as
`stillface/features/audio/ast_classes.csv` (`index, label, tier, group, reason`) and can be
filtered by tier or group in any spreadsheet.

## Prioritised classes

| Tier | Group | Index | Class | Why |
|---|---|---|---|---|
| 1 | Speech | 0 | Speech | who speaks: mother (female), other adult (male), child |
| 1 | Speech | 1 | Male speech, man speaking | who speaks: mother (female), other adult (male), child |
| 1 | Speech | 2 | Female speech, woman speaking | who speaks: mother (female), other adult (male), child |
| 1 | Speech | 3 | Child speech, kid speaking | who speaks: mother (female), other adult (male), child |
| 1 | Baby vocalisation | 6 | Babbling | baby vocalisation type (fusion: baby cry / fuss / laugh / babble) |
| 1 | Baby vocalisation | 14 | Screaming | baby vocalisation type (fusion: baby cry / fuss / laugh / babble) |
| 1 | Mother vocalisation | 15 | Whispering | maternal vocalisation type (fusion: singing / humming / laughter) |
| 1 | Laughter (either partner) | 16 | Laughter | laughter of either partner; the role comes from the diarisation |
| 1 | Baby vocalisation | 17 | Baby laughter | baby vocalisation type (fusion: baby cry / fuss / laugh / babble) |
| 1 | Laughter (either partner) | 18 | Giggle | laughter of either partner; the role comes from the diarisation |
| 1 | Mother vocalisation | 20 | Belly laugh | maternal vocalisation type (fusion: singing / humming / laughter) |
| 1 | Mother vocalisation | 21 | Chuckle, chortle | maternal vocalisation type (fusion: singing / humming / laughter) |
| 1 | Baby vocalisation | 22 | Crying, sobbing | baby vocalisation type (fusion: baby cry / fuss / laugh / babble) |
| 1 | Baby vocalisation | 23 | Baby cry, infant cry | baby vocalisation type (fusion: baby cry / fuss / laugh / babble) |
| 1 | Baby vocalisation | 24 | Whimper | baby vocalisation type (fusion: baby cry / fuss / laugh / babble) |
| 1 | Baby vocalisation | 25 | Wail, moan | baby vocalisation type (fusion: baby cry / fuss / laugh / babble) |
| 1 | Mother vocalisation | 27 | Singing | maternal vocalisation type (fusion: singing / humming / laughter) |
| 1 | Mother vocalisation | 33 | Female singing | maternal vocalisation type (fusion: singing / humming / laughter) |
| 1 | Mother vocalisation | 37 | Humming | maternal vocalisation type (fusion: singing / humming / laughter) |
| 1 | Mother vocalisation | 271 | Lullaby | maternal vocalisation type (fusion: singing / humming / laughter) |
| 2 | Context / confound | 7 | Speech synthesizer | background or confound that can mask the dyad (fusion: CONTEXT) |
| 2 | Bodily sound | 26 | Sigh | non-cry vocal or bodily sounds of the baby (descriptive only) |
| 2 | Bodily sound | 44 | Gasp | non-cry vocal or bodily sounds of the baby (descriptive only) |
| 2 | Bodily sound | 47 | Cough | non-cry vocal or bodily sounds of the baby (descriptive only) |
| 2 | Bodily sound | 49 | Sneeze | non-cry vocal or bodily sounds of the baby (descriptive only) |
| 2 | Context / confound | 53 | Walk, footsteps | background or confound that can mask the dyad (fusion: CONTEXT) |
| 2 | Bodily sound | 58 | Burping, eructation | non-cry vocal or bodily sounds of the baby (descriptive only) |
| 2 | Bodily sound | 59 | Hiccup | non-cry vocal or bodily sounds of the baby (descriptive only) |
| 2 | Touch and play | 63 | Clapping | touch or play sounds during the interaction |
| 2 | Context / confound | 135 | Rattle | background or confound that can mask the dyad (fusion: CONTEXT) |
| 2 | Context / confound | 137 | Music | background or confound that can mask the dyad (fusion: CONTEXT) |
| 2 | Touch and play | 175 | Rattle (instrument) | touch or play sounds during the interaction |
| 2 | Paradigm signal | 177 | Gong | phase-change gong (fusion: GONG, phase edges only) |
| 2 | Paradigm signal | 200 | Bell | phase-change gong (fusion: GONG, phase edges only) |
| 2 | Paradigm signal | 205 | Chime | phase-change gong (fusion: GONG, phase edges only) |
| 2 | Paradigm signal | 214 | Singing bowl | phase-change gong (fusion: GONG, phase edges only) |
| 2 | Context / confound | 354 | Door | background or confound that can mask the dyad (fusion: CONTEXT) |
| 2 | Context / confound | 359 | Knock | background or confound that can mask the dyad (fusion: CONTEXT) |
| 2 | Context / confound | 361 | Squeak | background or confound that can mask the dyad (fusion: CONTEXT) |
| 2 | Context / confound | 389 | Telephone | background or confound that can mask the dyad (fusion: CONTEXT) |
| 2 | Context / confound | 412 | Mechanical fan | background or confound that can mask the dyad (fusion: CONTEXT) |
| 2 | Context / confound | 413 | Air conditioning | background or confound that can mask the dyad (fusion: CONTEXT) |
| 2 | Touch and play | 467 | Slap, smack | touch or play sounds during the interaction |
| 2 | Context / confound | 500 | Silence | no sound in the window |
| 2 | Context / confound | 520 | White noise | background or confound that can mask the dyad (fusion: CONTEXT) |
| 2 | Context / confound | 524 | Television | background or confound that can mask the dyad (fusion: CONTEXT) |
| 2 | Context / confound | 525 | Radio | background or confound that can mask the dyad (fusion: CONTEXT) |

## All 527 classes

```{csv-table}
:file: ../stillface/features/audio/ast_classes.csv
:header-rows: 1
:widths: 6 30 5 12 47
```

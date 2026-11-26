# Changelog

## [0.3.0](https://github.com/fodorad/StillFace/compare/stillface-v0.2.0...stillface-v0.3.0) (2026-11-26)


### Features

* **audio:** Annie export, QC record and listening list ([#44](https://github.com/fodorad/StillFace/issues/44)) ([4d7bd6a](https://github.com/fodorad/StillFace/commit/4d7bd6a0e21277f714a4f53be1ebbcbb5c1e3e02))
* **audio:** audEERING dimensional and emotion2vec+ categorical affect workers ([#38](https://github.com/fodorad/StillFace/issues/38)) ([e742035](https://github.com/fodorad/StillFace/commit/e7420357226cbd76b43fc12c999d99644f6f2602))
* **audio:** audio config and session inventory command ([#29](https://github.com/fodorad/StillFace/issues/29)) ([95912fb](https://github.com/fodorad/StillFace/commit/95912fb7e7c7b2aa63ed1d73a55525b63c56d828))
* **audio:** AudioSet class table with priority tiers ([#50](https://github.com/fodorad/StillFace/issues/50)) ([8b7c5d6](https://github.com/fodorad/StillFace/commit/8b7c5d61a4e06bcf2584b5c27205e9e5db806fa0))
* **audio:** CrisperWhisper 2 transcripts and MFA alignment of the mother ([#42](https://github.com/fodorad/StillFace/issues/42)) ([f60325b](https://github.com/fodorad/StillFace/commit/f60325b8d22faf32bac2888a37681b9682d88b43))
* **audio:** defaults of the full database run ([#53](https://github.com/fodorad/StillFace/issues/53)) ([28caaad](https://github.com/fodorad/StillFace/commit/28caaadb691b6df4dfb669185f671b8d1d0ea3dc))
* **audio:** full audio chain command ([#45](https://github.com/fodorad/StillFace/issues/45)) ([2cc3763](https://github.com/fodorad/StillFace/commit/2cc3763a6ea60d9acc171b7f8f99c66f91614a2b))
* **audio:** lane, band and event schema with CSV/JSON round-trip ([#28](https://github.com/fodorad/StillFace/issues/28)) ([737b8c5](https://github.com/fodorad/StillFace/commit/737b8c5d1de8ae01db43551baa415957b2a853e8))
* **audio:** LittleBeats infant and adult vocalisation worker ([#37](https://github.com/fodorad/StillFace/issues/37)) ([4d6d1d3](https://github.com/fodorad/StillFace/commit/4d6d1d3765c88b9e0d68d6888e9f94b57fa64f2e))
* **audio:** model adapter protocol with subprocess workers and provenance ([#32](https://github.com/fodorad/StillFace/issues/32)) ([df593e4](https://github.com/fodorad/StillFace/commit/df593e4436970cf78d89ed0fed3a4f7c45c78eb9))
* **audio:** model environments with Silero VAD and AST workers ([#34](https://github.com/fodorad/StillFace/issues/34)) ([bcc1fa8](https://github.com/fodorad/StillFace/commit/bcc1fa875456161eeba14741108694985713e14e))
* **audio:** mother and baby affect lanes ([#43](https://github.com/fodorad/StillFace/issues/43)) ([13205c7](https://github.com/fodorad/StillFace/commit/13205c7ac01605caaa474cb7f495bd708705a616))
* **audio:** mother voice enrolment and female segment roles ([#39](https://github.com/fodorad/StillFace/issues/39)) ([fb8db2b](https://github.com/fodorad/StillFace/commit/fb8db2b3dbee66bc35a5768cca1fc5c0aafdab7f))
* **audio:** prepare 16 kHz mono wav per session and phase ([#31](https://github.com/fodorad/StillFace/issues/31)) ([3703dac](https://github.com/fodorad/StillFace/commit/3703dac1063598bf397671c8436b36eaa53bb276))
* **audio:** prosody lanes with Parselmouth and openSMILE eGeMAPS ([#33](https://github.com/fodorad/StillFace/issues/33)) ([2b20998](https://github.com/fodorad/StillFace/commit/2b209982d6b10db1f894f8e9fcb666d2e3d46ab9))
* **audio:** pyannote segmentation and speaker-embedding worker ([#36](https://github.com/fodorad/StillFace/issues/36)) ([5ffc826](https://github.com/fodorad/StillFace/commit/5ffc826e43f41b5a2931a0902c585a6d28e03685))
* **audio:** rule-based fusion lanes and decision tables ([#40](https://github.com/fodorad/StillFace/issues/40)) ([9a61d09](https://github.com/fodorad/StillFace/commit/9a61d0959ac03020cb36a1fbf50d538db68efc2c))
* **audio:** split window cameras and clip frame rate for early sessions ([#48](https://github.com/fodorad/StillFace/issues/48)) ([c5c2fe8](https://github.com/fodorad/StillFace/commit/c5c2fe8d32842d66c7779152a6da6b7d9204f813))
* **audio:** time-gated mother and baby tracks with overlap list ([#41](https://github.com/fodorad/StillFace/issues/41)) ([9b0dcd4](https://github.com/fodorad/StillFace/commit/9b0dcd422175a3300c1ed0d95152ef84e49c8846))
* **audio:** VTC voice-type and addressee workers ([#35](https://github.com/fodorad/StillFace/issues/35)) ([fb48bec](https://github.com/fodorad/StillFace/commit/fb48bec621141c1caba209bab220169fe51120f3))


### Bug Fixes

* **audio:** model fixes found by running every model on a real recording ([#51](https://github.com/fodorad/StillFace/issues/51)) ([7713747](https://github.com/fodorad/StillFace/commit/77137470471c3d358adabb67e1f9e046a455476a))
* **audio:** retry CrisperWhisper and MFA failures seen on the full database ([#54](https://github.com/fodorad/StillFace/issues/54)) ([a2515c8](https://github.com/fodorad/StillFace/commit/a2515c81798d09ee6e15d9fe5fdb4c1fe83c328f))
* **audio:** worker fixes found by the smoke test on real models ([#47](https://github.com/fodorad/StillFace/issues/47)) ([2dada48](https://github.com/fodorad/StillFace/commit/2dada485b1d56cd14f93a314350b6c58af68eedc))


### Documentation

* **readme:** analysis stack for touch, emotion and gaze ([#27](https://github.com/fodorad/StillFace/issues/27)) ([a1fe7d7](https://github.com/fodorad/StillFace/commit/a1fe7d75ce5c8e44d0e4868ffdbcbfd49353ad10))
* **readme:** audio pipeline overview and commands ([#46](https://github.com/fodorad/StillFace/issues/46)) ([5a14429](https://github.com/fodorad/StillFace/commit/5a14429b3b222201dedd7262a74dd04bd054f869))
* **readme:** baby vocal affect review results ([#23](https://github.com/fodorad/StillFace/issues/23)) ([848d70a](https://github.com/fodorad/StillFace/commit/848d70a8da29a429b5a1adb1518ccc44788ef841))

## [0.2.0](https://github.com/fodorad/StillFace/compare/stillface-v0.1.0...stillface-v0.2.0) (2026-10-28)


### Features

* **features:** per-camera overlay clips and stacked overlay grid ([bcefaf0](https://github.com/fodorad/StillFace/commit/bcefaf0170bb36261a1d4262c71067159da365a1))
* **features:** timeline schema, run record, config and atomic writes ([6e13ad6](https://github.com/fodorad/StillFace/commit/6e13ad616f27c1976bc73602746a221fbc09c2e4))
* **gaze:** exordium head tracker adapter and features heads pilot ([bd85b6e](https://github.com/fodorad/StillFace/commit/bd85b6e81b8fa5ff95b20858bd3972b69b2d9293))
* **gaze:** face-confirmed close-camera owners and face visibility flag ([13c8767](https://github.com/fodorad/StillFace/commit/13c8767c964292589b28ecab59ae8cc9b63343c5))
* **gaze:** gaze pipeline and features gaze CLI ([8e54938](https://github.com/fodorad/StillFace/commit/8e5493874e9d70b062fbe1716df8d9817ffc9348))
* **gaze:** head tracks and role assignment ([e66b513](https://github.com/fodorad/StillFace/commit/e66b51329d4b4a980ef7a6e70d2eb3c665dffe37))
* **gaze:** heatmap-to-class decision ([7b4e441](https://github.com/fodorad/StillFace/commit/7b4e441cf4f3c176b3a123c129e8d9f63d8cb202))
* **gaze:** reliable head detection for mother and baby on all cameras ([556b0f9](https://github.com/fodorad/StillFace/commit/556b0f92ce766e0a0caf047cf87d13c21d852ad8))
* **gaze:** unified two-detector head tracking with seating-based roles ([1476183](https://github.com/fodorad/StillFace/commit/147618365ffa43010acb2ece8a50107a3fdceb30))
* **paths:** read-only accessors for DB phase clips and gaze output paths ([c84973b](https://github.com/fodorad/StillFace/commit/c84973b459e3ee6a899d1969e2520ba7f051baaf))
* restructure into stillface package with data preparation, CLI, CI and docs ([b34fc16](https://github.com/fodorad/StillFace/commit/b34fc16eba0d1cd7a3a7c32921431638a24dec28))
* sync and cut videos ([0f01ccf](https://github.com/fodorad/StillFace/commit/0f01ccfed981282b36b2fcbf1b0e6bc7edbaa20e))


### Bug Fixes

* **gaze:** tolerate clips with fewer decodable frames than their metadata ([86de328](https://github.com/fodorad/StillFace/commit/86de3284c959dafc5c26190bbd72d6ee2b2aa971))


### Documentation

* **readme:** dataset statistics section ([9328206](https://github.com/fodorad/StillFace/commit/9328206dd1182e32965b62c77ff1a99c59978545))
* **readme:** probing results and failure lists ([#22](https://github.com/fodorad/StillFace/issues/22)) ([e6f22c0](https://github.com/fodorad/StillFace/commit/e6f22c0c0fd4d3821e39c5243946fc83b12420e7))
* widen logo and use experiment-focused tagline ([7a7ee2f](https://github.com/fodorad/StillFace/commit/7a7ee2fb0dfb4597e3ddb52f1ad74760290cf292))

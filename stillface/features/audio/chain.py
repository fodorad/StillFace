"""The whole audio chain, in dependency order, resumable.

Stages, each writing to disk before the next reads:

1. inventory, preparation;
2. first-pass models on the phase WAVs (VTC, Silero, pyannote speech, AST, LittleBeats,
   prosody);
3. second-pass models on VTC segments (speaker embeddings, addressee), then the roles;
4. models on role segments (eGeMAPS, audEERING, emotion2vec+);
5. fusion, gating, ASR on the gated mother track, forced alignment of its utterances, ASR
   flags;
6. affect lanes, Annie export, QC.

Inference stages keep up-to-date outputs, so a re-run only computes what is missing or stale;
the rule-based stages after it are cheap and always rerun, so a changed threshold reaches
every downstream table without touching the model outputs. Disabled and unavailable models
are skipped and recorded.
"""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from stillface.config import DatasetPaths
from stillface.constants import PHASES
from stillface.data_preparation.metadata import load_metadata
from stillface.features.audio.affect import run_affect
from stillface.features.audio.asr import builder_for, run_asr_flags
from stillface.features.audio.config import AudioConfig, ModelConfig
from stillface.features.audio.export import export_phase, session_qc, write_dataset_tables
from stillface.features.audio.fusion import run_fusion
from stillface.features.audio.gating import run_gating
from stillface.features.audio.infer import run_model
from stillface.features.audio.inventory import build_inventory, write_inventory
from stillface.features.audio.models import ModelAdapter
from stillface.features.audio.prepare import prepare_all
from stillface.features.audio.registry import build_adapter, env_python
from stillface.features.audio.roles import assign_roles, write_listening

logger = logging.getLogger(__name__)

AdapterFactory = Callable[[str, AudioConfig], ModelAdapter]
"""Builds the adapter of a model name (replaceable for tests)."""

INVENTORY_FILE: str = "inventory.csv"
"""Inventory table name in the dataset-level audio folder."""


def model_group(model: ModelConfig) -> int:
    """Chain position of a model.

    Args:
        model: Model settings.

    Returns:
        1 whole-phase models, 2 VTC-segment models, 3 role-segment models, 4 gated-track
        (ASR) models, 5 utterance (aligner) models.
    """
    if model.input == "wav":
        return 4 if model.options.get("track") else 1
    kind = model.segment_source.partition(":")[0]
    return {"vtc": 2, "roles": 3, "asr": 5}.get(kind, 3)


@dataclass
class ChainReport:
    """What a chain run did.

    Attributes:
        sessions: Sessions processed.
        models: Model name mapped to ``ran``, ``disabled``, ``not installed``, ``failed: ...``
            or ``partial: <n> failed``.
        stages: Stages completed, in order.
    """

    sessions: list[str] = field(default_factory=list)
    models: dict[str, str] = field(default_factory=dict)
    stages: list[str] = field(default_factory=list)


def default_factory(root: Path | None) -> AdapterFactory:
    """Adapter factory from the registry.

    Args:
        root: Folder of the model environments; from ``$STILLFACE_AUDIO_ENVS`` if None.

    Returns:
        The factory.
    """
    return lambda name, config: build_adapter(name, config, root)


def run_chain(
    paths: DatasetPaths,
    config: AudioConfig,
    sessions: Sequence[str] | None = None,
    factory: AdapterFactory | None = None,
    envs: Path | None = None,
    overwrite: bool = False,
    phases: Sequence[str] = PHASES,
) -> ChainReport:
    """Run every stage of the audio chain.

    Args:
        paths: Dataset locations.
        config: Audio configuration.
        sessions: Sessions to process; the in-scope sessions of the inventory if None.
        factory: Adapter factory; the registry with ``envs`` if None.
        envs: Folder of the model environments (used to detect missing environments).
        overwrite: Recompute up-to-date model outputs.
        phases: Phases to process.

    Returns:
        The report.
    """
    report = ChainReport()
    rows = build_inventory(paths, config, load_metadata(paths.metadata_path))
    write_inventory(rows, paths.audio_dir / INVENTORY_FILE)
    report.stages.append("inventory")
    selected = list(sessions) if sessions else sorted({r.session_id for r in rows if r.in_scope})
    report.sessions = selected
    prepare_all(paths, rows, config, set(selected), overwrite)
    report.stages.append("prepare")
    build = factory or default_factory(envs)
    by_group: dict[int, list[str]] = {}
    for name, model in config.models.items():
        by_group.setdefault(model_group(model), []).append(name)

    def infer(group: int) -> None:
        for name in by_group.get(group, []):
            model = config.models[name]
            if not model.enabled:
                report.models[name] = "disabled"
                continue
            if (
                model.env
                and factory is None
                and envs is not None
                and not env_python(envs, model.env).exists()
            ):
                report.models[name] = "not installed"
                continue
            try:
                adapter = build(name, config)
            except Exception as error:
                report.models[name] = f"failed: {error}"
                continue
            result = run_model(
                paths, adapter, model, selected, phases, builder_for(model, config), overwrite
            )
            report.models[name] = (
                f"partial: {len(result.failed)} failed" if result.failed else "ran"
            )
        report.stages.append(f"models:{group}")

    infer(1)
    infer(2)
    items = []
    for session in selected:
        items += assign_roles(paths, session, config, phases)
    write_listening(items, paths.audio_dir / "listening_roles.csv")
    report.stages.append("roles")
    infer(3)
    items = []
    for session in selected:
        items += run_fusion(paths, session, config, phases)
        run_gating(paths, session, config, phases)
    write_listening(items, paths.audio_dir / "listening_fusion.csv")
    report.stages += ["fusion", "gating"]
    infer(4)
    infer(5)
    records = []
    for session in selected:
        run_asr_flags(paths, session, config, phases)
        run_affect(paths, session, config, phases)
        for phase in phases:
            export_phase(paths, session, phase)
        records.append(session_qc(paths, session, config, phases))
    write_dataset_tables(paths, records)
    report.stages += ["asr", "affect", "export", "qc"]
    logger.info("Audio chain on %d sessions: %s", len(selected), report.models)
    return report

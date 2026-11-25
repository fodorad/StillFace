"""Decision tables: every rule and threshold that produced the audio outputs.

The tables are generated from the loaded configuration, never written by hand, so the
report always shows the rules that were actually applied. Every ``rule_id`` that appears in
a band or an event is listed here.
"""

import html
from dataclasses import dataclass

from stillface.features.audio.config import AudioConfig
from stillface.features.audio.fusion import RULES as FUSION_RULES
from stillface.features.audio.roles import RULES as ROLE_RULES


@dataclass(frozen=True)
class DecisionRow:
    """One rule.

    Attributes:
        rule_id: Identifier used in the outputs.
        description: What the rule decides.
        inputs: Models or classes it reads.
        threshold: Threshold value as text, empty if none.
        origin: Origin of the threshold, empty if none.
    """

    rule_id: str
    description: str
    inputs: str = ""
    threshold: str = ""
    origin: str = ""


THRESHOLD_OF: dict[str, str] = {
    "VAD-SILERO": "silero",
    "VAD-PYANNOTE": "pyannote",
    "GONG": "gong",
    "CONTEXT": "context",
    "MOTHER-TYPE": "ast_type",
    "BABY-TYPE-VOTE": "ast_type",
}
"""Rule id mapped to the name of the fusion threshold it applies."""


def decision_rows(config: AudioConfig) -> list[DecisionRow]:
    """All rules with the inputs and thresholds of a configuration.

    Args:
        config: Audio configuration.

    Returns:
        Role rules, fusion rules, then one row per configured vocalisation type.
    """
    fc, rc = config.fusion, config.roles
    inputs = {
        "ROLE-BABY": f"{rc.vtc_model}: {', '.join(rc.baby_labels)}",
        "ROLE-MAL": f"{rc.vtc_model}: MAL",
        "ROLE-FEM-MOTHER": f"{rc.embedding_model} cosine to the enrolled voice",
        "ROLE-FEM-OTHER": f"{rc.embedding_model} cosine to the enrolled voice",
        "ROLE-FEM-UNCERTAIN": f"{rc.embedding_model} cosine to the enrolled voice",
        "VAD-FUSED": f"{rc.vtc_model}: KCHI, OCH, MAL, FEM",
        "GONG": "ast: " + ", ".join(fc.gong.classes) + f" (edges {fc.gong.edge_s} s)",
        "CONTEXT": "ast: " + ", ".join(fc.context.classes),
    }
    special = {"ROLE-FEM-MOTHER": (rc.tau_high, "config"), "ROLE-FEM-OTHER": (rc.tau_low, "config")}
    rows = []
    for rule_id, description in [*ROLE_RULES.items(), *FUSION_RULES.items()]:
        value, origin = "", ""
        if rule_id in THRESHOLD_OF and THRESHOLD_OF[rule_id] in fc.thresholds:
            t = fc.thresholds[THRESHOLD_OF[rule_id]]
            value, origin = f"{t.value:g}", t.origin
        elif rule_id in special:
            value, origin = f"{special[rule_id][0]:g}", special[rule_id][1]
        rows.append(DecisionRow(rule_id, description, inputs.get(rule_id, ""), value, origin))
    for kind, table in (("baby", fc.baby_types), ("mother", fc.mother_types)):
        for name, rule in table.items():
            sources = []
            if rule.ast:
                sources.append("ast: " + ", ".join(rule.ast))
            if rule.littlebeats:
                sources.append("littlebeats: " + ", ".join(rule.littlebeats))
            rows.append(
                DecisionRow(
                    f"TYPE-{kind.upper()}-{name.upper()}",
                    f"{kind} type '{name}'",
                    "; ".join(sources),
                )
            )
    return rows


COLUMNS: tuple[str, ...] = ("rule", "decides", "inputs", "threshold", "origin")
"""Table column titles."""


def to_markdown(rows: list[DecisionRow]) -> str:
    """Render rows as a Markdown table.

    Args:
        rows: Decision rows.

    Returns:
        The table text.
    """

    def cell(text: str) -> str:
        return text.replace("|", "\\\\|")

    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    for r in rows:
        values = (r.rule_id, r.description, r.inputs, r.threshold, r.origin)
        lines.append("| " + " | ".join(cell(v) for v in values) + " |")
    return "\n".join(lines) + "\n"


def to_html(rows: list[DecisionRow]) -> str:
    """Render rows as an HTML table fragment.

    Args:
        rows: Decision rows.

    Returns:
        ``<table>`` markup with escaped cells.
    """
    head = "".join(f"<th>{c}</th>" for c in COLUMNS)
    body = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(v)}</td>"
            for v in (r.rule_id, r.description, r.inputs, r.threshold, r.origin)
        )
        + "</tr>"
        for r in rows
    )
    return f'<table class="decisions"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>\n'

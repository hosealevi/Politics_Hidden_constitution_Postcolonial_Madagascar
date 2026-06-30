#!/usr/bin/env python3
"""Create publication-oriented S2.1 tables and PDF figures."""

from __future__ import annotations

import json
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle

matplotlib.use("Agg")

COLORS = {
    "ink": "#263238",
    "muted": "#607D8B",
    "grid": "#D9E2E7",
    "blue": "#2F6F9F",
    "green": "#4F8A6A",
    "coral": "#C96B4B",
    "gold": "#C4A24A",
    "violet": "#7464A1",
    "teal": "#3C8D8A",
    "sand": "#D8C99B",
    "pale": "#F5F7F8",
}

CLAIM_TYPE_ORDER = [
    "legal_constitutional",
    "electoral",
    "popular_or_street",
    "democratic_restoration",
    "stability_order",
    "moral_reconciliation",
    "international_recognition",
    "revolutionary_or_refoundation",
]

FORMALIZATION_MODE_ORDER = [
    "constitutional court / HCC decision",
    "constitution or amendment",
    "transition charter / convention",
    "mediated roadmap",
    "election or referendum",
    "ordinance / executive act",
    "agreement or recognition",
    "appointment / office allocation",
]

EXTERNAL_ACTOR_ORDER = ["SADC", "AU/OAU", "OIF", "UN", "France", "Senegal/Dakar", "FFKM", "HCC"]
EXTERNAL_CRITERIA_ORDER = [
    "mediation / roadmap",
    "elections",
    "inclusiveness / consensus",
    "anti-coup / legality",
    "stability / order",
    "recognition / sanctions",
    "sovereignty tension",
]

DELEGITIMATION_FRAME_ORDER = [
    "authoritarianism",
    "corruption",
    "repression",
    "illegality",
    "institutional coup",
    "electoral fraud",
    "foreign dependence",
    "incapacity / state failure",
]


def _load_s21_output(repo_root: Path) -> pd.DataFrame:
    path = repo_root / "outputs" / "evidence_synthesis" / "S21_all_evidence_synthesis_v1.csv"
    return pd.read_csv(path, dtype=str).fillna("")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _records(output: pd.DataFrame, question_id: str) -> pd.DataFrame:
    row = output.loc[output["question_id"].eq(question_id)].iloc[0]
    structured = json.loads(row["structured_evidence"])
    df = pd.DataFrame(structured.get("records", [])).fillna("")
    df.insert(0, "question_id", question_id)
    return df


def _clean_cell(value: object) -> str:
    text = str(value).strip()
    if text.lower() in {"", "nan", "0", "0.0", "1", "1.0", "present", "true", "yes"}:
        return ""
    parts = []
    for part in text.replace("|", ";").split(";"):
        cleaned = part.strip()
        if cleaned.lower() in {"", "nan", "0", "0.0", "1", "1.0", "present", "true", "yes"}:
            continue
        parts.append(cleaned)
    return "; ".join(dict.fromkeys(parts))


def _present(value: object) -> bool:
    return bool(_clean_cell(value))


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.map(_clean_cell)


def _split_values(value: object) -> list[str]:
    text = _clean_cell(value)
    if not text:
        return []
    return [part.strip() for part in text.replace("|", ";").split(";") if part.strip()]


def _wrap(text: object, width: int = 28) -> str:
    cleaned = _clean_cell(text)
    if not cleaned:
        return ""
    return "\n".join(textwrap.wrap(cleaned, width=width, break_long_words=False))


def _short(text: object, length: int = 82) -> str:
    cleaned = " ".join(_clean_cell(text).split())
    if len(cleaned) <= length:
        return cleaned
    return cleaned[: length - 3].rstrip() + "..."


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _clean_df(df).to_csv(path, index=False)


def _save_pdf(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight", metadata={"Creator": "Codex evidence synthesis"})
    plt.close(fig)


def _new_page(figsize: tuple[float, float] = (11.7, 8.3)) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return fig, ax


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.tick_params(colors=COLORS["ink"], labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(COLORS["grid"])


def _add_count_colorbar(fig: plt.Figure, ax: plt.Axes, image: matplotlib.image.AxesImage, label: str, max_value: float) -> None:
    cbar = fig.colorbar(image, ax=ax, fraction=0.034, pad=0.022)
    cbar.set_label(label, fontsize=8.6, color=COLORS["ink"])
    cbar.ax.tick_params(labelsize=8, colors=COLORS["ink"])
    cbar.locator = matplotlib.ticker.MaxNLocator(integer=True, nbins=min(int(max(max_value, 1)) + 1, 6))
    cbar.update_ticks()


def _count_cmap(name: str) -> matplotlib.colors.LinearSegmentedColormap:
    return matplotlib.colors.LinearSegmentedColormap.from_list(name, ["#FFFFFF", "#CFE3DA", COLORS["green"], COLORS["blue"]])


def _crisis_sort_key(crisis_id: str) -> tuple[int, str]:
    digits = "".join(ch for ch in str(crisis_id) if ch.isdigit())
    return (int(digits[:4]) if digits else 9999, str(crisis_id))


def _crises(records: pd.DataFrame) -> list[str]:
    return sorted(records["crisis_id"].dropna().unique().tolist(), key=_crisis_sort_key)


def _claim_tokens(row: pd.Series) -> list[str]:
    values = _split_values(row.get("legitimacy_types", ""))
    return [token for token in CLAIM_TYPE_ORDER if token in values]


def _formalization_modes(row: pd.Series) -> list[str]:
    text = " ".join(
        [
            _clean_cell(row.get("formalization_path", "")),
            _clean_cell(row.get("settlement_context", "")),
            _clean_cell(row.get("constitutional_context", "")),
        ]
    ).lower()
    modes = []
    tests = [
        ("constitutional court / HCC decision", ["hcc", "constitutional court", "cour constitutionnelle", "court-mediated"]),
        ("constitution or amendment", ["constitution", "constitutional amendment", "formal_regular", "formal_or_regular_revision"]),
        ("transition charter / convention", ["charter", "convention", "transition_convention"]),
        ("mediated roadmap", ["roadmap", "feuille de route", "mediated_roadmap"]),
        ("election or referendum", ["election", "referendum", "electoral"]),
        ("ordinance / executive act", ["ordinance", "executive act", "ordonnance"]),
        ("agreement or recognition", ["agreement", "recognition", "signed_or_mediated_agreement", "accord"]),
        ("appointment / office allocation", ["appoint", "office allocation", "prime minister", "government"]),
    ]
    for label, needles in tests:
        if any(needle in text for needle in needles):
            modes.append(label)
    return modes


def _external_actor_tokens(row: pd.Series) -> list[str]:
    text = " ".join(
        [
            _clean_cell(row.get("international_criteria", "")),
            _clean_cell(row.get("bargain_actor_context", "")),
            _clean_cell(row.get("settlement_context", "")),
            _clean_cell(row.get("source_ids", "")),
        ]
    ).lower()
    tests = {
        "SADC": ["sadc"],
        "AU/OAU": ["african union", " au", "oau", "inst_au"],
        "OIF": ["oif", "francophonie"],
        "UN": [" un", "united nations"],
        "France": ["france", "french"],
        "Senegal/Dakar": ["senegal", "dakar"],
        "FFKM": ["ffkm", "churches"],
        "HCC": ["hcc", "haute cour constitutionnelle"],
    }
    return [label for label in EXTERNAL_ACTOR_ORDER if any(needle in text for needle in tests[label])]


def _external_criteria_tokens(row: pd.Series) -> list[str]:
    text = " ".join(
        [
            _clean_cell(row.get("international_criteria", "")),
            _clean_cell(row.get("settlement_context", "")),
            _clean_cell(row.get("formalization_path", "")),
        ]
    ).lower()
    tests = {
        "mediation / roadmap": ["mediat", "roadmap", "dakar", "guarantor", "oversight"],
        "elections": ["election", "electoral", "referendum"],
        "inclusiveness / consensus": ["inclusive", "consensus", "consultation", "power-sharing"],
        "anti-coup / legality": ["anti-coup", "coup", "constitutional", "legal", "hcc"],
        "stability / order": ["stability", "order", "security"],
        "recognition / sanctions": ["recognition", "sanction", "accept"],
        "sovereignty tension": ["sovereignty", "external pressure", "foreign"],
    }
    return [label for label in EXTERNAL_CRITERIA_ORDER if any(needle in text for needle in tests[label])]


def _delegitimation_frames(row: pd.Series) -> list[str]:
    text = " ".join([_clean_cell(row.get("delegitimation_claim", "")), _clean_cell(row.get("claim_summary", "")), _clean_cell(row.get("keywords", ""))]).lower()
    tests = {
        "authoritarianism": ["authoritarian", "dictator"],
        "corruption": ["corruption"],
        "repression": ["repression", "repressive"],
        "illegality": ["illegal", "illegality", "legal"],
        "institutional coup": ["institutional coup", "coup"],
        "electoral fraud": ["fraud", "recount", "electoral", "election"],
        "foreign dependence": ["foreign", "dependency", "neocolonial"],
        "incapacity / state failure": ["incapacity", "state failure", "abandonment", "unable"],
    }
    return [label for label in DELEGITIMATION_FRAME_ORDER if any(needle in text for needle in tests[label])]


def _delegitimation_target(row: pd.Series) -> str:
    claim = _clean_cell(row.get("delegitimation_claim", ""))
    for target in [
        "Philibert Tsiranana",
        "Didier Ratsiraka",
        "Albert Zafy",
        "Marc Ravalomanana",
        "Andry Rajoelina",
        "Michaël Randrianirina",
    ]:
        if target.lower() in claim.lower():
            return target
    parts = _split_values(claim)
    return parts[0] if parts else "target not explicit"


def _base_cols(extra: list[str]) -> list[str]:
    return [
        "crisis_id",
        "crisis_name",
        "discourse_record_id",
        "source_date",
        "actor_id",
        "actor_name",
        "claim_summary",
        "textual_evidence",
        *extra,
        "settlement_context",
        "constitutional_context",
        "source_ids",
    ]


def _select(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[[col for col in cols if col in df.columns]].copy()


def _dedupe_records(*dfs: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat(dfs, ignore_index=True, sort=False).fillna("")
    combined = combined.loc[combined["discourse_record_id"].astype(str).str.strip().ne("")]
    return combined.drop_duplicates(["crisis_id", "discourse_record_id", "actor_id"], keep="first")


def build_tables(output: pd.DataFrame, table_dir: Path) -> dict[str, pd.DataFrame]:
    iqa = _records(output, "QE-S21-IQA")
    iqb = _records(output, "QE-S21-IQB")
    iqc = _records(output, "QE-S21-IQC")
    iqd = _records(output, "QE-S21-IQD")
    iqe = _records(output, "QE-S21-IQE")
    iqf = _records(output, "QE-S21-IQF")
    iqg = _records(output, "QE-S21-IQG")
    iqh = _records(output, "QE-S21-IQH")

    tables = {
        "T_S21_1_legitimacy_claim_typology.csv": _select(iqa, _base_cols(["legitimacy_types", "keywords"])),
        "T_S21_2_people_invocation.csv": _select(iqb, _base_cols(["people_invocation", "keywords"])),
        "T_S21_3_democracy_legality_contrast.csv": _select(iqc, _base_cols(["democracy_legality_frame", "legality_gap"])),
        "T_S21_4_stability_reconciliation_settlements.csv": _select(iqd, _base_cols(["stability_reconciliation_frame"])),
        "T_S21_5_international_criteria.csv": _select(iqe, _base_cols(["international_criteria", "bargain_actor_context"])),
        "T_S21_6_delegitimation_claim_targets.csv": _select(iqf, _base_cols(["delegitimation_claim", "elite_context", "keywords"])),
        "T_S21_7_legality_gap_cases.csv": _select(iqg, _base_cols(["legality_gap", "democracy_legality_frame", "formalization_path"])),
        "T_S21_8_formalization_sequences.csv": _select(iqh, _base_cols(["legitimacy_types", "formalization_path"])),
    }

    index = _dedupe_records(iqa, iqb, iqc, iqd, iqe, iqf, iqg, iqh)
    index["claim_type_tokens"] = index.apply(lambda row: "; ".join(_claim_tokens(row)), axis=1)
    index["formalization_modes"] = index.apply(lambda row: "; ".join(_formalization_modes(row)), axis=1)
    index["external_actors"] = index.apply(lambda row: "; ".join(_external_actor_tokens(row)), axis=1)
    index["delegitimation_target"] = index.apply(_delegitimation_target, axis=1)
    tables["T_S21_9_discourse_source_anchor_index.csv"] = _select(
        index,
        _base_cols(
            [
                "claim_type_tokens",
                "formalization_modes",
                "external_actors",
                "delegitimation_target",
                "people_invocation",
                "democracy_legality_frame",
                "legality_gap",
            ]
        ),
    )

    for name, df in tables.items():
        _save_csv(df, table_dir / name)
    return tables


def figure_claim_type_matrix(records: pd.DataFrame, path: Path) -> None:
    df = _clean_df(records.copy())
    crises = _crises(df)
    matrix = np.zeros((len(crises), len(CLAIM_TYPE_ORDER)))
    for _, row in df.drop_duplicates(["crisis_id", "discourse_record_id"]).iterrows():
        for claim_type in _claim_tokens(row):
            matrix[crises.index(row["crisis_id"]), CLAIM_TYPE_ORDER.index(claim_type)] += 1

    fig, ax = plt.subplots(figsize=(12.2, max(5.8, len(crises) * 0.42)))
    cmap = _count_cmap("s21_claims")
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=max(matrix.max(), 1))
    _add_count_colorbar(fig, ax, image, "Distinct discourse records", matrix.max())
    ax.set_yticks(np.arange(len(crises)))
    ax.set_yticklabels(crises, fontsize=8.8)
    ax.set_xticks(np.arange(len(CLAIM_TYPE_ORDER)))
    ax.set_xticklabels([_wrap(label.replace("_", " "), 16) for label in CLAIM_TYPE_ORDER], rotation=35, ha="right", fontsize=8.1)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(CLAIM_TYPE_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(crises), 1), minor=True)
    ax.grid(which="minor", color=COLORS["grid"], lw=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title("S2.1 Legitimacy Claim Types by Crisis", loc="left", fontsize=15, color=COLORS["ink"], weight="bold")
    ax.text(0, -0.27, "Shading counts distinct discourse records, not the intensity or truth of the claim.", transform=ax.transAxes, fontsize=9, color=COLORS["muted"])
    _save_pdf(fig, path)


def figure_claim_to_formalization_flow(records: pd.DataFrame, path: Path) -> None:
    df = _clean_df(records.copy())
    edges = Counter()
    examples: dict[tuple[str, str], list[str]] = defaultdict(list)
    for _, row in df.drop_duplicates(["crisis_id", "discourse_record_id"]).iterrows():
        modes = _formalization_modes(row) or ["no coded formalization"]
        for claim_type in _claim_tokens(row) or ["claim type not explicit"]:
            for mode in modes:
                key = (claim_type, mode)
                edges[key] += 1
                examples[key].append(f"{row['crisis_id']}/{_short(row.get('actor_name', ''), 32)}")

    display_edges = Counter(dict(edges.most_common(14)))
    left_nodes = [node for node in CLAIM_TYPE_ORDER if any(edge[0] == node for edge in display_edges)]
    right_nodes = [node for node in FORMALIZATION_MODE_ORDER if any(edge[1] == node for edge in display_edges)]
    if any(edge[1] == "no coded formalization" for edge in display_edges):
        right_nodes.append("no coded formalization")
    palette = [COLORS["blue"], COLORS["green"], COLORS["coral"], COLORS["violet"], COLORS["gold"], COLORS["teal"], "#8E6E53", "#78909C"]
    max_count = max(edges.values()) if edges else 1

    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path, metadata={"Creator": "Codex evidence synthesis"}) as pdf:
        fig, ax = _new_page((13.0, 8.2))
        ax.text(0.03, 0.965, "S2.1 From Legitimacy Claim to Formalization Mode", fontsize=16, color=COLORS["ink"], weight="bold", va="top")
        ax.text(0.03, 0.92, "Main map shows the 14 most repeated discourse-to-institution pathways. Edge thickness marks repetition; examples are on page 2.", fontsize=9.5, color=COLORS["muted"])
        ax.text(0.07, 0.855, "Claim type", fontsize=10.8, color=COLORS["ink"], weight="bold")
        ax.text(0.67, 0.855, "Formalization mode", fontsize=10.8, color=COLORS["ink"], weight="bold")
        left_y = {node: 0.78 - i * (0.66 / max(len(left_nodes) - 1, 1)) for i, node in enumerate(left_nodes)}
        right_y = {node: 0.78 - i * (0.66 / max(len(right_nodes) - 1, 1)) for i, node in enumerate(right_nodes)}

        for idx, ((claim_type, mode), count) in enumerate(display_edges.items()):
            if claim_type not in left_y or mode not in right_y:
                continue
            y1, y2 = left_y[claim_type], right_y[mode]
            verts = [(0.34, y1), (0.48, y1), (0.54, y2), (0.66, y2)]
            path_obj = MplPath(verts, [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
            ax.add_patch(PathPatch(path_obj, facecolor="none", edgecolor=palette[idx % len(palette)], lw=0.8 + 4.0 * count / max_count, alpha=0.45, zorder=1))

        for node, yy in left_y.items():
            ax.add_patch(Rectangle((0.04, yy - 0.030), 0.30, 0.060, facecolor=COLORS["pale"], edgecolor=COLORS["grid"], lw=1.1, zorder=3))
            ax.text(0.055, yy, _wrap(node.replace("_", " "), 27), va="center", fontsize=8.5, color=COLORS["ink"], zorder=4)
        for node, yy in right_y.items():
            ax.add_patch(Rectangle((0.66, yy - 0.030), 0.31, 0.060, facecolor="#F7FAF8", edgecolor=COLORS["grid"], lw=1.1, zorder=3))
            ax.text(0.675, yy, _wrap(node, 30), va="center", fontsize=8.5, color=COLORS["ink"], zorder=4)
        ax.text(0.04, 0.045, "Reading note: the figure traces coded claims into institutional forms; it does not assume formalization equals acceptance.", fontsize=8.8, color=COLORS["muted"])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _new_page((13.0, 8.2))
        ax.text(0.04, 0.96, "S2.1 Flow Figure - Claim-to-Formalization Examples", fontsize=15, color=COLORS["ink"], weight="bold", va="top")
        ax.text(0.04, 0.915, "Examples preserve crisis and actor anchors. Full discourse/source IDs are in T_S21_8 and T_S21_9.", fontsize=9.5, color=COLORS["muted"])
        y = 0.84
        for idx, ((claim_type, mode), count) in enumerate(edges.most_common(12)):
            color = palette[idx % len(palette)]
            ax.add_patch(Rectangle((0.045, y - 0.018), 0.012, 0.036, facecolor=color, edgecolor=color))
            ax.text(0.07, y + 0.012, f"{claim_type.replace('_', ' ')} -> {mode}", fontsize=9.1, color=COLORS["ink"], weight="bold", va="center")
            example_text = _wrap("Examples: " + "; ".join(dict.fromkeys(examples[(claim_type, mode)][:6])), 136)
            ax.text(0.07, y - 0.020, example_text, fontsize=8.0, color=COLORS["muted"], va="top")
            ax.text(0.92, y, f"records: {count}", fontsize=8.5, color=COLORS["muted"], va="center", ha="right")
            y -= 0.071 + 0.018 * example_text.count("\n")
            if y < 0.08:
                break
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def figure_claim_to_formalization_all_edges_matrix(records: pd.DataFrame, path: Path) -> None:
    df = _clean_df(records.copy())
    matrix = np.zeros((len(CLAIM_TYPE_ORDER), len(FORMALIZATION_MODE_ORDER)))
    examples = defaultdict(list)
    for _, row in df.drop_duplicates(["crisis_id", "discourse_record_id"]).iterrows():
        modes = _formalization_modes(row)
        for claim_type in _claim_tokens(row):
            for mode in modes:
                i = CLAIM_TYPE_ORDER.index(claim_type)
                j = FORMALIZATION_MODE_ORDER.index(mode)
                matrix[i, j] += 1
                examples[(claim_type, mode)].append(f"{row['crisis_id']}/{_short(row.get('actor_name', ''), 34)}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path, metadata={"Creator": "Codex evidence synthesis"}) as pdf:
        fig, ax = plt.subplots(figsize=(12.2, 7.6))
        cmap = _count_cmap("s21_all_edges")
        image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=max(matrix.max(), 1))
        _add_count_colorbar(fig, ax, image, "Distinct discourse records", matrix.max())
        ax.set_yticks(np.arange(len(CLAIM_TYPE_ORDER)))
        ax.set_yticklabels([_wrap(label.replace("_", " "), 24) for label in CLAIM_TYPE_ORDER], fontsize=8.7)
        ax.set_xticks(np.arange(len(FORMALIZATION_MODE_ORDER)))
        ax.set_xticklabels([_wrap(mode, 18) for mode in FORMALIZATION_MODE_ORDER], rotation=35, ha="right", fontsize=8.2)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(FORMALIZATION_MODE_ORDER), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(CLAIM_TYPE_ORDER), 1), minor=True)
        ax.grid(which="minor", color=COLORS["grid"], lw=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_title("S2.1_B Exhaustive Claim-to-Formalization Edge Matrix", loc="left", fontsize=15, color=COLORS["ink"], weight="bold")
        ax.text(0, -0.25, "Every nonzero cell is a coded edge retained in the full data; the readable flow figure shows only the strongest repeated pathways.", transform=ax.transAxes, fontsize=9, color=COLORS["muted"])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _new_page((12.2, 8.3))
        ax.text(0.04, 0.96, "S2.1_B Exhaustive Edge Examples", fontsize=15, color=COLORS["ink"], weight="bold", va="top")
        ax.text(0.04, 0.915, "Nonzero cells from the matrix with crisis/actor examples. Full source IDs remain in T_S21_8 and T_S21_9.", fontsize=9.4, color=COLORS["muted"])
        y = 0.84
        for (claim_type, mode), vals in sorted(examples.items(), key=lambda item: (-len(item[1]), item[0][0], item[0][1])):
            ax.text(0.055, y, f"{claim_type.replace('_', ' ')} -> {mode}", fontsize=8.8, color=COLORS["ink"], weight="bold", va="center")
            example_text = _wrap("Examples: " + "; ".join(dict.fromkeys(vals[:8])), 134)
            ax.text(0.055, y - 0.024, example_text, fontsize=7.6, color=COLORS["muted"], va="top")
            y -= 0.066 + 0.017 * example_text.count("\n")
            if y < 0.08:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                fig, ax = _new_page((12.2, 8.3))
                ax.text(0.04, 0.96, "S2.1_B Exhaustive Edge Examples Continued", fontsize=15, color=COLORS["ink"], weight="bold", va="top")
                y = 0.88
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

def figure_democracy_legality_matrix(records: pd.DataFrame, path: Path) -> None:
    df = _clean_df(records.copy())
    crises = _crises(df)
    labels = ["formal regular", "regular revision", "transition / exception", "court adjustment", "legality gap"]
    matrix = np.zeros((len(crises), len(labels)))
    for _, row in df.drop_duplicates(["crisis_id", "discourse_record_id"]).iterrows():
        text = " ".join([row.get("democracy_legality_frame", ""), row.get("constitutional_context", ""), row.get("legality_gap", "")]).lower()
        hits = {
            "formal regular": "formal_regular" in text,
            "regular revision": "formal_or_regular_revision" in text,
            "transition / exception": "formalized_transition_or_exception" in text,
            "court adjustment": "court-mediated" in text or "hcc" in text,
            "legality gap": _present(row.get("legality_gap", "")),
        }
        for label, hit in hits.items():
            if hit:
                matrix[crises.index(row["crisis_id"]), labels.index(label)] += 1

    fig, ax = plt.subplots(figsize=(10.8, max(5.5, len(crises) * 0.42)))
    cmap = _count_cmap("s21_demleg")
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=max(matrix.max(), 1))
    _add_count_colorbar(fig, ax, image, "Distinct discourse records", matrix.max())
    ax.set_yticks(np.arange(len(crises)))
    ax.set_yticklabels(crises, fontsize=8.8)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels([_wrap(label, 18) for label in labels], rotation=30, ha="right", fontsize=8.6)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(crises), 1), minor=True)
    ax.grid(which="minor", color=COLORS["grid"], lw=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title("S2.1 Democracy-Legality Tension by Crisis", loc="left", fontsize=15, color=COLORS["ink"], weight="bold")
    ax.text(0, -0.20, "Cells mark coded discourse/legal-context anchors; the colorbar is a record count, not a judgment of democratic quality.", transform=ax.transAxes, fontsize=9, color=COLORS["muted"])
    _save_pdf(fig, path)


def figure_stability_cards(records: pd.DataFrame, path: Path, max_records: int | None = 12, title_suffix: str = "") -> None:
    df = _clean_df(records.copy())
    df = df.loc[df["stability_reconciliation_frame"].map(_present) | df["settlement_context"].map(_present)].copy()
    if max_records is not None:
        df = df.head(max_records)
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path, metadata={"Creator": "Codex evidence synthesis"}) as pdf:
        for page, start in enumerate(range(0, len(df), 6), start=1):
            fig, ax = _new_page((11.7, 8.3))
            ax.text(0.04, 0.96, f"S2.1{title_suffix} Stability, Reconciliation, and Settlement Frames - Page {page}", fontsize=15, color=COLORS["ink"], weight="bold", va="top")
            ax.text(0.04, 0.915, "Evidence cards retain actor, crisis, settlement context, and source anchors.", fontsize=9.5, color=COLORS["muted"])
            card_w = 0.42
            card_h = 0.22
            xs = [0.05, 0.53]
            y = 0.82
            for idx, (_, row) in enumerate(df.iloc[start : start + 6].iterrows()):
                col = idx % 2
                if idx and col == 0:
                    y -= card_h + 0.045
                x = xs[col]
                ax.add_patch(Rectangle((x, y - card_h), card_w, card_h, facecolor="#FFFFFF", edgecolor=COLORS["grid"], lw=1.1))
                ax.add_patch(Rectangle((x, y - 0.030), card_w, 0.030, facecolor=COLORS["pale"], edgecolor=COLORS["grid"], lw=0.6))
                title = f"{row['crisis_id']} / {_short(row.get('actor_name', ''), 36)}"
                ax.text(x + 0.012, y - 0.015, title, fontsize=8.6, color=COLORS["ink"], weight="bold", va="center")
                lines = [
                    "Frame: " + _short(row.get("stability_reconciliation_frame", ""), 88),
                    "Settlement: " + _short(row.get("settlement_context", ""), 86),
                    "Sources: " + _short(row.get("source_ids", ""), 86),
                ]
                for line_i, line in enumerate(lines):
                    ax.text(x + 0.012, y - 0.062 - line_i * 0.049, _wrap(line, 58), fontsize=7.0, color=COLORS["muted"], va="top")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def figure_international_matrix(records: pd.DataFrame, path: Path) -> None:
    df = _clean_df(records.copy())
    crises = _crises(df)
    matrix = np.zeros((len(crises), len(EXTERNAL_ACTOR_ORDER)))
    criteria_examples = defaultdict(list)
    for _, row in df.drop_duplicates(["crisis_id", "discourse_record_id"]).iterrows():
        for actor in _external_actor_tokens(row):
            matrix[crises.index(row["crisis_id"]), EXTERNAL_ACTOR_ORDER.index(actor)] += 1
            criteria = ", ".join(_external_criteria_tokens(row)) or _short(row.get("international_criteria", ""), 54)
            criteria_examples[actor].append(f"{row['crisis_id']}: {criteria}; src={_short(row.get('source_ids', ''), 52)}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path, metadata={"Creator": "Codex evidence synthesis"}) as pdf:
        fig, ax = plt.subplots(figsize=(11.2, max(5.8, len(crises) * 0.42)))
        cmap = _count_cmap("s21_ext")
        image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=max(matrix.max(), 1))
        _add_count_colorbar(fig, ax, image, "Distinct discourse records", matrix.max())
        ax.set_yticks(np.arange(len(crises)))
        ax.set_yticklabels(crises, fontsize=8.8)
        ax.set_xticks(np.arange(len(EXTERNAL_ACTOR_ORDER)))
        ax.set_xticklabels(EXTERNAL_ACTOR_ORDER, rotation=35, ha="right", fontsize=8.5)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(EXTERNAL_ACTOR_ORDER), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(crises), 1), minor=True)
        ax.grid(which="minor", color=COLORS["grid"], lw=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_title("S2.1 International Legitimacy Actors by Crisis", loc="left", fontsize=15, color=COLORS["ink"], weight="bold")
        ax.text(0, -0.19, "Shading marks discourse records mentioning each external or quasi-external legitimacy actor. Criteria examples are on page 2.", transform=ax.transAxes, fontsize=9, color=COLORS["muted"])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _new_page((11.7, 8.3))
        ax.text(0.04, 0.96, "S2.1 International Criteria Examples", fontsize=15, color=COLORS["ink"], weight="bold", va="top")
        ax.text(0.04, 0.915, "Examples list the coded criterion and source anchors behind each actor category.", fontsize=9.5, color=COLORS["muted"])
        y = 0.84
        palette = [COLORS["blue"], COLORS["green"], COLORS["coral"], COLORS["violet"], COLORS["gold"], COLORS["teal"], "#8E6E53", "#78909C"]
        for idx, actor in enumerate(EXTERNAL_ACTOR_ORDER):
            vals = list(dict.fromkeys(criteria_examples.get(actor, [])))[:5]
            if not vals:
                continue
            ax.add_patch(Rectangle((0.045, y - 0.018), 0.012, 0.036, facecolor=palette[idx % len(palette)], edgecolor=palette[idx % len(palette)]))
            ax.text(0.07, y + 0.012, actor, fontsize=9.5, color=COLORS["ink"], weight="bold", va="center")
            example_text = _wrap("Examples: " + "; ".join(vals), 132)
            ax.text(0.07, y - 0.020, example_text, fontsize=7.8, color=COLORS["muted"], va="top")
            y -= 0.084 + 0.018 * example_text.count("\n")
            if y < 0.08:
                break
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def figure_delegitimation_matrix(records: pd.DataFrame, path: Path) -> None:
    df = _clean_df(records.copy())
    df["target"] = df.apply(_delegitimation_target, axis=1)
    targets = [target for target in ["Philibert Tsiranana", "Didier Ratsiraka", "Albert Zafy", "Marc Ravalomanana", "Andry Rajoelina", "Michaël Randrianirina", "target not explicit"] if target in set(df["target"])]
    matrix = np.zeros((len(targets), len(DELEGITIMATION_FRAME_ORDER)))
    examples = defaultdict(list)
    for _, row in df.iterrows():
        for frame in _delegitimation_frames(row):
            i = targets.index(row["target"])
            j = DELEGITIMATION_FRAME_ORDER.index(frame)
            matrix[i, j] += 1
            examples[(row["target"], frame)].append(f"{row['crisis_id']}/{_short(row.get('actor_name', ''), 28)}: {_short(row.get('claim_summary', ''), 70)}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path, metadata={"Creator": "Codex evidence synthesis"}) as pdf:
        fig, ax = plt.subplots(figsize=(11.5, max(4.8, len(targets) * 0.70)))
        cmap = _count_cmap("s21_deleg")
        image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=max(matrix.max(), 1))
        _add_count_colorbar(fig, ax, image, "Distinct claim records", matrix.max())
        ax.set_yticks(np.arange(len(targets)))
        ax.set_yticklabels([_wrap(target, 22) for target in targets], fontsize=8.7)
        ax.set_xticks(np.arange(len(DELEGITIMATION_FRAME_ORDER)))
        ax.set_xticklabels([_wrap(frame, 16) for frame in DELEGITIMATION_FRAME_ORDER], rotation=30, ha="right", fontsize=8.3)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(DELEGITIMATION_FRAME_ORDER), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(targets), 1), minor=True)
        ax.grid(which="minor", color=COLORS["grid"], lw=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_title("S2.1 Delegitimation Targets and Frames", loc="left", fontsize=15, color=COLORS["ink"], weight="bold")
        ax.text(0, -0.25, "Cells count coded claim records. These are accusations/frames, not independently verified facts.", transform=ax.transAxes, fontsize=9, color=COLORS["muted"])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _new_page((11.7, 8.3))
        ax.text(0.04, 0.96, "S2.1 Delegitimation Examples", fontsize=15, color=COLORS["ink"], weight="bold", va="top")
        ax.text(0.04, 0.915, "Examples preserve target, crisis, actor, claim summary, and source anchors in the companion table.", fontsize=9.5, color=COLORS["muted"])
        y = 0.84
        for (target, frame), vals in sorted(examples.items(), key=lambda item: (-len(item[1]), item[0][0], item[0][1]))[:10]:
            ax.text(0.055, y, f"{target} -> {frame}", fontsize=9.1, color=COLORS["ink"], weight="bold", va="center")
            example_text = _wrap("Examples: " + "; ".join(dict.fromkeys(vals[:4])), 130)
            ax.text(0.055, y - 0.028, example_text, fontsize=7.9, color=COLORS["muted"], va="top")
            y -= 0.076 + 0.018 * example_text.count("\n")
            if y < 0.08:
                break
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def figure_legality_gap_pathways(records: pd.DataFrame, path: Path) -> None:
    df = _clean_df(records.copy()).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12.2, max(4.8, len(df) * 0.70)))
    _style_axis(ax)
    stages = ["legal context", "gap / insufficiency", "rival legitimacy", "formalization"]
    palette = [COLORS["blue"], COLORS["coral"], COLORS["gold"], COLORS["green"]]
    y = np.arange(len(df))
    ax.set_yticks(y)
    ax.set_yticklabels((df["crisis_id"] + " / " + df["actor_name"].map(lambda x: _short(x, 32))).tolist(), fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(-0.5, len(stages) - 0.5)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, fontsize=9)
    ax.grid(axis="x", color=COLORS["grid"], lw=0.8)
    for i, (_, row) in enumerate(df.iterrows()):
        present = []
        if _present(row.get("constitutional_context", "")) or _present(row.get("democracy_legality_frame", "")):
            present.append(0)
        if _present(row.get("legality_gap", "")):
            present.append(1)
        if _present(row.get("claim_summary", "")) or _present(row.get("delegitimation_claim", "")):
            present.append(2)
        if _present(row.get("formalization_path", "")) or _present(row.get("settlement_context", "")):
            present.append(3)
        if present:
            ax.plot(present, [i] * len(present), color=COLORS["grid"], lw=1.2, zorder=1)
            for idx in present:
                ax.scatter(idx, i, s=95, color=palette[idx], edgecolor="white", lw=0.9, zorder=3)
        ax.text(len(stages) - 0.25, i, _short(row.get("legality_gap", ""), 64), va="center", fontsize=7.2, color=COLORS["muted"], clip_on=False)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=palette[idx], markeredgecolor="white", markersize=7, label=stage)
        for idx, stage in enumerate(stages)
    ]
    legend = ax.legend(handles=handles, title="Stage legend", loc="upper center", bbox_to_anchor=(0.50, -0.11), ncol=4, frameon=True, fontsize=7.8, title_fontsize=8.2)
    legend.get_frame().set_edgecolor(COLORS["grid"])
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.94)
    ax.set_title("S2.1 Legality-Legitimacy Gap Pathways", loc="left", fontsize=15, color=COLORS["ink"], weight="bold")
    ax.text(0, -0.25, "Dots show which stages are documented for each gap record; right notes name the coded insufficiency.", transform=ax.transAxes, fontsize=9, color=COLORS["muted"])
    _save_pdf(fig, path)


def figure_formalization_matrix(records: pd.DataFrame, path: Path) -> None:
    df = _clean_df(records.copy())
    crises = _crises(df)
    matrix = np.zeros((len(crises), len(FORMALIZATION_MODE_ORDER)))
    for _, row in df.drop_duplicates(["crisis_id", "discourse_record_id"]).iterrows():
        for mode in _formalization_modes(row):
            matrix[crises.index(row["crisis_id"]), FORMALIZATION_MODE_ORDER.index(mode)] += 1

    fig, ax = plt.subplots(figsize=(11.7, max(5.8, len(crises) * 0.42)))
    cmap = _count_cmap("s21_formal")
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=max(matrix.max(), 1))
    _add_count_colorbar(fig, ax, image, "Distinct discourse records", matrix.max())
    ax.set_yticks(np.arange(len(crises)))
    ax.set_yticklabels(crises, fontsize=8.8)
    ax.set_xticks(np.arange(len(FORMALIZATION_MODE_ORDER)))
    ax.set_xticklabels([_wrap(mode, 18) for mode in FORMALIZATION_MODE_ORDER], rotation=35, ha="right", fontsize=8.1)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(FORMALIZATION_MODE_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(crises), 1), minor=True)
    ax.grid(which="minor", color=COLORS["grid"], lw=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title("S2.1 Formalization Modes by Crisis", loc="left", fontsize=15, color=COLORS["ink"], weight="bold")
    ax.text(0, -0.25, "Shading marks discourse records whose legitimacy claim is linked to each formalization mode.", transform=ax.transAxes, fontsize=9, color=COLORS["muted"])
    _save_pdf(fig, path)


def build_figures(output: pd.DataFrame, figure_dir: Path) -> None:
    iqa = _records(output, "QE-S21-IQA")
    iqc = _records(output, "QE-S21-IQC")
    iqd = _records(output, "QE-S21-IQD")
    iqe = _records(output, "QE-S21-IQE")
    iqf = _records(output, "QE-S21-IQF")
    iqg = _records(output, "QE-S21-IQG")
    iqh = _records(output, "QE-S21-IQH")

    figure_claim_type_matrix(iqa, figure_dir / "F_S21_1_legitimacy_claim_type_matrix.pdf")
    figure_claim_to_formalization_flow(iqh, figure_dir / "F_S21_2_claim_to_formalization_flow.pdf")
    figure_claim_to_formalization_all_edges_matrix(iqh, figure_dir / "F_S21_2_B_claim_to_formalization_all_edges_matrix.pdf")
    figure_democracy_legality_matrix(iqc, figure_dir / "F_S21_3_democracy_legality_tension_matrix.pdf")
    figure_stability_cards(iqd, figure_dir / "F_S21_4_stability_reconciliation_cards.pdf")
    figure_stability_cards(iqd, figure_dir / "F_S21_4_B_stability_reconciliation_all_cards.pdf", max_records=None, title_suffix="_B")
    figure_international_matrix(iqe, figure_dir / "F_S21_5_international_criteria_matrix.pdf")
    figure_delegitimation_matrix(iqf, figure_dir / "F_S21_6_delegitimation_target_frame_matrix.pdf")
    figure_legality_gap_pathways(iqg, figure_dir / "F_S21_7_legality_gap_pathways.pdf")
    figure_formalization_matrix(iqh, figure_dir / "F_S21_8_formalization_modes_matrix.pdf")


def main() -> None:
    repo_root = _repo_root()
    output = _load_s21_output(repo_root)
    table_dir = repo_root / "outputs" / "tables" / "s21_tables"
    figure_dir = repo_root / "outputs" / "figures" / "s21_figures"
    build_tables(output, table_dir)
    build_figures(output, figure_dir)
    print(f"Wrote S21 tables to {table_dir}")
    print(f"Wrote S21 figures to {figure_dir}")


if __name__ == "__main__":
    main()

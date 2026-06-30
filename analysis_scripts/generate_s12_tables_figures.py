#!/usr/bin/env python3
"""Create publication-oriented S1.2 tables and PDF figures."""

from __future__ import annotations

import json
import sys
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
from matplotlib.patches import FancyArrowPatch, PathPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "setup"))
from load_project_data import resolve_project_paths


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

ROUTE_ORDER = [
    "office_spell_or_constitutional_office",
    "military_transfer_gatekeeping_or_direct_rule",
    "negotiated_bargain_or_settlement",
    "constitutional_rule_actor",
    "civil_moral_or_movement_authority",
    "business_media_or_international_support",
    "informal_network_access",
]

RESOURCE_ORDER = [
    "military",
    "party",
    "professional",
    "business",
    "media",
    "international",
    "civil",
    "moral",
]


def _load_s12_output(repo_root: Path) -> pd.DataFrame:
    path = repo_root / "outputs" / "evidence_synthesis" / "S12_all_evidence_synthesis_v1.csv"
    return pd.read_csv(path, dtype=str).fillna("")


def _records(output: pd.DataFrame, question_id: str) -> pd.DataFrame:
    row = output.loc[output["question_id"].eq(question_id)].iloc[0]
    structured = json.loads(row["structured_evidence"])
    df = pd.DataFrame(structured.get("records", [])).fillna("")
    df.insert(0, "question_id", question_id)
    return df


def _clean_cell(value: object) -> str:
    text = str(value).strip()
    if text.lower() in {"", "nan", "0", "0.0", "1", "1.0", "yes_or_potentially", "present"}:
        return ""
    parts = []
    for part in text.replace("|", ";").split(";"):
        cleaned = part.strip()
        if cleaned.lower() in {"", "nan", "0", "0.0", "1", "1.0", "yes_or_potentially", "present"}:
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


def _short(text: object, length: int = 78) -> str:
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


def _route_tokens(row: pd.Series) -> list[str]:
    tokens = []
    for token in _split_values(row.get("entry_route_types", "")):
        if token in ROUTE_ORDER:
            tokens.append(token)
    primary = _clean_cell(row.get("primary_entry_route", ""))
    if primary in ROUTE_ORDER and primary not in tokens:
        tokens.insert(0, primary)
    return tokens


def _resource_tokens(row: pd.Series) -> list[str]:
    text = " ".join(
        [
            _clean_cell(row.get("resource_profile", "")),
            _clean_cell(row.get("support_networks", "")),
            _clean_cell(row.get("network_form", "")),
            _clean_cell(row.get("civil_moral_role", "")),
        ]
    ).lower()
    tokens = []
    for resource in RESOURCE_ORDER:
        if resource in text:
            tokens.append(resource)
    if "church" in text and "moral" not in tokens:
        tokens.append("moral")
    return tokens


def _outcome_bucket(row: pd.Series) -> str:
    text = _clean_cell(row.get("outcome_status", "")).lower()
    if any(term in text for term in ["president", "head of state", "chief of state", "transition president"]):
        return "presidency_or_head_of_state"
    if any(term in text for term in ["prime minister", "head of government"]):
        return "head_of_government_or_pm"
    if any(term in text for term in ["minister", "cabinet", "council", "vice-president", "vice president"]):
        return "cabinet_council_or_vp"
    if "veto" in text:
        return "veto_or_recognition"
    if _present(text):
        return "other_executive_or_symbolic_outcome"
    return "no_coded_executive_outcome"


def _unique_actor_crisis(records: pd.DataFrame) -> pd.DataFrame:
    df = _clean_df(records.copy())
    df["outcome_bucket"] = df.apply(_outcome_bucket, axis=1)
    rows = []
    for (_crisis, _actor), group in df.groupby(["crisis_id", "actor_id"], dropna=False):
        first = group.iloc[0].copy()
        merge_fields = [
            "entry_route_types",
            "entry_status",
            "office_evidence",
            "bargain_evidence",
            "constitutional_rule_context",
            "resource_profile",
            "support_networks",
            "network_form",
            "military_role",
            "civil_moral_role",
            "outcome_status",
            "source_ids",
        ]
        for field in merge_fields:
            first[field] = "; ".join(dict.fromkeys(part for value in group[field].tolist() for part in _split_values(value)))
        first["primary_entry_route"] = group["primary_entry_route"].iloc[0]
        first["outcome_bucket"] = group["outcome_bucket"].iloc[0]
        rows.append(first)
    return pd.DataFrame(rows).fillna("")


def build_tables(output: pd.DataFrame, table_dir: Path) -> dict[str, pd.DataFrame]:
    iqa = _unique_actor_crisis(_records(output, "QE-S12-IQA"))
    iqb = _unique_actor_crisis(_records(output, "QE-S12-IQB"))
    iqc = _unique_actor_crisis(_records(output, "QE-S12-IQC"))
    iqd = _unique_actor_crisis(_records(output, "QE-S12-IQD"))
    iqe = _unique_actor_crisis(_records(output, "QE-S12-IQE"))
    iqf = _unique_actor_crisis(_records(output, "QE-S12-IQF"))
    iqg = _unique_actor_crisis(_records(output, "QE-S12-IQG"))
    iqh = _unique_actor_crisis(_records(output, "QE-S12-IQH"))

    tables = {
        "T_S12_1_actor_entry_routes.csv": iqa[
            [
                "crisis_id",
                "actor_id",
                "actor_name",
                "primary_entry_route",
                "entry_route_types",
                "entry_status",
                "office_evidence",
                "bargain_evidence",
                "outcome_status",
                "source_ids",
            ]
        ],
        "T_S12_2_constitutional_eligibility_assessment.csv": iqb[
            [
                "crisis_id",
                "actor_id",
                "actor_name",
                "constitutional_eligibility_assessment",
                "barrier_status",
                "constitutional_rule_context",
                "office_evidence",
                "source_ids",
            ]
        ],
        "T_S12_3_office_bargain_evidence.csv": iqa[
            [
                "crisis_id",
                "actor_id",
                "actor_name",
                "office_evidence",
                "bargain_evidence",
                "formal_or_informal_access",
                "outcome_status",
                "source_ids",
            ]
        ],
        "T_S12_4_resources_networks_support.csv": iqc[
            [
                "crisis_id",
                "actor_id",
                "actor_name",
                "primary_entry_route",
                "resource_profile",
                "network_form",
                "support_networks",
                "outcome_status",
                "source_ids",
            ]
        ],
        "T_S12_5_military_civil_moral_roles.csv": pd.concat([iqe, iqf], ignore_index=True, sort=False)[
            [
                "crisis_id",
                "actor_id",
                "actor_name",
                "primary_entry_route",
                "military_role",
                "civil_moral_role",
                "bargain_evidence",
                "discourse_evidence",
                "outcome_status",
                "source_ids",
            ]
        ],
        "T_S12_6_opportunity_outcomes.csv": iqh[
            [
                "crisis_id",
                "actor_id",
                "actor_name",
                "primary_entry_route",
                "entry_status",
                "outcome_bucket",
                "outcome_status",
                "constitutional_eligibility_assessment",
                "source_ids",
            ]
        ],
    }

    crisis_summary_rows = []
    for crisis_id, group in iqa.groupby("crisis_id", dropna=False):
        route_counts = Counter()
        outcome_counts = Counter(group["outcome_bucket"])
        actors_by_route = defaultdict(list)
        for _, row in group.iterrows():
            for route in _route_tokens(row):
                route_counts[route] += 1
                actors_by_route[route].append(row.get("actor_name", ""))
        crisis_summary_rows.append(
            {
                "crisis_id": crisis_id,
                "actors": "; ".join(group["actor_name"].drop_duplicates().tolist()),
                "entry_routes_present": "; ".join(route for route in ROUTE_ORDER if route_counts.get(route)),
                "actors_by_entry_route": " | ".join(
                    f"{route}: {', '.join(dict.fromkeys(actors_by_route[route]))}"
                    for route in ROUTE_ORDER
                    if actors_by_route.get(route)
                ),
                "outcome_buckets": "; ".join(f"{k}={v}" for k, v in outcome_counts.items()),
                "source_ids": "; ".join(dict.fromkeys(part for value in group["source_ids"].tolist() for part in _split_values(value))),
            }
        )
    tables["T_S12_7_crisis_actor_entry_summary.csv"] = pd.DataFrame(crisis_summary_rows)

    evidence_index = iqa[
        [
            "crisis_id",
            "actor_id",
            "actor_name",
            "primary_entry_route",
            "constitutional_eligibility_assessment",
            "office_evidence",
            "bargain_evidence",
            "resource_profile",
            "military_role",
            "civil_moral_role",
            "support_networks",
            "outcome_status",
            "source_ids",
        ]
    ]
    tables["T_S12_8_actor_evidence_index.csv"] = evidence_index

    for name, df in tables.items():
        _save_csv(df, table_dir / name)
    return tables


def figure_entry_route_matrix(records: pd.DataFrame, path: Path) -> None:
    df = _unique_actor_crisis(records)
    crises = sorted(df["crisis_id"].unique())
    matrix = np.zeros((len(crises), len(ROUTE_ORDER)))
    examples = defaultdict(list)
    for _, row in df.iterrows():
        i = crises.index(row["crisis_id"])
        for route in _route_tokens(row):
            j = ROUTE_ORDER.index(route)
            matrix[i, j] += 1
            examples[(row["crisis_id"], route)].append(row["actor_name"])

    fig, ax = plt.subplots(figsize=(12, max(5.8, len(crises) * 0.42)))
    cmap = _count_cmap("s12_routes")
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=max(matrix.max(), 1))
    _add_count_colorbar(fig, ax, image, "Distinct actor-crisis records", matrix.max())
    ax.set_yticks(np.arange(len(crises)))
    ax.set_yticklabels(crises, fontsize=8.8)
    ax.set_xticks(np.arange(len(ROUTE_ORDER)))
    ax.set_xticklabels([_wrap(route.replace("_", " "), 16) for route in ROUTE_ORDER], rotation=35, ha="right", fontsize=8)
    ax.set_title("S1.2 Actor Entry Routes by Crisis", loc="left", fontsize=15, color=COLORS["ink"], weight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(ROUTE_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(crises), 1), minor=True)
    ax.grid(which="minor", color=COLORS["grid"], lw=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.text(0, -0.24, "Shading marks number of distinct actor-crisis records with each route. Details are in T_S12_1 and T_S12_7.", transform=ax.transAxes, fontsize=9, color=COLORS["muted"])
    _save_pdf(fig, path)


def figure_route_to_outcome_flow(records: pd.DataFrame, path: Path) -> None:
    df = _unique_actor_crisis(records)
    edges = Counter()
    examples = defaultdict(list)
    for _, row in df.iterrows():
        route = row["primary_entry_route"] or "entry_route_not_explicitly_coded"
        outcome = row["outcome_bucket"]
        edges[(route, outcome)] += 1
        examples[(route, outcome)].append(f"{row['crisis_id']}/{row['actor_name']}")

    left_nodes = list(dict.fromkeys(route for route, _outcome in edges))
    right_nodes = list(dict.fromkeys(outcome for _route, outcome in edges))
    palette = [COLORS["blue"], COLORS["green"], COLORS["coral"], COLORS["violet"], COLORS["gold"], COLORS["teal"], "#8E6E53"]
    max_count = max(edges.values()) if edges else 1
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path, metadata={"Creator": "Codex evidence synthesis"}) as pdf:
        fig, ax = _new_page((12.8, 8.0))
        ax.text(0.03, 0.965, "S1.2 From Entry Route to Executive Outcome", fontsize=16, color=COLORS["ink"], weight="bold", va="top")
        ax.text(0.03, 0.92, "Main map. Page 2 lists concrete actor-crisis examples for the largest pathways.", fontsize=9.5, color=COLORS["muted"])
        ax.text(0.08, 0.855, "Primary entry route", fontsize=10.8, color=COLORS["ink"], weight="bold")
        ax.text(0.70, 0.855, "Outcome bucket", fontsize=10.8, color=COLORS["ink"], weight="bold")
        left_y = {node: 0.78 - i * (0.66 / max(len(left_nodes) - 1, 1)) for i, node in enumerate(left_nodes)}
        right_y = {node: 0.78 - i * (0.66 / max(len(right_nodes) - 1, 1)) for i, node in enumerate(right_nodes)}
        for idx, ((route, outcome), count) in enumerate(edges.items()):
            y1, y2 = left_y[route], right_y[outcome]
            verts = [(0.37, y1), (0.50, y1), (0.54, y2), (0.67, y2)]
            path_obj = MplPath(verts, [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
            ax.add_patch(PathPatch(path_obj, facecolor="none", edgecolor=palette[idx % len(palette)], lw=0.9 + 4.0 * count / max_count, alpha=0.48, zorder=1))
        for node, yy in left_y.items():
            ax.add_patch(Rectangle((0.04, yy - 0.030), 0.33, 0.060, facecolor=COLORS["pale"], edgecolor=COLORS["grid"], lw=1.1, zorder=3))
            ax.text(0.055, yy, _wrap(node.replace("_", " "), 34), va="center", fontsize=8.6, color=COLORS["ink"], zorder=4)
        for node, yy in right_y.items():
            ax.add_patch(Rectangle((0.67, yy - 0.030), 0.29, 0.060, facecolor="#F7FAF8", edgecolor=COLORS["grid"], lw=1.1, zorder=3))
            ax.text(0.685, yy, _wrap(node.replace("_", " "), 28), va="center", fontsize=8.8, color=COLORS["ink"], zorder=4)
        ax.text(0.04, 0.055, "Reading note: routes are actor-crisis classifications; outcomes are not treated as equivalent offices.", fontsize=8.8, color=COLORS["muted"])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _new_page((12.8, 8.0))
        ax.text(0.04, 0.96, "S1.2 Flow Figure - Actor Examples", fontsize=15, color=COLORS["ink"], weight="bold", va="top")
        ax.text(0.04, 0.915, "Examples preserve the crisis and actor names behind the route-to-outcome pathways.", fontsize=9.5, color=COLORS["muted"])
        y = 0.84
        for idx, ((route, outcome), count) in enumerate(edges.most_common(12)):
            color = palette[idx % len(palette)]
            ax.add_patch(Rectangle((0.045, y - 0.018), 0.012, 0.036, facecolor=color, edgecolor=color))
            ax.text(0.07, y + 0.012, f"{_short(route.replace('_', ' '), 48)} -> {_short(outcome.replace('_', ' '), 40)}", fontsize=9.2, color=COLORS["ink"], weight="bold", va="center")
            example_text = _wrap("Examples: " + "; ".join(examples[(route, outcome)][:5]), 132)
            ax.text(0.07, y - 0.020, example_text, fontsize=8.1, color=COLORS["muted"], va="top")
            ax.text(0.91, y, f"records: {count}", fontsize=8.6, color=COLORS["muted"], va="center", ha="right")
            y -= 0.072 + 0.018 * example_text.count("\n")
            if y < 0.08:
                break
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def figure_route_to_outcome_all_edges_matrix(records: pd.DataFrame, path: Path) -> None:
    df = _unique_actor_crisis(records)
    outcomes = list(dict.fromkeys(df["outcome_bucket"].tolist()))
    matrix = np.zeros((len(ROUTE_ORDER), len(outcomes)))
    examples = defaultdict(list)
    for _, row in df.iterrows():
        for route in _route_tokens(row):
            if route not in ROUTE_ORDER:
                continue
            outcome = row["outcome_bucket"]
            matrix[ROUTE_ORDER.index(route), outcomes.index(outcome)] += 1
            examples[(route, outcome)].append(f"{row['crisis_id']}/{row['actor_name']}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path, metadata={"Creator": "Codex evidence synthesis"}) as pdf:
        fig, ax = plt.subplots(figsize=(11.7, 7.2))
        cmap = _count_cmap("s12_route_outcome_all")
        image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=max(matrix.max(), 1))
        _add_count_colorbar(fig, ax, image, "Distinct actor-crisis records", matrix.max())
        ax.set_yticks(np.arange(len(ROUTE_ORDER)))
        ax.set_yticklabels([_wrap(route.replace("_", " "), 28) for route in ROUTE_ORDER], fontsize=8.4)
        ax.set_xticks(np.arange(len(outcomes)))
        ax.set_xticklabels([_wrap(outcome.replace("_", " "), 22) for outcome in outcomes], rotation=30, ha="right", fontsize=8.3)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(outcomes), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(ROUTE_ORDER), 1), minor=True)
        ax.grid(which="minor", color=COLORS["grid"], lw=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_title("S1.2_B Exhaustive Entry-Route to Outcome Edge Matrix", loc="left", fontsize=15, color=COLORS["ink"], weight="bold")
        ax.text(0, -0.25, "Every nonzero cell is a coded route-to-outcome edge. Counts are actor-crisis records, not equivalent offices.", transform=ax.transAxes, fontsize=9, color=COLORS["muted"])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _new_page((11.7, 8.3))
        ax.text(0.04, 0.96, "S1.2_B Exhaustive Route-to-Outcome Examples", fontsize=15, color=COLORS["ink"], weight="bold", va="top")
        ax.text(0.04, 0.915, "Nonzero matrix cells with crisis/actor examples. Full source IDs are in T_S12_1 and T_S12_7.", fontsize=9.4, color=COLORS["muted"])
        y = 0.84
        for (route, outcome), vals in sorted(examples.items(), key=lambda item: (-len(item[1]), item[0][0], item[0][1])):
            ax.text(0.055, y, f"{route.replace('_', ' ')} -> {outcome.replace('_', ' ')}", fontsize=8.8, color=COLORS["ink"], weight="bold", va="center")
            example_text = _wrap("Examples: " + "; ".join(dict.fromkeys(vals[:8])), 132)
            ax.text(0.055, y - 0.024, example_text, fontsize=7.6, color=COLORS["muted"], va="top")
            y -= 0.066 + 0.017 * example_text.count("\n")
            if y < 0.08:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                fig, ax = _new_page((11.7, 8.3))
                ax.text(0.04, 0.96, "S1.2_B Exhaustive Route-to-Outcome Examples Continued", fontsize=15, color=COLORS["ink"], weight="bold", va="top")
                y = 0.88
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def figure_eligibility_by_route(records: pd.DataFrame, path: Path) -> None:
    df = _unique_actor_crisis(records)
    assessments = list(dict.fromkeys(df["constitutional_eligibility_assessment"].map(_clean_cell)))
    assessments = [a for a in assessments if a]
    matrix = np.zeros((len(assessments), len(ROUTE_ORDER)))
    for _, row in df.iterrows():
        assessment = _clean_cell(row["constitutional_eligibility_assessment"])
        if not assessment:
            continue
        i = assessments.index(assessment)
        for route in _route_tokens(row):
            matrix[i, ROUTE_ORDER.index(route)] += 1
    fig, ax = plt.subplots(figsize=(12, max(5.5, len(assessments) * 0.8)))
    cmap = _count_cmap("s12_elig")
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=max(matrix.max(), 1))
    _add_count_colorbar(fig, ax, image, "Distinct actor-crisis records", matrix.max())
    ax.set_yticks(np.arange(len(assessments)))
    ax.set_yticklabels([_wrap(a, 42) for a in assessments], fontsize=8.5)
    ax.set_xticks(np.arange(len(ROUTE_ORDER)))
    ax.set_xticklabels([_wrap(route.replace("_", " "), 16) for route in ROUTE_ORDER], rotation=35, ha="right", fontsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(ROUTE_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(assessments), 1), minor=True)
    ax.grid(which="minor", color=COLORS["grid"], lw=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title("S1.2 Constitutional Eligibility Assessment by Entry Route", loc="left", fontsize=15, color=COLORS["ink"], weight="bold")
    ax.text(0, -0.23, "Cells summarize distinct actor-crisis records; table T_S12_2 retains actor/source details.", transform=ax.transAxes, fontsize=9, color=COLORS["muted"])
    _save_pdf(fig, path)


def figure_resource_network_matrix(records: pd.DataFrame, path: Path) -> None:
    df = _unique_actor_crisis(records)
    crises = sorted(df["crisis_id"].unique())
    matrix = np.zeros((len(crises), len(RESOURCE_ORDER)))
    examples = defaultdict(list)
    for _, row in df.iterrows():
        for token in _resource_tokens(row):
            if token in RESOURCE_ORDER:
                matrix[crises.index(row["crisis_id"]), RESOURCE_ORDER.index(token)] += 1
                examples[token].append(f"{row['crisis_id']}/{row['actor_name']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path, metadata={"Creator": "Codex evidence synthesis"}) as pdf:
        fig, ax = plt.subplots(figsize=(11.7, max(5.8, len(crises) * 0.42)))
        cmap = _count_cmap("s12_res")
        image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=max(matrix.max(), 1))
        _add_count_colorbar(fig, ax, image, "Distinct actor-crisis records", matrix.max())
        ax.set_yticks(np.arange(len(crises)))
        ax.set_yticklabels(crises, fontsize=8.8)
        ax.set_xticks(np.arange(len(RESOURCE_ORDER)))
        ax.set_xticklabels(RESOURCE_ORDER, rotation=35, ha="right", fontsize=8.5)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(RESOURCE_ORDER), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(crises), 1), minor=True)
        ax.grid(which="minor", color=COLORS["grid"], lw=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_title("S1.2 Resources and Support Networks by Crisis", loc="left", fontsize=15, color=COLORS["ink"], weight="bold")
        ax.text(0, -0.18, "Shading marks distinct actor-crisis records with each resource type. Actor examples are on page 2.", transform=ax.transAxes, fontsize=9, color=COLORS["muted"])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _new_page((11.7, 8.3))
        ax.text(0.04, 0.96, "S1.2 Resource and Support-Network Examples", fontsize=15, color=COLORS["ink"], weight="bold", va="top")
        ax.text(0.04, 0.915, "Examples preserve actor and crisis identifiers; full evidence and source IDs are in T_S12_4.", fontsize=9.5, color=COLORS["muted"])
        y = 0.84
        for idx, token in enumerate(RESOURCE_ORDER):
            vals = list(dict.fromkeys(examples.get(token, [])))[:7]
            if not vals:
                continue
            color = [COLORS["blue"], COLORS["green"], COLORS["coral"], COLORS["violet"], COLORS["gold"], COLORS["teal"], "#8E6E53", "#78909C"][idx % 8]
            ax.add_patch(Rectangle((0.045, y - 0.018), 0.012, 0.036, facecolor=color, edgecolor=color))
            ax.text(0.07, y + 0.012, token, fontsize=9.7, color=COLORS["ink"], weight="bold", va="center")
            example_text = _wrap("Examples: " + "; ".join(vals), 128)
            ax.text(0.07, y - 0.020, example_text, fontsize=8.1, color=COLORS["muted"], va="top")
            y -= 0.076 + 0.018 * example_text.count("\n")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def figure_military_sequence(records: pd.DataFrame, path: Path) -> None:
    df = _unique_actor_crisis(records)
    df = df.loc[df["military_role"].map(_present) | df["primary_entry_route"].str.contains("military", na=False)].copy()
    df = df.sort_values(["crisis_id", "actor_name"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, max(5.5, len(df) * 0.42)))
    _style_axis(ax)
    y = np.arange(len(df))
    ax.set_yticks(y)
    ax.set_yticklabels((df["crisis_id"] + " / " + df["actor_name"].map(lambda x: _short(x, 30))).tolist(), fontsize=8)
    ax.invert_yaxis()
    cols = ["entry_status", "military_role", "office_evidence", "outcome_status"]
    xlabels = ["entry", "military role", "office evidence", "outcome"]
    ax.set_xlim(-0.5, len(cols) - 0.5)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.grid(axis="x", color=COLORS["grid"], lw=0.8)
    palette = [COLORS["coral"], COLORS["gold"], COLORS["blue"], COLORS["green"]]
    for i, (_, row) in enumerate(df.iterrows()):
        present = [idx for idx, col in enumerate(cols) if _present(row.get(col, ""))]
        if present:
            ax.plot(present, [i] * len(present), color=COLORS["grid"], lw=1.3, zorder=1)
            for idx in present:
                ax.scatter(idx, i, s=95, color=palette[idx], edgecolor="white", lw=0.9, zorder=3)
        ax.text(len(cols) - 0.25, i, _short(row.get("outcome_status", ""), 60), va="center", fontsize=7.3, color=COLORS["muted"], clip_on=False)
    ax.set_title("S1.2 Military Gatekeeping and Office Conversion", loc="left", fontsize=15, color=COLORS["ink"], weight="bold")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["coral"], markeredgecolor="white", markersize=7, label="entry status documented"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["gold"], markeredgecolor="white", markersize=7, label="military role documented"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["blue"], markeredgecolor="white", markersize=7, label="office evidence documented"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["green"], markeredgecolor="white", markersize=7, label="outcome documented"),
    ]
    legend = ax.legend(handles=handles, title="Stage legend", loc="upper center", bbox_to_anchor=(0.50, -0.10), ncol=2, frameon=True, fontsize=7.8, title_fontsize=8.2)
    legend.get_frame().set_edgecolor(COLORS["grid"])
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.94)
    ax.text(0, -0.29, "Dots show which sequence stages are documented for military-linked actor-crisis records.", transform=ax.transAxes, fontsize=9, color=COLORS["muted"])
    _save_pdf(fig, path)


def figure_civil_support_roles(records: pd.DataFrame, path: Path, max_records: int | None = 8, title_suffix: str = "") -> None:
    df = _unique_actor_crisis(records)
    df = df.loc[df["civil_moral_role"].map(_present) | df["support_networks"].map(_present)].copy()
    if df.empty:
        fig, ax = _new_page()
        ax.text(0.05, 0.95, f"S1.2{title_suffix} Civil, Moral, and Support-Network Actors", fontsize=15, color=COLORS["ink"], weight="bold", va="top")
        ax.text(0.05, 0.85, "No coded civil/moral/support-network records were available.", fontsize=10, color=COLORS["muted"])
        _save_pdf(fig, path)
        return
    if max_records is not None:
        df = df.head(max_records)
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path, metadata={"Creator": "Codex evidence synthesis"}) as pdf:
        for page, start in enumerate(range(0, len(df), 8), start=1):
            fig, ax = _new_page((11.7, 8.3))
            ax.text(0.04, 0.96, f"S1.2{title_suffix} Civil, Moral, and Support-Network Actors - Page {page}", fontsize=15, color=COLORS["ink"], weight="bold", va="top")
            ax.text(0.04, 0.915, "Evidence cards for actors whose crisis relevance is not reducible to ordinary office-spell access.", fontsize=9.5, color=COLORS["muted"])
            card_w = 0.42
            card_h = 0.19
            xs = [0.05, 0.53]
            y = 0.82
            for idx, (_, row) in enumerate(df.iloc[start : start + 8].iterrows()):
                col = idx % 2
                if idx and col == 0:
                    y -= card_h + 0.035
                x = xs[col]
                ax.add_patch(Rectangle((x, y - card_h), card_w, card_h, facecolor="#FFFFFF", edgecolor=COLORS["grid"], lw=1.1))
                ax.add_patch(Rectangle((x, y - 0.028), card_w, 0.028, facecolor=COLORS["pale"], edgecolor=COLORS["grid"], lw=0.6))
                title = f"{row['crisis_id']} / {_short(row['actor_name'], 38)}"
                ax.text(x + 0.012, y - 0.014, title, fontsize=8.7, color=COLORS["ink"], weight="bold", va="center")
                evidence = row.get("civil_moral_role", "") or row.get("support_networks", "")
                bargain = row.get("bargain_evidence", "")
                discourse = row.get("discourse_evidence", "")
                lines = [
                    "Role: " + _short(evidence, 90),
                    "Bargain: " + _short(bargain, 85),
                    "Discourse: " + _short(discourse, 82),
                ]
                for line_i, line in enumerate(lines):
                    ax.text(x + 0.012, y - 0.058 - line_i * 0.042, _wrap(line, 58), fontsize=7.0, color=COLORS["muted"], va="top")
            ax.text(0.05, 0.045, "Full source IDs and uncropped text are retained in T_S12_5 and T_S12_8.", fontsize=8.5, color=COLORS["muted"])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def figure_evidence_completeness(records: pd.DataFrame, path: Path) -> None:
    df = _unique_actor_crisis(records)
    fields = ["office_evidence", "bargain_evidence", "constitutional_rule_context", "resource_profile", "military_role", "civil_moral_role", "support_networks", "outcome_status"]
    labels = ["office", "bargain", "constitutional", "resource", "military", "civil/moral", "support", "outcome"]
    crises = sorted(df["crisis_id"].unique())
    matrix = np.zeros((len(crises), len(fields)))
    for i, crisis_id in enumerate(crises):
        group = df.loc[df["crisis_id"].eq(crisis_id)]
        for j, field in enumerate(fields):
            matrix[i, j] = group[field].map(_present).sum()
    fig, ax = plt.subplots(figsize=(11.5, max(5.5, len(crises) * 0.42)))
    cmap = _count_cmap("s12_complete")
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=max(matrix.max(), 1))
    _add_count_colorbar(fig, ax, image, "Distinct actor-crisis records", matrix.max())
    ax.set_yticks(np.arange(len(crises)))
    ax.set_yticklabels(crises, fontsize=8.8)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(crises), 1), minor=True)
    ax.grid(which="minor", color=COLORS["grid"], lw=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title("S1.2 Evidence Coverage by Crisis", loc="left", fontsize=15, color=COLORS["ink"], weight="bold")
    ax.text(0, -0.18, "Cells mark how many distinct actor-crisis records contain each kind of evidence.", transform=ax.transAxes, fontsize=9, color=COLORS["muted"])
    _save_pdf(fig, path)


def build_figures(output: pd.DataFrame, figure_dir: Path) -> None:
    iqa = _clean_df(_records(output, "QE-S12-IQA"))
    iqb = _clean_df(_records(output, "QE-S12-IQB"))
    iqc = _clean_df(_records(output, "QE-S12-IQC"))
    iqe = _clean_df(_records(output, "QE-S12-IQE"))
    iqf = _clean_df(_records(output, "QE-S12-IQF"))
    iqg = _clean_df(_records(output, "QE-S12-IQG"))
    iqh = _clean_df(_records(output, "QE-S12-IQH"))

    figure_entry_route_matrix(iqa, figure_dir / "F_S12_1_actor_entry_route_matrix.pdf")
    figure_route_to_outcome_flow(iqh, figure_dir / "F_S12_2_entry_route_to_outcome_flow.pdf")
    figure_route_to_outcome_all_edges_matrix(iqh, figure_dir / "F_S12_2_B_entry_route_to_outcome_all_edges_matrix.pdf")
    figure_eligibility_by_route(iqb, figure_dir / "F_S12_3_eligibility_by_entry_route.pdf")
    figure_resource_network_matrix(iqc, figure_dir / "F_S12_4_resource_network_matrix.pdf")
    figure_military_sequence(iqe, figure_dir / "F_S12_5_military_gatekeeping_sequence.pdf")
    figure_civil_support_roles(pd.concat([iqf, iqg], ignore_index=True, sort=False), figure_dir / "F_S12_6_civil_moral_support_roles.pdf")
    figure_civil_support_roles(pd.concat([iqf, iqg], ignore_index=True, sort=False), figure_dir / "F_S12_6_B_civil_moral_support_roles_all.pdf", max_records=None, title_suffix="_B")
    figure_evidence_completeness(iqa, figure_dir / "F_S12_7_evidence_coverage_by_crisis.pdf")


def main() -> None:
    paths = resolve_project_paths()
    output = _load_s12_output(paths.repo_root)
    table_dir = paths.repo_root / "outputs" / "tables" / "s12_tables"
    figure_dir = paths.repo_root / "outputs" / "figures" / "s12_figures"
    build_tables(output, table_dir)
    build_figures(output, figure_dir)
    print(f"Wrote S12 tables to {table_dir}")
    print(f"Wrote S12 figures to {figure_dir}")


if __name__ == "__main__":
    main()

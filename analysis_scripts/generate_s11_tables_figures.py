#!/usr/bin/env python3
"""Create publication-oriented S1.1 tables and PDF figures."""

from __future__ import annotations

import json
import sys
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.patches import FancyArrowPatch, Patch, PathPatch, Rectangle

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


def _load_s11_output(repo_root: Path) -> pd.DataFrame:
    path = repo_root / "outputs" / "evidence_synthesis" / "S11_all_evidence_synthesis_v1.csv"
    return pd.read_csv(path, dtype=str).fillna("")


def _records(output: pd.DataFrame, question_id: str) -> pd.DataFrame:
    row = output.loc[output["question_id"].eq(question_id)].iloc[0]
    structured = json.loads(row["structured_evidence"])
    items = structured.get("records", structured.get("rule_contestation_matrix", []))
    df = pd.DataFrame(items).fillna("")
    df.insert(0, "question_id", question_id)
    return df


def _clean_cell(value: object) -> str:
    text = str(value).strip()
    if text.lower() in {"", "nan", "0", "0.0", "1", "1.0", "yes_or_potentially"}:
        return ""
    parts = []
    for part in text.replace("|", ";").split(";"):
        cleaned = part.strip()
        if cleaned.lower() in {"", "nan", "0", "0.0", "1", "1.0", "yes_or_potentially"}:
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


def _first_date(*values: object) -> pd.Timestamp | pd.NaT:
    for value in values:
        for part in _split_values(value) or [str(value).strip()]:
            try:
                date = pd.to_datetime(part, errors="coerce")
            except (TypeError, ValueError):
                date = pd.NaT
            if pd.notna(date):
                return date
    return pd.NaT


def _period_start(period: object) -> pd.Timestamp | pd.NaT:
    text = str(period)
    start = text.split(" to ")[0].strip()
    return pd.to_datetime(start, errors="coerce")


def _wrap(text: object, width: int = 28) -> str:
    cleaned = _clean_cell(text)
    if not cleaned:
        return ""
    return "\n".join(textwrap.wrap(cleaned, width=width, break_long_words=False))


def _short(text: object, length: int = 80) -> str:
    cleaned = " ".join(_clean_cell(text).split())
    if len(cleaned) <= length:
        return cleaned
    return cleaned[: length - 3].rstrip() + "..."


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _clean_df(df).to_csv(path, index=False)


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.tick_params(colors=COLORS["ink"], labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(COLORS["grid"])


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


def _add_count_colorbar(fig: plt.Figure, ax: plt.Axes, image: matplotlib.image.AxesImage, label: str, max_value: float) -> None:
    cbar = fig.colorbar(image, ax=ax, fraction=0.034, pad=0.022)
    cbar.set_label(label, fontsize=8.6, color=COLORS["ink"])
    cbar.ax.tick_params(labelsize=8, colors=COLORS["ink"])
    cbar.locator = matplotlib.ticker.MaxNLocator(integer=True, nbins=min(int(max(max_value, 1)) + 1, 6))
    cbar.update_ticks()


def _count_cmap(name: str) -> matplotlib.colors.LinearSegmentedColormap:
    return matplotlib.colors.LinearSegmentedColormap.from_list(name, ["#FFFFFF", "#CFE3DA", COLORS["green"], COLORS["blue"]])


def _add_trigger_legend_page(pdf: PdfPages, title: str, color_map: dict[str, str]) -> None:
    fig, ax = _new_page((11.7, 8.3))
    ax.text(0.05, 0.955, f"{title} - Color Legend", fontsize=15, color=COLORS["ink"], weight="bold", va="top")
    ax.text(0.05, 0.91, "Marker colors identify the coded trigger-event type. They are categorical, not ordered.", fontsize=9.4, color=COLORS["muted"])
    items = list(color_map.items())
    cols = 2
    row_h = 0.072
    for idx, (trigger, color) in enumerate(items):
        col = idx % cols
        row = idx // cols
        x = 0.06 + col * 0.46
        y = 0.82 - row * row_h
        ax.add_patch(Rectangle((x, y - 0.016), 0.020, 0.032, facecolor=color, edgecolor=color))
        ax.text(x + 0.032, y, trigger.replace("_", " "), fontsize=8.9, color=COLORS["ink"], va="center")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_tables(output: pd.DataFrame, table_dir: Path) -> dict[str, pd.DataFrame]:
    iqa = _records(output, "QE-S11-IQA")
    iqb = _records(output, "QE-S11-IQB")
    iqc = _records(output, "QE-S11-IQC")
    iqd = _records(output, "QE-S11-IQD")
    iqe = _records(output, "QE-S11-IQE")
    iqf = _records(output, "QE-S11-IQF")
    iqg = _records(output, "QE-S11-IQG")
    iqh = _records(output, "QE-S11-IQH")

    tables = {
        "T_S11_1_rule_contestation_matrix.csv": iqa[
            [
                "crisis_id",
                "period",
                "contested_rule_type",
                "coded_constitutional_instruments",
                "mode_of_contestation",
                "actors_linked_to_rule",
                "rule_or_outcome_evidence",
                "settlement_or_bargain_evidence",
                "evidence_strength",
                "source_ids",
            ]
        ],
        "T_S11_2_rupture_sequence.csv": iqb[
            [
                "crisis_id",
                "period",
                "trigger_event_type",
                "trigger_event_date",
                "rupture_moment_date",
                "turning_point_or_threshold",
                "manifestation_reasons",
                "discourse_trace",
                "settlement_response_to_trigger",
                "source_ids",
            ]
        ],
        "T_S11_3_arena_shift_matrix.csv": iqc[
            [
                "crisis_id",
                "formal_arena_before",
                "decisive_arena_after",
                "secondary_arena",
                "actors_relocated_to_new_arena",
                "actor_roles_in_new_arena",
                "arena_legitimacy_claims",
                "source_ids",
            ]
        ],
        "T_S11_5_legality_resource_substitution.csv": iqe[
            [
                "crisis_id",
                "competing_resource_types",
                "dominant_resource_in_outcome",
                "actors_linked_to_substitution",
                "actor_resource_profiles",
                "bargain_recognition_of_resources",
                "discourse_resource_claims",
                "source_ids",
            ]
        ],
    }

    frame_table = iqd.merge(
        iqf,
        on=["crisis_id", "crisis_label"],
        how="outer",
        suffixes=("_delegitimation", "_exception"),
    )
    tables["T_S11_4_delegitimation_exception_frames.csv"] = frame_table[
        [
            "crisis_id",
            "delegitimation_frames",
            "delegitimated_institutions",
            "leading_actors",
            "actor_position_mapping",
            "justification_frames",
            "actor_justification_types",
            "settlement_link",
            "discourse_examples_delegitimation",
            "discourse_examples_exception",
            "source_ids_delegitimation",
            "source_ids_exception",
        ]
    ]

    chain_table = iqg.merge(
        iqh,
        on=["crisis_id", "crisis_label"],
        how="outer",
        suffixes=("_suspension", "_legalization"),
    )
    tables["T_S11_6_suspension_legalization_chain.csv"] = chain_table[
        [
            "crisis_id",
            "classification",
            "date_order_suspended_or_reinterpreted",
            "document_chain",
            "suspension_clauses",
            "reinterpretation_clauses",
            "continuity_claims",
            "validation_or_relegalization",
            "extra_constitutional_fact_created",
            "fact_creation_date",
            "legalization_instruments",
            "legalization_actor",
            "legalization_date",
            "settlement_chain",
            "implementation_statuses",
            "source_ids_suspension",
            "source_ids_legalization",
        ]
    ]

    pathway_table = iqa[
        [
            "crisis_id",
            "period",
            "contested_rule_type",
            "mode_of_contestation",
            "actors_linked_to_rule",
            "source_ids",
        ]
    ].merge(
        iqc[
            [
                "crisis_id",
                "decisive_arena_after",
                "secondary_arena",
                "actors_relocated_to_new_arena",
                "actor_roles_in_new_arena",
                "arena_legitimacy_claims",
                "source_ids",
            ]
        ],
        on="crisis_id",
        how="outer",
        suffixes=("_rule", "_arena"),
    )
    tables["T_S11_7_rule_to_arena_pathways.csv"] = pathway_table

    mechanism_index = iqa[["crisis_id", "period", "contested_rule_type", "mode_of_contestation"]].merge(
        iqb[["crisis_id", "trigger_event_type", "turning_point_or_threshold"]],
        on="crisis_id",
        how="outer",
    ).merge(
        iqc[["crisis_id", "decisive_arena_after"]],
        on="crisis_id",
        how="outer",
    ).merge(
        iqe[["crisis_id", "dominant_resource_in_outcome", "competing_resource_types"]],
        on="crisis_id",
        how="outer",
    ).merge(
        chain_table[["crisis_id", "classification", "legalization_instruments", "implementation_statuses"]],
        on="crisis_id",
        how="outer",
    )
    tables["T_S11_8_crisis_mechanism_index.csv"] = mechanism_index

    for name, df in tables.items():
        _save_csv(df, table_dir / name)
    return tables


def figure_timeline(iqb: pd.DataFrame, iqa: pd.DataFrame, path: Path) -> None:
    df = iqb.copy()
    df["plot_date"] = df.apply(
        lambda r: _first_date(r.get("rupture_moment_date", ""), r.get("trigger_event_date", ""), _period_start(r.get("period", ""))),
        axis=1,
    )
    df = df.loc[pd.notna(df["plot_date"])].sort_values("plot_date")
    rule_lookup = iqa.drop_duplicates("crisis_id").set_index("crisis_id")["contested_rule_type"].to_dict()
    df["rule"] = df["crisis_id"].map(rule_lookup).fillna("")

    categories = sorted(df["trigger_event_type"].map(_clean_cell).unique())
    palette = [COLORS["blue"], COLORS["coral"], COLORS["green"], COLORS["gold"], COLORS["violet"], COLORS["teal"], "#8E6E53", "#78909C"]
    color_map = {cat: palette[i % len(palette)] for i, cat in enumerate(categories)}

    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path, metadata={"Creator": "Codex evidence synthesis"}) as pdf:
        fig, ax = plt.subplots(figsize=(12.5, 7.2))
        _style_axis(ax)
        y = np.arange(len(df))
        ax.hlines(y, df["plot_date"] - pd.Timedelta(days=160), df["plot_date"] + pd.Timedelta(days=160), color=COLORS["grid"], lw=2)
        for idx, (_, row) in enumerate(df.iterrows()):
            color = color_map.get(row["trigger_event_type"], COLORS["blue"])
            ax.scatter(row["plot_date"], idx, s=95, color=color, edgecolor="white", linewidth=1.1, zorder=3)
            label = f"{row['crisis_id']}  {_short(row['trigger_event_type'], 42)}"
            ax.text(row["plot_date"] + pd.Timedelta(days=210), idx, label, va="center", fontsize=8.7, color=COLORS["ink"])
            if _present(row["rule"]):
                ax.text(row["plot_date"] - pd.Timedelta(days=210), idx, _short(row["rule"], 46), va="center", ha="right", fontsize=8, color=COLORS["muted"])

        ax.set_yticks([])
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(axis="x", color=COLORS["grid"], lw=0.8)
        ax.set_title("S1.1 Timeline of Constitutional Ruptures and Trigger Events", loc="left", fontsize=15, color=COLORS["ink"], weight="bold")
        ax.set_xlabel("Approximate trigger or rupture date", color=COLORS["ink"])
        ax.text(0, -0.12, "Left labels show contested rule type where coded; right labels show crisis and trigger type. Marker-color legend is on page 2.", transform=ax.transAxes, fontsize=9, color=COLORS["muted"])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        _add_trigger_legend_page(pdf, "S1.1 Timeline of Constitutional Ruptures and Trigger Events", color_map)


def figure_flow(iqa: pd.DataFrame, iqc: pd.DataFrame, path: Path) -> None:
    merged = iqa[["crisis_id", "contested_rule_type"]].merge(
        iqc[["crisis_id", "decisive_arena_after", "actors_relocated_to_new_arena"]],
        on="crisis_id",
        how="inner",
    )
    edges = Counter()
    examples: dict[tuple[str, str], list[str]] = defaultdict(list)
    for _, row in merged.iterrows():
        rule = _short(row["contested_rule_type"], 52) or "uncoded rule"
        arenas = _split_values(row["decisive_arena_after"]) or ["uncoded arena"]
        for arena in arenas:
            key = (rule, arena)
            edges[key] += 1
            examples[key].append(row["crisis_id"])

    left_nodes = list(dict.fromkeys([edge[0] for edge in edges]))
    right_nodes = list(dict.fromkeys([edge[1] for edge in edges]))
    left_y = {node: 0.92 - i * (0.82 / max(len(left_nodes) - 1, 1)) for i, node in enumerate(left_nodes)}
    right_y = {node: 0.92 - i * (0.82 / max(len(right_nodes) - 1, 1)) for i, node in enumerate(right_nodes)}

    path.parent.mkdir(parents=True, exist_ok=True)
    edge_palette = [COLORS["blue"], COLORS["green"], COLORS["coral"], COLORS["violet"], COLORS["gold"], COLORS["teal"]]
    max_count = max(edges.values()) if edges else 1
    with PdfPages(path, metadata={"Creator": "Codex evidence synthesis"}) as pdf:
        fig, ax = _new_page((12.8, 8.0))
        ax.text(0.03, 0.965, "S1.1 From Contested Rule to Decisive Arena", fontsize=16, color=COLORS["ink"], weight="bold", va="top")
        ax.text(0.03, 0.92, "Main map. Edge thickness marks repeated crisis pathways; crisis examples are on the next page.", fontsize=9.5, color=COLORS["muted"])
        ax.text(0.08, 0.855, "Contested constitutional rule", fontsize=10.8, color=COLORS["ink"], weight="bold")
        ax.text(0.70, 0.855, "Arena that becomes decisive", fontsize=10.8, color=COLORS["ink"], weight="bold")

        # Rescale node bands below the title area.
        left_y = {node: 0.78 - i * (0.66 / max(len(left_nodes) - 1, 1)) for i, node in enumerate(left_nodes)}
        right_y = {node: 0.78 - i * (0.66 / max(len(right_nodes) - 1, 1)) for i, node in enumerate(right_nodes)}

        for i, ((rule, arena), count) in enumerate(edges.items()):
            y1, y2 = left_y[rule], right_y[arena]
            verts = [(0.37, y1), (0.50, y1), (0.54, y2), (0.67, y2)]
            path_obj = MplPath(verts, [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
            patch = PathPatch(
                path_obj,
                facecolor="none",
                edgecolor=edge_palette[i % len(edge_palette)],
                lw=0.9 + 3.5 * count / max_count,
                alpha=0.48,
                zorder=1,
            )
            ax.add_patch(patch)

        for node, yy in left_y.items():
            ax.add_patch(Rectangle((0.04, yy - 0.030), 0.33, 0.060, facecolor=COLORS["pale"], edgecolor=COLORS["grid"], lw=1.1, zorder=3))
            ax.text(0.055, yy, _wrap(node, 34), va="center", fontsize=8.8, color=COLORS["ink"], zorder=4)
        for node, yy in right_y.items():
            ax.add_patch(Rectangle((0.67, yy - 0.030), 0.26, 0.060, facecolor="#F7FAF8", edgecolor=COLORS["grid"], lw=1.1, zorder=3))
            ax.text(0.685, yy, _wrap(node, 25), va="center", fontsize=9.2, color=COLORS["ink"], zorder=4)

        ax.text(0.04, 0.055, "Reading note: this figure shows movement away from a formal rule/procedure into the arena that becomes outcome-making.", fontsize=8.8, color=COLORS["muted"])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _new_page((12.8, 8.0))
        ax.text(0.04, 0.96, "S1.1 Flow Figure - Pathway Examples", fontsize=15, color=COLORS["ink"], weight="bold", va="top")
        ax.text(0.04, 0.915, "Examples are taken from the crisis IDs attached to each rule-to-arena edge.", fontsize=9.5, color=COLORS["muted"])
        y = 0.84
        for idx, ((rule, arena), count) in enumerate(edges.most_common()):
            crisis_ids = ", ".join(examples[(rule, arena)][:7])
            color = edge_palette[idx % len(edge_palette)]
            ax.add_patch(Rectangle((0.045, y - 0.018), 0.012, 0.036, facecolor=color, edgecolor=color))
            ax.text(0.07, y + 0.012, f"{_short(rule, 52)} -> {_short(arena, 28)}", fontsize=9.4, color=COLORS["ink"], weight="bold", va="center")
            ax.text(0.07, y - 0.020, f"Crisis examples: {crisis_ids}", fontsize=8.7, color=COLORS["muted"], va="center")
            ax.text(0.88, y, f"pathways: {count}", fontsize=8.6, color=COLORS["muted"], va="center", ha="right")
            y -= 0.068
            if y < 0.08:
                break
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def figure_flow_all_pathways_matrix(iqa: pd.DataFrame, iqc: pd.DataFrame, path: Path) -> None:
    merged = iqa[["crisis_id", "contested_rule_type"]].merge(
        iqc[["crisis_id", "decisive_arena_after", "actors_relocated_to_new_arena"]],
        on="crisis_id",
        how="inner",
    )
    edges = Counter()
    examples: dict[tuple[str, str], list[str]] = defaultdict(list)
    for _, row in merged.iterrows():
        rule = _short(row["contested_rule_type"], 60) or "uncoded rule"
        arenas = _split_values(row["decisive_arena_after"]) or ["uncoded arena"]
        for arena in arenas:
            key = (rule, arena)
            edges[key] += 1
            examples[key].append(row["crisis_id"])

    rules = list(dict.fromkeys(rule for rule, _arena in edges))
    arenas = list(dict.fromkeys(arena for _rule, arena in edges))
    matrix = np.zeros((len(rules), len(arenas)))
    for (rule, arena), count in edges.items():
        matrix[rules.index(rule), arenas.index(arena)] = count

    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path, metadata={"Creator": "Codex evidence synthesis"}) as pdf:
        fig, ax = plt.subplots(figsize=(11.7, max(5.8, len(rules) * 0.58)))
        cmap = _count_cmap("s11_all_pathways")
        image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=max(matrix.max(), 1))
        _add_count_colorbar(fig, ax, image, "Crisis-rule-arena pathways", matrix.max())
        ax.set_yticks(np.arange(len(rules)))
        ax.set_yticklabels([_wrap(rule, 32) for rule in rules], fontsize=8.4)
        ax.set_xticks(np.arange(len(arenas)))
        ax.set_xticklabels([_wrap(arena, 18) for arena in arenas], rotation=30, ha="right", fontsize=8.5)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(arenas), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(rules), 1), minor=True)
        ax.grid(which="minor", color=COLORS["grid"], lw=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_title("S1.1_B Exhaustive Rule-to-Arena Pathway Matrix", loc="left", fontsize=15, color=COLORS["ink"], weight="bold")
        ax.text(0, -0.23, "Every nonzero cell is a coded contested-rule to decisive-arena pathway. Counts are pathway records, not causal weights.", transform=ax.transAxes, fontsize=9, color=COLORS["muted"])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _new_page((11.7, 8.3))
        ax.text(0.04, 0.96, "S1.1_B Exhaustive Rule-to-Arena Examples", fontsize=15, color=COLORS["ink"], weight="bold", va="top")
        ax.text(0.04, 0.915, "Nonzero matrix cells with all attached crisis IDs. Full source details are in T_S11_3 and T_S11_7.", fontsize=9.4, color=COLORS["muted"])
        y = 0.84
        for (rule, arena), vals in sorted(examples.items(), key=lambda item: (-len(item[1]), item[0][0], item[0][1])):
            ax.text(0.055, y, f"{_short(rule, 56)} -> {_short(arena, 32)}", fontsize=8.8, color=COLORS["ink"], weight="bold", va="center")
            example_text = _wrap("Crisis IDs: " + ", ".join(dict.fromkeys(vals)), 132)
            ax.text(0.055, y - 0.024, example_text, fontsize=7.8, color=COLORS["muted"], va="top")
            y -= 0.062 + 0.017 * example_text.count("\n")
            if y < 0.08:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                fig, ax = _new_page((11.7, 8.3))
                ax.text(0.04, 0.96, "S1.1_B Exhaustive Rule-to-Arena Examples Continued", fontsize=15, color=COLORS["ink"], weight="bold", va="top")
                y = 0.88
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def figure_heatmap(iqa: pd.DataFrame, path: Path) -> None:
    df = iqa.copy()
    df["start"] = df["period"].map(_period_start)
    df = df.sort_values("start")
    crises = df["crisis_id"].tolist()
    rule_types = list(dict.fromkeys(df["contested_rule_type"].map(_clean_cell)))
    strength_map = {"weak": 1, "medium": 2, "strong": 3}
    matrix = np.zeros((len(crises), len(rule_types)))
    for i, (_, row) in enumerate(df.iterrows()):
        j = rule_types.index(_clean_cell(row["contested_rule_type"]))
        matrix[i, j] = strength_map.get(_clean_cell(row.get("evidence_strength", "")).lower(), 2)

    fig, ax = plt.subplots(figsize=(11.5, max(5.8, len(crises) * 0.36)))
    cmap = _count_cmap("s11_strength")
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=3)
    cbar = fig.colorbar(image, ax=ax, fraction=0.034, pad=0.022, ticks=[0, 1, 2, 3])
    cbar.set_label("Coded evidence strength", fontsize=8.6, color=COLORS["ink"])
    cbar.ax.set_yticklabels(["no coded link", "weak", "medium", "strong"])
    cbar.ax.tick_params(labelsize=8, colors=COLORS["ink"])
    ax.set_yticks(np.arange(len(crises)))
    ax.set_yticklabels(crises, fontsize=8.5)
    ax.set_xticks(np.arange(len(rule_types)))
    ax.set_xticklabels([_wrap(rule, 18) for rule in rule_types], rotation=35, ha="right", fontsize=8.2)
    ax.set_title("S1.1 Contested Constitutional Rule Types by Crisis", loc="left", fontsize=15, color=COLORS["ink"], weight="bold")
    ax.set_xlabel("Contested rule type", color=COLORS["ink"])
    ax.set_ylabel("Crisis", color=COLORS["ink"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(rule_types), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(crises), 1), minor=True)
    ax.grid(which="minor", color=COLORS["grid"], linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.text(0, -0.28, "Shading marks coded evidence strength for crisis-rule links, not frequency.", transform=ax.transAxes, fontsize=9, color=COLORS["muted"])
    _save_pdf(fig, path)


def figure_mechanism(path: Path) -> None:
    steps = [
        ("Rule\nContestation", "A formal rule or procedure\nbecomes politically contested", "S11-IQA"),
        ("Rupture\nTrigger", "Protest, election dispute,\nrepression, elite split, or transfer", "S11-IQB"),
        ("Arena\nShift", "Outcome-making moves to\nstreet, military, mediation,\nchurch, court, or diplomacy", "S11-IQC"),
        ("Resource\nSubstitution", "Legality competes with\nmilitary, popular, religious,\ninternational, or media capital", "S11-IQE"),
        ("Exception\nJustification", "Actors narrate necessity,\ndemocracy, order, people,\nor reconciliation", "S11-IQD/IQF"),
        ("After-the-fact\nLegalization", "Facts are translated into\ncharters, decisions, roadmaps,\nconstitutions, or elections", "S11-IQG/IQH"),
    ]
    fig, ax = plt.subplots(figsize=(13, 5.8))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.02, 0.96, "S1.1 Hidden Constitution Mechanism", fontsize=16, color=COLORS["ink"], weight="bold", va="top")
    ax.text(0.02, 0.90, "A data-grounded mechanism linking constitutional contestation to extra-ordinary executive formation.", fontsize=9.5, color=COLORS["muted"])
    xs = np.linspace(0.08, 0.92, len(steps))
    box_colors = [COLORS["blue"], COLORS["coral"], COLORS["green"], COLORS["gold"], COLORS["violet"], COLORS["teal"]]
    for i, (title, subtitle, qid) in enumerate(steps):
        x = xs[i]
        ax.add_patch(Rectangle((x - 0.065, 0.42), 0.13, 0.26, facecolor="#FFFFFF", edgecolor=box_colors[i], lw=1.8))
        ax.text(x, 0.63, title, ha="center", va="center", fontsize=10.5, color=COLORS["ink"], weight="bold")
        ax.text(x, 0.515, subtitle, ha="center", va="center", fontsize=7.6, color=COLORS["muted"])
        ax.text(x, 0.445, qid, ha="center", va="center", fontsize=7.5, color=box_colors[i], weight="bold")
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x + 0.068, 0.55), (xs[i + 1] - 0.068, 0.55), arrowstyle="-|>", mutation_scale=12, lw=1.4, color=COLORS["ink"], alpha=0.75))
    ax.text(0.08, 0.25, "Representative anchors:", fontsize=9, color=COLORS["ink"], weight="bold")
    ax.text(0.08, 0.18, "CR-1972: student-worker mobilization + military transfer -> provisional constitutional law", fontsize=8.8, color=COLORS["muted"])
    ax.text(0.08, 0.13, "CR-1991: Forces Vives pressure -> Panorama Convention -> constitutional incorporation", fontsize=8.8, color=COLORS["muted"])
    ax.text(0.08, 0.08, "CR-2009: coercive transfer + HAT ordinances/charters -> transition authority and later roadmap", fontsize=8.8, color=COLORS["muted"])
    _save_pdf(fig, path)


def figure_resource_matrix(iqe: pd.DataFrame, path: Path) -> None:
    df = iqe.copy()
    resources = ["legal", "electoral", "street", "military", "church", "mediation", "international", "media", "wealth"]
    crises = df["crisis_id"].tolist()
    fig, ax = plt.subplots(figsize=(11.5, max(5.2, len(crises) * 0.45)))
    ax.set_xlim(-0.5, len(resources) - 0.5)
    ax.set_ylim(-0.5, len(crises) - 0.5)
    ax.set_xticks(range(len(resources)))
    ax.set_xticklabels(resources, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(crises)))
    ax.set_yticklabels(crises, fontsize=9)
    ax.invert_yaxis()
    _style_axis(ax)
    ax.grid(color=COLORS["grid"], lw=0.8)
    for i, (_, row) in enumerate(df.iterrows()):
        present = set()
        for field in ["competing_resource_types", "actor_resource_profiles", "bargain_recognition_of_resources", "discourse_resource_claims"]:
            for value in _split_values(row.get(field, "")):
                lower = value.lower()
                for resource in resources:
                    if resource in lower:
                        present.add(resource)
        dominant = set()
        for value in _split_values(row.get("dominant_resource_in_outcome", "")):
            lower = value.lower()
            for resource in resources:
                if resource in lower:
                    dominant.add(resource)
        for j, resource in enumerate(resources):
            if resource in present:
                ax.scatter(j, i, s=70, color=COLORS["sand"], edgecolor=COLORS["muted"], linewidth=0.7)
            if resource in dominant:
                ax.scatter(j, i, s=155, color=COLORS["coral"], marker="D", edgecolor="white", linewidth=0.9, zorder=3)
    ax.set_title("S1.1 Resource Substitution in Executive Formation", loc="left", fontsize=15, color=COLORS["ink"], weight="bold")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["sand"], markeredgecolor=COLORS["muted"], markersize=7, label="resource present"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=COLORS["coral"], markeredgecolor="white", markersize=8, label="dominant in outcome"),
    ]
    ax.legend(handles=handles, title="Marker legend", loc="upper right", frameon=False, fontsize=8.5, title_fontsize=8.8)
    ax.text(0, -0.22, "Circles mark coded resource presence; diamonds mark resources coded as dominant in outcome.", transform=ax.transAxes, fontsize=9, color=COLORS["muted"])
    _save_pdf(fig, path)


def figure_rupture_triggers(iqb: pd.DataFrame, path: Path) -> None:
    df = iqb.copy()
    df["date"] = df.apply(
        lambda r: _first_date(r.get("rupture_moment_date", ""), r.get("trigger_event_date", ""), _period_start(r.get("period", ""))),
        axis=1,
    )
    df = df.loc[pd.notna(df["date"])].sort_values("date")
    trigger_types = list(dict.fromkeys(df["trigger_event_type"].map(_clean_cell)))
    palette = [COLORS["blue"], COLORS["green"], COLORS["coral"], COLORS["violet"], COLORS["gold"], COLORS["teal"], "#8E6E53", "#78909C"]
    trigger_color = {trigger: palette[i % len(palette)] for i, trigger in enumerate(trigger_types)}

    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path, metadata={"Creator": "Codex evidence synthesis"}) as pdf:
        fig, ax = plt.subplots(figsize=(12.5, 7.4))
        _style_axis(ax)
        y = np.arange(len(df))
        ax.hlines(y, pd.Timestamp("1958-01-01"), df["date"], color=COLORS["grid"], lw=1.2)
        for idx, (_, row) in enumerate(df.iterrows()):
            color = trigger_color.get(_clean_cell(row["trigger_event_type"]), COLORS["blue"])
            ax.scatter(row["date"], idx, s=90, color=color, edgecolor="white", lw=1.0, zorder=3)
            ax.text(row["date"] + pd.Timedelta(days=180), idx, f"{row['crisis_id']}: {_short(row['trigger_event_type'], 44)}", va="center", fontsize=8.6, color=COLORS["ink"])
        ax.set_yticks([])
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(axis="x", color=COLORS["grid"], lw=0.8)
        ax.set_title("S1.1 Rupture Triggers Over Time", loc="left", fontsize=15, color=COLORS["ink"], weight="bold")
        ax.set_xlabel("Approximate trigger or rupture date", color=COLORS["ink"])
        ax.text(0, -0.12, "Each marker is a crisis. Labels name the coded trigger; marker-color legend is on page 2.", transform=ax.transAxes, fontsize=9, color=COLORS["muted"])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        _add_trigger_legend_page(pdf, "S1.1 Rupture Triggers Over Time", trigger_color)


def figure_frame_matrix(iqd: pd.DataFrame, iqf: pd.DataFrame, path: Path) -> None:
    frame_table = iqd.merge(iqf, on=["crisis_id", "crisis_label"], how="outer", suffixes=("_delegitimation", "_exception"))
    frame_groups = {
        "delegitimation": ["delegitimation_frames", "delegitimated_institutions", "supporting_keywords"],
        "popular\nsovereignty": ["delegitimation_frames", "justification_frames"],
        "necessity /\nexception": ["justification_frames", "exceptional_context_claimed"],
        "stability /\norder": ["justification_frames", "transition_or_order_language"],
        "democratic\nrestoration": ["actor_justification_types", "justification_frames"],
        "settlement\nlink": ["settlement_link"],
    }
    crises = frame_table["crisis_id"].tolist()
    matrix = np.zeros((len(crises), len(frame_groups)))
    labels = list(frame_groups)
    for i, (_, row) in enumerate(frame_table.iterrows()):
        row_text = " ".join(str(row.get(col, "")) for col in frame_table.columns).lower()
        for j, (label, fields) in enumerate(frame_groups.items()):
            text = " ".join(_clean_cell(row.get(field, "")) for field in fields).lower()
            if label.startswith("popular"):
                hit = any(term in text or term in row_text for term in ["people", "popular", "sovereignty", "street"])
            elif label.startswith("necessity"):
                hit = any(term in text for term in ["necessity", "exception", "emergency", "extra"])
            elif label.startswith("stability"):
                hit = any(term in text for term in ["stability", "order", "security", "return"])
            elif label.startswith("democratic"):
                hit = any(term in text for term in ["democracy", "democratic", "anti-authoritarian"])
            else:
                hit = bool(text)
            matrix[i, j] = 1 if hit else 0

    fig, ax = plt.subplots(figsize=(10.8, max(5.8, len(crises) * 0.38)))
    cmap = matplotlib.colors.ListedColormap(["#FFFFFF", COLORS["violet"]])
    ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(crises)))
    ax.set_yticklabels(crises, fontsize=8.5)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=8.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(crises), 1), minor=True)
    ax.grid(which="minor", color=COLORS["grid"], linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title("S1.1 Delegitimation and Exception Frames", loc="left", fontsize=15, color=COLORS["ink"], weight="bold")
    handles = [
        Patch(facecolor=COLORS["violet"], edgecolor=COLORS["violet"], label="coded frame present"),
        Patch(facecolor="#FFFFFF", edgecolor=COLORS["grid"], label="not coded in available evidence"),
    ]
    ax.legend(handles=handles, title="Cell legend", loc="upper right", bbox_to_anchor=(1.0, 1.09), frameon=False, fontsize=8.3, title_fontsize=8.6)
    ax.text(0, -0.15, "Filled cells mark coded frame presence in discourse or settlement-linked evidence.", transform=ax.transAxes, fontsize=9, color=COLORS["muted"])
    _save_pdf(fig, path)


def figure_legalization_chain(iqg: pd.DataFrame, iqh: pd.DataFrame, path: Path) -> None:
    chain = iqg.merge(iqh, on=["crisis_id", "crisis_label"], how="outer", suffixes=("_suspension", "_legalization"))
    chain["event_date"] = chain.apply(
        lambda r: _first_date(
            r.get("date_order_suspended_or_reinterpreted", ""),
            r.get("fact_creation_date", ""),
            r.get("legalization_date", ""),
        ),
        axis=1,
    )
    chain = chain.sort_values(["event_date", "crisis_id"], na_position="last").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, max(5.8, len(chain) * 0.38)))
    _style_axis(ax)
    y = np.arange(len(chain))
    ax.set_yticks(y)
    ax.set_yticklabels(chain["crisis_id"], fontsize=8.6)
    ax.invert_yaxis()
    ax.set_xlim(0, 3)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["extra-ordinary\nfact", "suspension /\nreinterpretation", "formal\ninstrument", "implementation /\nrecognition"], fontsize=9)
    ax.grid(axis="x", color=COLORS["grid"], lw=0.8)
    for i, (_, row) in enumerate(chain.iterrows()):
        points = []
        if _present(row.get("extra_constitutional_fact_created", "")) or _present(row.get("fact_creation_date", "")):
            points.append((0, COLORS["coral"]))
        if _present(row.get("classification", "")) or _present(row.get("date_order_suspended_or_reinterpreted", "")):
            points.append((1, COLORS["gold"]))
        if _present(row.get("legalization_instruments", "")) or _present(row.get("document_chain", "")):
            points.append((2, COLORS["blue"]))
        if _present(row.get("implementation_statuses", "")) or _present(row.get("validation_or_relegalization", "")):
            points.append((3, COLORS["green"]))
        if points:
            xs = [p[0] for p in points]
            ax.plot(xs, [i] * len(xs), color=COLORS["grid"], lw=1.4, zorder=1)
            for x, color in points:
                ax.scatter(x, i, s=90, color=color, edgecolor="white", lw=0.9, zorder=3)
        note = _short(row.get("classification", "") or row.get("legalization_instruments", "") or row.get("settlement_chain", ""), 60)
        ax.text(3.08, i, note, va="center", fontsize=7.8, color=COLORS["muted"], clip_on=False)
    ax.set_title("S1.1 Suspension, Reinterpretation, and Legalization Chain", loc="left", fontsize=15, color=COLORS["ink"], weight="bold")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["coral"], markeredgecolor="white", markersize=7, label="extra-ordinary fact"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["gold"], markeredgecolor="white", markersize=7, label="suspension/reinterpretation"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["blue"], markeredgecolor="white", markersize=7, label="formal instrument"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["green"], markeredgecolor="white", markersize=7, label="implementation/recognition"),
    ]
    legend = ax.legend(handles=handles, title="Stage legend", loc="upper center", bbox_to_anchor=(0.50, -0.10), ncol=4, frameon=True, fontsize=7.8, title_fontsize=8.2)
    legend.get_frame().set_edgecolor(COLORS["grid"])
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.94)
    ax.text(0, -0.24, "Dots show which stages are documented for each crisis; right notes name the coded legal move.", transform=ax.transAxes, fontsize=9, color=COLORS["muted"])
    _save_pdf(fig, path)


def build_figures(output: pd.DataFrame, figure_dir: Path) -> None:
    iqa = _clean_df(_records(output, "QE-S11-IQA"))
    iqb = _clean_df(_records(output, "QE-S11-IQB"))
    iqc = _clean_df(_records(output, "QE-S11-IQC"))
    iqd = _clean_df(_records(output, "QE-S11-IQD"))
    iqe = _clean_df(_records(output, "QE-S11-IQE"))
    iqf = _clean_df(_records(output, "QE-S11-IQF"))
    iqg = _clean_df(_records(output, "QE-S11-IQG"))
    iqh = _clean_df(_records(output, "QE-S11-IQH"))

    figure_timeline(iqb, iqa, figure_dir / "F_S11_1_constitutional_ruptures_timeline.pdf")
    figure_flow(iqa, iqc, figure_dir / "F_S11_2_formal_rule_to_decisive_arena_flow.pdf")
    figure_flow_all_pathways_matrix(iqa, iqc, figure_dir / "F_S11_2_B_formal_rule_to_decisive_arena_all_pathways_matrix.pdf")
    figure_heatmap(iqa, figure_dir / "F_S11_3_contested_rule_heatmap.pdf")
    figure_mechanism(figure_dir / "F_S11_4_hidden_constitution_mechanism.pdf")
    figure_resource_matrix(iqe, figure_dir / "F_S11_5_resource_substitution_matrix.pdf")
    figure_rupture_triggers(iqb, figure_dir / "F_S11_6_rupture_triggers_over_time.pdf")
    figure_frame_matrix(iqd, iqf, figure_dir / "F_S11_7_delegitimation_exception_frame_matrix.pdf")
    figure_legalization_chain(iqg, iqh, figure_dir / "F_S11_8_suspension_legalization_chain.pdf")


def main() -> None:
    paths = resolve_project_paths()
    output = _load_s11_output(paths.repo_root)
    table_dir = paths.repo_root / "outputs" / "tables" / "s11_tables"
    figure_dir = paths.repo_root / "outputs" / "figures" / "s11_figures"
    build_tables(output, table_dir)
    build_figures(output, figure_dir)
    print(f"Wrote S11 tables to {table_dir}")
    print(f"Wrote S11 figures to {figure_dir}")


if __name__ == "__main__":
    main()

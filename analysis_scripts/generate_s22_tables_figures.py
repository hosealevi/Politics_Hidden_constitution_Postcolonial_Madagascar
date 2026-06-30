#!/usr/bin/env python3
"""Create publication-oriented S2.2 tables and PDF figures."""

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
from matplotlib.patches import PathPatch, Rectangle

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.setup.load_project_data import load_project_data, resolve_project_paths, validate_output_crisis_ids


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

DOCUMENT_FORM_ORDER = [
    "signed_or_mediated_agreement",
    "roadmap",
    "transition_charter",
    "constitutional_court_decision",
    "ordinance_or_executive_transfer",
    "referendum_or_provisional_constitution",
    "consultation_process",
    "foundational_or_bilateral_settlement",
]

POWER_ORDER = [
    "power_sharing_or_controlled_inclusion",
    "authority_concentration_or_transfer",
    "symbolic_or_consultative_inclusion",
    "electoral_exit_formula",
    "power_distribution_not_explicitly_coded",
]

ACTOR_SECTOR_ORDER = [
    "state_elite",
    "opposition_elite",
    "military",
    "military_elite",
    "regional_organization",
    "international_organization",
    "foreign_state",
    "domestic_collective",
    "judicial_constitutional",
    "religious_actor",
    "civil_society",
    "transition_institution",
    "political_actor",
    "opposition",
    "legislature",
    "opposition_movement",
    "election_institution",
    "political_bloc",
]

ACTOR_ROLE_ORDER = [
    "claimant_or_signatory",
    "claimant_or_transition_actor",
    "coercive_or_transfer_actor",
    "recipient_or_transfer_actor",
    "mediator_or_guarantor",
    "mediator_observer_or_norm_enforcer",
    "constitutional_court_or_formalizer",
    "transition_authority",
    "opposition_actor",
    "foreign_state_counterparty",
    "participant_or_code_subject",
    "mediator_or_observer",
    "moral_or_religious_mediator",
    "consulted_actor_categories",
    "host_or_mediator",
    "participant_or_consulted_actor",
    "formal_state_actor",
    "incumbent_or_transition_actor",
    "transition_head_or_recipient",
    "constitutional_framework",
    "adopting_body",
    "incumbent_or_counterparty",
    "military_directorate_head",
    "military_directorate",
    "opposition_movement_signatory",
    "election_management_actor",
    "political_movement_bloc",
    "consensus_prime_minister",
    "proposal_author_or_opposition_bloc",
]

SECTOR_FAMILY_ORDER = [
    "state and transition institutions",
    "government and state elites",
    "opposition and political blocs",
    "military",
    "judicial and electoral institutions",
    "regional and international organizations",
    "foreign states",
    "religious and civil-society actors",
    "domestic collective actors",
]

ROLE_FAMILY_ORDER = [
    "claimant, signatory, or counterparty",
    "transfer, recipient, or transition authority",
    "mediator, guarantor, host, or observer",
    "constitutional or formal state actor",
    "opposition or political movement",
    "consulted or general participant",
    "military directorate role",
    "executive office role",
]

CONDITION_ORDER = [
    "elections_referendum_or_roadmap",
    "election_timetable",
    "transition_institution_design",
    "executive_power_sharing",
    "inclusive_government",
    "consensus_government",
    "electoral_framework",
    "eligibility",
    "leader_status_or_return",
    "amnesty_or_accountability",
    "claim_resolution",
    "political_conduct",
    "return_to_constitutional_order",
]

DURABILITY_ORDER = [
    "implemented_or_formally_recognized",
    "partial_or_process_needs_verification",
    "signed_but_failed_or_partially_implemented",
    "ongoing_or_draft",
    "not_adopted_or_failed_proposal",
]

LATER_EFFECT_ORDER = [
    "transition_to_new_constitution",
    "constitutional_amendment",
    "new constitution",
    "transition charter",
    "roadmap / election exit",
    "executive precedent",
    "refoundation / fifth republic",
]


def _repo_root() -> Path:
    return REPO_ROOT


def _load_s22_output(repo_root: Path) -> pd.DataFrame:
    path = repo_root / "outputs" / "evidence_synthesis" / "S22_all_evidence_synthesis_v1.csv"
    return pd.read_csv(path, dtype=str).fillna("")


def _records(output: pd.DataFrame, question_id: str) -> pd.DataFrame:
    row = output.loc[output["question_id"].eq(question_id)].iloc[0]
    structured = json.loads(row["structured_evidence"])
    df = pd.DataFrame(structured.get("records", [])).fillna("")
    df.insert(0, "question_id", question_id)
    return df


def _clean_cell(value: object) -> str:
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""
    parts = []
    for part in text.replace("|", ";").split(";"):
        cleaned = part.strip()
        if cleaned.lower() in {"", "nan", "none", "null"}:
            continue
        parts.append(cleaned)
    return "; ".join(dict.fromkeys(parts))


def _present(value: object) -> bool:
    return _clean_cell(value).lower() not in {"", "0", "0.0", "false", "no"}


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.map(_clean_cell)


def _split_values(value: object) -> list[str]:
    text = _clean_cell(value)
    if not text:
        return []
    return [
        part.strip()
        for part in text.replace("|", ";").split(";")
        if part.strip().lower() not in {"", "0", "0.0", "1", "1.0", "true", "false", "yes", "no", "present"}
    ]


def _label(value: object) -> str:
    return _clean_cell(value).replace("_", " ")


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


def _count_heatmap(ax: plt.Axes, matrix: np.ndarray, name: str) -> matplotlib.image.AxesImage:
    maximum = max(int(matrix.max()), 1)
    if maximum == 1:
        colors = ["#FFFFFF", "#B8D8C7"]
    else:
        positive = matplotlib.colors.LinearSegmentedColormap.from_list(
            f"{name}_positive", ["#CFE3DA", COLORS["green"], COLORS["blue"]]
        )(np.linspace(0, 1, maximum))
        colors = ["#FFFFFF", *positive]
    cmap = matplotlib.colors.ListedColormap(colors, name=name)
    norm = matplotlib.colors.BoundaryNorm(np.arange(-0.5, maximum + 1.5), cmap.N)
    return ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)


def _crisis_sort_key(crisis_id: str) -> tuple[int, str]:
    digits = "".join(ch for ch in str(crisis_id) if ch.isdigit())
    return (int(digits[:4]) if digits else 9999, str(crisis_id))


def _crises(records: pd.DataFrame) -> list[str]:
    return sorted(records["crisis_id"].dropna().unique().tolist(), key=_crisis_sort_key)


def _dedupe_bargains(df: pd.DataFrame) -> pd.DataFrame:
    return _clean_df(df.copy()).drop_duplicates(["crisis_id", "bargain_id"], keep="first")


def _tokens(row: pd.Series, field: str, order: list[str]) -> list[str]:
    values = _split_values(row.get(field, ""))
    return [token for token in order if token in values]


def _document_form_tokens(row: pd.Series) -> list[str]:
    return _tokens(row, "settlement_document_form", DOCUMENT_FORM_ORDER)


def _power_tokens(row: pd.Series) -> list[str]:
    value = _clean_cell(row.get("power_distribution_type", ""))
    return [value] if value in POWER_ORDER else []


def _actor_sector_tokens(row: pd.Series) -> list[str]:
    text = _clean_cell(row.get("participant_sectors", ""))
    return [token for token in ACTOR_SECTOR_ORDER if token in _split_values(text)]


def _actor_role_tokens(row: pd.Series) -> list[str]:
    text = _clean_cell(row.get("participant_roles", ""))
    return [token for token in ACTOR_ROLE_ORDER if token in _split_values(text)]


def _sector_family(token: str) -> str:
    mapping = {
        "state_elite": "government and state elites",
        "transition_institution": "state and transition institutions",
        "legislature": "state and transition institutions",
        "opposition_elite": "opposition and political blocs",
        "opposition": "opposition and political blocs",
        "opposition_movement": "opposition and political blocs",
        "political_actor": "opposition and political blocs",
        "political_bloc": "opposition and political blocs",
        "military": "military",
        "military_elite": "military",
        "judicial_constitutional": "judicial and electoral institutions",
        "election_institution": "judicial and electoral institutions",
        "regional_organization": "regional and international organizations",
        "international_organization": "regional and international organizations",
        "foreign_state": "foreign states",
        "religious_actor": "religious and civil-society actors",
        "civil_society": "religious and civil-society actors",
        "domestic_collective": "domestic collective actors",
    }
    return mapping[token]


def _role_family(token: str) -> str:
    if token in {
        "claimant_or_signatory", "claimant_or_transition_actor", "foreign_state_counterparty",
        "incumbent_or_counterparty", "opposition_movement_signatory", "proposal_author_or_opposition_bloc",
    }:
        return "claimant, signatory, or counterparty"
    if token in {
        "coercive_or_transfer_actor", "recipient_or_transfer_actor", "transition_authority",
        "incumbent_or_transition_actor", "transition_head_or_recipient",
    }:
        return "transfer, recipient, or transition authority"
    if token in {
        "mediator_or_guarantor", "mediator_observer_or_norm_enforcer", "mediator_or_observer",
        "moral_or_religious_mediator", "host_or_mediator",
    }:
        return "mediator, guarantor, host, or observer"
    if token in {
        "constitutional_court_or_formalizer", "constitutional_framework", "adopting_body",
        "formal_state_actor", "election_management_actor",
    }:
        return "constitutional or formal state actor"
    if token in {"opposition_actor", "political_movement_bloc"}:
        return "opposition or political movement"
    if token in {"participant_or_code_subject", "consulted_actor_categories", "participant_or_consulted_actor"}:
        return "consulted or general participant"
    if token in {"military_directorate_head", "military_directorate"}:
        return "military directorate role"
    if token == "consensus_prime_minister":
        return "executive office role"
    raise ValueError(f"Unmapped actor role token: {token}")


def _record_label(row: pd.Series) -> str:
    return f"{row.get('crisis_id', '')} / {row.get('bargain_id', '')}"


def _observed_tokens(df: pd.DataFrame, field: str, preferred_order: list[str] | None = None) -> list[str]:
    observed = {token for value in df[field] for token in _split_values(value)}
    ordered = [token for token in (preferred_order or []) if token in observed]
    return ordered + sorted(observed - set(ordered))


def _condition_tokens(row: pd.Series) -> list[str]:
    text = " ".join([_clean_cell(row.get("condition_summary", "")), _clean_cell(row.get("settlement_effect", ""))]).lower()
    tokens = []
    tests = {
        "elections_referendum_or_roadmap": ["elections_referendum_or_roadmap", "referendum", "roadmap"],
        "election_timetable": ["election_timetable", "timetable", "election"],
        "transition_institution_design": ["transition_institution_design", "institution"],
        "executive_power_sharing": ["executive_power_sharing", "shared executive", "power-sharing"],
        "inclusive_government": ["inclusive_government", "inclusive"],
        "consensus_government": ["consensus_government", "consensus"],
        "electoral_framework": ["electoral_framework", "electoral legal framework"],
        "eligibility": ["eligibility", "candidate"],
        "leader_status_or_return": ["leader_status_or_return", "return/status"],
        "amnesty_or_accountability": ["amnesty_or_accountability", "charges", "prosecutions"],
        "claim_resolution": ["claim_resolution", "competing presidential claims"],
        "political_conduct": ["political_conduct", "ethical"],
        "return_to_constitutional_order": ["return_to_constitutional_order", "constitutional order"],
    }
    for label, needles in tests.items():
        if any(needle in text for needle in needles):
            tokens.append(label)
    return tokens


def _durability_token(row: pd.Series) -> str:
    values = _split_values(row.get("durability_status", ""))
    for token in DURABILITY_ORDER:
        if token in values:
            return token
    return values[0] if values else ""


def _later_effect_tokens(row: pd.Series) -> list[str]:
    text = " ".join([_clean_cell(row.get("later_effect", "")), _clean_cell(row.get("constitutional_context", ""))]).lower()
    tests = {
        "transition_to_new_constitution": ["transition_to_new_constitution"],
        "constitutional_amendment": ["constitutional_amendment", "amendment"],
        "new constitution": ["new constitution", "creates third republic", "creates fourth republic", "fifth republic", "constitution-making"],
        "transition charter": ["transition_charter", "charter"],
        "roadmap / election exit": ["roadmap", "election", "referendum"],
        "executive precedent": ["executive", "head-of-state", "prime minister", "authority"],
        "refoundation / fifth republic": ["refoundation", "fifth republic"],
    }
    return [label for label in LATER_EFFECT_ORDER if any(needle in text for needle in tests[label])]


def _base_cols(extra: list[str]) -> list[str]:
    return [
        "crisis_id",
        "crisis_name",
        "bargain_id",
        "bargain_date",
        "bargain_name",
        "document_title",
        "document_type",
        "bargain_type",
        *extra,
        "settlement_effect",
        "formalization_sequence",
        "constitutional_context",
        "discourse_evidence",
        "source_ids",
    ]


def _select(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[[col for col in cols if col in df.columns]].copy()


def _add_status(df: pd.DataFrame, field: str, status_col: str) -> pd.DataFrame:
    result = df.copy()
    result[status_col] = result[field].map(lambda value: "coded" if _present(value) else "not coded in current source tables")
    return result


def _plot_data(records: pd.DataFrame) -> pd.DataFrame:
    df = _dedupe_bargains(records)
    rows: list[dict[str, object]] = []

    def add(chart: str, row_key: str, column_key: str, matched: pd.DataFrame) -> None:
        if matched.empty:
            return
        rows.append(
            {
                "figure_id": chart,
                "row_key": row_key,
                "column_key": column_key,
                "distinct_settlement_records": len(matched),
                "evidence_ids": "; ".join(matched.apply(_record_label, axis=1)),
            }
        )

    for crisis in _crises(df):
        crisis_df = df.loc[df["crisis_id"].eq(crisis)]
        for token in DOCUMENT_FORM_ORDER:
            add("F_S22_1", crisis, token, crisis_df.loc[crisis_df.apply(lambda row: token in _document_form_tokens(row), axis=1)])
        for token in POWER_ORDER:
            add("F_S22_4", crisis, token, crisis_df.loc[crisis_df.apply(lambda row: token in _power_tokens(row), axis=1)])
        for token in CONDITION_ORDER:
            add("F_S22_6", crisis, token, crisis_df.loc[crisis_df.apply(lambda row: token in _condition_tokens(row), axis=1)])
        for token in LATER_EFFECT_ORDER:
            add("F_S22_8", crisis, token, crisis_df.loc[crisis_df.apply(lambda row: token in _later_effect_tokens(row), axis=1)])

    for form in DOCUMENT_FORM_ORDER:
        form_df = df.loc[df.apply(lambda row: form in _document_form_tokens(row), axis=1)]
        for power in POWER_ORDER:
            add("F_S22_2", form, power, form_df.loc[form_df.apply(lambda row: power in _power_tokens(row), axis=1)])

    for sector in ACTOR_SECTOR_ORDER:
        sector_df = df.loc[df.apply(lambda row: sector in _actor_sector_tokens(row), axis=1)]
        for role in ACTOR_ROLE_ORDER:
            add("F_S22_3_B", sector, role, sector_df.loc[sector_df.apply(lambda row: role in _actor_role_tokens(row), axis=1)])
    return pd.DataFrame(rows)


def _category_dictionary() -> pd.DataFrame:
    rows = []
    families = [
        ("F_S22_1", "settlement_document_form", DOCUMENT_FORM_ORDER, "classified from document_type, bargain_type, and formalization fields"),
        ("F_S22_2/F_S22_4", "power_distribution_type", POWER_ORDER, "classified from coded power-sharing, authority-transfer, office, and actor-role fields"),
        ("F_S22_3_B", "participant_sectors", ACTOR_SECTOR_ORDER, "raw DB_Bargain_Actors actor_sector code"),
        ("F_S22_3_B", "participant_roles", ACTOR_ROLE_ORDER, "raw DB_Bargain_Actors actor_role code"),
        ("F_S22_6", "condition_family", CONDITION_ORDER, "rule-based family derived from coded condition types and text"),
        ("F_S22_7", "durability_status", DURABILITY_ORDER, "implementation and durability status from settlement records"),
        ("F_S22_8", "later_effect_family", LATER_EFFECT_ORDER, "rule-based family derived from later_effect and constitutional_context"),
    ]
    for figure_id, field, values, derivation in families:
        for value in values:
            rows.append({"figure_id": figure_id, "encoded_field": field, "category_code": value, "display_label": _label(value), "derivation": derivation})
    for value in SECTOR_FAMILY_ORDER:
        rows.append({"figure_id": "F_S22_3", "encoded_field": "sector_family", "category_code": value, "display_label": value, "derivation": "analytical grouping of raw actor_sector codes; raw matrix retained in F_S22_3_B"})
    for value in ROLE_FAMILY_ORDER:
        rows.append({"figure_id": "F_S22_3", "encoded_field": "role_family", "category_code": value, "display_label": value, "derivation": "analytical grouping of raw actor_role codes; raw matrix retained in F_S22_3_B"})
    return pd.DataFrame(rows)


def build_tables(output: pd.DataFrame, table_dir: Path) -> dict[str, pd.DataFrame]:
    iqa = _dedupe_bargains(_records(output, "QE-S22-IQA"))
    iqb = _dedupe_bargains(_records(output, "QE-S22-IQB"))
    iqc = _dedupe_bargains(_records(output, "QE-S22-IQC"))
    iqd = _dedupe_bargains(_records(output, "QE-S22-IQD"))
    iqe = _dedupe_bargains(_records(output, "QE-S22-IQE"))
    iqf = _dedupe_bargains(_records(output, "QE-S22-IQF"))
    iqg = _dedupe_bargains(_records(output, "QE-S22-IQG"))
    iqh = _dedupe_bargains(_records(output, "QE-S22-IQH"))

    tables = {
        "T_S22_1_settlement_document_typology.csv": _select(iqa, _base_cols(["settlement_document_form", "durability_status", "implementation_status"])),
        "T_S22_2_actor_roles_inclusion.csv": _select(_add_status(iqb, "excluded_or_limited_actors", "exclusion_evidence_status"), _base_cols(["participant_actor_names", "participant_roles", "participant_sectors", "mediators_or_guarantors", "excluded_or_limited_actors", "exclusion_evidence_status"])),
        "T_S22_3_executive_power_distribution.csv": _select(_add_status(iqc, "cabinet_or_office_allocation", "allocation_evidence_status"), _base_cols(["cabinet_or_office_allocation", "allocation_evidence_status", "power_distribution_type"])),
        "T_S22_4_power_sharing_concentration.csv": _select(iqd, _base_cols(["power_distribution_type", "participant_actor_names", "cabinet_or_office_allocation"])),
        "T_S22_5_informal_to_formal_sequences.csv": _select(iqe, _base_cols(["settlement_document_form", "formalization_sequence"])),
        "T_S22_6_conditions_timetables_limits.csv": _select(iqf, _base_cols(["condition_summary", "implementation_status"])),
        "T_S22_7_settlement_durability.csv": _select(iqg, _base_cols(["durability_status", "implementation_status", "later_effect"])),
        "T_S22_8_later_constitutional_effects.csv": _select(iqh, _base_cols(["later_effect", "durability_status"])),
    }

    index = _dedupe_bargains(iqa)
    index["document_form_tokens"] = index.apply(lambda row: "; ".join(_document_form_tokens(row)), axis=1)
    index["condition_tokens"] = index.apply(lambda row: "; ".join(_condition_tokens(row)), axis=1)
    index["later_effect_tokens"] = index.apply(lambda row: "; ".join(_later_effect_tokens(row)), axis=1)
    tables["T_S22_9_settlement_source_anchor_index.csv"] = _select(
        index,
        _base_cols(["document_form_tokens", "power_distribution_type", "condition_tokens", "later_effect_tokens", "participant_actor_names", "mediators_or_guarantors", "durability_status"]),
    )
    tables["T_S22_10_figure_plot_data.csv"] = _plot_data(index)
    tables["T_S22_11_figure_category_dictionary.csv"] = _category_dictionary()

    for name, df in tables.items():
        _save_csv(df, table_dir / name)
    return tables


def figure_document_form_matrix(records: pd.DataFrame, path: Path) -> None:
    df = _dedupe_bargains(records)
    crises = _crises(df)
    matrix = np.zeros((len(crises), len(DOCUMENT_FORM_ORDER)))
    for _, row in df.iterrows():
        for token in _document_form_tokens(row):
            matrix[crises.index(row["crisis_id"]), DOCUMENT_FORM_ORDER.index(token)] += 1
    fig, ax = plt.subplots(figsize=(12, max(5.8, len(crises) * 0.43)))
    image = _count_heatmap(ax, matrix, "s22_docs")
    _add_count_colorbar(fig, ax, image, "Distinct settlement records", matrix.max())
    ax.set_yticks(np.arange(len(crises)))
    ax.set_yticklabels(crises, fontsize=8.8)
    ax.set_xticks(np.arange(len(DOCUMENT_FORM_ORDER)))
    ax.set_xticklabels([_wrap(label.replace("_", " "), 16) for label in DOCUMENT_FORM_ORDER], rotation=35, ha="right", fontsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(DOCUMENT_FORM_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(crises), 1), minor=True)
    ax.grid(which="minor", color=COLORS["grid"], lw=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title("S2.2 Settlement Document and Process Forms by Crisis", loc="left", fontsize=15, color=COLORS["ink"], weight="bold")
    ax.text(0, -0.26, "Shading counts distinct settlement/process records. White means no coded settlement of that form.", transform=ax.transAxes, fontsize=9, color=COLORS["muted"])
    _save_pdf(fig, path)


def figure_record_category_matrix(
    records: pd.DataFrame,
    path: Path,
    categories: list[str],
    token_function,
    title: str,
    note: str,
) -> None:
    df = _dedupe_bargains(records).copy()
    df["date"] = pd.to_datetime(df["bargain_date"], errors="coerce")
    df = df.sort_values(["date", "crisis_id", "bargain_id"], na_position="last")
    matrix = np.zeros((len(df), len(categories)))
    for row_index, (_, row) in enumerate(df.iterrows()):
        for token in token_function(row):
            if token in categories:
                matrix[row_index, categories.index(token)] = 1
    fig, ax = plt.subplots(figsize=(max(12.0, len(categories) * 0.58 + 5.2), max(7.2, len(df) * 0.26 + 2.5)))
    image = _count_heatmap(ax, matrix, path.stem)
    _add_count_colorbar(fig, ax, image, "Coded for settlement record (0/1)", matrix.max())
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df.apply(_record_label, axis=1), fontsize=7.1)
    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels([_wrap(_label(value), 17) for value in categories], rotation=40, ha="right", fontsize=7.4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(categories), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(df), 1), minor=True)
    ax.grid(which="minor", color=COLORS["grid"], lw=0.65)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title(title, loc="left", fontsize=15, color=COLORS["ink"], weight="bold")
    ax.text(0, -0.22, note, transform=ax.transAxes, fontsize=9, color=COLORS["muted"])
    _save_pdf(fig, path)


def _raw_condition_tokens(row: pd.Series) -> list[str]:
    return [token for token in _split_values(row.get("condition_summary", "")) if "_" in token and " " not in token]


def _raw_later_effect_tokens(row: pd.Series) -> list[str]:
    return [
        token
        for token in _split_values(row.get("later_effect", ""))
        if ("_" in token and " " not in token) or token in {"constitution"}
    ]


def figure_document_to_power_flow(records: pd.DataFrame, path: Path) -> None:
    df = _dedupe_bargains(records)
    edges = Counter()
    examples = defaultdict(list)
    for _, row in df.iterrows():
        for form in _document_form_tokens(row) or ["form not explicit"]:
            for power in _power_tokens(row) or ["power type not explicit"]:
                edges[(form, power)] += 1
                examples[(form, power)].append(f"{row['crisis_id']}/{row['bargain_id']}")
    display_edges = Counter(dict(edges.most_common(12)))
    left_nodes = [node for node in DOCUMENT_FORM_ORDER if any(edge[0] == node for edge in display_edges)]
    right_nodes = [node for node in POWER_ORDER if any(edge[1] == node for edge in display_edges)]
    palette = [COLORS["blue"], COLORS["green"], COLORS["coral"], COLORS["violet"], COLORS["gold"], COLORS["teal"], "#8E6E53", "#78909C"]
    form_colors = {form: palette[idx] for idx, form in enumerate(DOCUMENT_FORM_ORDER)}
    max_count = max(edges.values()) if edges else 1
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path, metadata={"Creator": "Codex evidence synthesis"}) as pdf:
        fig, ax = _new_page((13, 8.2))
        ax.text(0.03, 0.965, "S2.2 From Settlement Form to Power Distribution", fontsize=16, color=COLORS["ink"], weight="bold", va="top")
        ax.text(0.03, 0.92, "Main map shows the 12 most repeated form-to-power pathways. Exhaustive matrix is in the _B figure.", fontsize=9.5, color=COLORS["muted"])
        ax.text(0.03, 0.887, "Color = settlement form (matched by left-node strip). Line width = distinct settlement records.", fontsize=9.2, color=COLORS["muted"])
        ax.text(0.07, 0.855, "Settlement form", fontsize=10.8, color=COLORS["ink"], weight="bold")
        ax.text(0.67, 0.855, "Power distribution", fontsize=10.8, color=COLORS["ink"], weight="bold")
        left_y = {node: 0.78 - i * (0.66 / max(len(left_nodes) - 1, 1)) for i, node in enumerate(left_nodes)}
        right_y = {node: 0.78 - i * (0.66 / max(len(right_nodes) - 1, 1)) for i, node in enumerate(right_nodes)}
        for (form, power), count in display_edges.items():
            if form not in left_y or power not in right_y:
                continue
            verts = [(0.34, left_y[form]), (0.48, left_y[form]), (0.54, right_y[power]), (0.66, right_y[power])]
            path_obj = MplPath(verts, [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
            ax.add_patch(PathPatch(path_obj, facecolor="none", edgecolor=form_colors[form], lw=0.8 + 4.0 * count / max_count, alpha=0.45, zorder=1))
        for node, yy in left_y.items():
            ax.add_patch(Rectangle((0.04, yy - 0.030), 0.30, 0.060, facecolor=COLORS["pale"], edgecolor=COLORS["grid"], lw=1.1, zorder=3))
            ax.add_patch(Rectangle((0.047, yy - 0.023), 0.013, 0.046, facecolor=form_colors[node], edgecolor=form_colors[node], lw=0.8, zorder=4))
            ax.text(0.067, yy, _wrap(node.replace("_", " "), 25), va="center", fontsize=8.3, color=COLORS["ink"], zorder=4)
        for node, yy in right_y.items():
            ax.add_patch(Rectangle((0.66, yy - 0.030), 0.31, 0.060, facecolor="#F7FAF8", edgecolor=COLORS["grid"], lw=1.1, zorder=3))
            ax.text(0.675, yy, _wrap(node.replace("_", " "), 29), va="center", fontsize=8.3, color=COLORS["ink"], zorder=4)
        ax.text(0.79, 0.902, "Line width", fontsize=8.5, color=COLORS["ink"], weight="bold", ha="left")
        for legend_idx, value in enumerate(sorted({1, max_count})):
            legend_y = 0.884 - legend_idx * 0.020
            ax.plot(
                [0.79, 0.84],
                [legend_y, legend_y],
                color=COLORS["muted"],
                lw=0.8 + 4.0 * value / max_count,
                alpha=0.6,
                solid_capstyle="round",
            )
            ax.text(
                0.845,
                legend_y,
                f"{value} record" + ("" if value == 1 else "s"),
                fontsize=8.0,
                color=COLORS["muted"],
                va="center",
                ha="left",
            )
        ax.text(0.04, 0.045, "Reading note: power sharing and concentration are classifications of settlement effects, not normative scores.", fontsize=8.8, color=COLORS["muted"])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _new_page((13, 8.2))
        ax.text(0.04, 0.96, "S2.2 Flow Figure - Settlement Examples", fontsize=15, color=COLORS["ink"], weight="bold", va="top")
        ax.text(0.04, 0.915, "Examples preserve crisis and bargain IDs. Swatch color matches settlement form. Full source IDs are in T_S22_1, T_S22_3, and T_S22_9.", fontsize=9.5, color=COLORS["muted"])
        y = 0.84
        for (form, power), count in edges.most_common(12):
            color = form_colors[form]
            ax.add_patch(Rectangle((0.045, y - 0.018), 0.012, 0.036, facecolor=color, edgecolor=color))
            ax.text(0.07, y + 0.012, f"{form.replace('_', ' ')} -> {power.replace('_', ' ')}", fontsize=9.0, color=COLORS["ink"], weight="bold", va="center")
            example_text = _wrap("Examples: " + "; ".join(dict.fromkeys(examples[(form, power)][:7])), 136)
            ax.text(0.07, y - 0.020, example_text, fontsize=8.0, color=COLORS["muted"], va="top")
            ax.text(0.92, y, f"records: {count}", fontsize=8.5, color=COLORS["muted"], va="center", ha="right")
            y -= 0.071 + 0.018 * example_text.count("\n")
            if y < 0.08:
                break
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def figure_document_to_power_all_matrix(records: pd.DataFrame, path: Path) -> None:
    df = _dedupe_bargains(records)
    matrix = np.zeros((len(DOCUMENT_FORM_ORDER), len(POWER_ORDER)))
    examples = defaultdict(list)
    for _, row in df.iterrows():
        for form in _document_form_tokens(row):
            for power in _power_tokens(row):
                matrix[DOCUMENT_FORM_ORDER.index(form), POWER_ORDER.index(power)] += 1
                examples[(form, power)].append(f"{row['crisis_id']}/{row['bargain_id']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path, metadata={"Creator": "Codex evidence synthesis"}) as pdf:
        fig, ax = plt.subplots(figsize=(11.5, 7.6))
        image = _count_heatmap(ax, matrix, "s22_doc_power_all")
        _add_count_colorbar(fig, ax, image, "Distinct settlement records", matrix.max())
        ax.set_yticks(np.arange(len(DOCUMENT_FORM_ORDER)))
        ax.set_yticklabels([_wrap(label.replace("_", " "), 26) for label in DOCUMENT_FORM_ORDER], fontsize=8.5)
        ax.set_xticks(np.arange(len(POWER_ORDER)))
        ax.set_xticklabels([_wrap(label.replace("_", " "), 20) for label in POWER_ORDER], rotation=35, ha="right", fontsize=8.2)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(POWER_ORDER), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(DOCUMENT_FORM_ORDER), 1), minor=True)
        ax.grid(which="minor", color=COLORS["grid"], lw=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_title("S2.2_B Exhaustive Settlement-Form to Power-Distribution Matrix", loc="left", fontsize=15, color=COLORS["ink"], weight="bold")
        ax.text(0, -0.27, "Every nonzero cell is a coded settlement-form to power-distribution edge. Counts are settlement records.", transform=ax.transAxes, fontsize=9, color=COLORS["muted"])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _new_page((11.7, 8.3))
        ax.text(0.04, 0.96, "S2.2_B Exhaustive Settlement-Form Examples", fontsize=15, color=COLORS["ink"], weight="bold", va="top")
        ax.text(0.04, 0.915, "Nonzero matrix cells with attached crisis/bargain IDs.", fontsize=9.4, color=COLORS["muted"])
        y = 0.84
        for (form, power), vals in sorted(examples.items(), key=lambda item: (-len(item[1]), item[0][0], item[0][1])):
            ax.text(0.055, y, f"{form.replace('_', ' ')} -> {power.replace('_', ' ')}", fontsize=8.8, color=COLORS["ink"], weight="bold", va="center")
            example_text = _wrap("Examples: " + "; ".join(dict.fromkeys(vals)), 132)
            ax.text(0.055, y - 0.024, example_text, fontsize=7.7, color=COLORS["muted"], va="top")
            y -= 0.064 + 0.017 * example_text.count("\n")
            if y < 0.08:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                fig, ax = _new_page((11.7, 8.3))
                ax.text(0.04, 0.96, "S2.2_B Exhaustive Settlement-Form Examples Continued", fontsize=15, color=COLORS["ink"], weight="bold", va="top")
                y = 0.88
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def figure_actor_role_sector_matrix(records: pd.DataFrame, path: Path) -> None:
    df = _dedupe_bargains(records)
    matrix = np.zeros((len(SECTOR_FAMILY_ORDER), len(ROLE_FAMILY_ORDER)))
    for _, row in df.iterrows():
        sectors = {_sector_family(token) for token in _actor_sector_tokens(row)}
        roles = {_role_family(token) for token in _actor_role_tokens(row)}
        for sector in sectors:
            for role in roles:
                matrix[SECTOR_FAMILY_ORDER.index(sector), ROLE_FAMILY_ORDER.index(role)] += 1
    fig, ax = plt.subplots(figsize=(12.2, 8.0))
    image = _count_heatmap(ax, matrix, "s22_actor_roles")
    _add_count_colorbar(fig, ax, image, "Distinct settlement records", matrix.max())
    ax.set_yticks(np.arange(len(SECTOR_FAMILY_ORDER)))
    ax.set_yticklabels([_wrap(label, 25) for label in SECTOR_FAMILY_ORDER], fontsize=8.6)
    ax.set_xticks(np.arange(len(ROLE_FAMILY_ORDER)))
    ax.set_xticklabels([_wrap(label, 20) for label in ROLE_FAMILY_ORDER], rotation=35, ha="right", fontsize=8.1)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(ROLE_FAMILY_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(SECTOR_FAMILY_ORDER), 1), minor=True)
    ax.grid(which="minor", color=COLORS["grid"], lw=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title("S2.2 Settlement Actor Families and Functions", loc="left", fontsize=15, color=COLORS["ink"], weight="bold")
    ax.text(0, -0.24, "Families support comparison; F_S22_3_B retains all 18 raw sectors and 29 raw roles. Co-occurrence does not imply equal power.", transform=ax.transAxes, fontsize=9, color=COLORS["muted"])
    _save_pdf(fig, path)


def figure_actor_role_sector_raw_matrix(records: pd.DataFrame, path: Path) -> None:
    df = _dedupe_bargains(records)
    sectors = _observed_tokens(df, "participant_sectors", ACTOR_SECTOR_ORDER)
    roles = _observed_tokens(df, "participant_roles", ACTOR_ROLE_ORDER)
    matrix = np.zeros((len(sectors), len(roles)))
    for _, row in df.iterrows():
        for sector in _split_values(row.get("participant_sectors", "")):
            for role in _split_values(row.get("participant_roles", "")):
                matrix[sectors.index(sector), roles.index(role)] += 1
    fig, ax = plt.subplots(figsize=(18.5, 10.5))
    image = _count_heatmap(ax, matrix, "s22_actor_roles_raw")
    _add_count_colorbar(fig, ax, image, "Distinct settlement records", matrix.max())
    ax.set_yticks(np.arange(len(sectors)))
    ax.set_yticklabels([_wrap(_label(value), 24) for value in sectors], fontsize=7.8)
    ax.set_xticks(np.arange(len(roles)))
    ax.set_xticklabels([_wrap(_label(value), 17) for value in roles], rotation=45, ha="right", fontsize=6.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(roles), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(sectors), 1), minor=True)
    ax.grid(which="minor", color=COLORS["grid"], lw=0.65)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title("S2.2_B Exhaustive Raw Actor-Sector by Actor-Role Matrix", loc="left", fontsize=15, color=COLORS["ink"], weight="bold")
    ax.text(0, -0.27, "All observed DB_Bargain_Actors sector and role codes are retained. White = 0; pale green = low nonzero; blue = highest count.", transform=ax.transAxes, fontsize=9, color=COLORS["muted"])
    _save_pdf(fig, path)


def figure_power_distribution_matrix(records: pd.DataFrame, path: Path) -> None:
    df = _dedupe_bargains(records)
    crises = _crises(df)
    matrix = np.zeros((len(crises), len(POWER_ORDER)))
    for _, row in df.iterrows():
        for token in _power_tokens(row):
            matrix[crises.index(row["crisis_id"]), POWER_ORDER.index(token)] += 1
    fig, ax = plt.subplots(figsize=(11.5, max(5.6, len(crises) * 0.42)))
    image = _count_heatmap(ax, matrix, "s22_power")
    _add_count_colorbar(fig, ax, image, "Distinct settlement records", matrix.max())
    ax.set_yticks(np.arange(len(crises)))
    ax.set_yticklabels(crises, fontsize=8.8)
    ax.set_xticks(np.arange(len(POWER_ORDER)))
    ax.set_xticklabels([_wrap(label.replace("_", " "), 18) for label in POWER_ORDER], rotation=30, ha="right", fontsize=8.3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(POWER_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(crises), 1), minor=True)
    ax.grid(which="minor", color=COLORS["grid"], lw=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title("S2.2 Power Sharing and Authority Concentration by Crisis", loc="left", fontsize=15, color=COLORS["ink"], weight="bold")
    ax.text(0, -0.22, "Shading marks settlement records by power-distribution classification.", transform=ax.transAxes, fontsize=9, color=COLORS["muted"])
    _save_pdf(fig, path)


def figure_formalization_sequence(records: pd.DataFrame, path: Path) -> None:
    df = _dedupe_bargains(records)
    df["date"] = pd.to_datetime(df["bargain_date"], errors="coerce")
    df = df.sort_values(["date", "crisis_id", "bargain_id"], na_position="last").reset_index(drop=True)
    stages = ["informal / political bargain", "settlement document", "constitutional/legal form", "implemented / later effect"]
    palette = [COLORS["coral"], COLORS["gold"], COLORS["blue"], COLORS["green"]]
    fig, ax = plt.subplots(figsize=(12.3, max(6.0, len(df) * 0.24)))
    _style_axis(ax)
    y = np.arange(len(df))
    ax.set_yticks(y)
    ax.set_yticklabels((df["crisis_id"] + " / " + df["bargain_id"]).tolist(), fontsize=6.9)
    ax.invert_yaxis()
    ax.set_xlim(-0.5, len(stages) - 0.5)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels([_wrap(stage, 18) for stage in stages], fontsize=8.6)
    ax.grid(axis="x", color=COLORS["grid"], lw=0.8)
    for i, (_, row) in enumerate(df.iterrows()):
        present = []
        if _present(row.get("bargain_type", "")) or _present(row.get("participant_actor_names", "")):
            present.append(0)
        if _present(row.get("document_type", "")) or _present(row.get("settlement_document_form", "")):
            present.append(1)
        if _present(row.get("constitutional_context", "")) or _present(row.get("formalization_sequence", "")):
            present.append(2)
        if _present(row.get("implementation_status", "")) or _present(row.get("later_effect", "")):
            present.append(3)
        if present:
            ax.plot(present, [i] * len(present), color=COLORS["grid"], lw=1.0, zorder=1)
            for idx in present:
                ax.scatter(idx, i, s=54, color=palette[idx], edgecolor="white", lw=0.7, zorder=3)
        ax.text(len(stages) - 0.25, i, _short(row.get("bargain_name", ""), 58), va="center", fontsize=6.5, color=COLORS["muted"], clip_on=False)
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=palette[idx], markeredgecolor="white", markersize=7, label=stage) for idx, stage in enumerate(stages)]
    legend = ax.legend(handles=handles, title="Stage legend", loc="upper center", bbox_to_anchor=(0.50, -0.08), ncol=4, frameon=True, fontsize=7.5, title_fontsize=8.0)
    legend.get_frame().set_edgecolor(COLORS["grid"])
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.94)
    ax.set_title("S2.2 Informal Bargain to Formal Authority Sequence", loc="left", fontsize=15, color=COLORS["ink"], weight="bold")
    ax.text(0, -0.20, "Dots show which sequence stages are documented for each settlement; right notes identify the bargain.", transform=ax.transAxes, fontsize=9, color=COLORS["muted"])
    _save_pdf(fig, path)


def figure_conditions_matrix(records: pd.DataFrame, path: Path) -> None:
    df = _dedupe_bargains(records)
    crises = _crises(df)
    matrix = np.zeros((len(crises), len(CONDITION_ORDER)))
    examples = defaultdict(list)
    for _, row in df.iterrows():
        for token in _condition_tokens(row):
            matrix[crises.index(row["crisis_id"]), CONDITION_ORDER.index(token)] += 1
            examples[token].append(f"{row['crisis_id']}/{row['bargain_id']}: {_short(row.get('condition_summary', ''), 70)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path, metadata={"Creator": "Codex evidence synthesis"}) as pdf:
        fig, ax = plt.subplots(figsize=(12.2, max(5.6, len(crises) * 0.42)))
        image = _count_heatmap(ax, matrix, "s22_conditions")
        _add_count_colorbar(fig, ax, image, "Distinct settlement records", matrix.max())
        ax.set_yticks(np.arange(len(crises)))
        ax.set_yticklabels(crises, fontsize=8.8)
        ax.set_xticks(np.arange(len(CONDITION_ORDER)))
        ax.set_xticklabels([_wrap(label.replace("_", " "), 15) for label in CONDITION_ORDER], rotation=35, ha="right", fontsize=7.8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(CONDITION_ORDER), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(crises), 1), minor=True)
        ax.grid(which="minor", color=COLORS["grid"], lw=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_title("S2.2 Settlement Conditions, Timetables, and Limits", loc="left", fontsize=15, color=COLORS["ink"], weight="bold")
        ax.text(0, -0.27, "Cells count settlement records carrying each condition family. Examples are on page 2.", transform=ax.transAxes, fontsize=9, color=COLORS["muted"])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _new_page((11.7, 8.3))
        ax.text(0.04, 0.96, "S2.2 Condition Examples", fontsize=15, color=COLORS["ink"], weight="bold", va="top")
        ax.text(0.04, 0.915, "Examples preserve crisis/bargain IDs. Full condition and source detail is in T_S22_6.", fontsize=9.5, color=COLORS["muted"])
        y = 0.84
        for token in CONDITION_ORDER:
            vals = list(dict.fromkeys(examples.get(token, [])))[:6]
            if not vals:
                continue
            ax.text(0.055, y, token.replace("_", " "), fontsize=8.8, color=COLORS["ink"], weight="bold", va="center")
            example_text = _wrap("Examples: " + "; ".join(vals), 132)
            ax.text(0.055, y - 0.024, example_text, fontsize=7.7, color=COLORS["muted"], va="top")
            y -= 0.068 + 0.017 * example_text.count("\n")
            if y < 0.08:
                break
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def figure_durability_timeline(records: pd.DataFrame, path: Path) -> None:
    df = _dedupe_bargains(records)
    df["date"] = pd.to_datetime(df["bargain_date"], errors="coerce")
    df = df.loc[pd.notna(df["date"])].sort_values("date")
    color_map = {
        "implemented_or_formally_recognized": COLORS["blue"],
        "partial_or_process_needs_verification": COLORS["green"],
        "signed_but_failed_or_partially_implemented": COLORS["gold"],
        "ongoing_or_draft": COLORS["teal"],
        "not_adopted_or_failed_proposal": COLORS["coral"],
    }
    fig, ax = plt.subplots(figsize=(12.4, max(5.8, len(df) * 0.28)))
    _style_axis(ax)
    y = np.arange(len(df))
    ax.hlines(y, df["date"] - pd.Timedelta(days=95), df["date"] + pd.Timedelta(days=95), color=COLORS["grid"], lw=1.2)
    for idx, (_, row) in enumerate(df.iterrows()):
        status = _durability_token(row)
        ax.scatter(row["date"], idx, s=72, color=color_map.get(status, COLORS["muted"]), edgecolor="white", lw=0.8, zorder=3)
        ax.text(row["date"] + pd.Timedelta(days=150), idx, f"{row['crisis_id']}/{row['bargain_id']}: {_short(row.get('bargain_name', ''), 50)}", fontsize=7.0, va="center", color=COLORS["ink"])
    ax.set_yticks([])
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="x", color=COLORS["grid"], lw=0.8)
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=color_map[token], markeredgecolor="white", markersize=7, label=token.replace("_", " ")) for token in DURABILITY_ORDER]
    ax.legend(handles=handles, title="Durability status", loc="upper left", frameon=True, fontsize=7.2, title_fontsize=7.8)
    ax.set_title("S2.2 Settlement Durability Over Time", loc="left", fontsize=15, color=COLORS["ink"], weight="bold")
    ax.set_xlabel("Bargain or settlement date", color=COLORS["ink"])
    ax.text(0, -0.13, "Each marker is a settlement record. Color identifies coded durability/implementation status.", transform=ax.transAxes, fontsize=9, color=COLORS["muted"])
    _save_pdf(fig, path)


def figure_later_effect_matrix(records: pd.DataFrame, path: Path) -> None:
    df = _dedupe_bargains(records)
    crises = _crises(df)
    matrix = np.zeros((len(crises), len(LATER_EFFECT_ORDER)))
    for _, row in df.iterrows():
        for token in _later_effect_tokens(row):
            matrix[crises.index(row["crisis_id"]), LATER_EFFECT_ORDER.index(token)] += 1
    fig, ax = plt.subplots(figsize=(11.8, max(5.6, len(crises) * 0.42)))
    image = _count_heatmap(ax, matrix, "s22_later")
    _add_count_colorbar(fig, ax, image, "Distinct settlement records", matrix.max())
    ax.set_yticks(np.arange(len(crises)))
    ax.set_yticklabels(crises, fontsize=8.8)
    ax.set_xticks(np.arange(len(LATER_EFFECT_ORDER)))
    ax.set_xticklabels([_wrap(label, 18) for label in LATER_EFFECT_ORDER], rotation=35, ha="right", fontsize=8.2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(LATER_EFFECT_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(crises), 1), minor=True)
    ax.grid(which="minor", color=COLORS["grid"], lw=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title("S2.2 Later Constitutional and Executive Effects", loc="left", fontsize=15, color=COLORS["ink"], weight="bold")
    ax.text(0, -0.25, "Shading marks settlement records linked to later constitutional effects, crises, or executive precedents.", transform=ax.transAxes, fontsize=9, color=COLORS["muted"])
    _save_pdf(fig, path)


def figure_settlement_cards(records: pd.DataFrame, path: Path, max_records: int | None = 12, title_suffix: str = "") -> None:
    df = _dedupe_bargains(records)
    df["date"] = pd.to_datetime(df["bargain_date"], errors="coerce")
    df = df.sort_values(["date", "crisis_id", "bargain_id"], na_position="last")
    if max_records is not None:
        df = df.head(max_records)
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path, metadata={"Creator": "Codex evidence synthesis"}) as pdf:
        for page, start in enumerate(range(0, len(df), 6), start=1):
            fig, ax = _new_page((11.7, 8.3))
            ax.text(0.04, 0.96, f"S2.2{title_suffix} Settlement Evidence Cards - Page {page}", fontsize=15, color=COLORS["ink"], weight="bold", va="top")
            ax.text(0.04, 0.915, "Cards retain settlement form, power distribution, durability, and source anchors.", fontsize=9.5, color=COLORS["muted"])
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
                title = f"{row['crisis_id']} / {row['bargain_id']}"
                ax.text(x + 0.012, y - 0.015, title, fontsize=8.5, color=COLORS["ink"], weight="bold", va="center")
                lines = [
                    "Doc: " + _short(row.get("bargain_name", "") or row.get("document_title", ""), 82),
                    "Power: " + _short(row.get("power_distribution_type", ""), 82),
                    "Durability: " + _short(row.get("durability_status", ""), 82),
                    "Sources: " + _short(row.get("source_ids", ""), 82),
                ]
                for line_i, line in enumerate(lines):
                    ax.text(x + 0.012, y - 0.058 - line_i * 0.038, _wrap(line, 58), fontsize=6.8, color=COLORS["muted"], va="top")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def build_figures(output: pd.DataFrame, figure_dir: Path) -> None:
    iqa = _records(output, "QE-S22-IQA")
    iqb = _records(output, "QE-S22-IQB")
    iqc = _records(output, "QE-S22-IQC")
    iqd = _records(output, "QE-S22-IQD")
    iqe = _records(output, "QE-S22-IQE")
    iqf = _records(output, "QE-S22-IQF")
    iqg = _records(output, "QE-S22-IQG")
    iqh = _records(output, "QE-S22-IQH")

    figure_document_form_matrix(iqa, figure_dir / "F_S22_1_settlement_document_form_matrix.pdf")
    figure_record_category_matrix(
        iqa,
        figure_dir / "F_S22_1_B_settlement_document_form_all_records.pdf",
        DOCUMENT_FORM_ORDER,
        _document_form_tokens,
        "S2.2_B Settlement Document Forms - All 34 Records",
        "One row per crisis/bargain record. Green marks a coded form; white marks absence. Full source anchors are in T_S22_1 and T_S22_9.",
    )
    figure_document_to_power_flow(iqd, figure_dir / "F_S22_2_settlement_form_to_power_flow.pdf")
    figure_document_to_power_all_matrix(iqd, figure_dir / "F_S22_2_B_settlement_form_to_power_all_edges_matrix.pdf")
    figure_actor_role_sector_matrix(iqb, figure_dir / "F_S22_3_actor_sector_role_matrix.pdf")
    figure_actor_role_sector_raw_matrix(iqb, figure_dir / "F_S22_3_B_actor_sector_role_all_raw_codes.pdf")
    figure_power_distribution_matrix(iqd, figure_dir / "F_S22_4_power_distribution_by_crisis.pdf")
    figure_record_category_matrix(
        iqd,
        figure_dir / "F_S22_4_B_power_distribution_all_records.pdf",
        [token for token in POWER_ORDER if token in set(iqd["power_distribution_type"])],
        _power_tokens,
        "S2.2_B Power Distribution - All 34 Settlement Records",
        "One row per crisis/bargain record. The classification is descriptive; it does not treat participation as proof of effective power-sharing.",
    )
    figure_formalization_sequence(iqe, figure_dir / "F_S22_5_informal_to_formal_sequence.pdf")
    figure_conditions_matrix(iqf, figure_dir / "F_S22_6_conditions_timetables_limits_matrix.pdf")
    raw_condition_categories = _observed_tokens(
        iqf.assign(condition_codes=iqf.apply(lambda row: "; ".join(_raw_condition_tokens(row)), axis=1)),
        "condition_codes",
    )
    figure_record_category_matrix(
        iqf,
        figure_dir / "F_S22_6_B_conditions_all_records_raw_codes.pdf",
        raw_condition_categories,
        _raw_condition_tokens,
        "S2.2_B Conditions and Timetables - All Coded Records",
        "Every settlement with coded conditions is shown against every observed condition code. Narrative clause text remains in T_S22_6.",
    )
    figure_durability_timeline(iqg, figure_dir / "F_S22_7_settlement_durability_timeline.pdf")
    figure_later_effect_matrix(iqh, figure_dir / "F_S22_8_later_effects_matrix.pdf")
    raw_later_categories = _observed_tokens(
        iqh.assign(later_effect_codes=iqh.apply(lambda row: "; ".join(_raw_later_effect_tokens(row)), axis=1)),
        "later_effect_codes",
    )
    figure_record_category_matrix(
        iqh,
        figure_dir / "F_S22_8_B_later_effects_all_records_raw_codes.pdf",
        raw_later_categories,
        _raw_later_effect_tokens,
        "S2.2_B Later Effects - All Settlement Records and Raw Codes",
        "Raw later-effect codes are shown without collapsing them into the seven interpretive families used in the primary figure.",
    )
    figure_settlement_cards(iqa, figure_dir / "F_S22_9_settlement_evidence_cards.pdf")
    figure_settlement_cards(iqa, figure_dir / "F_S22_9_B_settlement_evidence_all_cards.pdf", max_records=None, title_suffix="_B")


def validate_inputs(output: pd.DataFrame, repo_root: Path) -> None:
    expected_questions = {f"QE-S22-IQ{letter}" for letter in "ABCDEFGH"}
    observed_questions = set(output["question_id"])
    if expected_questions != observed_questions:
        raise ValueError(f"Unexpected S2.2 question set: {sorted(observed_questions)}")

    records = _dedupe_bargains(_records(output, "QE-S22-IQA"))
    if len(records) != 34:
        raise ValueError(f"Expected 34 distinct S2.2 settlement records, found {len(records)}")
    if records.duplicated(["crisis_id", "bargain_id"]).any():
        raise ValueError("Duplicate crisis_id + bargain_id keys remain after normalization")

    project_tables = load_project_data(repo_root)
    validate_output_crisis_ids(records["crisis_id"], project_tables["DB_Crisis"])

    observed_sectors = set(_observed_tokens(records, "participant_sectors"))
    observed_roles = set(_observed_tokens(records, "participant_roles"))
    if observed_sectors - set(ACTOR_SECTOR_ORDER):
        raise ValueError(f"Unmapped actor sectors: {sorted(observed_sectors - set(ACTOR_SECTOR_ORDER))}")
    if observed_roles - set(ACTOR_ROLE_ORDER):
        raise ValueError(f"Unmapped actor roles: {sorted(observed_roles - set(ACTOR_ROLE_ORDER))}")
    for token in observed_sectors:
        _sector_family(token)
    for token in observed_roles:
        _role_family(token)

    source_index = pd.read_csv(
        resolve_project_paths(repo_root).master_root / "source_indices" / "MASTER_Source_Index_UNIFIED_v1.csv",
        dtype=str,
    ).fillna("")
    valid_sources = set(source_index["source_id"])
    used_sources = {token for value in records["source_ids"] for token in _split_values(value)}
    missing_sources = sorted(used_sources - valid_sources)
    if missing_sources:
        raise ValueError(f"S2.2 records reference source IDs absent from the unified index: {missing_sources}")

    print(
        "Validated S2.2 inputs: "
        f"{len(records)} settlements, {records['crisis_id'].nunique()} crises, "
        f"{len(used_sources)} indexed sources, {len(observed_sectors)} sectors, {len(observed_roles)} roles."
    )


def main() -> None:
    repo_root = _repo_root()
    output = _load_s22_output(repo_root)
    validate_inputs(output, repo_root)
    table_dir = repo_root / "outputs" / "tables" / "s22_tables"
    figure_dir = repo_root / "outputs" / "figures" / "s22_figures"
    build_tables(output, table_dir)
    build_figures(output, figure_dir)
    print(f"Wrote S22 tables to {table_dir}")
    print(f"Wrote S22 figures to {figure_dir}")


if __name__ == "__main__":
    main()

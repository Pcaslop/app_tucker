#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tucker Yield Curve Explorer
"""

import io
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tensorly.decomposition import tucker

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Tucker Yield Curve Explorer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .app-header {
    background: #0d1117;
    padding: 1.6rem 2rem 1.2rem;
    margin: -1rem -1rem 1.5rem -1rem;
    border-bottom: 1px solid #21262d;
  }
  .app-header h1 { color: #e6edf3; font-size: 1.55rem; font-weight: 700;
                   margin: 0 0 .2rem 0; letter-spacing: -0.02em; }
  .app-header p  { color: #8b949e; font-size: .85rem; margin: 0; }
  .curve-row {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: .75rem 1rem;
    margin-bottom: .5rem;
  }
  .badge {
    display: inline-block;
    background: #1f6feb22; color: #58a6ff;
    border: 1px solid #1f6feb55; border-radius: 4px;
    padding: 1px 7px; font-size: .75rem; margin: 2px;
  }
  .metric-box {
    background: #161b22; border: 1px solid #21262d;
    border-radius: 8px; padding: .8rem 1.2rem; text-align: center;
  }
  .metric-box .val { font-size: 1.6rem; font-weight: 700; color: #58a6ff; }
  .metric-box .lbl { font-size: .75rem; color: #8b949e; margin-top: .2rem; }
  #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
  <h1>Tucker Yield Curve Explorer</h1>
  <p>Tensor decomposition of yield curves — up to 7 countries / sources</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
if "n_curves" not in st.session_state:
    st.session_state["n_curves"] = 2

for key in ["tucker_result", "aligned_dates", "aligned_mats",
            "curve_labels", "n_components"]:
    if key not in st.session_state:
        st.session_state[key] = None

COLORS = ["#58a6ff", "#3fb950", "#ff7b72",
          "#d2a8ff", "#ffa657", "#79c0ff", "#56d364"]
MAX_CURVES = 7

SAMPLE_DATA_URLS = {
    "UK": "https://raw.githubusercontent.com/Pcaslop/app_tucker/main/UK_DATA.csv",
    "FED": "https://raw.githubusercontent.com/Pcaslop/app_tucker/main/FED_DATA.csv",
}


def hex_to_rgba(hex_color, alpha=0.13):
    """Convert '#rrggbb' to 'rgba(r,g,b,alpha)' — Plotly doesn't accept 8-digit hex."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def parse_csv(uploaded_file, label):
    try:
        raw = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
        raw.index.name = "Date"
        raw.index = pd.to_datetime(raw.index, errors="coerce")
        raw = raw[~raw.index.isna()]
        float_cols = {}
        for c in raw.columns:
            try:
                float_cols[c] = float(str(c).strip())
            except ValueError:
                pass
        if not float_cols:
            st.error(f"**{label}**: no numeric maturity columns found.")
            return None, None
        df = raw[list(float_cols.keys())].rename(columns=float_cols)
        df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
        mats = sorted(df.columns.tolist())
        return df[mats], mats
    except Exception as e:
        st.error(f"**{label}**: parse error — {e}")
        return None, None


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_sample_csv(url):
    """Download a sample CSV from GitHub raw and return its raw bytes."""
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.content


def load_sample_data():
    """
    Load the default UK + FED sample curves from GitHub.
    Returns {label: DataFrame} using the same parsing logic as uploads.
    Returns {} if any download/parse fails (caller should handle fallback).
    """
    result = {}
    for label, url in SAMPLE_DATA_URLS.items():
        try:
            raw_bytes = fetch_sample_csv(url)
            df, mats = parse_csv(io.BytesIO(raw_bytes), label)
            if df is not None:
                if df.median().median() > 1:
                    df = df / 100
                result[label] = df
        except Exception as e:
            st.error(f"Could not load sample data for **{label}**: {e}")
    return result


def align_all(curves_dict):
    all_mats = [set(df.columns.tolist()) for df in curves_dict.values()]
    common_mats = sorted(all_mats[0].intersection(*all_mats[1:]))
    if len(common_mats) < 3:
        st.error(f"Only {len(common_mats)} common maturities. Need at least 3.")
        return None, None, None
    all_idx = [set(df.index.tolist()) for df in curves_dict.values()]
    common_dates = sorted(all_idx[0].intersection(*all_idx[1:]))
    if len(common_dates) < 10:
        st.error(f"Only {len(common_dates)} common dates. Check the files share a period.")
        return None, None, None
    aligned = {}
    for label, df in curves_dict.items():
        aligned[label] = df.loc[common_dates, common_mats].dropna()
    idx2 = list(aligned.values())[0].index
    for df in list(aligned.values())[1:]:
        idx2 = idx2.intersection(df.index)
    for label in aligned:
        aligned[label] = aligned[label].loc[idx2].values
    return aligned, pd.DatetimeIndex(idx2), common_mats


def run_tucker(aligned, n_components):
    arrays = list(aligned.values())
    T, P   = arrays[0].shape
    tensor = np.stack(arrays, axis=2)
    core, factors = tucker(tensor, rank=[T, n_components, len(arrays)])
    return tensor, core, factors


def explained_variance(tensor, core, factors):
    from tensorly import tucker_to_tensor
    recon  = tucker_to_tensor((core, factors))
    ss_res = np.sum((tensor - recon) ** 2)
    ss_tot = np.sum((tensor - tensor.mean()) ** 2)
    return 1 - ss_res / ss_tot


def dark_layout(title="", height=340):
    return dict(
        title=title, height=height,
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        font=dict(color="#e6edf3"),
        xaxis=dict(gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d"),
        legend=dict(bgcolor="#0d1117"),
        margin=dict(t=40, b=40),
        hovermode="x unified",
    )


# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["⚙️  Data & Tucker", "📊  Factors", "🧪  Stress Test"])


# ══════════════════════════════════════════════
# TAB 1
# ══════════════════════════════════════════════
with tab1:

    st.markdown("### Yield curves")
    st.caption(
        "CSV format: column 1 = dates, headers = maturities in years "
        "(e.g. `0.5`, `1`, `2`, `5`, `10`). "
        "Yields in decimal (0.04) or percent (4.0) — auto-detected."
    )

    data_source = st.radio(
        "Data source",
        options=["Sample data (UK + FED)", "Upload my own CSVs"],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
    )

    curves_parsed = {}

    if data_source == "Sample data (UK + FED)":
        with st.spinner("Loading sample data from GitHub…"):
            curves_parsed = load_sample_data()

        if curves_parsed:
            st.success(f"Loaded {len(curves_parsed)} sample curves: "
                       f"{', '.join(curves_parsed.keys())}")
            info_cols = st.columns(len(curves_parsed))
            for col, (label, df) in zip(info_cols, curves_parsed.items()):
                rng = (f"{df.index.min().strftime('%Y-%m-%d')} → "
                       f"{df.index.max().strftime('%Y-%m-%d')}")
                badges = " ".join(
                    f'<span class="badge">{m}y</span>' for m in df.columns
                )
                col.markdown(
                    f"<div class='curve-row'><b>{label}</b><br>"
                    f"<small style='color:#8b949e'>{rng} &nbsp;|&nbsp; "
                    f"{len(df):,} obs</small><br>{badges}</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.warning(
                "Sample data could not be loaded. Switch to "
                "**Upload my own CSVs** to continue."
            )

    else:
        btn_col1, btn_col2, _ = st.columns([1, 1, 5])
        with btn_col1:
            if st.button("＋  Add curve",
                         disabled=st.session_state["n_curves"] >= MAX_CURVES):
                st.session_state["n_curves"] += 1
        with btn_col2:
            if st.button("－  Remove curve",
                         disabled=st.session_state["n_curves"] <= 2):
                st.session_state["n_curves"] -= 1

        st.markdown("")

        for i in range(st.session_state["n_curves"]):
            with st.container():
                st.markdown('<div class="curve-row">', unsafe_allow_html=True)
                c_name, c_file, c_info = st.columns([2, 3, 4])

                with c_name:
                    label = st.text_input(
                        "Name", value=f"Curve {i+1}",
                        key=f"label_{i}",
                        label_visibility="collapsed",
                        placeholder=f"Curve {i+1}",
                    )
                with c_file:
                    f = st.file_uploader(
                        "Upload CSV", type=["csv"],
                        key=f"upload_{i}",
                        label_visibility="collapsed",
                    )
                with c_info:
                    if f is not None:
                        df, mats = parse_csv(f, label)
                        if df is not None:
                            if df.median().median() > 1:
                                df = df / 100
                            curves_parsed[label] = df
                            rng = (f"{df.index.min().strftime('%Y-%m-%d')} → "
                                   f"{df.index.max().strftime('%Y-%m-%d')}")
                            badges = " ".join(
                                f'<span class="badge">{m}y</span>' for m in mats
                            )
                            st.markdown(
                                f"<small style='color:#8b949e'>{rng} &nbsp;|&nbsp; "
                                f"{len(df):,} obs</small><br>{badges}",
                                unsafe_allow_html=True,
                            )
                    else:
                        st.markdown(
                            "<small style='color:#484f58'>No file uploaded</small>",
                            unsafe_allow_html=True,
                        )
                st.markdown("</div>", unsafe_allow_html=True)

    if len(curves_parsed) < 2:
        st.info("Upload at least 2 CSV files to continue.")
        st.stop()

    # ── Alignment ─────────────────────────────
    st.markdown("---")
    st.markdown("### Alignment")

    aligned, dates, common_mats = align_all(curves_parsed)
    if aligned is None:
        st.stop()

    c1, c2, c3 = st.columns(3)
    for col, val, lbl in zip(
        [c1, c2, c3],
        [f"{len(dates):,}", len(common_mats), len(aligned)],
        ["Common dates", "Common maturities", "Active curves"],
    ):
        col.markdown(
            f'<div class="metric-box"><div class="val">{val}</div>'
            f'<div class="lbl">{lbl}</div></div>',
            unsafe_allow_html=True,
        )

    badges = " ".join(f'<span class="badge">{m}y</span>' for m in common_mats)
    st.markdown(f"<br>**Shared maturities:** {badges}", unsafe_allow_html=True)

    fig_prev = go.Figure()
    for i, (label, arr) in enumerate(aligned.items()):
        fig_prev.add_trace(go.Scatter(
            x=common_mats, y=arr.mean(axis=0),
            mode="lines+markers", name=label,
            line=dict(color=COLORS[i % len(COLORS)], width=2),
            marker=dict(size=5),
        ))
    fig_prev.update_layout(
        **dark_layout("Mean curve by source", height=300),
        xaxis_title="Maturity (years)", yaxis_title="Yield",
    )
    st.plotly_chart(fig_prev, use_container_width=True)

    # ── Tucker config ─────────────────────────
    st.markdown("---")
    st.markdown("### Tucker decomposition")

    cfg1, cfg2 = st.columns([1, 2])
    with cfg1:
        n_comp = st.slider(
            "Maturity components (rank)",
            min_value=2, max_value=min(len(common_mats), 6),
            value=min(3, len(common_mats)),
            help="3 = level / slope / curvature (NSS interpretation).",
        )
        run_btn = st.button("▶  Run Tucker", type="primary",
                            use_container_width=True)
    with cfg2:
        st.markdown("""
        <div style='background:#161b22;border:1px solid #21262d;border-radius:8px;
                    padding:.8rem 1.1rem;font-size:.82rem;color:#8b949e;'>
        <b style='color:#e6edf3'>Tensor layout</b><br>
        <code style='color:#58a6ff'>shape = (T, P, N)</code><br>
        T = dates &nbsp;|&nbsp; P = maturities &nbsp;|&nbsp; N = curves<br><br>
        <code>tucker(rank = [T, rank_maturity, N])</code><br>
        No temporal compression — consistent with the base model.
        </div>
        """, unsafe_allow_html=True)

    if run_btn:
        with st.spinner("Building tensor and running Tucker…"):
            try:
                tensor, core, factors = run_tucker(aligned, n_comp)
                ev = explained_variance(tensor, core, factors)
                st.session_state["tucker_result"] = (tensor, core, factors)
                st.session_state["aligned_dates"]  = dates
                st.session_state["aligned_mats"]   = common_mats
                st.session_state["curve_labels"]   = list(aligned.keys())
                st.session_state["n_components"]   = n_comp
                st.success(f"Tucker complete — explained variance: **{ev:.2%}**")
                r1, r2, r3 = st.columns(3)
                for col, val, lbl in zip(
                    [r1, r2, r3],
                    [str(tensor.shape), str(core.shape), str(factors[1].shape)],
                    ["Tensor (T×P×N)", "Core (T×comp×N)", "Maturity factor (P×comp)"],
                ):
                    col.markdown(
                        f'<div class="metric-box"><div class="val" '
                        f'style="font-size:1.1rem">{val}</div>'
                        f'<div class="lbl">{lbl}</div></div>',
                        unsafe_allow_html=True,
                    )
                st.info("Go to the **📊 Factors** tab to explore the results.")
            except Exception as e:
                st.error(f"Tucker error: {e}")


# ══════════════════════════════════════════════
# TAB 2
# ══════════════════════════════════════════════
with tab2:

    if st.session_state["tucker_result"] is None:
        st.info("Run Tucker in the ⚙️ tab first.")
        st.stop()

    tensor, core, factors = st.session_state["tucker_result"]
    dates  = st.session_state["aligned_dates"]
    mats   = st.session_state["aligned_mats"]
    labels = st.session_state["curve_labels"]
    n_comp = st.session_state["n_components"]
    COMP_NAMES = ["Level", "Slope", "Curvature",
                  "Comp 4", "Comp 5", "Comp 6"][:n_comp]

    st.markdown("### Factor visualisation")

    ctrl1, ctrl2, ctrl3 = st.columns(3)
    with ctrl1:
        sel_comp = st.selectbox(
            "Component",
            options=list(range(n_comp)),
            format_func=lambda x: f"{x+1} — {COMP_NAMES[x]}",
        )
    with ctrl2:
        sel_curves = st.multiselect(
            "Curves", options=labels, default=labels,
        )
    with ctrl3:
        date_range = st.date_input(
            "Date range",
            value=(dates.min().date(), dates.max().date()),
            min_value=dates.min().date(),
            max_value=dates.max().date(),
        )

    if len(date_range) == 2:
        mask = (dates >= pd.Timestamp(date_range[0])) & \
               (dates <= pd.Timestamp(date_range[1]))
    else:
        mask = np.ones(len(dates), dtype=bool)

    dates_f = dates[mask]
    core_f  = core[mask]

    # 1. Loadings
    st.markdown("---")
    st.markdown("#### Maturity loadings")
    fig_load = go.Figure()
    for j in range(n_comp):
        fig_load.add_trace(go.Scatter(
            x=mats, y=factors[1][:, j],
            mode="lines+markers", name=COMP_NAMES[j],
            line=dict(color=COLORS[j % len(COLORS)], width=2.5),
            marker=dict(size=6),
            opacity=1.0 if j == sel_comp else 0.25,
        ))
    fig_load.update_layout(
        **dark_layout(height=320),
        xaxis_title="Maturity (years)", yaxis_title="Loading",
    )
    st.plotly_chart(fig_load, use_container_width=True)

    # 2. Score over time
    st.markdown("---")
    st.markdown(f"#### {COMP_NAMES[sel_comp]} score over time")
    fig_score = go.Figure()
    for i, label in enumerate(labels):
        if label not in sel_curves:
            continue
        score = tensor[mask, :, i] @ factors[1][:, sel_comp]
        fig_score.add_trace(go.Scatter(
            x=dates_f, y=score, mode="lines", name=label,
            line=dict(color=COLORS[i % len(COLORS)], width=1.8),
        ))
    fig_score.update_layout(
        **dark_layout(height=340),
        xaxis_title="Date", yaxis_title="Score",
    )
    st.plotly_chart(fig_score, use_container_width=True)

    # 3. Core intensity
    st.markdown("---")
    st.markdown(f"#### Core intensity — {COMP_NAMES[sel_comp]} by curve")

    # Tucker leaves the time axis uncompressed (rank = T), so the first
    # records carry a large, distorting spike. Skip them for display.
    SKIP_N = 10
    skip_n = min(SKIP_N, len(core_f) - 1)
    dates_core = dates_f[skip_n:]
    core_skip  = core_f[skip_n:]

    sel_valid = [l for l in sel_curves if l in labels]
    if sel_valid:
        fig_core = make_subplots(
            rows=1, cols=len(sel_valid),
            shared_yaxes=True,
            subplot_titles=sel_valid,
        )
        for ci, label in enumerate(sel_valid):
            gi = labels.index(label)
            fig_core.add_trace(
                go.Scatter(
                    x=dates_core, y=core_skip[:, sel_comp, gi],
                    mode="lines", name=label,
                    line=dict(color=COLORS[gi % len(COLORS)], width=1.5),
                    showlegend=False,
                ),
                row=1, col=ci + 1,
            )
        fig_core.update_layout(
            height=300,
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font=dict(color="#e6edf3"),
            margin=dict(t=40, b=40),
        )
        fig_core.update_xaxes(gridcolor="#21262d")
        fig_core.update_yaxes(gridcolor="#21262d")
        st.plotly_chart(fig_core, use_container_width=True)

    # 4. Decoupling
    if len(sel_curves) >= 2:
        st.markdown("---")
        st.markdown(f"#### Decoupling — {COMP_NAMES[sel_comp]}")
        ref_label = st.selectbox(
            "Reference curve", options=sel_curves, index=0, key="ref_label",
        )
        ref_score = tensor[mask, :, labels.index(ref_label)] @ factors[1][:, sel_comp]
        fig_diff = go.Figure()
        for i, label in enumerate(labels):
            if label not in sel_curves or label == ref_label:
                continue
            diff = tensor[mask, :, i] @ factors[1][:, sel_comp] - ref_score
            fig_diff.add_trace(go.Scatter(
                x=dates_f, y=diff, mode="lines",
                name=f"{label} − {ref_label}",
                line=dict(color=COLORS[i % len(COLORS)], width=1.5),
                fill="tozeroy",
                fillcolor=hex_to_rgba(COLORS[i % len(COLORS)]),
            ))
        fig_diff.add_hline(y=0, line_dash="dash",
                           line_color="#8b949e", line_width=1)
        fig_diff.update_layout(
            **dark_layout(height=300),
            xaxis_title="Date",
            yaxis_title=f"Δ Score ({COMP_NAMES[sel_comp]})",
        )
        st.plotly_chart(fig_diff, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 3 — Stress Test
# ══════════════════════════════════════════════
with tab3:

    if st.session_state["tucker_result"] is None:
        st.info("Run Tucker in the ⚙️ tab first.")
        st.stop()

    tensor, core, factors = st.session_state["tucker_result"]
    dates  = st.session_state["aligned_dates"]
    mats   = st.session_state["aligned_mats"]
    labels = st.session_state["curve_labels"]
    n_comp = st.session_state["n_components"]
    N      = len(labels)
    COMP_NAMES = ["Level", "Slope", "Curvature",
                  "Comp 4", "Comp 5", "Comp 6"][:n_comp]

    st.markdown("### Stress test")
    st.caption(
        "Shock the last observed curve of one country through its level, "
        "slope and curvature scores. The shock propagates to the other "
        "countries through the country-mode loading (factors[2], column 0), "
        "scaled by each country's relative weight in that common factor."
    )

    # ── Controls ──────────────────────────────
    ctrl1, ctrl2 = st.columns([1, 2])
    with ctrl1:
        origin_label = st.selectbox(
            "Shock origin country",
            options=labels,
        )
    origin_idx = labels.index(origin_label)

    st.markdown("")
    slider_cols = st.columns(min(n_comp, 3))
    shock_pct = {}
    for j in range(min(n_comp, 3)):
        with slider_cols[j]:
            shock_pct[j] = st.slider(
                f"{COMP_NAMES[j]} shock (%)",
                min_value=-100, max_value=100, value=0, step=5,
                key=f"shock_{j}",
            )

    # ── Core computation ──────────────────────
    last_curve = tensor[-1, :, :]                       # (P, N)
    country_w0 = factors[2][:, 0]                        # (N,) common factor weights
    w_origin   = country_w0[origin_idx]

    # Current scores for every country, every component (last date)
    current_scores = np.zeros((N, n_comp))
    for k in range(N):
        for j in range(n_comp):
            current_scores[k, j] = last_curve[:, k] @ factors[1][:, j]

    # Delta at the origin country, for the shocked components only
    delta_origin = np.zeros(n_comp)
    for j in shock_pct:
        delta_origin[j] = current_scores[origin_idx, j] * (shock_pct[j] / 100.0)

    # Propagate delta to every country via the country-mode common factor
    stressed_scores = current_scores.copy()
    for k in range(N):
        if w_origin == 0:
            scale = 0.0
        else:
            scale = country_w0[k] / w_origin
        for j in shock_pct:
            stressed_scores[k, j] = current_scores[k, j] + delta_origin[j] * scale

    # Reconstruct curves: original vs stressed, per country
    original_curves = {}
    stressed_curves  = {}
    for k, label in enumerate(labels):
        original_curves[label] = current_scores[k, :] @ factors[1].T     # (P,)
        stressed_curves[label] = stressed_scores[k, :] @ factors[1].T    # (P,)

    # ── Summary table of applied shocks ───────
    st.markdown("---")
    st.markdown("#### Propagated shock by country (score space)")

    summary_cols = st.columns(N)
    for k, label in enumerate(labels):
        with summary_cols[k]:
            tag = " (origin)" if k == origin_idx else ""
            st.markdown(
                f"<div class='metric-box'><div class='val' style='font-size:1rem'>"
                f"{label}{tag}</div>"
                f"<div class='lbl'>w₀ = {country_w0[k]:.3f}</div></div>",
                unsafe_allow_html=True,
            )

    # ── Chart 1: original vs stressed curves, in yields ──
    st.markdown("---")
    st.markdown("#### Yield curve — original vs stressed")

    fig_curves = make_subplots(
        rows=1, cols=N,
        shared_yaxes=True,
        subplot_titles=labels,
    )
    for k, label in enumerate(labels):
        color = COLORS[k % len(COLORS)]
        fig_curves.add_trace(
            go.Scatter(
                x=mats, y=original_curves[label],
                mode="lines+markers", name="Original",
                line=dict(color=color, width=2, dash="dot"),
                marker=dict(size=5),
                showlegend=(k == 0),
                legendgroup="orig",
            ),
            row=1, col=k + 1,
        )
        fig_curves.add_trace(
            go.Scatter(
                x=mats, y=stressed_curves[label],
                mode="lines+markers", name="Stressed",
                line=dict(color=color, width=2.5),
                marker=dict(size=6),
                showlegend=(k == 0),
                legendgroup="stress",
            ),
            row=1, col=k + 1,
        )
    fig_curves.update_layout(
        height=340,
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        font=dict(color="#e6edf3"),
        legend=dict(bgcolor="#0d1117"),
        margin=dict(t=40, b=40),
    )
    fig_curves.update_xaxes(title_text="Maturity (years)",
                            gridcolor="#21262d")
    fig_curves.update_yaxes(title_text="Yield", gridcolor="#21262d", col=1)
    fig_curves.update_yaxes(gridcolor="#21262d")
    st.plotly_chart(fig_curves, use_container_width=True)

    # ── Chart 2: difference in basis points ───
    st.markdown("---")
    st.markdown("#### Shock impact — stressed minus original (bps)")

    fig_bps = make_subplots(
        rows=1, cols=N,
        shared_yaxes=False,
        subplot_titles=labels,
    )
    for k, label in enumerate(labels):
        diff_bps = (stressed_curves[label] - original_curves[label]) * 10000
        color = COLORS[k % len(COLORS)]
        fig_bps.add_trace(
            go.Scatter(
                x=mats, y=diff_bps,
                mode="lines+markers", name=label,
                line=dict(color=color, width=2),
                marker=dict(size=5),
                fill="tozeroy",
                fillcolor=hex_to_rgba(color),
                showlegend=False,
            ),
            row=1, col=k + 1,
        )
    fig_bps.add_hline(y=0, line_dash="dash",
                      line_color="#8b949e", line_width=1)
    fig_bps.update_layout(
        height=300,
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        font=dict(color="#e6edf3"),
        margin=dict(t=40, b=40),
    )
    fig_bps.update_xaxes(title_text="Maturity (years)", gridcolor="#21262d")
    fig_bps.update_yaxes(title_text="Δ bps", gridcolor="#21262d",
                         showticklabels=True, col=1)
    fig_bps.update_yaxes(gridcolor="#21262d", showticklabels=True)
    st.plotly_chart(fig_bps, use_container_width=True)

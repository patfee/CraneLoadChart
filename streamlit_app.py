import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from pathlib import Path

st.set_page_config(page_title="Crane Capacity Viewer", layout="wide")

st.title("🪝 Crane Capacity Viewer")
st.caption("Works with a long/normalized dataset: FoldingJib_deg, MainJib_deg, Outreach_m, Height_m, Duty, Capacity_t")

# --- Data source ---
with st.sidebar:
    st.header("Data")

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "CraneData_long_clean.xlsx"   # your sample path

use_sample = st.checkbox("Use sample data from /data folder", value=True)
uploaded = st.file_uploader("Or upload a CSV/XLSX", type=["csv", "xlsx"])

def read_any(fp_or_buffer):
    name = getattr(fp_or_buffer, "name", str(fp_or_buffer)).lower()
    if name.endswith(".csv"):
        return pd.read_csv(fp_or_buffer)
    else:
        return pd.read_excel(fp_or_buffer)

df = None
if uploaded is not None:
    df = read_any(uploaded)
elif use_sample and DATA_PATH.exists():
    df = read_any(DATA_PATH)
else:
    st.info("Load data from the sidebar to begin (upload a file or enable 'Use sample data').")
    st.stop()

# --- Validate columns ---
required_cols = ["FoldingJib_deg", "MainJib_deg", "Outreach_m", "Height_m", "Duty", "Capacity_t"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# Coerce numerics
for num_col in ["FoldingJib_deg","MainJib_deg","Outreach_m","Height_m","Capacity_t"]:
    df[num_col] = pd.to_numeric(df[num_col], errors="coerce")
df = df.dropna(subset=["Outreach_m","Height_m","Capacity_t"])

# Sidebar global filters used by all tabs
with st.sidebar:
    st.header("Global Filters")
    duties = sorted(df["Duty"].dropna().unique().tolist())
    duty = st.selectbox("Duty", duties)

    fj_vals = sorted(df["FoldingJib_deg"].dropna().unique().tolist())
    mj_vals = sorted(df["MainJib_deg"].dropna().unique().tolist())
    sel_fj = st.multiselect("FoldingJib_deg (optional)", fj_vals, default=fj_vals)
    sel_mj = st.multiselect("MainJib_deg (optional)", mj_vals, default=mj_vals)

tabs = st.tabs(["Curve (Capacity vs Outreach)", "Iso-load Contour", "SWL Envelope (by Cdyn)"])

# ===================
# Tab 1: Curve
# ===================
with tabs[0]:
    st.subheader(f"Capacity vs Outreach · Duty = {duty}")
    work = df[df["Duty"] == duty].copy()
    if sel_fj:
        work = work[work["FoldingJib_deg"].isin(sel_fj)]
    if sel_mj:
        work = work[work["MainJib_deg"].isin(sel_mj)]

    if work.empty:
        st.warning("No data after filters.")
    else:
        left, right = st.columns([2,1], gap="large")
        with right:
            h_min, h_max = float(np.nanmin(work["Height_m"])), float(np.nanmax(work["Height_m"]))
            target_h = st.slider("Target height (m)", min_value=h_min, max_value=h_max,
                                 value=float(np.clip(0.0, h_min, h_max)))
            tol = st.number_input("Nearest-height tolerance (m)", min_value=0.0, value=0.75, step=0.25)
            conservative = st.checkbox("Conservative curve (min Capacity per Outreach)", value=False)

        work = work.assign(h_err=(work["Height_m"] - target_h).abs())
        idx = work.groupby("Outreach_m")["h_err"].transform("min") == work["h_err"]
        nearest = work[idx & (work["h_err"] <= tol)].copy()

        if nearest.empty:
            st.warning("No data within the chosen height tolerance.")
        else:
            agg_func = "min" if conservative else "max"
            agg_df = nearest.groupby("Outreach_m", as_index=False).agg(
                Capacity_t=("Capacity_t", agg_func),
                Height_at=("Height_m", "mean"),
            ).sort_values("Outreach_m")

            st.line_chart(data=agg_df.set_index("Outreach_m")["Capacity_t"], height=420)

            st.subheader("Filtered data (used for chart)")
            st.dataframe(agg_df, use_container_width=True)

            csv_bytes = agg_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV (filtered)", data=csv_bytes,
                               file_name="capacity_vs_outreach_filtered.csv", mime="text/csv")

# ===================
# Tab 2: Iso-load Contour
# ===================
with tabs[1]:
    st.subheader(f"Iso-load Contour · Duty = {duty}")

    grid_left, grid_right = st.columns([2,1], gap="large")

    filt = df[df["Duty"] == duty].copy()
    if sel_fj:
        filt = filt[filt["FoldingJib_deg"].isin(sel_fj)]
    if sel_mj:
        filt = filt[filt["MainJib_deg"].isin(sel_mj)]

    if filt.empty:
        st.warning("No data after filters.")
        st.stop()

    agg_mode = grid_right.selectbox("Aggregate duplicates by", ["max", "min", "mean"], index=0)
    agg_df = filt.groupby(["Outreach_m","Height_m"], as_index=False).agg(Capacity_t=("Capacity_t", agg_mode))

    with grid_right:
        n_x = st.number_input("Grid points (Outreach)", min_value=30, max_value=400, value=120, step=10)
        n_y = st.number_input("Grid points (Height)", min_value=30, max_value=400, value=120, step=10)

    x = agg_df["Outreach_m"].to_numpy()
    y = agg_df["Height_m"].to_numpy()
    z = agg_df["Capacity_t"].to_numpy()

    tri = mtri.Triangulation(x, y)
    interpolator = mtri.LinearTriInterpolator(tri, z)

    xi = np.linspace(np.nanmin(x), np.nanmax(x), int(n_x))
    yi = np.linspace(np.nanmin(y), np.nanmax(y), int(n_y))
    XI, YI = np.meshgrid(xi, yi)
    ZI = interpolator(XI, YI)

    with grid_right:
        zmin = float(np.nanmin(z))
        zmax = float(np.nanmax(z))
        st.caption(f"Capacity range in data: {zmin:.2f} – {zmax:.2f} t")

        levels_mode = st.radio("Levels", ["Auto", "N levels", "Custom list"], index=0, horizontal=True)
        filled = st.checkbox("Filled contours", value=True)
        show_points = st.checkbox("Show data points", value=True)

        if levels_mode == "Auto":
            levels = None
        elif levels_mode == "N levels":
            n_levels = st.slider("Number of levels", min_value=5, max_value=30, value=12)
            levels = np.linspace(zmin, zmax, n_levels)
        else:
            raw = st.text_input("Comma-separated levels (e.g., 5,7.5,10,12.5,15)", "")
            try:
                levels = [float(v.strip()) for v in raw.split(",") if v.strip()]
                if len(levels) == 0:
                    levels = None
            except Exception:
                st.warning("Could not parse custom levels; using auto.")
                levels = None

        st.markdown("**Iso-load threshold**")
        thresh = st.number_input("Threshold (t): plot isoline at this load, and optionally shade region ≥ threshold",
                                 min_value=0.0, value=50.0, step=1.0)
        shade_region = st.checkbox("Shade region ≥ threshold", value=True)
        draw_isoline = st.checkbox("Draw isoline at threshold", value=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    Zmask = np.ma.array(ZI, mask=np.isnan(ZI))

    if filled:
        cs = ax.contourf(XI, YI, Zmask, levels=levels)
        cbar = fig.colorbar(cs, ax=ax)
        cbar.set_label("Capacity (t)")
    else:
        cs = ax.contour(XI, YI, Zmask, levels=levels)
        cbar = fig.colorbar(cs, ax=ax)
        cbar.set_label("Capacity (t)")

    if shade_region and np.isfinite(zmax):
        try:
            ax.contourf(XI, YI, Zmask, levels=[thresh, zmax], alpha=0.35)
        except Exception:
            pass

    if draw_isoline:
        try:
            cs_th = ax.contour(XI, YI, Zmask, levels=[thresh], linewidths=2.0)
            ax.clabel(cs_th, fmt={thresh: f"{thresh:g} t"}, inline=True, fontsize=9)
        except Exception:
            pass

    if show_points:
        ax.plot(x, y, ".", ms=2)

    ax.set_xlabel("Outreach (m)")
    ax.set_ylabel("Height (m)")
    ax.set_title("Iso-load Contours")
    st.pyplot(fig, clear_figure=True)

    grid_df = pd.DataFrame({
        "Outreach_m": XI.ravel(),
        "Height_m": YI.ravel(),
        "Capacity_t": ZI.ravel()
    }).dropna()
    csv_bytes = grid_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download gridded surface (CSV)", data=csv_bytes,
                       file_name="capacity_grid.csv", mime="text/csv")

    with st.expander("Raw points used (after aggregation)"):
        st.dataframe(agg_df.sort_values(["Outreach_m","Height_m"]), use_container_width=True)

# ============================
# Tab 3: SWL Envelope (by Cdyn)
# ============================
with tabs[2]:
    st.subheader("SWL Envelope (by Cdyn) — SWL [t] vs RADIUS [m]")

    # Map Duty -> Cdyn (e.g., Harbour_Cd115 -> 1.15)
    def duty_to_cdyn(d):
        m = re.search(r"Cd\s*([0-9]+)", d, flags=re.IGNORECASE)
        if not m:
            return None
        return float(m.group(1)) / 100.0

    # All duties available in data
    all_duties = sorted(df["Duty"].unique().tolist())
    duty_cdyn_pairs = [(d, duty_to_cdyn(d)) for d in all_duties]
    # allow multi-select to compare
    selected_duties = st.multiselect("Select Cdyn/Duties to plot",
                                     all_duties,
                                     default=[duty] if duty in all_duties else all_duties[:1])

    # style options
    col1, col2 = st.columns([1,1])
    with col1:
        fill_under = st.checkbox("Fill under curve", value=False)
    with col2:
        thick_line = st.checkbox("Use thick line style", value=True)

    if len(selected_duties) == 0:
        st.info("Select at least one Duty to plot.")
    else:
        fig, ax = plt.subplots(figsize=(9, 5.5))

        for dsel in selected_duties:
            w = df[df["Duty"] == dsel].copy()
            if sel_fj:
                w = w[w["FoldingJib_deg"].isin(sel_fj)]
            if sel_mj:
                w = w[w["MainJib_deg"].isin(sel_mj)]

            if w.empty:
                continue

            # Envelope: for each outreach, take the maximum capacity across all heights / angles
            env = (w.groupby("Outreach_m", as_index=False)
                    .agg(SWL_t=("Capacity_t", "max"))
                    .sort_values("Outreach_m"))

            # Plot
            lw = 3.0 if thick_line else 1.5
            line, = ax.plot(env["Outreach_m"], env["SWL_t"], linewidth=lw, label=dsel)
            if fill_under:
                ax.fill_between(env["Outreach_m"], env["SWL_t"], step=None, alpha=0.15)

        # Labels / grid styled to resemble the sample
        ax.set_xlabel("RADIUS [m]")
        ax.set_ylabel("SWL [t]")
        ax.grid(True, which="both", linestyle="-", linewidth=0.5, alpha=0.6)
        ax.set_title("OFFSHORE LIFT CAPACITY")

        # If a single duty is selected and has a Cdyn, show it like the sample subtitle
        if len(selected_duties) == 1:
            cd = duty_to_cdyn(selected_duties[0])
            if cd is not None:
                st.caption(f"DESIGN DYNAMIC FACTOR: {cd:.2f}")

        ax.legend(title="Duty / Cdyn", loc="best")
        st.pyplot(fig, clear_figure=True)

        # Offer download of the last plotted envelope if only one duty is selected
        if len(selected_duties) == 1 and not w.empty:
            env_csv = env.rename(columns={"SWL_t": "SWL_t_at_envelope"})
            csv_bytes = env_csv.to_csv(index=False).encode("utf-8")
            st.download_button("Download SWL envelope (CSV)", data=csv_bytes,
                               file_name="swl_envelope.csv", mime="text/csv")


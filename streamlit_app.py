import io
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
# default sample inside /data
DATA_PATH = ROOT / "data" / "CraneData_long_clean.xlsx"

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

# Coerce numerics (in case of strings in XLSX/CSV)
for num_col in ["FoldingJib_deg","MainJib_deg","Outreach_m","Height_m","Capacity_t"]:
    df[num_col] = pd.to_numeric(df[num_col], errors="coerce")
df = df.dropna(subset=["Outreach_m","Height_m","Capacity_t"])

with st.sidebar:
    st.header("Global Filters")
    duties = sorted(df["Duty"].dropna().unique().tolist())
    duty = st.selectbox("Duty", duties)

    fj_vals = sorted(df["FoldingJib_deg"].dropna().unique().tolist())
    mj_vals = sorted(df["MainJib_deg"].dropna().unique().tolist())
    sel_fj = st.multiselect("FoldingJib_deg (optional)", fj_vals, default=fj_vals)
    sel_mj = st.multiselect("MainJib_deg (optional)", mj_vals, default=mj_vals)

tabs = st.tabs(["Curve (Capacity vs Outreach)", "Iso-load Contour"])

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
            target_h = st.slider("Target height (m)", min_value=h_min, max_value=h_max, value=float(np.clip(0.0, h_min, h_max)))
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
            st.download_button("Download CSV (filtered)", data=csv_bytes, file_name="capacity_vs_outreach_filtered.csv", mime="text/csv")

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

    # If multiple points exist for the same (Outreach, Height), aggregate by chosen statistic
    agg_mode = grid_right.selectbox("Aggregate duplicates by", ["max", "min", "mean"], index=0)
    agg_df = filt.groupby(["Outreach_m","Height_m"], as_index=False).agg(Capacity_t=("Capacity_t", agg_mode))

    # Define grid
    with grid_right:
        n_x = st.number_input("Grid points (Outreach)", min_value=30, max_value=400, value=120, step=10)
        n_y = st.number_input("Grid points (Height)", min_value=30, max_value=400, value=120, step=10)

    x = agg_df["Outreach_m"].to_numpy()
    y = agg_df["Height_m"].to_numpy()
    z = agg_df["Capacity_t"].to_numpy()

    # Triangulation-based interpolation (no SciPy required)
    tri = mtri.Triangulation(x, y)
    interpolator = mtri.LinearTriInterpolator(tri, z)

    xi = np.linspace(np.nanmin(x), np.nanmax(x), int(n_x))
    yi = np.linspace(np.nanmin(y), np.nanmax(y), int(n_y))
    XI, YI = np.meshgrid(xi, yi)
    ZI = interpolator(XI, YI)

    # Levels & threshold controls
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
        thresh = st.number_input("Threshold (t): plot isoline at this load, and optionally shade region ≥ threshold", min_value=0.0, value=50.0, step=1.0)
        shade_region = st.checkbox("Shade region ≥ threshold", value=True)
        draw_isoline = st.checkbox("Draw isoline at threshold", value=True)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    Zmask = np.ma.array(ZI, mask=np.isnan(ZI))

    # Base contour plot
    if filled:
        cs = ax.contourf(XI, YI, Zmask, levels=levels)
        cbar = fig.colorbar(cs, ax=ax)
        cbar.set_label("Capacity (t)")
    else:
        cs = ax.contour(XI, YI, Zmask, levels=levels)
        cbar = fig.colorbar(cs, ax=ax)
        cbar.set_label("Capacity (t)")

    # Threshold shading (region ≥ threshold)
    if shade_region and np.isfinite(zmax):
        # Two-level fill from threshold to max to shade the "safe" / higher-capacity region
        try:
            ax.contourf(XI, YI, Zmask, levels=[thresh, zmax], alpha=0.35)
        except Exception:
            pass  # in case thresh is outside data range

    # Draw the exact isoline at the threshold
    if draw_isoline:
        try:
            cs_th = ax.contour(XI, YI, Zmask, levels=[thresh], linewidths=2.0)
            # label the isoline
            ax.clabel(cs_th, fmt={thresh: f"{thresh:g} t"}, inline=True, fontsize=9)
        except Exception:
            pass

    if show_points:
        ax.plot(x, y, ".", ms=2)

    ax.set_xlabel("Outreach (m)")
    ax.set_ylabel("Height (m)")
    ax.set_title("Iso-load Contours")

    st.pyplot(fig, clear_figure=True)

    # Optional: download the gridded surface
    grid_df = pd.DataFrame({
        "Outreach_m": XI.ravel(),
        "Height_m": YI.ravel(),
        "Capacity_t": ZI.ravel()
    }).dropna()
    csv_bytes = grid_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download gridded surface (CSV)", data=csv_bytes, file_name="capacity_grid.csv", mime="text/csv")

    with st.expander("Raw points used (after aggregation)"):
        st.dataframe(agg_df.sort_values(["Outreach_m","Height_m"]), use_container_width=True)

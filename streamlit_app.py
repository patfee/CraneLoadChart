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

# ---------------- Data Source ----------------
with st.sidebar:
    st.header("Data")

ROOT = Path(__file__).parent
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
    st.info("Load data from the sidebar to begin (upload a file or enable sample data).")
    st.stop()

required_cols = ["FoldingJib_deg", "MainJib_deg", "Outreach_m", "Height_m", "Duty", "Capacity_t"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

for num_col in ["FoldingJib_deg","MainJib_deg","Outreach_m","Height_m","Capacity_t"]:
    df[num_col] = pd.to_numeric(df[num_col], errors="coerce")
df = df.dropna(subset=["Outreach_m","Height_m","Capacity_t"])

# ---------------- Global Filters ----------------
with st.sidebar:
    st.header("Global Filters")
    duties = sorted(df["Duty"].dropna().unique().tolist())
    duty = st.selectbox("Duty", duties)

    fj_vals = sorted(df["FoldingJib_deg"].dropna().unique().tolist())
    mj_vals = sorted(df["MainJib_deg"].dropna().unique().tolist())
    sel_fj = st.multiselect("FoldingJib_deg (optional)", fj_vals, default=fj_vals)
    sel_mj = st.multiselect("MainJib_deg (optional)", mj_vals, default=mj_vals)

# ---------------- Tabs ----------------
tab_curve, tab_iso, tab_envelope, tab_diag = st.tabs([
    "Curve (Capacity vs Outreach)",
    "Iso-load Contour",
    "SWL Envelope (by Cdyn)",
    "SWL vs MainJib (fixed FJ)"
])

# ==========================================================
# TAB 1 – Curve (Capacity vs Outreach)
# ==========================================================
with tab_curve:
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
            st.dataframe(agg_df, use_container_width=True)
            st.download_button("Download CSV (filtered)",
                               data=agg_df.to_csv(index=False).encode("utf-8"),
                               file_name="capacity_vs_outreach_filtered.csv")

# ==========================================================
# TAB 2 – Iso-load Contour
# ==========================================================
with tab_iso:
    st.subheader(f"Iso-load Contour · Duty = {duty}")
    filt = df[df["Duty"] == duty].copy()
    if sel_fj:
        filt = filt[filt["FoldingJib_deg"].isin(sel_fj)]
    if sel_mj:
        filt = filt[filt["MainJib_deg"].isin(sel_mj)]

    if filt.empty:
        st.warning("No data after filters.")
        st.stop()

    left, right = st.columns([2,1], gap="large")
    agg_mode = right.selectbox("Aggregate duplicates by", ["max", "min", "mean"], index=0)
    agg_df = filt.groupby(["Outreach_m","Height_m"], as_index=False).agg(Capacity_t=("Capacity_t", agg_mode))
    n_x = right.number_input("Grid points (Outreach)", 30, 400, 120, 10)
    n_y = right.number_input("Grid points (Height)", 30, 400, 120, 10)

    x, y, z = agg_df["Outreach_m"], agg_df["Height_m"], agg_df["Capacity_t"]
    tri = mtri.Triangulation(x, y)
    interpolator = mtri.LinearTriInterpolator(tri, z)
    xi = np.linspace(x.min(), x.max(), int(n_x))
    yi = np.linspace(y.min(), y.max(), int(n_y))
    XI, YI = np.meshgrid(xi, yi)
    ZI = interpolator(XI, YI)

    zmin, zmax = float(np.nanmin(z)), float(np.nanmax(z))
    right.caption(f"Capacity range: {zmin:.2f}–{zmax:.2f} t")
    levels_mode = right.radio("Levels", ["Auto", "N levels", "Custom list"], index=0, horizontal=True)
    filled = right.checkbox("Filled contours", value=True)
    show_points = right.checkbox("Show data points", value=True)
    thresh = right.number_input("Threshold (t)", 0.0, 9999.0, 50.0)
    shade_region = right.checkbox("Shade ≥ threshold", value=True)
    draw_isoline = right.checkbox("Draw isoline", value=True)

    if levels_mode == "Auto":
        levels = None
    elif levels_mode == "N levels":
        n_levels = right.slider("Number of levels", 5, 30, 12)
        levels = np.linspace(zmin, zmax, n_levels)
    else:
        raw = right.text_input("Comma-separated levels", "")
        try:
            levels = [float(v.strip()) for v in raw.split(",") if v.strip()]
            if not levels:
                levels = None
        except Exception:
            st.warning("Could not parse levels; using auto.")
            levels = None

    fig, ax = plt.subplots(figsize=(8,6))
    Zmask = np.ma.array(ZI, mask=np.isnan(ZI))
    cs = ax.contourf(XI, YI, Zmask, levels=levels) if filled else ax.contour(XI, YI, Zmask, levels=levels)
    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label("Capacity (t)")

    if shade_region:
        try:
            ax.contourf(XI, YI, Zmask, levels=[thresh, zmax], alpha=0.35)
        except Exception:
            pass
    if draw_isoline:
        try:
            cs_th = ax.contour(XI, YI, Zmask, levels=[thresh], colors="k", linewidths=2)
            ax.clabel(cs_th, fmt={thresh: f"{thresh:g} t"})
        except Exception:
            pass
    if show_points:
        ax.plot(x, y, ".", ms=2)

    ax.set_xlabel("Outreach (m)")
    ax.set_ylabel("Height (m)")
    st.pyplot(fig, clear_figure=True)

# ==========================================================
# TAB 3 – SWL Envelope (by Cdyn, fixed FJ)
# ==========================================================
with tab_envelope:
    st.subheader("SWL Envelope (by Cdyn) — SWL [t] vs RADIUS [m] (fixed Folding Jib)")

    def duty_to_cdyn(d):
        m = re.search(r"Cd\s*([0-9]+)", d, re.I)
        return float(m.group(1))/100 if m else None

    fj_all = sorted(df["FoldingJib_deg"].dropna().unique().tolist())
    target_fj = st.number_input("Fixed FoldingJib (deg)",
                                float(min(fj_all)), float(max(fj_all)),
                                float(fj_all[0]) if fj_all else 0.0, 0.01)
    fj_tol = st.number_input("FoldingJib tolerance (deg)", 0.0, 5.0, 0.25, 0.05)

    w = df[(df["Duty"]==duty) &
           (df["FoldingJib_deg"].between(target_fj-fj_tol, target_fj+fj_tol))]
    if sel_mj:
        w = w[w["MainJib_deg"].isin(sel_mj)]
    if w.empty:
        st.warning("No data for this Duty/FJ.")
        st.stop()

    col_a,col_b,col_c=st.columns(3)
    with col_a:
        bin_step=st.number_input("Radius bin (m)",0.1,2.0,0.5,0.1)
    with col_b:
        enforce_monotonic=st.checkbox("Force monotonic decrease",True)
    with col_c:
        interpolate_grid=st.checkbox("Interpolate regular grid",True)

    w["Outreach_bin"]=(np.round(w["Outreach_m"]/bin_step)*bin_step).astype(float)
    env=(w.groupby("Outreach_bin",as_index=False)
           .agg(SWL_t=("Capacity_t","max"))
           .sort_values("Outreach_bin"))

    if interpolate_grid and len(env)>=2:
        xi=np.arange(env["Outreach_bin"].min(), env["Outreach_bin"].max()+1e-9, bin_step)
        yi=np.interp(xi, env["Outreach_bin"], env["SWL_t"])
        env=pd.DataFrame({"Outreach_bin":xi,"SWL_t":yi})

    if enforce_monotonic:
        env["SWL_t"]=env["SWL_t"].cummin()

    fig,ax=plt.subplots(figsize=(9,5.5))
    ax.plot(env["Outreach_bin"],env["SWL_t"],lw=3,
            label=f"{duty} @ FJ≈{target_fj:.2f}°")
    ax.set_xlabel("RADIUS [m]")
    ax.set_ylabel("SWL [t]")
    ax.set_title("OFFSHORE LIFT CAPACITY")
    ax.grid(True,which="both",ls="-",lw=0.5,alpha=0.6)
    ax.legend(title="Duty / Cdyn")
    cd=duty_to_cdyn(duty)
    if cd: st.caption(f"DESIGN DYNAMIC FACTOR: {cd:.2f}")
    st.pyplot(fig,clear_figure=True)

    st.download_button("Download SWL Envelope (CSV)",
                       data=env.rename(columns={"Outreach_bin":"Radius_m"}).to_csv(index=False).encode("utf-8"),
                       file_name="swl_envelope_fixed_fj.csv")

# ==========================================================
# TAB 4 – SWL vs MainJib (fixed FJ) Diagnostic
# ==========================================================
with tab_diag:
    st.subheader("SWL vs MainJib (fixed Folding Jib) — Diagnostic")

    fj_all = sorted(df["FoldingJib_deg"].dropna().unique().tolist())
    target_fj = st.number_input("Fixed FoldingJib (deg) for diagnostic",
                                float(min(fj_all)), float(max(fj_all)),
                                float(fj_all[0]) if fj_all else 0.0, 0.01, key="diag_fj")
    fj_tol = st.number_input("FoldingJib tolerance (deg)",0.0,5.0,0.25,0.05,key="diag_tol")

    w = df[(df["Duty"]==duty) &
           (df["FoldingJib_deg"].between(target_fj-fj_tol, target_fj+fj_tol))]
    if w.empty:
        st.warning("No rows for this Duty/FJ.")
        st.stop()

    grp=(w.groupby("MainJib_deg")
           .apply(lambda g: pd.Series({
               "SWL_t": g["Capacity_t"].max(),
               "Radius_at_max": g.loc[g["Capacity_t"].idxmax(),"Outreach_m"]
           }))
           .reset_index()
           .sort_values("MainJib_deg"))

    interp=st.checkbox("Interpolate missing MainJib angles",True)
    if interp and len(grp)>=2:
        mj=grp["MainJib_deg"].to_numpy()
        swl=grp["SWL_t"].to_numpy()
        mj_i=np.arange(mj.min(), mj.max()+1e-9, 0.5)
        swl_i=np.interp(mj_i, mj, swl)
        grp_plot=pd.DataFrame({"MainJib_deg":mj_i,"SWL_t":swl_i})
    else:
        grp_plot=grp[["MainJib_deg","SWL_t"]]

    fig,ax=plt.subplots(figsize=(9,4.5))
    ax.plot(grp_plot["MainJib_deg"],grp_plot["SWL_t"],lw=2.5)
    ax.set_xlabel("MainJib [deg]")
    ax.set_ylabel("SWL [t]")
    ax.set_title(f"SWL vs MainJib — Duty {duty}, FJ≈{target_fj:.2f}°")
    ax.grid(True,ls="-",lw=0.5,alpha=0.6)
    st.pyplot(fig,clear_figure=True)

    st.subheader("Table (max SWL per MainJib)")
    st.dataframe(grp,use_container_width=True)
    st.download_button("Download table (CSV)",
                       data=grp.to_csv(index=False).encode("utf-8"),
                       file_name="swl_vs_mainjib_fixed_fj.csv")

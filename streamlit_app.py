import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.interpolate import PchipInterpolator, LinearNDInterpolator

from pathlib import Path
from scipy.interpolate import PchipInterpolator  # Monotonic cubic spline

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
tab_curve, tab_iso, tab_envelope, tab_diag, tab_outreach_load, tab_optimal_curve = st.tabs([
    "Curve (Capacity vs Outreach)",
    "Iso-load Contour",
    "SWL Envelope (by Cdyn)",
    "SWL vs MainJib (fixed FJ)",
    "Outreach vs Load (booming modes)",
    "Optimal FJ curve (interp toggle)"  # <-- NEW TAB (6)
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

# ==========================================================
# TAB 5 – Outreach vs Load (booming with Main Jib) + Monotonic cubic interpolation
# ==========================================================
with tab_outreach_load:
    st.subheader("Outreach vs Load — booming with Main Jib")

    # Select Duties (Cdyn)
    all_duties = sorted(df["Duty"].dropna().unique().tolist())
    duties_sel = st.multiselect(
        "Select Duty (Cdyn) to plot",
        all_duties,
        default=[duty] if duty in all_duties else (all_duties[:1] if all_duties else []),
        key="t5_duties_sel"
    )

    # Optional: honor sidebar filters
    use_sidebar_filters = st.checkbox(
        "Apply current FoldingJib/MainJib filters",
        value=False,
        key="t5_use_filters"
    )

    # Modes per your spec
    mode = st.radio(
        "Mode",
        [
            "A) Max load per MainJib (optimal FoldingJib for capacity)",
            "B) Force FoldingJib to endpoints for clearance (0° = shortest, 102° = longest)"
        ],
        index=0,
        horizontal=False,
        key="t5_mode"
    )

    # Endpoint controls for Mode B
    if mode.startswith("B)"):
        endpoint_choice = st.radio(
            "Endpoint path",
            ["Shortest (FJ = 0°)", "Longest (FJ = 102°)"],
            index=0,
            horizontal=True,
            key="t5_endpoint_choice"
        )
        fj_target = 0.0 if endpoint_choice.startswith("Shortest") else 102.0
        fj_tol = st.number_input(
            "FoldingJib matching tolerance (deg)",
            min_value=0.0, max_value=5.0, value=0.25, step=0.05,
            help="At each MainJib, we pick rows whose FJ is nearest to the target (0° or 102°). "
                 "If none within tolerance, we use the absolute nearest.",
            key="t5_fj_tol"
        )

    st.divider()
    st.markdown("**Interpolation (monotonic cubic)** — densify between provided MainJib angles (PCHIP)")
    do_interp = st.checkbox(
        "Interpolate between MainJib steps (PCHIP)",
        value=True,
        key="t5_do_interp"
    )
    interp_step = st.slider(
        "Interpolation step (deg)",
        min_value=0.1, max_value=5.0, value=0.5, step=0.1,
        key="t5_interp_step"
    )

    # --- Selection helpers ---
    def pick_optimal_FJ_for_max_load(g: pd.DataFrame) -> pd.Series:
        return g.sort_values(["Capacity_t", "Outreach_m"], ascending=[False, False]).iloc[0]

    def pick_nearest_endpoint_FJ_then_max_load(g: pd.DataFrame, target_fj: float, tol: float) -> pd.Series:
        gg = g.copy()
        gg["fj_err"] = (gg["FoldingJib_deg"] - target_fj).abs()
        min_err = float(gg["fj_err"].min())
        cand = gg[gg["fj_err"] <= max(min_err, tol)]
        return cand.sort_values(["Capacity_t", "Outreach_m"], ascending=[False, False]).iloc[0]

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    last_df_for_download = None
    last_label = None

    for dname in duties_sel:
        W = df[df["Duty"] == dname].copy()
        if use_sidebar_filters:
            if sel_fj:
                W = W[W["FoldingJib_deg"].isin(sel_fj)]
            if sel_mj:
                W = W[W["MainJib_deg"].isin(sel_mj)]

        W = W.dropna(subset=["MainJib_deg", "FoldingJib_deg", "Outreach_m", "Capacity_t"])
        if W.empty:
            continue

        # Build the discrete curve first (one point per MainJib angle)
        rows = []
        for mj, g in W.groupby("MainJib_deg"):
            g = g.dropna(subset=["Outreach_m", "Capacity_t"])
            if g.empty:
                continue
            if mode.startswith("A)"):
                row = pick_optimal_FJ_for_max_load(g)
            else:
                row = pick_nearest_endpoint_FJ_then_max_load(g, fj_target, fj_tol)

            rows.append({
                "MainJib_deg": float(mj),
                "FoldingJib_deg": float(row["FoldingJib_deg"]),
                "Outreach_m": float(row["Outreach_m"]),
                "Load_t": float(row["Capacity_t"])
            })

        if not rows:
            continue

        curve = pd.DataFrame(rows).sort_values("MainJib_deg")

        # --- Monotonic cubic interpolation over MainJib_deg (PCHIP) ---
        if do_interp and len(curve) >= 2:
            mj = curve["MainJib_deg"].to_numpy()
            o  = curve["Outreach_m"].to_numpy()
            l  = curve["Load_t"].to_numpy()

            f_outreach = PchipInterpolator(mj, o)
            f_load     = PchipInterpolator(mj, l)

            mj_i = np.arange(mj.min(), mj.max() + 1e-9, st.session_state["t5_interp_step"])
            o_i  = f_outreach(mj_i)
            l_i  = f_load(mj_i)

            curve_plot = pd.DataFrame({
                "MainJib_deg": mj_i,
                "Outreach_m": o_i,
                "Load_t": l_i
            })
        else:
            curve_plot = curve

        label_txt = (f"{dname} – Mode A (optimal FJ)"
                     if mode.startswith("A)")
                     else f"{dname} – Mode B ({'FJ=0° shortest' if (mode.startswith('B)') and fj_target==0.0) else 'FJ=102° longest'})")
        ax.plot(curve_plot["Outreach_m"], curve_plot["Load_t"], linewidth=2.5, label=label_txt)

        last_df_for_download = curve_plot.copy()
        last_label = label_txt.replace(" ", "_").replace("°","deg")

    ax.set_xlabel("Outreach (m)")
    ax.set_ylabel("Load (t)")
    ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.6)
    ax.set_title("Outreach vs Load while booming Main Jib")
    if duties_sel:
        ax.legend(loc="best")
    st.pyplot(fig, clear_figure=True)

    if last_df_for_download is not None:
        st.subheader("Data points used for the last curve plotted")
        st.dataframe(last_df_for_download, use_container_width=True)
        st.download_button(
            "Download plotted curve (CSV)",
            data=last_df_for_download.to_csv(index=False).encode("utf-8"),
            file_name=f"outreach_vs_load_{last_label}.csv",
            mime="text/csv",
            key="t5_dl"
        )

# ==========================================================
# TAB 6 – Optimal FJ curve (interp toggle) — with ordering, envelope,
#          straight-line comparison, and Simultaneous MJ+FJ path
# ==========================================================
with tab_optimal_curve:
    st.subheader("Optimal Folding Jib per Main Jib — Load vs Outreach")

    all_duties = sorted(df["Duty"].dropna().unique().tolist())
    duties_sel = st.multiselect(
        "Select Duty (Cdyn) to plot",
        all_duties,
        default=[duty] if duty in all_duties else (all_duties[:1] if all_duties else []),
        key="t6_duties_sel"
    )

    use_sidebar_filters = st.checkbox(
        "Apply current FoldingJib/MainJib filters",
        value=False,
        key="t6_use_filters"
    )

    st.markdown("**Interpolation mode for picking optimal points (PCHIP):**")
    interp_mode = st.radio(
        "Choose how to interpolate",
        [
            "None (discrete points)",
            "Interpolate FoldingJib only (per MainJib)",
            "Interpolate MainJib only (densify curve)",
            "Interpolate both (FJ optimum + MJ densify)",
        ],
        index=3,
        horizontal=False,
        key="t6_interp_mode"
    )

    col_interp = st.columns(2)
    with col_interp[0]:
        fj_step = st.slider(
            "FoldingJib sampling step (deg) when interpolating FJ",
            0.1, 5.0, 0.5, 0.1,
            key="t6_fj_step"
        )
    with col_interp[1]:
        mj_step = st.slider(
            "MainJib interpolation step (deg) when interpolating MJ",
            0.1, 5.0, 0.5, 0.1,
            key="t6_mj_step"
        )

    st.divider()
    st.markdown("**Plot behaviour**")
    order_mode = st.radio(
        "X-ordering",
        ["Sort by Outreach (recommended)", "Sort by MainJib (booming order)"],
        index=0, horizontal=True, key="t6_order_mode"
    )
    enforce_envelope = st.checkbox(
        "Enforce monotonic capacity envelope (Load ↓ as Outreach ↑)",
        value=True, key="t6_enforce_env"
    )
    R_bin_step = st.slider(
        "Outreach bin size (m) for envelope",
        0.05, 1.0, 0.25, 0.05, key="t6_r_bin_step"
    )

    st.divider()
    st.markdown("**Straight-line comparison in Outreach–Load space**")
    display_mode = st.radio(
        "What to show",
        ["Original only", "Straight-line only", "Both (overlay)"],
        index=2, horizontal=True, key="t6_display_mode"
    )
    straight_step = st.slider(
        "Outreach step (m) for straight-line curve",
        0.05, 1.0, 0.25, 0.05, key="t6_straight_step"
    )

    st.divider()
    st.markdown("**Simultaneous MJ + FJ interpolation path** (continuous motion across (MJ,FJ))")
    show_mjfj_path = st.checkbox(
        "Overlay coordinated Main-Jib + Folding-Jib path",
        value=True, key="t6_show_path"
    )
    n_path_pts = st.slider(
        "Path samples",
        10, 200, 60, 5, key="t6_path_n"
    )

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    last_plot_df = None
    last_label = None

    # ---------- helpers ----------
    def build_optimal_curve_for_duty(W: pd.DataFrame) -> pd.DataFrame:
        """Return DF(MainJib_deg,FoldingJib_deg,Outreach_m,Load_t) with optimal FJ per MJ.
           Applies FJ and/or MJ PCHIP if selected."""
        rows = []
        for mj, g in W.groupby("MainJib_deg"):
            g = g.dropna(subset=["FoldingJib_deg","Outreach_m","Capacity_t"])
            if g.empty:
                continue

            if st.session_state["t6_interp_mode"] in [
                "Interpolate FoldingJib only (per MainJib)",
                "Interpolate both (FJ optimum + MJ densify)"
            ]:
                g_sorted = g.sort_values("FoldingJib_deg")
                fj_arr = g_sorted["FoldingJib_deg"].to_numpy()
                fj_unique, uniq_idx = np.unique(fj_arr, return_index=True)
                if fj_unique.size >= 2:
                    cap = g_sorted["Capacity_t"].to_numpy()[uniq_idx]
                    out = g_sorted["Outreach_m"].to_numpy()[uniq_idx]
                    fj_min, fj_max = float(fj_unique.min()), float(fj_unique.max())
                    fj_grid = np.arange(fj_min, fj_max + 1e-9, st.session_state["t6_fj_step"])

                    f_cap = PchipInterpolator(fj_unique, cap)
                    f_out = PchipInterpolator(fj_unique, out)

                    cap_i = f_cap(fj_grid)
                    out_i = f_out(fj_grid)
                    k = int(np.argmax(cap_i))
                    rows.append({
                        "MainJib_deg": float(mj),
                        "FoldingJib_deg": float(fj_grid[k]),
                        "Outreach_m": float(out_i[k]),
                        "Load_t": float(cap_i[k])
                    })
                else:
                    gg = g.sort_values(["Capacity_t","Outreach_m"], ascending=[False, False]).iloc[0]
                    rows.append({
                        "MainJib_deg": float(mj),
                        "FoldingJib_deg": float(gg["FoldingJib_deg"]),
                        "Outreach_m": float(gg["Outreach_m"]),
                        "Load_t": float(gg["Capacity_t"])
                    })
            else:
                gg = g.sort_values(["Capacity_t","Outreach_m"], ascending=[False, False]).iloc[0]
                rows.append({
                    "MainJib_deg": float(mj),
                    "FoldingJib_deg": float(gg["FoldingJib_deg"]),
                    "Outreach_m": float(gg["Outreach_m"]),
                    "Load_t": float(gg["Capacity_t"])
                })

        if not rows:
            return pd.DataFrame(columns=["MainJib_deg","FoldingJib_deg","Outreach_m","Load_t"])

        curve = pd.DataFrame(rows).sort_values("MainJib_deg")

        # PCHIP across MJ if selected
        if st.session_state["t6_interp_mode"] in [
            "Interpolate MainJib only (densify curve)",
            "Interpolate both (FJ optimum + MJ densify)"
        ] and len(curve) >= 2:
            mj = curve["MainJib_deg"].to_numpy()
            o  = curve["Outreach_m"].to_numpy()
            l  = curve["Load_t"].to_numpy()
            f_o = PchipInterpolator(mj, o)
            f_l = PchipInterpolator(mj, l)
            mj_i = np.arange(mj.min(), mj.max() + 1e-9, st.session_state["t6_mj_step"])
            o_i  = f_o(mj_i)
            l_i  = f_l(mj_i)
            curve = pd.DataFrame({"MainJib_deg": mj_i, "Outreach_m": o_i, "Load_t": l_i})

        return curve

    def order_curve(df_in: pd.DataFrame) -> pd.DataFrame:
        if order_mode.startswith("Sort by Outreach"):
            return df_in.sort_values("Outreach_m", kind="mergesort")
        return df_in.sort_values("MainJib_deg", kind="mergesort")

    def enforce_capacity_envelope(df_in: pd.DataFrame) -> pd.DataFrame:
        cp = df_in.copy()
        cp["R_bin"] = (np.round(cp["Outreach_m"] / st.session_state["t6_r_bin_step"]) * st.session_state["t6_r_bin_step"]).astype(float)
        env = (cp.groupby("R_bin", as_index=False)
                 .agg(Outreach_m=("Outreach_m","mean"),
                      Load_t=("Load_t","max"))
                 .sort_values("R_bin"))
        env["Load_t"] = env["Load_t"].cummin()
        return env[["Outreach_m","Load_t"]]

    def straight_line_curve(df_in: pd.DataFrame) -> pd.DataFrame:
        """Piecewise linear curve in R–L space."""
        cp = order_curve(df_in)
        if len(cp) < 2:
            return cp[["Outreach_m","Load_t"]]
        x = cp["Outreach_m"].to_numpy()
        y = cp["Load_t"].to_numpy()
        xu, uniq_idx = np.unique(x, return_index=True)
        yu = y[uniq_idx]
        x_grid = np.arange(xu.min(), xu.max() + 1e-9, st.session_state["t6_straight_step"])
        y_grid = np.interp(x_grid, xu, yu)
        return pd.DataFrame({"Outreach_m": x_grid, "Load_t": y_grid})

    # ---------- loop per duty ----------
    for dname in duties_sel:
        W = df[df["Duty"] == dname].copy()
        if use_sidebar_filters:
            if sel_fj:
                W = W[W["FoldingJib_deg"].isin(sel_fj)]
            if sel_mj:
                W = W[W["MainJib_deg"].isin(sel_mj)]
        W = W.dropna(subset=["MainJib_deg","FoldingJib_deg","Outreach_m","Capacity_t"])
        if W.empty:
            continue

        base_curve = build_optimal_curve_for_duty(W)
        if base_curve.empty:
            continue

        # Original (ordered + optional envelope)
        orig = order_curve(base_curve)[["Outreach_m","Load_t"]]
        if enforce_envelope:
            orig = enforce_capacity_envelope(orig)

        # Straight-line variant (and optional envelope)
        sl = straight_line_curve(base_curve)
        if enforce_envelope:
            sl = enforce_capacity_envelope(sl)

        # ---- NEW: Coordinated MJ+FJ path over the raw (MJ,FJ) surface ----
        if show_mjfj_path:
            # Interpolators from scattered data (no need for full rectangular grid)
            pts = W[["MainJib_deg","FoldingJib_deg"]].to_numpy()
            load_vals = W["Capacity_t"].to_numpy()
            rad_vals  = W["Outreach_m"].to_numpy()
            f_load2d = LinearNDInterpolator(pts, load_vals)
            f_rad2d  = LinearNDInterpolator(pts, rad_vals)

            # Use endpoints from the non-interpolated optimal curve (preserves real MJ values)
            opt_per_mj = (W.groupby("MainJib_deg")
                            .apply(lambda g: g.loc[g["Capacity_t"].idxmax()][["FoldingJib_deg","Outreach_m","Capacity_t"]])
                            .reset_index()
                         )
            mj_min, mj_max = opt_per_mj["MainJib_deg"].min(), opt_per_mj["MainJib_deg"].max()
            # Pick two endpoints via sliders
            st.markdown("**MJ+FJ Path Endpoints**")
            c1, c2 = st.columns(2)
            with c1:
                mj_start = st.slider("Start MainJib (deg)", float(mj_min), float(mj_max), float(mj_min), 0.01, key="t6_path_mj_start")
            with c2:
                mj_end   = st.slider("End MainJib (deg)",   float(mj_min), float(mj_max), float(mj_max), 0.01, key="t6_path_mj_end")

            # Find optimal FJ at those MJ (discrete; we interpolate FJ linearly between them)
            def best_fj_at(mj_val: float) -> float:
                # nearest MJ group
                row = opt_per_mj.iloc[(opt_per_mj["MainJib_deg"] - mj_val).abs().argmin()]
                return float(row["FoldingJib_deg"])

            fj_start = best_fj_at(mj_start)
            fj_end   = best_fj_at(mj_end)

            tt = np.linspace(0, 1, st.session_state["t6_path_n"])
            MJp = mj_start + tt * (mj_end - mj_start)
            FJp = fj_start + tt * (fj_end - fj_start)

            path_load = f_load2d(np.column_stack([MJp, FJp]))
            path_R    = f_rad2d (np.column_stack([MJp, FJp]))

            path_df = pd.DataFrame({"MainJib_deg": MJp, "FoldingJib_deg": FJp,
                                    "Outreach_m": path_R, "Load_t": path_load}).dropna()

            # Order / enforce envelope to match the current display settings
            path_show = order_curve(path_df)[["Outreach_m","Load_t"]]
            if enforce_envelope and len(path_show):
                path_show = enforce_capacity_envelope(path_show)

        # Plot according to display mode
        if display_mode == "Original only":
            ax.plot(orig["Outreach_m"], orig["Load_t"], linewidth=2.8, label=f"{dname} — Original")
            last_plot_df = orig.copy(); last_plot_df["Variant"] = "Original"
            last_label = f"{dname}_Original"
        elif display_mode == "Straight-line only":
            ax.plot(sl["Outreach_m"], sl["Load_t"], linewidth=2.8, linestyle="--", label=f"{dname} — Straight-line")
            last_plot_df = sl.copy(); last_plot_df["Variant"] = "Straight-line"
            last_label = f"{dname}_StraightLine"
        else:  # Both (overlay)
            ax.plot(orig["Outreach_m"], orig["Load_t"], linewidth=2.8, label=f"{dname} — Original")
            ax.plot(sl["Outreach_m"], sl["Load_t"], linewidth=2.5, linestyle="--", label=f"{dname} — Straight-line")
            both = pd.concat([
                orig.assign(Variant="Original"),
                sl.assign(Variant="Straight-line")
            ], ignore_index=True)
            last_plot_df = both
            last_label = f"{dname}_Both"

        # Overlay the coordinated MJ+FJ path
        if show_mjfj_path and 'path_show' in locals() and len(path_show):
            ax.plot(path_show["Outreach_m"], path_show["Load_t"], linewidth=2.2, linestyle=":", label=f"{dname} — MJ+FJ path")
            # append to download
            if last_plot_df is not None:
                last_plot_df = pd.concat([last_plot_df, path_show.assign(Variant="MJ+FJ path")], ignore_index=True)
            else:
                last_plot_df = path_show.assign(Variant="MJ+FJ path")
            last_label = f"{last_label}_withPath" if last_label else f"{dname}_PathOnly"

    ax.set_xlabel("Outreach (m)")
    ax.set_ylabel("Load (t)")
    ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.6)
    ax.set_title("Maximum load vs outreach (Main Jib booming; Folding Jib at optimal position)")
    if duties_sel:
        ax.legend(loc="best")
    st.pyplot(fig, clear_figure=True)

    if last_plot_df is not None:
        st.subheader("Data used for the last curve plotted")
        st.dataframe(last_plot_df, use_container_width=True)
        st.download_button(
            "Download plotted curve (CSV)",
            data=last_plot_df.to_csv(index=False).encode("utf-8"),
            file_name=f"optimal_fj_curve_{last_label}.csv",
            mime="text/csv",
            key="t6_dl"
        )

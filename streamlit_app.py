
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator

st.set_page_config(page_title="Crane Curve Viewer", layout="wide")

DATA_PATH = "./data/CraneData_streamlit_long.csv"

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    req = ["FoldingJib","MainJib","Outreach","Height","Condition","Environment","Cd","Load"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    for c in ["FoldingJib","MainJib","Outreach","Height","Cd","Load"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["FoldingJib","MainJib","Outreach","Height","Load"])
    df["Environment"] = df["Environment"].astype("category")
    return df

def nearest_point(df, fold_angle, main_angle):
    if df.empty:
        return np.nan, np.nan, np.nan
    idx = ((df["FoldingJib"] - fold_angle).abs() + (df["MainJib"] - main_angle).abs()).idxmin()
    row = df.loc[idx]
    return float(row["Outreach"]), float(row["Height"]), float(row["Load"])

def extract_iso_polyline(XI, YI, ZI, target):
    \"\"\"Return a list of (x, y) points along the iso-load contour.
    XI, YI are 2D meshgrids; ZI is same shape; target is load value.
    \"\"\"
    if XI is None or YI is None or ZI is None:
        return []
    try:
        fig_tmp, ax_tmp = plt.subplots()
        cs = ax_tmp.contour(XI, YI, ZI, levels=[target])
        plt.close(fig_tmp)
    except Exception:
        return []

    pts = []
    if len(cs.allsegs) and len(cs.allsegs[0]):
        # Take all segments and concatenate (keep order per segment)
        for seg in cs.allsegs[0]:
            for (xx, yy) in seg:
                pts.append((float(xx), float(yy)))
    # Deduplicate consecutive duplicates
    dedup = []
    for p in pts:
        if not dedup or (abs(dedup[-1][0]-p[0])>1e-9 or abs(dedup[-1][1]-p[1])>1e-9):
            dedup.append(p)
    return dedup

def contour_plot(df_env, target=None, marker=None, show_colorbar=True, title="Rated capacity related to height and radius"):
    if df_env.empty:
        st.warning("No data to plot for this environment/filters.")
        return None, None, None, None
    x = df_env["Outreach"].values
    y = df_env["Height"].values
    z = df_env["Load"].values

    # Ensure enough unique points for triangulation
    if len(np.unique(x)) < 3 or len(np.unique(y)) < 3:
        st.warning("Not enough unique (Outreach, Height) points to build contour.")
        return None, None, None, None

    tri = Triangulation(x, y)
    interp = LinearTriInterpolator(tri, z)

    n = 400
    xi = np.linspace(x.min(), x.max(), n)
    yi = np.linspace(y.min(), y.max(), n)
    XI, YI = np.meshgrid(xi, yi)
    ZI = interp(XI, YI)

    fig, ax = plt.subplots(figsize=(8, 7))
    hm = ax.contourf(XI, YI, ZI, levels=20)
    iso_pts = []
    if target is not None:
        try:
            cs = ax.contour(XI, YI, ZI, levels=[target], linewidths=2.0)
        except Exception:
            cs = None
        # Also extract iso polyline points for later plotting / export
        iso_pts = extract_iso_polyline(XI, YI, ZI, target)

    if marker is not None and all(np.isfinite(marker)):
        ox, hz = marker
        # RED current point
        ax.plot([ox], [hz], marker="o", markersize=7, markerfacecolor="red", markeredgecolor="red")

    ax.set_xlabel("y Radius [m]")
    ax.set_ylabel("z Height [m]")
    ax.set_title(title)

    if show_colorbar:
        cbar = fig.colorbar(hm, ax=ax, shrink=0.9)
        cbar.set_label("Load [t]")
    return fig, XI, YI, ZI, iso_pts

def capacity_envelope_vs_radius(df_env):
    if df_env.empty:
        return np.array([]), np.array([])
    g = (
        df_env
        .groupby(pd.cut(df_env["Outreach"], bins=200, include_lowest=True))["Load"]
        .max()
        .dropna()
    )
    xs = np.array([float(iv.mid) for iv in g.index])
    ys = g.values
    order = np.argsort(xs)
    return xs[order], ys[order]

def main():
    st.markdown("## DCN Picasso DSV Crane Curve Interface")

    try:
        df = load_data(DATA_PATH)
    except Exception as e:
        st.error(f"Could not load data from {DATA_PATH}: {e}")
        st.stop()

    # Environment selector
    env_options = ["Harbour", "Deck", "SeaSubsea", "Subsea"]
    chosen_env = st.radio("Environment", env_options, horizontal=True, index=0)
    df_env = df[df["Environment"] == chosen_env]
    if df_env.empty:
        st.warning(f"No rows found for environment: {chosen_env}")
        st.stop()

    # Layout: Left plot, Right controls
    left, right = st.columns([2.2, 1.0])

    with right:
        st.markdown("### Current point")
        mj_min, mj_max = float(df_env["MainJib"].min()), float(df_env["MainJib"].max())
        fj_min, fj_max = float(df_env["FoldingJib"].min()), float(df_env["FoldingJib"].max())

        main_angle = st.slider("Main angle [deg]", mj_min, mj_max, float(np.round((mj_min+mj_max)/2,2)), step=0.01)
        fold_angle = st.slider("Folding angle [deg]", fj_min, fj_max, float(np.round((fj_min+fj_max)/2,2)), step=0.01)

        ox, hz, rated = nearest_point(df_env, fold_angle, main_angle)
        daf = float(df_env["Cd"].iloc[0]) if "Cd" in df_env.columns and not df_env.empty else np.nan

        k1, k2 = st.columns(2)
        with k1:
            st.text("Radius [m]")
            st.header(f"{ox:.2f}" if np.isfinite(ox) else "-")
            st.text("Rated load [t]")
            st.header(f"{rated:.1f}" if np.isfinite(rated) else "-")
        with k2:
            st.text("Height [m]")
            st.header(f"{hz:.2f}" if np.isfinite(hz) else "-")
            st.text("DAF")
            st.header(f"{daf:.2f}" if np.isfinite(daf) else "-")

        minL, maxL = float(df_env["Load"].min()), float(df_env["Load"].max())
        iso_default = float(np.round((minL + maxL)/2, 1))
        target = st.slider("Iso-load [t] (overlay)", minL, maxL, iso_default, step=0.1)

    with left:
        # --- Main contour chart with iso-overlay and RED current point ---
        fig, XI, YI, ZI, iso_pts = contour_plot(
            df_env,
            target=target,
            marker=(ox, hz),
            title="Rated capacity related to height and radius"
        )
        if fig is not None:
            st.pyplot(fig, use_container_width=True)

        # --- Capacity curve (Print chart) ---
        st.markdown("### Print chart")
        xs, ys = capacity_envelope_vs_radius(df_env)

        if len(xs) and len(ys):
            fig2, ax2 = plt.subplots(figsize=(8, 4.5))
            ax2.plot(xs, ys, linewidth=2.0)
            ax2.set_xlabel("RADIUS (m)")
            ax2.set_ylabel("SWL (t)")
            ax2.set_title("OFFSHORE LIFT CAPACITY")
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2, use_container_width=True)

            env_csv = pd.DataFrame({"Radius_m": xs, "SWL_t": ys})
            st.download_button("Download capacity curve CSV",
                               env_csv.to_csv(index=False).encode("utf-8"),
                               file_name=f"capacity_curve_{chosen_env}.csv", mime="text/csv")
        else:
            st.info("No data available to build capacity curve for the selected environment.")

        # --- New bottom chart: Iso-load polyline Radius vs Height ---
        st.markdown("### Iso-load profile (Radius vs Height)")
        if iso_pts:
            iso_df = pd.DataFrame(iso_pts, columns=["Radius_m","Height_m"]).sort_values("Radius_m")
            fig3, ax3 = plt.subplots(figsize=(8, 4.5))
            ax3.plot(iso_df["Radius_m"], iso_df["Height_m"], linewidth=2.0)
            ax3.set_xlabel("RADIUS (m)")
            ax3.set_ylabel("HEIGHT (m)")
            ax3.set_title(f"Iso-load polyline @ {target:.2f} t")
            ax3.grid(True, alpha=0.3)
            st.pyplot(fig3, use_container_width=True)

            st.download_button("Download iso-load polyline CSV",
                               iso_df.to_csv(index=False).encode("utf-8"),
                               file_name=f"iso_polyline_{chosen_env}_{target:.2f}t.csv",
                               mime="text/csv")
        else:
            st.info("Iso-load contour could not be extracted for this target. Try another value.")

if __name__ == "__main__":
    main()

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
    # Pick the row nearest to the requested angles
    idx = ((df["FoldingJib"] - fold_angle).abs() + (df["MainJib"] - main_angle).abs()).idxmin()
    row = df.loc[idx]
    return float(row["Outreach"]), float(row["Height"]), float(row["Load"])

def contour_plot(df_env, target=None, marker=None, show_colorbar=True, title="Rated capacity related to height and radius"):
    # Build interpolation in (Outreach, Height) plane
    x = df_env["Outreach"].values
    y = df_env["Height"].values
    z = df_env["Load"].values

    tri = Triangulation(x, y)
    interp = LinearTriInterpolator(tri, z)

    n = 400
    xi = np.linspace(x.min(), x.max(), n)
    yi = np.linspace(y.min(), y.max(), n)
    XI, YI = np.meshgrid(xi, yi)
    ZI = interp(XI, YI)

    fig, ax = plt.subplots(figsize=(8, 7))
    hm = ax.contourf(XI, YI, ZI, levels=20)
    if target is not None:
        ax.contour(XI, YI, ZI, levels=[target], linewidths=2.0)

    if marker is not None:
        ox, hz = marker
        ax.plot([ox], [hz], marker="o", markersize=6)

    ax.set_xlabel("y Radius [m]")
    ax.set_ylabel("z Height [m]")
    ax.set_title(title)

    if show_colorbar:
        cbar = fig.colorbar(hm, ax=ax, shrink=0.9)
        cbar.set_label("Load [t]")

    return fig

def capacity_envelope_vs_radius(df_env):
    # Build the capacity curve similar to the 2nd screenshot:
    # For each outreach bucket, take the maximum load over all heights / angle combos.
    g = (
        df_env
        .groupby(pd.cut(df_env["Outreach"], bins=200, include_lowest=True))["Load"]
        .max()
        .dropna()
    )
    # Convert IntervalIndex to midpoints
    xs = np.array([float(iv.mid) for iv in g.index])
    ys = g.values
    # Sort by radius
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    return xs, ys

def main():
    st.markdown("## MACGREGOR-like Crane Curve Interface")

    try:
        df = load_data(DATA_PATH)
    except Exception as e:
        st.error(f"Could not load data from {DATA_PATH}: {e}")
        st.stop()

    # Top environment selector (mimic the dot legend)
    env_options = ["Harbour", "Deck", "SeaSubsea", "Subsea"]
    # Map back to rows
    env_label_map = {
        "Harbour":"Harbour",
        "Deck":"Deck",
        "SeaSubsea":"SeaSubsea",
        "Subsea":"Subsea",
    }
    cols = st.columns(len(env_options))
    chosen_env = env_options[0]
    for i, name in enumerate(env_options):
        with cols[i]:
            if st.toggle(name, value=(i==0), key=f"env_{name}"):
                chosen_env = name
                # Turn the others off visually by resetting state — simple selection effect
                for j, n2 in enumerate(env_options):
                    if n2 != name and f"env_{n2}" in st.session_state:
                        st.session_state[f"env_{n2}"] = False

    # Filter df by environment string
    df_env = df[df["Environment"] == env_label_map[chosen_env]]
    if df_env.empty:
        st.warning(f"No rows found for environment: {chosen_env}")
        st.stop()

    # Layout like screenshot: Left = plot area, Right = controls
    left, right = st.columns([2.2, 1.0])

    with right:
        st.markdown("### Current point")
        # Angle sliders
        mj_min, mj_max = float(df_env["MainJib"].min()), float(df_env["MainJib"].max())
        fj_min, fj_max = float(df_env["FoldingJib"].min()), float(df_env["FoldingJib"].max())

        main_angle = st.slider("Main angle [deg]", mj_min, mj_max, float(np.round((mj_min+mj_max)/2,2)), step=0.01)
        fold_angle = st.slider("Folding angle [deg]", fj_min, fj_max, float(np.round((fj_min+fj_max)/2,2)), step=0.01)

        # Find nearest data point + DAF (Cd)
        ox, hz, rated = nearest_point(df_env, fold_angle, main_angle)
        daf = float(df_env["Cd"].iloc[0]) if "Cd" in df_env.columns else np.nan

        # Show readouts (styled like screenshot)
        k1, k2 = st.columns(2)
        with k1:
            st.text("Radius [m]")
            st.header(f"{ox:.2f}")
            st.text("Rated load [t]")
            st.header(f"{rated:.1f}")
        with k2:
            st.text("Height [m]")
            st.header(f"{hz:.2f}")
            st.text("DAF")
            st.header(f"{daf:.2f}" if not np.isnan(daf) else "-")

        # Iso-load selector (optional)
        minL, maxL = float(df_env["Load"].min()), float(df_env["Load"].max())
        iso_default = float(np.round((minL + maxL)/2, 1))
        target = st.slider("Iso-load [t] (overlay)", minL, maxL, iso_default, step=0.1)

    with left:
        # Main contour plot with marker and iso-load line
        fig = contour_plot(df_env, target=target, marker=(ox, hz))
        st.pyplot(fig, use_container_width=True)

        # --- "Print chart" area (capacity curve) ---
        st.markdown("### Print chart")
        xs, ys = capacity_envelope_vs_radius(df_env)

        fig2, ax2 = plt.subplots(figsize=(8, 4.5))
        ax2.plot(xs, ys, linewidth=2.0)
        ax2.set_xlabel("RADIUS (m)")
        ax2.set_ylabel("SWL (t)")
        ax2.set_title("OFFSHORE LIFT CAPACITY")
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2, use_container_width=True)

        # Allow CSV download of the envelope and current point
        env_csv = pd.DataFrame({"Radius_m": xs, "SWL_t": ys})
        st.download_button("Download capacity curve CSV", env_csv.to_csv(index=False).encode("utf-8"),
                           file_name=f"capacity_curve_{chosen_env}.csv", mime="text/csv")

if __name__ == "__main__":
    main()


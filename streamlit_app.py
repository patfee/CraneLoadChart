import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator

st.set_page_config(page_title="Knuckle Boom Crane Curves", layout="wide")

DATA_PATH = "./data/CraneData_streamlit_long.csv"

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Basic schema sanity
    required = ["FoldingJib","MainJib","Outreach","Height","Condition","Environment","Cd","Load"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    # Enforce dtypes
    for col in ["FoldingJib","MainJib","Outreach","Height","Cd","Load"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Environment"] = df["Environment"].astype("category")
    df["Condition"] = df["Condition"].astype("category")
    # Drop clearly invalid rows
    df = df.dropna(subset=["FoldingJib","MainJib","Outreach","Height","Load"])
    return df

def download_link(df: pd.DataFrame, filename: str, label: str = "Download CSV"):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, data=csv, file_name=filename, mime="text/csv")

def sidebar_filters(df: pd.DataFrame):
    st.sidebar.header("Filters")
    envs = sorted(df["Environment"].dropna().unique().tolist())
    env = st.sidebar.selectbox("Environment", envs, index=0 if envs else None)
    df = df[df["Environment"] == env]

    # Optional angle filters
    st.sidebar.subheader("Angle filters (optional)")
    fj_min, fj_max = float(df["FoldingJib"].min()), float(df["FoldingJib"].max())
    mj_min, mj_max = float(df["MainJib"].min()), float(df["MainJib"].max())
    fj_range = st.sidebar.slider("FoldingJib range (deg)", fj_min, fj_max, (fj_min, fj_max), step=0.01)
    mj_range = st.sidebar.slider("MainJib range (deg)", mj_min, mj_max, (mj_min, mj_max), step=0.01)
    df = df[(df["FoldingJib"].between(*fj_range)) & (df["MainJib"].between(*mj_range))]

    st.sidebar.markdown("---")
    st.sidebar.caption("Data file")
    st.sidebar.code(os.path.abspath(DATA_PATH))

    return df, env, fj_range, mj_range

def plot_isoload_contour(df: pd.DataFrame, env: str):
    st.subheader(f"Iso-load curve — {env}")
    minL, maxL = float(df["Load"].min()), float(df["Load"].max())
    default_target = np.clip((minL + maxL) / 2, minL, maxL)
    target = st.slider("Iso-load target (t)", minL, maxL, float(np.round(default_target, 1)), step=0.1)

    # Build triangulation in the XY-plane: Outreach vs Height
    x = df["Outreach"].values
    y = df["Height"].values
    z = df["Load"].values

    # Defensive: ensure we have enough unique points
    if len(x) < 10 or len(np.unique(x)) < 3 or len(np.unique(y)) < 3:
        st.warning("Not enough unique (Outreach, Height) points to compute contour.")
        return

    tri = Triangulation(x, y)
    # Interpolator for smooth contouring
    interp = LinearTriInterpolator(tri, z)

    # Create a regular grid for plotting
    n = 400  # dense grid
    xi = np.linspace(x.min(), x.max(), n)
    yi = np.linspace(y.min(), y.max(), n)
    XI, YI = np.meshgrid(xi, yi)
    ZI = interp(XI, YI)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    # filled background heatmap
    hm = ax.contourf(XI, YI, ZI, levels=16)
    # iso-load line
    cs = ax.contour(XI, YI, ZI, levels=[target], linewidths=2.0)

    # Labels and colorbar
    ax.set_xlabel("Outreach (m)")
    ax.set_ylabel("Hook Height (m)")
    ax.set_title(f"Iso-load ≈ {target:.2f} t — {env}")
    cbar = fig.colorbar(hm, ax=ax, shrink=0.85)
    cbar.set_label("Load (t)")

    st.pyplot(fig)

    # Extract the iso-line as a dataframe (if contour exists)
    iso_points = []
    if len(cs.allsegs) and len(cs.allsegs[0]):
        for seg in cs.allsegs[0]:
            for (xx, yy) in seg:
                iso_points.append((xx, yy, target))
    if iso_points:
        iso_df = pd.DataFrame(iso_points, columns=["Outreach","Height","LoadTarget"])
        st.caption("Extracted iso-load polyline points")
        st.dataframe(iso_df.head(200))
        download_link(iso_df, f"iso_{env}_{target:.2f}t.csv", "Download iso-load polyline")

def plot_heatmap(df: pd.DataFrame, env: str):
    st.subheader(f"Heatmap — Load over (Outreach, Height) — {env}")

    x = df["Outreach"].values
    y = df["Height"].values
    z = df["Load"].values

    tri = Triangulation(x, y)
    interp = LinearTriInterpolator(tri, z)

    n = 300
    xi = np.linspace(x.min(), x.max(), n)
    yi = np.linspace(y.min(), y.max(), n)
    XI, YI = np.meshgrid(xi, yi)
    ZI = interp(XI, YI)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    hm = ax.contourf(XI, YI, ZI, levels=20)
    ax.set_xlabel("Outreach (m)")
    ax.set_ylabel("Hook Height (m)")
    ax.set_title(f"Load (t) — {env}")
    cbar = fig.colorbar(hm, ax=ax, shrink=0.85)
    cbar.set_label("Load (t)")

    st.pyplot(fig)

def plot_angle_slices(df: pd.DataFrame, env: str):
    st.subheader(f"Slices — angle-curves — {env}")
    mode = st.radio("Slice mode", ["Fix FoldingJib (vary MainJib)", "Fix MainJib (vary FoldingJib)"], horizontal=True)
    measure = st.selectbox("Y-axis", ["Load","Height","Outreach"], index=0)

    if mode.startswith("Fix Folding"):
        fj_vals = sorted(df["FoldingJib"].unique())
        fj_pick = st.select_slider("FoldingJib (deg)", options=[float(np.round(v,2)) for v in fj_vals], value=float(np.round(fj_vals[len(fj_vals)//2],2)))
        d = df[np.isclose(df["FoldingJib"], fj_pick)]
        d = d.sort_values("MainJib")
        x_label = "MainJib (deg)"
        x = d["MainJib"]
    else:
        mj_vals = sorted(df["MainJib"].unique())
        mj_pick = st.select_slider("MainJib (deg)", options=[float(np.round(v,2)) for v in mj_vals], value=float(np.round(mj_vals[len(mj_vals)//2],2)))
        d = df[np.isclose(df["MainJib"], mj_pick)]
        d = d.sort_values("FoldingJib")
        x_label = "FoldingJib (deg)"
        x = d["FoldingJib"]

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.plot(x, d[measure], marker="o", linewidth=1.6)
    ax.set_xlabel(x_label)
    ax.set_ylabel(measure)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.caption("Slice data")
    st.dataframe(d[["FoldingJib","MainJib","Outreach","Height","Load"]].reset_index(drop=True).head(200))
    download_link(d, f"slice_{env}_{measure}.csv", "Download slice CSV")

def main():
    st.title("Knuckle Boom Crane — Curve Explorer")
    st.markdown("Load: **CraneData_streamlit_long.csv** in `./data/`")

    # Data loader
    try:
        df = load_data(DATA_PATH)
    except Exception as e:
        st.error(f"Could not load data from {DATA_PATH}: {e}")
        st.stop()

    # Sidebar filtering
    df_filt, env, fj_range, mj_range = sidebar_filters(df)

    tabs = st.tabs(["Iso-load curve", "Heatmap", "Angle slices", "Table"])

    with tabs[0]:
        plot_isoload_contour(df_filt, env)

    with tabs[1]:
        plot_heatmap(df_filt, env)

    with tabs[2]:
        plot_angle_slices(df_filt, env)

    with tabs[3]:
        st.subheader(f"Filtered rows — {env}")
        st.dataframe(df_filt.head(500))
        download_link(df_filt, f"filtered_{env}.csv", "Download filtered CSV")

    st.markdown("---")
    st.caption("Tips: Use the sidebar to refine angles. Iso-load curve extracts the polyline of points at a chosen tonnage.")

if __name__ == "__main__":
    main()



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


def contour_plot(df_env, target=None, marker=None, show_colorbar=True, title="Rated capacity related to height and radius"):
    if df_env.empty:
        st.warning("No data to plot for this environment/filters.")
        return None, None, None, None
    x = df_env["Outreach"].values
    y = df_env["Height"].values
    z = df_env["Load"].values

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
    if target is not None:
        try:
            ax.contour(XI, YI, ZI, levels=[target], linewidths=2.0)
        except Exception:
            pass

    if marker is not None and all(np.isfinite(marker)):
        ox, hz = marker
        ax.plot([ox], [hz], marker="o", markersize=7, markerfacecolor="red", markeredgecolor="red")

    ax.set_xlabel("y Radius [m]")
    ax.set_ylabel("z Height [m]")
    ax.set_title(title)

    if show_colorbar:
        cbar = fig.colorbar(hm, ax=ax, shrink=0.9)
        cbar.set_label("Load [t]")
    return fig, XI, YI, ZI, ax


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


def synced_slider_number(label, key_base, min_val, max_val, default, step=0.01, fmt="%.2f"):
    if key_base not in st.session_state:
        st.session_state[key_base] = float(np.clip(default, min_val, max_val))
        st.session_state[f"{key_base}_num"] = st.session_state[key_base]

    def from_num():
        v = float(st.session_state[f"{key_base}_num"])
        v = float(np.clip(v, min_val, max_val))
        st.session_state[key_base] = v

    def from_sld():
        v = float(st.session_state[key_base])
        st.session_state[f"{key_base}_num"] = v

    col_num, col_sld = st.columns([1, 3])
    with col_num:
        st.number_input(label, min_value=float(min_val), max_value=float(max_val),
                        value=float(st.session_state[f"{key_base}_num"]), step=step,
                        format=fmt, key=f"{key_base}_num", on_change=from_num)
    with col_sld:
        st.slider("", min_val, max_val, value=float(st.session_state[key_base]), step=step,
                  key=key_base, on_change=from_sld)
    return float(st.session_state[key_base])


# -------- Geometry estimation and drawing --------
@st.cache_data
def estimate_geometry(df_all: pd.DataFrame, relative=True):
    """Least-squares estimate of base (x0,y0), main length L1, folding length L2."""
    # Use all unique rows to fit
    thetas1 = np.deg2rad(df_all["MainJib"].values.astype(float))
    if relative:
        thetas2 = np.deg2rad((df_all["MainJib"] + df_all["FoldingJib"]).values.astype(float))
    else:
        thetas2 = np.deg2rad(df_all["FoldingJib"].values.astype(float))

    x = df_all["Outreach"].values.astype(float)
    y = df_all["Height"].values.astype(float)

    # Build linear system for parameters [x0, y0, L1, L2]
    # x = x0 + L1*cos(t1) + L2*cos(t2)
    # y = y0 + L1*sin(t1) + L2*sin(t2)
    # Stack equations
    n = len(x)
    A = np.zeros((2*n, 4))
    b = np.zeros(2*n)
    A[0:n, 0] = 1.0       # x0
    A[0:n, 2] = np.cos(thetas1)  # L1
    A[0:n, 3] = np.cos(thetas2)  # L2
    b[0:n] = x

    A[n:, 1] = 1.0        # y0
    A[n:, 2] = np.sin(thetas1)   # L1
    A[n:, 3] = np.sin(thetas2)   # L2
    b[n:] = y

    # Solve least squares
    params, *_ = np.linalg.lstsq(A, b, rcond=None)
    x0, y0, L1, L2 = params
    return float(x0), float(y0), float(L1), float(L2)


def forward_kinematics(main_deg, fold_deg, x0, y0, L1, L2, relative=True):
    t1 = np.deg2rad(main_deg)
    if relative:
        t2 = np.deg2rad(main_deg + fold_deg)
    else:
        t2 = np.deg2rad(fold_deg)
    x_elbow = x0 + L1*np.cos(t1)
    y_elbow = y0 + L1*np.sin(t1)
    x_tip = x_elbow + L2*np.cos(t2)
    y_tip = y_elbow + L2*np.sin(t2)
    return (x0, y0), (x_elbow, y_elbow), (x_tip, y_tip)


def draw_crane(ax, main_deg, fold_deg, df_env, relative=True, color="white"):
    # Estimate geometry from the selected environment (should be same for all envs)
    try:
        x0, y0, L1, L2 = estimate_geometry(df_env[["MainJib","FoldingJib","Outreach","Height"]], relative=relative)
    except Exception:
        # fallback to using all data
        x0, y0, L1, L2 = estimate_geometry(df, relative=relative)

    base, elbow, tip = forward_kinematics(main_deg, fold_deg, x0, y0, L1, L2, relative=relative)
    # Draw base square for reference
    bx, by = base
    size = max(L1, L2) * 0.04
    rect_x = [bx - size, bx + size, bx + size, bx - size, bx - size]
    rect_y = [by - size, by - size, by + size, by + size, by - size]
    ax.plot(rect_x, rect_y, color=color, linewidth=1.5)

    # Links
    ax.plot([base[0], elbow[0]], [base[1], elbow[1]], color=color, linewidth=2.0)
    ax.plot([elbow[0], tip[0]], [elbow[1], tip[1]], color=color, linewidth=2.0)
    # Joints
    ax.plot(base[0], base[1], "o", color=color, markersize=4, markerfacecolor=color)
    ax.plot(elbow[0], elbow[1], "o", color=color, markersize=4, markerfacecolor=color)
    ax.plot(tip[0], tip[1], "o", color=color, markersize=4, markerfacecolor=color)


def main():
    st.markdown("## DCN Picasso DSV Crane Curve Interface")

    try:
        df_all = load_data(DATA_PATH)
    except Exception as e:
        st.error(f"Could not load data from {DATA_PATH}: {e}")
        st.stop()

    env_options = ["Harbour", "Deck", "SeaSubsea", "Subsea"]
    chosen_env = st.radio("Environment", env_options, horizontal=True, index=0)
    df_env = df_all[df_all["Environment"] == chosen_env]
    if df_env.empty:
        st.warning(f"No rows found for environment: {chosen_env}")
        st.stop()

    left, right = st.columns([2.2, 1.0])

    with right:
        st.markdown("### Current point")
        mj_min, mj_max = float(df_env["MainJib"].min()), float(df_env["MainJib"].max())
        fj_min, fj_max = float(df_env["FoldingJib"].min()), float(df_env["FoldingJib"].max())

        main_angle = synced_slider_number("Main angle [deg]", "main_angle",
                                          mj_min, mj_max, default=(mj_min+mj_max)/2, step=0.01)
        fold_angle = synced_slider_number("Folding angle [deg]", "fold_angle",
                                          fj_min, fj_max, default=(fj_min+fj_max)/2, step=0.01)

        ox, hz, rated = nearest_point(df_env, fold_angle, main_angle)
        daf = float(df_env["Cd"].iloc[0]) if "Cd" in df_env.columns and not df_env.empty else np.nan

        # Geometry options
        st.markdown("#### Geometry options")
        relative = st.checkbox("Folding angle is **relative** to main angle", value=True, help="If unchecked, folding angle is treated as absolute.")

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
        target = synced_slider_number("Iso-load [t] (overlay)", "iso_load", minL, maxL,
                                      default=(minL + maxL) / 2, step=0.1, fmt="%.1f")

    with left:
        fig, XI, YI, ZI, ax = contour_plot(
            df_env,
            target=target,
            marker=(ox, hz),
            title="Rated capacity related to height and radius"
        )
        if fig is not None:
            # Overlay dynamic crane geometry
            draw_crane(ax, main_angle, fold_angle, df_env, relative=relative, color="white")
            st.pyplot(fig, use_container_width=True)

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


if __name__ == "__main__":
    main()


import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator

st.set_page_config(page_title="Crane Curve Viewer", layout="wide")

DATA_PATH = "./data/CraneData_streamlit_long.csv"
PEDESTAL_INBOARD_M = 2.26  # centre of slew bearing is 2.26 m inboard of hull side


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


def contour_plot(x, y, z, target=None, marker=None, show_colorbar=True, title="Rated capacity related to height and radius"):
    if len(x) < 3 or len(y) < 3:
        st.warning("Not enough points to build contour.")
        return None, None, None, None, None
    if len(np.unique(x)) < 3 or len(np.unique(y)) < 3:
        st.warning("Not enough unique (Outreach, Height) points to build contour.")
        return None, None, None, None, None

    tri = Triangulation(x, y)
    interp = LinearTriInterpolator(tri, z)

    n = 400
    xi = np.linspace(np.min(x), np.max(x), n)
    yi = np.linspace(np.min(y), np.max(y), n)
    XI, YI = np.meshgrid(xi, yi)
    ZI = interp(XI, YI)

    fig, ax = plt.subplots(figsize=(8, 7))
    hm = ax.contourf(XI, YI, ZI, levels=20)
    if target is not None:
        try:
            ax.contour(XI, YI, ZI, levels=[target], linewidths=2.0, colors=['#34113F'])
        except Exception:
            pass

    # Hull side guideline
    ax.axvline(PEDESTAL_INBOARD_M, linestyle="--", linewidth=1.2)
    ax.text(PEDESTAL_INBOARD_M, ax.get_ylim()[1], " Hull side", va="top", ha="left", rotation=90)

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


@st.cache_data
def estimate_geometry(df_subset: pd.DataFrame, relative=True):
    thetas1 = np.deg2rad(df_subset["MainJib"].values.astype(float))
    if relative:
        thetas2 = np.deg2rad((df_subset["MainJib"] + df_subset["FoldingJib"]).values.astype(float))
    else:
        thetas2 = np.deg2rad(df_subset["FoldingJib"].values.astype(float))

    x = df_subset["Outreach"].values.astype(float)
    y = df_subset["Height"].values.astype(float)

    n = len(x)
    A = np.zeros((2*n, 4)); b = np.zeros(2*n)
    A[0:n, 0] = 1.0; A[0:n, 2] = np.cos(thetas1); A[0:n, 3] = np.cos(thetas2); b[0:n] = x
    A[n:, 1] = 1.0; A[n:, 2] = np.sin(thetas1); A[n:, 3] = np.sin(thetas2); b[n:] = y
    params, *_ = np.linalg.lstsq(A, b, rcond=None)
    x0, y0, L1, L2 = params
    return float(x0), float(y0), float(L1), float(L2)


def elbow_point(main_deg, x0, y0, L1):
    t1 = np.deg2rad(main_deg)
    return (x0 + L1*np.cos(t1), y0 + L1*np.sin(t1))


def draw_crane_aligned(ax, main_deg, fold_deg, base_offset_y, df_env, relative=True, hook_target=None, color="white"):
    try:
        x0, y0, L1, L2 = estimate_geometry(df_env[["MainJib","FoldingJib","Outreach","Height"]], relative=relative)
    except Exception:
        x0, y0, L1, L2 = 0.0, 0.0, 10.0, 10.0

    y0 = y0 + base_offset_y

    ex, ey = elbow_point(main_deg, x0, y0, L1)
    bx, by = x0, y0

    size = max(L1, L2) * 0.04
    rect_x = [bx - size, bx + size, bx + size, bx - size, bx - size]
    rect_y = [by - size, by - size, by + size, by + size, by - size]
    ax.plot(rect_x, rect_y, color=color, linewidth=1.5)

    ax.plot([bx, ex], [by, ey], color=color, linewidth=2.0)
    ax.plot(bx, by, "o", color=color, markersize=4)
    ax.plot(ex, ey, "o", color=color, markersize=4)

    if hook_target is not None and all(np.isfinite(hook_target)):
        hx, hy = hook_target
        ax.plot([ex, hx], [ey, hy], color=color, linewidth=2.0)
        ax.plot(hx, hy, "o", color=color, markersize=4)


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

        ox, hz_raw, rated = nearest_point(df_env, fold_angle, main_angle)
        # Find the exact nearest angles used (from current environment df)
        try:
            idx_env = ((df_env["FoldingJib"] - fold_angle).abs() + (df_env["MainJib"] - main_angle).abs()).idxmin()
            nearest_main = float(df_env.loc[idx_env, "MainJib"])
            nearest_fold = float(df_env.loc[idx_env, "FoldingJib"])
        except Exception:
            nearest_main, nearest_fold = np.nan, np.nan

        # Max load at these angles across ALL conditions/environments
        if np.isfinite(nearest_main) and np.isfinite(nearest_fold):
            same_angles = df_all[(df_all["MainJib"] == nearest_main) & (df_all["FoldingJib"] == nearest_fold)]
            if not same_angles.empty:
                max_load_at_angles = float(same_angles["Load"].max())
                # Identify which condition/environment yields that max (for info)
                row_max = same_angles.loc[same_angles["Load"].idxmax()]
                max_env_label = str(row_max.get("Environment", ""))
                max_cond_label = str(row_max.get("Condition", ""))
            else:
                max_load_at_angles = np.nan
                max_env_label = max_cond_label = ""
        else:
            max_load_at_angles = np.nan
            max_env_label = max_cond_label = ""
    
        daf = float(df_env["Cd"].iloc[0]) if "Cd" in df_env.columns and not df_env.empty else np.nan

        st.markdown("#### Height reference")
        use_deck = st.checkbox("Reference height to **deck** (deck below slew bearing)", value=True)
        deck_offset = st.number_input("Deck offset [m] (positive if deck is below bearing)", value=6.0, step=0.1, format="%.1f")

        hz = hz_raw + (deck_offset if use_deck else 0.0)

        st.markdown("#### Geometry options")
        relative = st.checkbox("Folding angle is **relative** to main angle", value=True, help="If unchecked, folding angle is treated as absolute.")

        # Distance from hull
        dist_from_hull = ox - PEDESTAL_INBOARD_M if np.isfinite(ox) else np.nan

        k1, k2 = st.columns(2)
        with k1:
            st.text("Radius [m]")
            st.header(f"{ox:.2f}" if np.isfinite(ox) else "-")
            st.text("Rated load [t]")
            st.header(f"{rated:.1f}" if np.isfinite(rated) else "-")
            st.text("Max load @ angles [t]")
            if np.isfinite(max_load_at_angles):
                angle_info = ""
                if np.isfinite(nearest_main) and np.isfinite(nearest_fold):
                    angle_info = f" — Main: {nearest_main:.2f}°, Folding: {nearest_fold:.2f}°"
                st.markdown(
                    f"<h3>{max_load_at_angles:.1f}</h3>"
                    f"<div style='font-size:0.85rem;color:#aaa'>({max_env_label} / {max_cond_label}){angle_info}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.header("-")
            st.text("Distance from hull [m]")
            if np.isfinite(dist_from_hull) and dist_from_hull < 0:
                st.markdown(f"<h3 style='color:#cc0000'>{dist_from_hull:.2f}</h3>", unsafe_allow_html=True)
            else:
                st.header(f"{dist_from_hull:.2f}" if np.isfinite(dist_from_hull) else "-")
        with k2:
            st.text(("Height above deck [m]" if use_deck else "Height [m]"))
            st.header(f"{hz:.2f}" if np.isfinite(hz) else "-")
            st.text("DAF")
            st.header(f"{daf:.2f}" if np.isfinite(daf) else "-")

        minL, maxL = float(df_env["Load"].min()), float(df_env["Load"].max())
        target = synced_slider_number("Iso-load [t] (overlay)", "iso_load", minL, maxL,
                                      default=(minL + maxL) / 2, step=0.1, fmt="%.1f")

    with left:
        x = df_env["Outreach"].values.astype(float)
        y = (df_env["Height"] + (deck_offset if use_deck else 0.0)).values.astype(float)
        z = df_env["Load"].values.astype(float)

        fig, XI, YI, ZI, ax = contour_plot(
            x, y, z,
            target=target,
            marker=(ox, hz),
            title="Rated capacity related to height and radius"
        )
        if fig is not None:
            draw_crane_aligned(ax, main_angle, fold_angle,
                               base_offset_y=(deck_offset if use_deck else 0.0),
                               df_env=df_env, relative=relative, hook_target=(ox, hz), color="white")
            st.pyplot(fig, use_container_width=True)

        # ----- Offshore lift capacity chart with RED dot + hull guideline -----
        st.markdown("### Print chart")
        xs, ys = capacity_envelope_vs_radius(df_env)

        if len(xs) and len(ys):
            fig2, ax2 = plt.subplots(figsize=(8, 4.5))
            ax2.plot(xs, ys, linewidth=2.0, label="Envelope")
            # Hull side guideline
            ax2.axvline(PEDESTAL_INBOARD_M, linestyle="--", linewidth=1.2)
            ax2.text(PEDESTAL_INBOARD_M, ax2.get_ylim()[1], " Hull side", va="top", ha="left", rotation=90)
            # Current point marker
            if np.isfinite(ox) and np.isfinite(rated):
                ax2.plot([ox], [rated], marker="o", markersize=8, markerfacecolor="red", markeredgecolor="red", linestyle="None", label="Current point")
            ax2.set_xlabel("RADIUS (m)")
            ax2.set_ylabel("SWL (t)")
            ax2.set_title("OFFSHORE LIFT CAPACITY")
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc="best")
            st.pyplot(fig2, use_container_width=True)

            env_csv = pd.DataFrame({"Radius_m": xs, "SWL_t": ys})
            st.download_button("Download capacity curve CSV",
                               env_csv.to_csv(index=False).encode("utf-8"),
                               file_name=f"capacity_curve_{chosen_env}.csv", mime="text/csv")
        else:
            st.info("No data available to build capacity curve for the selected environment.")


if __name__ == "__main__":
    main()

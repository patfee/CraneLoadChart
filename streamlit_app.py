import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator
import plotly.graph_objects as go  # Plotly for interactive capacity chart

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


def contour_plot(
    x, y, z, target=None, marker=None, show_colorbar=True,
    title="Rated capacity related to height and radius",
    nlevels=20
):
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

    vmin = np.nanmin(z)
    vmax = np.nanmax(z)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    levels = np.linspace(vmin, vmax, nlevels)

    fig, ax = plt.subplots(figsize=(8, 7))
    hm = ax.contourf(XI, YI, ZI, levels=levels, vmin=vmin, vmax=vmax)

    # Single iso-load overlay with label
    if target is not None:
        try:
            cs = ax.contour(XI, YI, ZI, levels=[target], linewidths=2.0)
            ax.clabel(cs, fmt=lambda v: f"{v:.1f} t", inline=True, fontsize=10)
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


def capacity_table_with_meta(df_env: pd.DataFrame, deck_offset: float, use_deck: bool) -> pd.DataFrame:
    """
    Build a per-radius table carrying meta info (angles & height), plus a monotone SWL envelope.

    Steps:
    - Round radius to 2 decimals
    - For each rounded radius, pick the row with MAX Load (keep its MainJib/FoldingJib/Height)
    - Sort by radius
    - Compute a non-increasing envelope SWL_monotone by forward pass: SWL_mono[i] = min(SWL_mono[i-1], SWL_raw[i])
    - Compute Distance_from_hull
    - Provide Height_display consistent with the UI choice (deck-referenced or raw)
    """
    if df_env.empty:
        return pd.DataFrame(columns=[
            "Radius_m","SWL_raw_t","SWL_monotone_t","MainJib_deg","FoldingJib_deg",
            "Height_m","Distance_from_hull_m"
        ])

    df = df_env.copy()
    df["r_round"] = np.round(df["Outreach"].astype(float), 2)

    # idx of rows with max Load inside each rounded-radius bin
    idx_max = df.groupby("r_round")["Load"].idxmax()
    picks = df.loc[idx_max, ["r_round","Load","MainJib","FoldingJib","Height"]].copy()

    picks = picks.sort_values("r_round").reset_index(drop=True)

    # Height_display according to deck reference choice
    height_display = picks["Height"].astype(float) + (deck_offset if use_deck else 0.0)

    out = pd.DataFrame({
        "Radius_m": picks["r_round"].astype(float).to_numpy(),
        "SWL_raw_t": picks["Load"].astype(float).to_numpy(),
        "MainJib_deg": picks["MainJib"].astype(float).to_numpy(),
        "FoldingJib_deg": picks["FoldingJib"].astype(float).to_numpy(),
        "Height_m": height_display.astype(float).to_numpy()
    })

    # Monotone (non-increasing) envelope
    swl = out["SWL_raw_t"].to_numpy().copy()
    swl_mono = swl.copy()
    for i in range(1, len(swl_mono)):
        swl_mono[i] = min(swl_mono[i-1], swl_mono[i])
    out["SWL_monotone_t"] = swl_mono

    # Distance from hull
    out["Distance_from_hull_m"] = out["Radius_m"] - PEDESTAL_INBOARD_M

    return out[["Radius_m","SWL_raw_t","SWL_monotone_t","MainJib_deg","FoldingJib_deg","Height_m","Distance_from_hull_m"]]


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
def estimate_geometry(df_subset: pd.DataFrame):
    # Folding angle is ALWAYS treated as relative to main angle
    thetas1 = np.deg2rad(df_subset["MainJib"].values.astype(float))
    thetas2 = np.deg2rad((df_subset["MainJib"] + df_subset["FoldingJib"]).values.astype(float))

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


def draw_crane_aligned(ax, main_deg, fold_deg, base_offset_y, df_env, hook_target=None, color="white"):
    try:
        x0, y0, L1, L2 = estimate_geometry(df_env[["MainJib","FoldingJib","Outreach","Height"]])
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
        daf = float(df_env["Cd"].iloc[0]) if "Cd" in df_env.columns and not df_env.empty else np.nan

        st.markdown("#### Height reference")
        use_deck = st.checkbox("Reference height to **deck** (deck below slew bearing)", value=True)
        deck_offset = st.number_input("Deck offset [m] (positive if deck is below bearing)", value=6.0, step=0.1, format="%.1f")

        hz = hz_raw + (deck_offset if use_deck else 0.0)

        # Distance from hull
        dist_from_hull = ox - PEDESTAL_INBOARD_M if np.isfinite(ox) else np.nan

        k1, k2 = st.columns(2)
        with k1:
            st.text("Radius [m]")
            st.header(f"{ox:.2f}" if np.isfinite(ox) else "-")
            st.text("Rated load [t]")
            st.header(f"{rated:.1f}" if np.isfinite(rated) else "-")
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

        # Iso-load overlay control (value is also printed on the contour line in the chart)
        minL, maxL = float(df_env["Load"].min()), float(df_env["Load"].max())
        target = synced_slider_number("Iso-load [t] (overlay)", "iso_load", minL, maxL,
                                      default=(minL + maxL) / 2, step=0.1, fmt="%.1f")

    with left:
        # ----- Matplotlib contour (as before) -----
        x = df_env["Outreach"].values.astype(float)
        y = (df_env["Height"] + (deck_offset if use_deck else 0.0)).values.astype(float)
        z = df_env["Load"].values.astype(float)

        title_contour = f"Rated Capacity - {chosen_env} Cdyn {daf:.2f}" if np.isfinite(daf) else f"Rated Capacity - {chosen_env}"
        fig, XI, YI, ZI, ax = contour_plot(
            x, y, z,
            target=target,
            marker=(ox, hz),
            title=title_contour
        )
        if fig is not None:
            draw_crane_aligned(
                ax, main_angle, fold_angle,
                base_offset_y=(deck_offset if use_deck else 0.0),
                df_env=df_env, hook_target=(ox, hz), color="white"
            )
            st.pyplot(fig, use_container_width=True)

        # ----- Build Capacity Table (with meta + distance) -----
        st.markdown("### Interactive capacity chart")
        cap_table = capacity_table_with_meta(df_env, deck_offset=deck_offset, use_deck=use_deck)

        if not cap_table.empty:
            xs = cap_table["Radius_m"].to_numpy()
            ys_mono = cap_table["SWL_monotone_t"].to_numpy()

            # Distances for hover
            dist_from_hull_all = cap_table["Distance_from_hull_m"].to_numpy()
            dist_point = ox - PEDESTAL_INBOARD_M if np.isfinite(ox) else np.nan

            # Extra meta for hover (angles & height)
            main_j = cap_table["MainJib_deg"].to_numpy()
            fold_j = cap_table["FoldingJib_deg"].to_numpy()
            h_disp = cap_table["Height_m"].to_numpy()
            swl_raw = cap_table["SWL_raw_t"].to_numpy()

            # Build customdata: [dist, main, folding, height, swl_raw]
            custom = np.stack((dist_from_hull_all, main_j, fold_j, h_disp, swl_raw), axis=-1)

            # ===== Plotly interactive =====
            fig2 = go.Figure()

            # Envelope line (monotone)
            fig2.add_trace(go.Scatter(
                x=xs,
                y=ys_mono,
                mode="lines",
                name="Envelope (monotone)",
                customdata=custom,
                hovertemplate=(
                    "Radius: %{x:.2f} m<br>"
                    "SWL (env): %{y:.2f} t<br>"
                    "SWL (raw @r): %{customdata[4]:.2f} t<br>"
                    "Dist. from hull: %{customdata[0]:.2f} m<br>"
                    "MainJib: %{customdata[1]:.2f}°<br>"
                    "FoldingJib: %{customdata[2]:.2f}°<br>"
                    "Height: %{customdata[3]:.2f} m<extra></extra>"
                ),
            ))

            # Current point marker (if available)
            if np.isfinite(ox) and np.isfinite(rated):
                fig2.add_trace(go.Scatter(
                    x=[ox],
                    y=[rated],
                    mode="markers",
                    name="Current point",
                    marker=dict(size=10, color="red"),
                    customdata=[[dist_point, main_angle, fold_angle, hz, rated]],
                    hovertemplate=(
                        "Current point<br>"
                        "Radius: %{x:.2f} m<br>"
                        "SWL: %{y:.2f} t<br>"
                        "Dist. from hull: %{customdata[0]:.2f} m<br>"
                        "MainJib: %{customdata[1]:.2f}°<br>"
                        "FoldingJib: %{customdata[2]:.2f}°<br>"
                        "Height: %{customdata[3]:.2f} m<extra></extra>"
                    ),
                ))

            # Hull side vertical guide
            fig2.add_shape(
                type="line",
                x0=PEDESTAL_INBOARD_M, x1=PEDESTAL_INBOARD_M,
                y0=float(np.nanmin(ys_mono)) if len(ys_mono) else 0,
                y1=float(np.nanmax(ys_mono)) if len(ys_mono) else 1,
                line=dict(dash="dash"),
                xref="x", yref="y"
            )
            fig2.add_annotation(
                x=PEDESTAL_INBOARD_M,
                y=float(np.nanmax(ys_mono)) if len(ys_mono) else 0,
                text="Hull side",
                showarrow=False,
                yshift=10
            )

            title_capacity = f"Offshore Lift Capacity - {chosen_env} Cdyn {daf:.2f}" if np.isfinite(daf) else f"Offshore Lift Capacity - {chosen_env}"
            fig2.update_layout(
                title=title_capacity,
                xaxis_title="RADIUS (m)",
                yaxis_title="SWL (t)",
                hovermode="closest",
                xaxis=dict(rangeslider=dict(visible=True)),
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            st.plotly_chart(fig2, use_container_width=True, config={
                "displaylogo": False,
                "modeBarButtonsToRemove": ["lasso2d"]
            })

            # ===== Matplotlib capacity print chart (with red marker) =====
            st.markdown("### Print chart (Matplotlib)")
            fig3, ax3 = plt.subplots(figsize=(8, 4.5))
            ax3.plot(xs, ys_mono, linewidth=2.0, label="Envelope (monotone)")
            # Hull side guideline
            ax3.axvline(PEDESTAL_INBOARD_M, linestyle="--", linewidth=1.2)
            ax3.text(PEDESTAL_INBOARD_M, ax3.get_ylim()[1], " Hull side", va="top", ha="left", rotation=90)
            # Current point marker
            if np.isfinite(ox) and np.isfinite(rated):
                ax3.plot([ox], [rated], marker="o", markersize=8,
                         markerfacecolor="red", markeredgecolor="red",
                         linestyle="None", label="Current point")
            ax3.set_xlabel("RADIUS (m)")
            ax3.set_ylabel("SWL (t)")
            ax3.set_title(title_capacity)
            ax3.grid(True, alpha=0.3)
            ax3.legend(loc="best")
            st.pyplot(fig3, use_container_width=True)

            # ===== Enriched CSV (angles + height + distance + raw vs monotone SWL) =====
            env_csv = cap_table.rename(columns={
                "Radius_m": "Radius_m",
                "SWL_raw_t": "SWL_raw_t",
                "SWL_monotone_t": "SWL_monotone_t",
                "MainJib_deg": "MainJib_deg",
                "FoldingJib_deg": "FoldingJib_deg",
                "Height_m": "Height_m",
                "Distance_from_hull_m": "Distance_from_hull_m",
            })
            st.download_button(
                "Download capacity curve CSV",
                env_csv.to_csv(index=False).encode("utf-8"),
                file_name=f"capacity_curve_{chosen_env}.csv",
                mime="text/csv"
            )
        else:
            st.info("No data available to build capacity curve for the selected environment.")


if __name__ == "__main__":
    main()

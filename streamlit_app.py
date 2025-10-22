import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator
import plotly.graph_objects as go

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


def capacity_envelope_vs_radius(df_env):
    """
    Ripple-free envelope:
    1) Round radius to 2 decimals
    2) Max SWL per rounded radius
    3) Sort by radius
    4) Enforce non-increasing SWL with radius
    """
    if df_env.empty:
        return np.array([]), np.array([])

    r = df_env["Outreach"].to_numpy(dtype=float)
    L = df_env["Load"].to_numpy(dtype=float)

    r_round = np.round(r, 2)
    agg = (pd.DataFrame({"r": r_round, "L": L})
             .groupby("r", as_index=False)["L"].max()
             .sort_values("r"))

    xs = agg["r"].to_numpy()
    ys = agg["L"].to_numpy()

    ys = np.maximum.accumulate(ys[::-1])[::-1]
    return xs, ys


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


def build_plotly_contour(df_env, deck_offset, use_deck, target, main_angle, fold_angle, daf, chosen_env, ox, hz):
    # Build dense grid via triangulation, then render with Plotly
    x = df_env["Outreach"].values.astype(float)
    y = (df_env["Height"] + (deck_offset if use_deck else 0.0)).values.astype(float)
    z = df_env["Load"].values.astype(float)

    if len(x) < 3 or len(np.unique(x)) < 3 or len(np.unique(y)) < 3:
        return None

    tri = Triangulation(x, y)
    interp = LinearTriInterpolator(tri, z)

    n = 400
    xi = np.linspace(np.nanmin(x), np.nanmax(x), n)
    yi = np.linspace(np.nanmin(y), np.nanmax(y), n)
    XI, YI = np.meshgrid(xi, yi)
    ZI = interp(XI, YI)

    vmin = float(np.nanmin(z))
    vmax = float(np.nanmax(z))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    title_contour = f"Rated Capacity - {chosen_env} Cdyn {daf:.2f}" if np.isfinite(daf) else f"Rated Capacity - {chosen_env}"

    fig = go.Figure()

    # Filled contour (surface)
    fig.add_trace(go.Contour(
        x=xi, y=yi, z=ZI,
        ncontours=20,
        colorscale="Viridis",
        contours=dict(coloring="heatmap"),
        colorbar=dict(title="Load [t]"),
        hovertemplate="Radius: %{x:.2f} m<br>Height: %{y:.2f} m<br>Load: %{z:.2f} t<extra></extra>"
    ))

    # Single iso-load overlay with label
    if target is not None and np.isfinite(target):
        fig.add_trace(go.Contour(
            x=xi, y=yi, z=ZI,
            contours=dict(start=target, end=target, size=1e-9, coloring="lines", showlabels=True,
                          labelfont=dict(size=12)),
            showscale=False,
            line=dict(width=2),
            hoverinfo="skip",
            name=f"Iso-load {target:.1f} t"
        ))

    # Crane overlay (as shapes)
    try:
        x0, y0, L1, L2 = estimate_geometry(df_env[["MainJib","FoldingJib","Outreach","Height"]])
    except Exception:
        x0, y0, L1, L2 = 0.0, 0.0, 10.0, 10.0
    y0 = y0 + (deck_offset if use_deck else 0.0)

    ex, ey = elbow_point(main_angle, x0, y0, L1)

    size = max(L1, L2) * 0.04
    base_rect = dict(type="rect",
                     x0=x0 - size, x1=x0 + size, y0=y0 - size, y1=y0 + size,
                     line=dict(width=1.5), fillcolor="rgba(0,0,0,0)")

    main_link = dict(type="line", x0=x0, y0=y0, x1=ex, y1=ey, line=dict(width=2))
    hook_link = dict(type="line", x0=ex, y0=ey, x1=float(ox) if np.isfinite(ox) else ex,
                     y1=float(hz) if np.isfinite(hz) else ey, line=dict(width=2))

    # Hull side guideline
    hull_line = dict(type="line",
                     x0=PEDESTAL_INBOARD_M, x1=PEDESTAL_INBOARD_M,
                     y0=float(np.nanmin(yi)), y1=float(np.nanmax(yi)),
                     line=dict(width=1.2, dash="dash"))

    shapes = [base_rect, main_link, hook_link, hull_line]
    fig.update_layout(shapes=shapes)

    # Add joints + hook as scatter markers
    scatter_pts_x = [x0, ex]
    scatter_pts_y = [y0, ey]
    fig.add_trace(go.Scatter(
        x=scatter_pts_x, y=scatter_pts_y, mode="markers",
        marker=dict(size=6),
        name="Crane joints",
        hovertemplate="x: %{x:.2f} m<br>y: %{y:.2f} m<extra></extra>"
    ))

    if np.isfinite(ox) and np.isfinite(hz):
        fig.add_trace(go.Scatter(
            x=[ox], y=[hz], mode="markers",
            marker=dict(size=9),
            name="Hook position",
            hovertemplate="Hook<br>Radius: %{x:.2f} m<br>Height: %{y:.2f} m<extra></extra>"
        ))

    # Hull side label
    fig.add_annotation(
        x=PEDESTAL_INBOARD_M, y=float(np.nanmax(yi)),
        text="Hull side", showarrow=False, yshift=10
    )

    fig.update_layout(
        title=title_contour,
        xaxis_title="y Radius [m]",
        yaxis_title="z Height [m]",
        template="plotly_white",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


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
        # ---- INTERACTIVE PLOTLY CONTOUR ----
        fig_contour = build_plotly_contour(
            df_env=df_env,
            deck_offset=deck_offset,
            use_deck=use_deck,
            target=target,
            main_angle=main_angle,
            fold_angle=fold_angle,
            daf=daf,
            chosen_env=chosen_env,
            ox=ox, hz=hz
        )
        if fig_contour is not None:
            st.plotly_chart(fig_contour, use_container_width=True, config={"displaylogo": False})

        # ---- INTERACTIVE CAPACITY (Plotly) ----
        st.markdown("### Interactive capacity chart")
        xs, ys = capacity_envelope_vs_radius(df_env)

        if len(xs) and len(ys):
            fig2 = go.Figure()

            # Envelope line
            fig2.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines",
                name="Envelope",
                hovertemplate="Radius: %{x:.2f} m<br>SWL: %{y:.2f} t<extra></extra>"
            ))

            # Current point marker (if available)
            if np.isfinite(ox) and np.isfinite(rated):
                fig2.add_trace(go.Scatter(
                    x=[ox], y=[rated], mode="markers",
                    name="Current point",
                    marker=dict(size=10),
                    hovertemplate="Current point<br>Radius: %{x:.2f} m<br>SWL: %{y:.2f} t<extra></extra>"
                ))

            # Hull side vertical guide
            fig2.add_shape(
                type="line",
                x0=PEDESTAL_INBOARD_M, x1=PEDESTAL_INBOARD_M,
                y0=min(ys) if len(ys) else 0, y1=max(ys) if len(ys) else 1,
                line=dict(dash="dash"),
                xref="x", yref="y"
            )
            fig2.add_annotation(
                x=PEDESTAL_INBOARD_M, y=max(ys) if len(ys) else 0,
                text="Hull side", showarrow=False, yshift=10
            )

            title_capacity = f"Offshore Lift Capacity - {chosen_env} Cdyn {daf:.2f}" if np.isfinite(daf) else f"Offshore Lift Capacity - {chosen_env}"
            fig2.update_layout(
                title=title_capacity,
                xaxis_title="RADIUS (m)",
                yaxis_title="SWL (t)",
                hovermode="x unified",
                xaxis=dict(rangeslider=dict(visible=True)),
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            st.plotly_chart(fig2, use_container_width=True, config={
                "displaylogo": False,
                "modeBarButtonsToRemove": ["lasso2d"]
            })

            env_csv = pd.DataFrame({"Radius_m": xs, "SWL_t": ys})
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

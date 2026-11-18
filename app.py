
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from pathlib import Path
from core.color_enhancement import apply_ycrcb_clahe
from core.metrics import calculate_metrics


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SmartLight Analyzer",
    page_icon="💡",
    layout="wide"
)

OUTPUT_DIR = Path("outputs")

st.title("💡 SmartLight Analyzer")
st.caption(
    "Real-Time Image Enhancement and Histogram Analysis"
)



# ============================================================
# EXPERIMENT DISCOVERY
# ============================================================

conditions = sorted([
    folder.name
    for folder in OUTPUT_DIR.iterdir()
    if folder.is_dir()
]) if OUTPUT_DIR.exists() else []

if not conditions:
    st.error(
        "No processed experiments found. "
        "Run run_pipeline.py first."
    )
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Overview",
        "Image Processing",
        "Histogram Analysis",
        "Performance",
        "Live Enhancement",
        "Export"
    ]
)

condition = st.sidebar.selectbox(
    "Lighting Condition",
    conditions
)

folder = OUTPUT_DIR / condition


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(filename, color=False):

    path = folder / filename

    if not path.exists():
        return None

    image = cv2.imread(str(path))

    if image is None:
        return None

    if color:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

    return image


original_color = load_image(
    "original_color.png",
    color=True
)

original_color_bgr = load_image(
    "original_color.png"
)

original = load_image("grayscale.png")
global_he = load_image("global_he.png")
clahe = load_image("clahe.png")
gamma = load_image("gamma.png")

color_clahe = load_image(
    "color_clahe.png",
    color=True
)


# Grayscale processing outputs
images = {
    "Original": original,
    "Global HE": global_he,
    "CLAHE": clahe,
    "Gamma": gamma
}


# Downloadable outputs
export_images = {
    **images,
    "YCrCb + CLAHE": color_clahe
}


# ============================================================
# METRICS LOADING
# ============================================================

metrics_path = folder / "metrics.csv"

if metrics_path.exists():

    try:
        metrics_df = pd.read_csv(metrics_path)

    except Exception:
        metrics_df = pd.DataFrame()

else:
    metrics_df = pd.DataFrame()


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown("""
    ### Welcome!

    SmartLight Analyzer is an interactive image processing
    application developed to study the effect of illumination
    on image quality and explore different image enhancement
    techniques.

    It analyzes images captured under four lighting conditions
    and demonstrates how histogram-based methods can improve
    image visibility, contrast and intensity distribution.

    **Explore the application to:**
    - Analyze image histograms and CDF.
    - Apply Global Histogram Equalization.
    - Compare CLAHE and Gamma Correction.
    - Explore colour-preserving YCrCb enhancement.
    - Evaluate results using image quality metrics.

    ---
    """)

    st.header("Experiment Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Available Conditions",
        len(conditions)
    )

    col2.metric(
        "Current Condition",
        condition.title()
    )

    col3.metric(
        "Compared Outputs",
        5
    )

    st.divider()

    st.subheader("Original Photograph")

    if original_color is not None:

        st.image(
            original_color,
            use_container_width=True
        )

    else:
        st.warning(
            "Original photograph unavailable."
        )

    st.subheader("Experiment Results")

    if not metrics_df.empty:

        st.dataframe(
            metrics_df,
            use_container_width=True,
            hide_index=True
        )

    else:
        st.warning("Metrics file unavailable.")


# ============================================================
# IMAGE PROCESSING
# ============================================================

elif page == "Image Processing":

    st.header("Image Processing Comparison")

    if original is None:
        st.error("Original grayscale image not found.")
        st.stop()

    columns = st.columns(2)

    for index, (name, image) in enumerate(
        images.items()
    ):

        with columns[index % 2]:

            if image is not None:

                st.subheader(name)

                st.image(
                    image,
                    use_container_width=True,
                    clamp=True
                )

            else:

                st.warning(
                    f"{name} image unavailable."
                )

    st.divider()

    st.subheader(
        "Colour-Preserving Enhancement"
    )

    if color_clahe is not None:

        col1, col2 = st.columns(2)

        with col1:

            st.caption("Original Colour Image")

            if original_color is not None:

                st.image(
                    original_color,
                    use_container_width=True
                )

        with col2:

            st.caption("YCrCb + CLAHE")

            st.image(
                color_clahe,
                use_container_width=True
            )

        st.caption(
            "CLAHE is applied to the luminance channel "
            "in YCrCb colour space while preserving "
            "the chrominance channels."
        )

    else:

        st.warning(
            "Colour-enhanced image unavailable."
        )

    st.caption(
        "Global HE, CLAHE and Gamma outputs are "
        "generated from the same grayscale input."
    )


# ============================================================
# HISTOGRAM ANALYSIS
# ============================================================

elif page == "Histogram Analysis":

    st.header("Intensity Histogram Analysis")

    selected = st.multiselect(
        "Select images",
        list(images.keys()),
        default=["Original", "Global HE"]
    )

    fig = go.Figure()

    for name in selected:

        image = images[name]

        if image is not None:

            histogram = np.bincount(
                image.ravel(),
                minlength=256
            )

            fig.add_trace(
                go.Scatter(
                    x=list(range(256)),
                    y=histogram,
                    mode="lines",
                    name=name
                )
            )

    fig.update_layout(
        xaxis_title="Pixel Intensity",
        yaxis_title="Frequency",
        title="Grayscale Intensity Distribution",
        hovermode="x unified",
        template="plotly_white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# PERFORMANCE ANALYSIS
# ============================================================

elif page == "Performance":

    st.header("Image Quality Performance")

    if not metrics_df.empty:

        st.dataframe(
            metrics_df,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        st.subheader("Contrast Comparison")

        contrast_chart = go.Figure()

        contrast_chart.add_trace(
            go.Bar(
                x=metrics_df["Algorithm"],
                y=metrics_df["Contrast (Std Dev)"],
                name="Contrast"
            )
        )

        contrast_chart.update_layout(
            xaxis_title="Algorithm",
            yaxis_title="Standard Deviation",
            template="plotly_white"
        )

        st.plotly_chart(
            contrast_chart,
            use_container_width=True
        )

        st.subheader("Mean Intensity Comparison")

        st.bar_chart(
            metrics_df.set_index(
                "Algorithm"
            )["Mean Intensity"]
        )

        st.subheader("Entropy Comparison")

        st.bar_chart(
            metrics_df.set_index(
                "Algorithm"
            )["Entropy (bits)"]
        )

    else:

        st.warning(
            "Metrics file not found."
        )


# ============================================================
# LIVE ENHANCEMENT
# ============================================================

elif page == "Live Enhancement":

    st.header("Interactive Image Enhancement")

    # --------------------------------------------------------
    # IMAGE INPUT
    # --------------------------------------------------------

    st.subheader("Choose Your Image")

    uploaded_file = st.file_uploader(
        "Upload an image (optional)",
        type=["jpg", "jpeg", "png"],
        key="live_image_uploader"
    )

    if uploaded_file is not None:

        file_bytes = np.asarray(
            bytearray(uploaded_file.getvalue()),
            dtype=np.uint8
        )

        live_image_bgr = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )

        if live_image_bgr is None:

            st.error(
                "Unable to read the uploaded image."
            )

            st.stop()

        st.success(
            f"Using uploaded image: {uploaded_file.name}"
        )

    elif original_color_bgr is not None:

        live_image_bgr = original_color_bgr.copy()

    else:

        st.error(
            "No image available. Please upload an image."
        )

        st.stop()

    # --------------------------------------------------------
    # PREPARE IMAGE
    # --------------------------------------------------------

    live_image_rgb = cv2.cvtColor(
        live_image_bgr,
        cv2.COLOR_BGR2RGB
    )

    original_gray = cv2.cvtColor(
        live_image_bgr,
        cv2.COLOR_BGR2GRAY
    )

    st.image(
        live_image_rgb,
        caption="Selected Input Image",
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # GRAYSCALE CLAHE
    # --------------------------------------------------------

    st.subheader("1. Grayscale CLAHE")

    clip_limit = st.slider(
        "Grayscale Clip Limit",
        1.0,
        10.0,
        2.0,
        0.5
    )

    tile_size = st.slider(
        "Grayscale Tile Grid Size",
        2,
        16,
        8
    )

    live_clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(
            tile_size,
            tile_size
        )
    )

    enhanced = live_clahe.apply(
        original_gray
    )

    col1, col2 = st.columns(2)

    with col1:

        st.caption("Original Grayscale")

        st.image(
            original_gray,
            use_container_width=True
        )

    with col2:

        st.caption("Enhanced Grayscale")

        st.image(
            enhanced,
            use_container_width=True
        )

    success, gray_encoded = cv2.imencode(
        ".png",
        enhanced
    )

    if success:

        st.download_button(
            "Download Grayscale CLAHE Image",
            data=gray_encoded.tobytes(),
            file_name="enhanced_grayscale.png",
            mime="image/png"
        )

    st.divider()

    # --------------------------------------------------------
    # NOISE-AWARE COLOUR ENHANCEMENT
    # --------------------------------------------------------

    st.subheader(
        "2. Noise-Aware Colour Enhancement"
    )

    col1, col2 = st.columns(2)

    with col1:

        color_clip = st.slider(
            "Colour CLAHE Clip Limit",
            1.0,
            5.0,
            1.8,
            0.1
        )

        color_tile = st.slider(
            "Colour Tile Grid Size",
            4,
            32,
            16
        )

        alpha = st.slider(
            "Enhancement Strength (Alpha)",
            0.1,
            1.0,
            0.6,
            0.05
        )

    with col2:

        bilateral_d = st.select_slider(
            "Bilateral Filter Diameter",
            options=[3, 5, 7, 9, 11],
            value=5
        )

        sigma_color = st.slider(
            "Bilateral Sigma Color",
            5,
            100,
            35
        )

        sigma_space = st.slider(
            "Bilateral Sigma Space",
            5,
            100,
            35
        )

    color_enhanced = apply_ycrcb_clahe(
        live_image_bgr,
        clip_limit=color_clip,
        tile_size=color_tile,
        alpha=alpha,
        bilateral_d=bilateral_d,
        sigma_color=sigma_color,
        sigma_space=sigma_space
    )

    color_enhanced_rgb = cv2.cvtColor(
        color_enhanced,
        cv2.COLOR_BGR2RGB
    )

    col1, col2 = st.columns(2)

    with col1:

        st.caption("Original Colour Image")

        st.image(
            live_image_rgb,
            use_container_width=True
        )

    with col2:

        st.caption("Enhanced Colour Image")

        st.image(
            color_enhanced_rgb,
            use_container_width=True
        )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    st.subheader(
        "Colour Enhancement Metrics"
    )

    original_metrics = calculate_metrics(
        live_image_bgr
    )

    color_metrics = calculate_metrics(
        color_enhanced
    )

    metric_cols = st.columns(3)

    for col, (name, value) in zip(
        metric_cols,
        color_metrics.items()
    ):

        col.metric(
            name,
            value,
            delta=round(
                value - original_metrics[name],
                2
            )
        )

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    success, encoded = cv2.imencode(
        ".png",
        color_enhanced
    )

    if success:

        st.download_button(
            "Download Colour-Enhanced Image",
            data=encoded.tobytes(),
            file_name="enhanced_colour_image.png",
            mime="image/png"
        )


# ============================================================
# EXPORT
# ============================================================

elif page == "Export":

    st.header("Export Experiment Results")

    if not metrics_df.empty:

        csv_data = metrics_df.to_csv(
            index=False
        )

        st.download_button(
            "Download Metrics CSV",
            data=csv_data,
            file_name=f"{condition}_metrics.csv",
            mime="text/csv"
        )

    else:

        st.warning(
            "Metrics file unavailable."
        )

    st.divider()

    st.subheader("Download Processed Images")

    for name, image in export_images.items():

        if image is not None:

            image_to_encode = (
                cv2.cvtColor(
                    image,
                    cv2.COLOR_RGB2BGR
                )
                if name == "YCrCb + CLAHE"
                else image
            )

            success, encoded = cv2.imencode(
                ".png",
                image_to_encode
            )

            if success:

                st.download_button(
                    f"Download {name}",
                    data=encoded.tobytes(),
                    file_name=(
                        f"{condition}_"
                        f"{name.replace(' ', '_')}.png"
                    ),
                    mime="image/png",
                    key=f"{condition}_{name}"
                )

    st.success(
        "Your experiment outputs are ready for export."
    )

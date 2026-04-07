import logging
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.optimize import differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# =============================================================================
# CONFIGURATION
# =============================================================================
FEATURES: List[str] = [
    "albedo_w",
    "Height_canyon",
    "Width_canyon",
    "Width_roof",
    "fveg_G",
    "fveg_R",
    "Height_tree",
    "Radius_tree",
]

PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "albedo_w": (0.25, 0.90),
    "Height_canyon": (13.0, 24.9),
    "Width_canyon": (15.0, 30.0),
    "Width_roof": (8.0, 25.0),
    "fveg_G": (0.0, 1.0),
    "fveg_R": (0.0, 1.0),
    "Height_tree": (4.5, 9.5),
    "Radius_tree": (0.02, 6.0),
}

PARAM_META: Dict[str, Dict[str, str]] = {
    "albedo_w": {
        "symbol_ui": "αw",
        "symbol_latex": r"\alpha_w",
        "label": "Wall albedo",
        "unit": "-",
        "description": "Measure of the wall surface reflectivity; higher values mean more incoming shortwave radiation is reflected.",
    },
    "Height_canyon": {
        "symbol_ui": "Hr",
        "symbol_latex": r"H_r",
        "label": "Canyon height",
        "unit": "m",
        "description": "Average building height within the urban street canyon.",
    },
    "Width_canyon": {
        "symbol_ui": "Wc",
        "symbol_latex": r"W_c",
        "label": "Canyon width",
        "unit": "m",
        "description": "Width of the urban street canyon.",
    },
    "Width_roof": {
        "symbol_ui": "Wr",
        "symbol_latex": r"W_r",
        "label": "Roof width",
        "unit": "m",
        "description": "Width of the roof, perpendicular to the canyon orientation.",
    },
    "fveg_G": {
        "symbol_ui": "λg",
        "symbol_latex": r"\lambda_g",
        "label": "Vegetated ground fraction",
        "unit": "-",
        "description": "Fraction of vegetated ground surface, from fully impervious (0) to fully vegetated (1).",
    },
    "fveg_R": {
        "symbol_ui": "λr",
        "symbol_latex": r"\lambda_r",
        "label": "Vegetated roof fraction",
        "unit": "-",
        "description": "Fraction of vegetated roof surface, from fully impervious (0) to fully vegetated (1).",
    },
    "Height_tree": {
        "symbol_ui": "Ht",
        "symbol_latex": r"H_t",
        "label": "Tree height",
        "unit": "m",
        "description": "Average height of the trees lining the urban canyon.",
    },
    "Radius_tree": {
        "symbol_ui": "rt",
        "symbol_latex": r"r_t",
        "label": "Tree crown radius",
        "unit": "m",
        "description": "Average crown radius of the trees lining the urban canyon.",
    },
}

SECTION_ORDER = [
    "Project Description",
    "Parameter Guide",
    "City Visualisation",
    "Surrogate & Optimisation",
    "Response Explorer",
    "Summary",
]

DEFAULT_INPUT_PATH = "GPinput.csv"
DEFAULT_OUTPUT_PATH = "GPoutput.csv"
DEFAULT_TARGET_NAME = "target"
RANDOM_STATE = 42
KELVIN_TO_CELSIUS = 273.15

PROJECT_ABSTRACT = """
This application visualises a Gaussian-process surrogate model for urban heat mitigation. The goal is to
approximate how peak canyon air temperature changes as urban design variables such as wall reflectivity,
canyon geometry, vegetation fractions, and tree geometry are varied. The interface supports scenario
exploration, interpretable response curves, and surrogate-based optimisation to identify parameter
configurations associated with lower peak temperatures during heat-wave conditions.
""".strip()

OPTIMISATION_NOTES = """
The surrogate is trained on simulation data and then used as a fast proxy for design-space exploration.
In this app, optimisation is performed directly on the Gaussian-process mean prediction within the
feasible parameter bounds. This gives a computationally cheap estimate of the parameter set that minimises
predicted peak temperature.
""".strip()


# =============================================================================
# LOGGING
# =============================================================================
def setup_logging() -> logging.Logger:
    logger = logging.getLogger("gp_app")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


LOGGER = setup_logging()


# =============================================================================
# DATA STRUCTURES
# =============================================================================
@dataclass
class DataBundle:
    X: pd.DataFrame
    y_kelvin: pd.Series
    y_celsius: pd.Series
    feature_names: List[str]
    target_name: str


@dataclass
class TrainBundle:
    model: Pipeline
    metrics: Dict[str, float]
    train_size: int
    test_size: int
    kernel_before: str
    kernel_after: str
    feature_names: List[str]


@dataclass
class OptimizationBundle:
    optimum_params: Dict[str, float]
    optimum_temp_c: float
    baseline_temp_c: float
    improvement_c: float
    success: bool
    message: str


# =============================================================================
# UTILITIES
# =============================================================================
def log_step(message: str) -> None:
    LOGGER.info(message)


def kelvin_to_celsius(values):
    return np.asarray(values) - KELVIN_TO_CELSIUS


@st.cache_data(show_spinner=False)
def read_csv_flexible(path: str, expected_kind: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    if os.path.getsize(path) == 0:
        raise ValueError(f"CSV file is empty: {path}")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"CSV file is empty or invalid: {path}") from exc

    if expected_kind == "output" and df.shape[1] == 1:
        col = df.columns[0]
        try:
            float(str(col))
            df = pd.read_csv(path, header=None)
        except ValueError:
            pass
    return df


@st.cache_data(show_spinner=False)
def load_data(input_path: str, output_path: str) -> DataBundle:
    log_step("Starting data loading...")
    df_inputs = read_csv_flexible(input_path, expected_kind="input")
    df_outputs = read_csv_flexible(output_path, expected_kind="output")

    log_step(f"Loaded input CSV: {input_path} with shape {df_inputs.shape}")
    log_step(f"Loaded output CSV: {output_path} with shape {df_outputs.shape}")

    missing = [col for col in FEATURES if col not in df_inputs.columns]
    if missing:
        raise ValueError("Input CSV is missing required columns: " + ", ".join(missing))

    X = df_inputs[FEATURES].copy().apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        bad_cols = X.columns[X.isna().any()].tolist()
        raise ValueError(
            "Input CSV contains non-numeric or missing values in: "
            + ", ".join(bad_cols)
        )

    if df_outputs.shape[1] == 0:
        raise ValueError("Output CSV has no columns.")

    y_kelvin = pd.to_numeric(df_outputs.iloc[:, 0], errors="coerce")
    if y_kelvin.isna().any():
        raise ValueError("Output CSV contains non-numeric or missing target values.")

    if len(X) != len(y_kelvin):
        raise ValueError(
            f"Row count mismatch: inputs contain {len(X)} rows while outputs contain {len(y_kelvin)} rows."
        )

    y_kelvin = y_kelvin.rename(DEFAULT_TARGET_NAME)
    y_celsius = pd.Series(
        kelvin_to_celsius(y_kelvin), name=f"{DEFAULT_TARGET_NAME}_celsius"
    )

    log_step(f"Validated dataset successfully: {len(X)} samples, {X.shape[1]} features")
    return DataBundle(
        X=X,
        y_kelvin=y_kelvin,
        y_celsius=y_celsius,
        feature_names=FEATURES,
        target_name=DEFAULT_TARGET_NAME,
    )


@st.cache_resource(show_spinner=False)
def train_gp(
    X: pd.DataFrame, y_celsius: pd.Series, test_size: float, random_state: int
) -> TrainBundle:
    log_step("Preparing data for Gaussian Process training...")
    log_step(f"Number of samples: {len(X)}")
    log_step(f"Number of features: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_celsius, test_size=test_size, random_state=random_state
    )
    log_step(
        f"Train/test split complete. Train samples: {len(X_train)}, Test samples: {len(X_test)}"
    )

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.ones(X.shape[1]),
        length_scale_bounds=(1e-2, 1e2),
        nu=2.5,
    ) + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-1))

    log_step(f"Kernel selected: {kernel}")
    log_step("Starting Gaussian Process model fitting...")

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "gpr",
                GaussianProcessRegressor(
                    kernel=kernel,
                    normalize_y=True,
                    n_restarts_optimizer=5,
                    random_state=random_state,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    log_step("Model fitting completed.")

    y_pred = model.predict(X_test)
    metrics = {
        "R2": float(r2_score(y_test, y_pred)),
        "RMSE_C": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "MAE_C": float(mean_absolute_error(y_test, y_pred)),
    }

    trained_kernel = str(model.named_steps["gpr"].kernel_)
    log_step(f"Optimized kernel: {trained_kernel}")
    log_step(
        "Evaluation metrics | "
        + " | ".join([f"{name}: {value:.5f}" for name, value in metrics.items()])
    )

    return TrainBundle(
        model=model,
        metrics=metrics,
        train_size=len(X_train),
        test_size=len(X_test),
        kernel_before=str(kernel),
        kernel_after=trained_kernel,
        feature_names=FEATURES,
    )


def make_default_controls(X: pd.DataFrame) -> Dict[str, float]:
    defaults = {}
    for feature in FEATURES:
        low, high = PARAM_BOUNDS[feature]
        median_val = float(X[feature].median())
        defaults[feature] = float(np.clip(median_val, low, high))
    return defaults


def feature_vector_from_controls(control_values: Dict[str, float]) -> np.ndarray:
    return np.array([control_values[f] for f in FEATURES], dtype=float)


def predict_temperature_curve(
    model: Pipeline,
    control_values: Dict[str, float],
    x_feature: str,
    n_points: int = 240,
) -> pd.DataFrame:
    low, high = PARAM_BOUNDS[x_feature]
    x_grid = np.linspace(low, high, n_points)

    frame = pd.DataFrame(
        {
            feature: np.full(n_points, control_values[feature], dtype=float)
            for feature in FEATURES
        }
    )
    frame[x_feature] = x_grid

    mean_pred_c, std_pred = model.predict(frame, return_std=True)
    ci95 = 1.96 * std_pred

    return pd.DataFrame(
        {
            x_feature: x_grid,
            "temperature_c": mean_pred_c,
            "std_c": std_pred,
            "lower_95_c": mean_pred_c - ci95,
            "upper_95_c": mean_pred_c + ci95,
        }
    )


@st.cache_data(show_spinner=False)
def compute_training_scatter(_X: pd.DataFrame, _y_celsius: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"Observed Temperature [°C]": _y_celsius})


@st.cache_resource(show_spinner=False)
def optimise_surrogate(_model: Pipeline, _X: pd.DataFrame) -> OptimizationBundle:
    default_controls = make_default_controls(_X)
    baseline_temp_c = float(_model.predict(pd.DataFrame([default_controls]))[0])

    bounds = [PARAM_BOUNDS[f] for f in FEATURES]

    def objective(values: np.ndarray) -> float:
        frame = pd.DataFrame([values], columns=FEATURES)
        pred = float(_model.predict(frame)[0])
        return pred

    result = differential_evolution(
        objective,
        bounds=bounds,
        seed=RANDOM_STATE,
        maxiter=80,
        popsize=12,
        polish=True,
        updating="deferred",
        workers=1,
    )

    optimum_params = {
        feature: float(value) for feature, value in zip(FEATURES, result.x)
    }
    optimum_temp_c = float(result.fun)
    improvement_c = baseline_temp_c - optimum_temp_c

    return OptimizationBundle(
        optimum_params=optimum_params,
        optimum_temp_c=optimum_temp_c,
        baseline_temp_c=baseline_temp_c,
        improvement_c=improvement_c,
        success=bool(result.success),
        message=str(result.message),
    )


# =============================================================================
# STYLING AND HELPERS
# =============================================================================
def add_global_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }
        .hero-card {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0b3b2e 100%);
            padding: 1.3rem 1.5rem;
            border-radius: 18px;
            color: white;
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 18px 38px rgba(15,23,42,0.18);
            margin-bottom: 1rem;
        }
        .section-card {
            background: #ffffff;
            border: 1px solid rgba(15,23,42,0.08);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 24px rgba(15,23,42,0.06);
        }
        .mini-muted {
            color: #475569;
            font-size: 0.95rem;
        }
        .metric-band {
            background: linear-gradient(90deg, rgba(14,165,233,0.08), rgba(16,185,129,0.08));
            border: 1px solid rgba(15,23,42,0.08);
            border-radius: 14px;
            padding: 0.75rem 0.9rem;
            margin-bottom: 1rem;
        }
        .eq-card {
            background: #f8fafc;
            border-left: 4px solid #0ea5e9;
            padding: 0.8rem 1rem;
            border-radius: 10px;
            margin-bottom: 0.7rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def albedo_to_gray(alpha_w: float) -> str:
    level = int(np.clip(50 + 180 * alpha_w, 0, 235))
    return f"rgb({level},{level},{level})"


# =============================================================================
# UI COMPONENTS
# =============================================================================
def build_sidebar(X: pd.DataFrame) -> Tuple[str, Dict[str, float], float]:
    st.sidebar.header("Controls")
    x_feature = st.sidebar.selectbox(
        "Response variable on x-axis",
        FEATURES,
        index=0,
        format_func=lambda f: f"{PARAM_META[f]['label']} ({PARAM_META[f]['symbol_ui']})",
    )

    test_size = st.sidebar.slider(
        "Test split fraction",
        min_value=0.10,
        max_value=0.40,
        value=0.20,
        step=0.05,
    )

    defaults = make_default_controls(X)
    control_values: Dict[str, float] = {}

    st.sidebar.markdown("---")
    st.sidebar.subheader("Scenario parameters")
    for feature in FEATURES:
        low, high = PARAM_BOUNDS[feature]
        step = float((high - low) / 250.0) if high > low else 0.01
        control_values[feature] = st.sidebar.slider(
            f"{PARAM_META[feature]['label']} ({PARAM_META[feature]['symbol_ui']})",
            min_value=float(low),
            max_value=float(high),
            value=float(defaults[feature]),
            step=step,
            disabled=(feature == x_feature),
        )

    return x_feature, control_values, test_size


def render_header() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <h1 style="margin-bottom:0.25rem;">Urban Heat Surrogate Explorer</h1>
            <div style="font-size:1.02rem; opacity:0.95; max-width:1000px;">
                A professional interactive dashboard for Gaussian-process-based urban heat analysis,
                scenario exploration, and surrogate optimisation of street-canyon design variables.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_project_description() -> None:
    st.markdown("### Project description")
    st.markdown(PROJECT_ABSTRACT)

    st.markdown("### Why this project matters")
    st.markdown(
        "This tool turns a costly urban-climate simulation workflow into an interpretable, fast surrogate. "
        "Instead of repeatedly running the underlying simulator for every scenario, the trained Gaussian "
        "process provides instantaneous predictions, uncertainty estimates, and optimisation guidance."
    )


def render_parameter_guide() -> None:
    st.markdown("### Parameter dictionary")
    rows = []
    for feature in FEATURES:
        meta = PARAM_META[feature]
        low, high = PARAM_BOUNDS[feature]
        rows.append(
            {
                "Feature": feature,
                "Symbol": meta["symbol_ui"],
                "Meaning": meta["label"],
                "Unit": meta["unit"],
                "Range": f"[{low}, {high}]",
                "Description": meta["description"],
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def build_city_figure(
    control_values: Dict[str, float], predicted_temp_c: float
) -> go.Figure:
    hc = control_values["Height_canyon"]  # Hr
    wc = control_values["Width_canyon"]  # Wc
    wr = control_values["Width_roof"]  # Wr
    fvg = control_values["fveg_G"]  # λg
    fvr = control_values["fveg_R"]  # λr
    ht = control_values["Height_tree"]  # Ht
    rt = control_values["Radius_tree"]  # rt
    alpha_w = control_values["albedo_w"]  # αw

    wall_color = albedo_to_gray(alpha_w)
    roof_green = f"rgba(34,197,94,{0.30 + 0.55 * fvr})"
    ground_green = f"rgba(22,163,74,{0.20 + 0.70 * fvg})"

    building_width = wr

    left_building_x0 = 0.0
    left_building_x1 = building_width
    canyon_x0 = left_building_x1
    canyon_x1 = canyon_x0 + wc
    right_building_x0 = canyon_x1
    right_building_x1 = right_building_x0 + building_width

    fig = go.Figure()

    # Base ground
    fig.add_shape(
        type="rect",
        x0=left_building_x0 - 6,
        x1=right_building_x1 + 6,
        y0=-1.5,
        y1=0,
        line=dict(width=0),
        fillcolor="#94a3b8",
    )

    # Vegetated ground inside canyon
    fig.add_shape(
        type="rect",
        x0=canyon_x0,
        x1=canyon_x1,
        y0=-0.55,
        y1=0,
        line=dict(width=0),
        fillcolor=ground_green,
    )

    # Left and right buildings
    fig.add_shape(
        type="rect",
        x0=left_building_x0,
        x1=left_building_x1,
        y0=0,
        y1=hc,
        fillcolor=wall_color,
        line=dict(color="#334155", width=2),
    )
    fig.add_shape(
        type="rect",
        x0=right_building_x0,
        x1=right_building_x1,
        y0=0,
        y1=hc,
        fillcolor=wall_color,
        line=dict(color="#334155", width=2),
    )

    # Roof layers
    fig.add_shape(
        type="rect",
        x0=left_building_x0,
        x1=left_building_x1,
        y0=hc,
        y1=hc + 1.2,
        fillcolor=roof_green,
        line=dict(color="#334155", width=2),
    )
    fig.add_shape(
        type="rect",
        x0=right_building_x0,
        x1=right_building_x1,
        y0=hc,
        y1=hc + 1.2,
        fillcolor=roof_green,
        line=dict(color="#334155", width=2),
    )

    # Street/canyon line
    fig.add_trace(
        go.Scatter(
            x=[canyon_x0, canyon_x1],
            y=[0, 0],
            mode="lines",
            line=dict(color="#475569", width=5),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Trees inside canyon
    tree_positions = [canyon_x0 + 0.30 * wc, canyon_x0 + 0.70 * wc]
    for idx, tx in enumerate(tree_positions):
        fig.add_trace(
            go.Scatter(
                x=[tx, tx],
                y=[0, ht],
                mode="lines",
                line=dict(color="#7c4a1d", width=8),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        theta = np.linspace(0, 2 * np.pi, 160)
        fig.add_trace(
            go.Scatter(
                x=tx + rt * np.cos(theta),
                y=ht + 0.60 * rt * np.sin(theta),
                mode="lines",
                fill="toself",
                fillcolor="rgba(34,197,94,0.60)",
                line=dict(color="rgba(21,128,61,0.95)", width=2),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Dimension arrows for Wc
    fig.add_annotation(
        x=canyon_x0,
        y=-1.0,
        ax=canyon_x1,
        ay=-1.0,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowside="end+start",
        arrowwidth=1.8,
        arrowcolor="#0f172a",
        text="",
    )

    # Dimension arrows for Wr (left building width)
    fig.add_annotation(
        x=left_building_x0,
        y=hc + 2.2,
        ax=left_building_x1,
        ay=hc + 2.2,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowside="end+start",
        arrowwidth=1.8,
        arrowcolor="#0f172a",
        text="",
    )

    # Dimension arrow for Hr
    fig.add_annotation(
        x=right_building_x1 + 1.5,
        y=0,
        ax=right_building_x1 + 1.5,
        ay=hc,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowside="end+start",
        arrowwidth=1.8,
        arrowcolor="#0f172a",
        text="",
    )

    # Main labels
    fig.add_annotation(
        x=(canyon_x0 + canyon_x1) / 2,
        y=-1.45,
        text=f"<b>Wc</b> = {wc:.2f} m",
        showarrow=False,
        font=dict(size=12, color="#0f172a"),
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor="#cbd5e1",
        borderwidth=1,
    )

    fig.add_annotation(
        x=(left_building_x0 + left_building_x1) / 2,
        y=hc + 2.8,
        text=f"<b>Wr</b> = {wr:.2f} m",
        showarrow=False,
        font=dict(size=12, color="#0f172a"),
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor="#cbd5e1",
        borderwidth=1,
    )

    fig.add_annotation(
        x=right_building_x1 + 3.0,
        y=hc / 2,
        text=f"<b>Hr</b> = {hc:.2f} m",
        showarrow=False,
        font=dict(size=12, color="#0f172a"),
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor="#cbd5e1",
        borderwidth=1,
    )

    fig.add_annotation(
        x=(left_building_x0 + left_building_x1) / 2,
        y=hc / 2,
        text=f"<b>αw</b> = {alpha_w:.2f}",
        showarrow=False,
        font=dict(size=12, color="#0f172a"),
        bgcolor="rgba(255,255,255,0.80)",
        bordercolor="#cbd5e1",
        borderwidth=1,
    )

    fig.add_annotation(
        x=(canyon_x0 + canyon_x1) / 2,
        y=0.55,
        text=f"<b>λg</b> = {fvg:.2f}",
        showarrow=False,
        font=dict(size=12, color="#14532d"),
        bgcolor="rgba(255,255,255,0.88)",
        bordercolor="#bbf7d0",
        borderwidth=1,
    )

    fig.add_annotation(
        x=(right_building_x0 + right_building_x1) / 2,
        y=hc + 1.8,
        text=f"<b>λr</b> = {fvr:.2f}",
        showarrow=False,
        font=dict(size=12, color="#14532d"),
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor="#bbf7d0",
        borderwidth=1,
    )

    # Tree labels
    tree_mid_x = tree_positions[0]
    fig.add_annotation(
        x=tree_mid_x,
        y=ht + 1.10 * rt,
        text=f"<b>Ht</b> = {ht:.2f} m<br><b>rt</b> = {rt:.2f} m",
        showarrow=False,
        font=dict(size=12, color="#14532d"),
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor="#bbf7d0",
        borderwidth=1,
    )

    # Compact legend panel with all parameters
    legend_text = (
        f"<b>Current parameter set</b><br>"
        f"αw = {alpha_w:.2f}<br>"
        f"Hr = {hc:.2f} m<br>"
        f"Wc = {wc:.2f} m<br>"
        f"Wr = {wr:.2f} m<br>"
        f"λg = {fvg:.2f}<br>"
        f"λr = {fvr:.2f}<br>"
        f"Ht = {ht:.2f} m<br>"
        f"rt = {rt:.2f} m"
    )

    fig.add_annotation(
        x=right_building_x1 + 4.8,
        y=max(hc + 4.5, ht + rt + 3.0),
        text=legend_text,
        showarrow=False,
        align="left",
        font=dict(size=11, color="#0f172a"),
        bgcolor="rgba(255,255,255,0.96)",
        bordercolor="#cbd5e1",
        borderwidth=1,
    )

    fig.update_layout(
        template="plotly_white",
        height=600,
        margin=dict(l=20, r=20, t=70, b=20),
        title=dict(
            text=f"Stylised urban canyon scenario · Predicted peak canyon temperature: {predicted_temp_c:.2f} °C",
            font=dict(size=18, color="#0f172a"),  # ← FIX HERE (dark slate)
            x=0.02,  # left align (cleaner look)
            xanchor="left",
        ),
        xaxis=dict(visible=False, range=[left_building_x0 - 6, right_building_x1 + 12]),
        yaxis=dict(visible=False, range=[-2.4, max(hc + 7.5, ht + 1.5 * rt + 6)]),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#ffffff",
    )

    return fig


def build_response_plot(
    curve_df: pd.DataFrame, x_feature: str, baseline_x: float
) -> go.Figure:
    x_label = f"{PARAM_META[x_feature]['label']} ({PARAM_META[x_feature]['symbol_ui']})"
    unit = PARAM_META[x_feature]["unit"]
    unit_suffix = f" [{unit}]" if unit != "-" else ""

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=curve_df[x_feature],
            y=curve_df["upper_95_c"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=curve_df[x_feature],
            y=curve_df["lower_95_c"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(37, 99, 235, 0.24)",
            name="95% confidence band",
            hovertemplate="Lower band: %{y:.2f} °C<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=curve_df[x_feature],
            y=curve_df["temperature_c"],
            mode="lines",
            line=dict(width=4, color="#1d4ed8"),
            name="Predicted peak temperature",
            hovertemplate=f"{x_label}: %{{x:.3f}}<br>Peak temperature: %{{y:.2f}} °C<extra></extra>",
        )
    )

    fig.add_vline(
        x=baseline_x,
        line_width=2,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text="Current scenario",
        annotation_position="top",
    )

    fig.update_layout(
        template="plotly_white",
        title=f"Surrogate response curve: {PARAM_META[x_feature]['label']}",
        xaxis_title=f"{x_label}{unit_suffix}",
        yaxis_title="Predicted peak canyon temperature [°C]",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=70, b=20),
        height=560,
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    return fig


def build_parity_plot(
    model: Pipeline, X: pd.DataFrame, y_celsius: pd.Series
) -> go.Figure:
    pred = model.predict(X)
    r2 = r2_score(y_celsius, pred)
    minv = float(min(np.min(y_celsius), np.min(pred)))
    maxv = float(max(np.max(y_celsius), np.max(pred)))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_celsius,
            y=pred,
            mode="markers",
            marker=dict(
                size=8,
                color="#2563eb",
                opacity=0.75,
                line=dict(width=0.5, color="#1e3a8a"),
            ),
            name="Samples",
            hovertemplate="Observed: %{x:.2f} °C<br>Predicted: %{y:.2f} °C<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[minv, maxv],
            y=[minv, maxv],
            mode="lines",
            line=dict(color="#ef4444", dash="dash", width=2),
            name="Ideal agreement",
        )
    )
    fig.update_layout(
        template="plotly_white",
        title=f"Surrogate fidelity check (R² = {r2:.4f})",
        xaxis_title="Observed peak temperature [°C]",
        yaxis_title="GP-predicted peak temperature [°C]",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def render_metrics(train_bundle: TrainBundle, optimisation: OptimizationBundle) -> None:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Samples", train_bundle.train_size + train_bundle.test_size)
    c2.metric("Features", len(train_bundle.feature_names))
    c3.metric("R²", f"{train_bundle.metrics['R2']:.4f}")
    c4.metric("RMSE", f"{train_bundle.metrics['RMSE_C']:.3f} °C")
    c5.metric("MAE", f"{train_bundle.metrics['MAE_C']:.3f} °C")
    c6.metric("Optimised reduction", f"{optimisation.improvement_c:.2f} °C")


def render_equations() -> None:
    st.markdown("### Mathematical formulation")
    st.markdown(
        "The surrogate and optimisation in this application can be summarised by the following equations."
    )

    st.latex(
        r"\mathbf{x} = [\alpha_w, H_r, W_c, W_r, \lambda_g, \lambda_r, H_t, r_t]^\top"
    )
    st.latex(
        r"T_{\mathrm{peak}}(\mathbf{x}) = \max_{k=1,\ldots,N} T_{2m}(t_k; \mathbf{x})"
    )
    st.latex(
        r"\hat{T}_{\mathrm{peak}}(\mathbf{x}) \sim \mathcal{GP}(\mu(\mathbf{x}), k(\mathbf{x},\mathbf{x'}))"
    )
    st.latex(
        r"k(\mathbf{x},\mathbf{x'}) = C \cdot k_{\mathrm{Matern},\nu=2.5}(\mathbf{x},\mathbf{x'}) + k_{\mathrm{white}}(\mathbf{x},\mathbf{x'})"
    )
    st.latex(r"\mathbf{x}^* = \arg\min_{\mathbf{x} \in \mathcal{B}} \mu(\mathbf{x})")
    st.latex(r"\mathcal{B} = \prod_{j=1}^{8} [\ell_j, u_j]")

    st.markdown(
        "In words: the simulator-generated peak temperature is approximated by a Gaussian process. "
        "The optimisation stage then searches the feasible design box for the parameter vector that minimises "
        "the surrogate mean prediction."
    )


def render_summary(
    data_bundle: DataBundle, train_bundle: TrainBundle, optimisation: OptimizationBundle
) -> None:
    st.markdown("### Executive summary")
    st.markdown(
        f"The surrogate was trained on **{len(data_bundle.X)} samples** with **{len(FEATURES)} design variables**. "
        f"Model quality is strong when R² is high and error metrics remain low. In the current run, the GP achieved "
        f"**R² = {train_bundle.metrics['R2']:.4f}**, **RMSE = {train_bundle.metrics['RMSE_C']:.3f} °C**, and "
        f"**MAE = {train_bundle.metrics['MAE_C']:.3f} °C** on the holdout split."
    )

    st.markdown(
        f"The surrogate-based optimisation estimates a best-case peak canyon temperature of **{optimisation.optimum_temp_c:.2f} °C**. "
        f"Relative to the median-based baseline scenario (**{optimisation.baseline_temp_c:.2f} °C**), this corresponds to an estimated reduction of **{optimisation.improvement_c:.2f} °C**."
    )

    summary_df = pd.DataFrame(
        {
            "Feature": FEATURES,
            "Symbol": [PARAM_META[f]["symbol_ui"] for f in FEATURES],
            "Optimised value": [optimisation.optimum_params[f] for f in FEATURES],
            "Unit": [PARAM_META[f]["unit"] for f in FEATURES],
        }
    )
    st.dataframe(summary_df, use_container_width=True, hide_index=True)


# =============================================================================
# MAIN UI
# =============================================================================
def build_ui() -> None:
    st.set_page_config(page_title="Urban Heat Surrogate Explorer", layout="wide")
    add_global_styles()
    render_header()

    with st.expander("Data source settings", expanded=False):
        c1, c2 = st.columns(2)
        input_path = c1.text_input("Input CSV path", value=DEFAULT_INPUT_PATH)
        output_path = c2.text_input("Output CSV path", value=DEFAULT_OUTPUT_PATH)
        st.caption(
            "The app uses 8 inputs only: albedo_w, Height_canyon, Width_canyon, Width_roof, fveg_G, fveg_R, Height_tree, Radius_tree. Extra columns such as albedo_r and alpha are ignored."
        )

    try:
        data_bundle = load_data(input_path, output_path)
        x_feature, control_values, test_size = build_sidebar(data_bundle.X)
        train_bundle = train_gp(
            data_bundle.X,
            data_bundle.y_celsius,
            test_size=test_size,
            random_state=RANDOM_STATE,
        )
        curve_df = predict_temperature_curve(
            model=train_bundle.model,
            control_values=control_values,
            x_feature=x_feature,
            n_points=260,
        )
        current_temp_c = float(
            train_bundle.model.predict(pd.DataFrame([control_values]))[0]
        )
        optimisation = optimise_surrogate(train_bundle.model, data_bundle.X)
    except Exception as exc:
        st.error(f"Application error: {exc}")
        st.stop()

    render_metrics(train_bundle, optimisation)

    tabs = st.tabs(SECTION_ORDER)

    with tabs[0]:
        render_project_description()
        parity_fig = build_parity_plot(
            train_bundle.model, data_bundle.X, data_bundle.y_celsius
        )
        st.plotly_chart(parity_fig, use_container_width=True)

    with tabs[1]:
        render_parameter_guide()

    with tabs[2]:
        st.markdown("### Scenario-based city visualisation")
        st.markdown(
            "This conceptual section turns the current parameter set into an imaginative urban canyon sketch. "
            "It is not a CFD rendering; it is an interpretable design visual that helps connect abstract parameters to urban form."
        )
        city_fig = build_city_figure(control_values, current_temp_c)
        st.plotly_chart(city_fig, use_container_width=True)

    with tabs[3]:
        st.markdown("### Surrogate modelling and optimisation")
        st.markdown(OPTIMISATION_NOTES)
        render_equations()

        left, right = st.columns([1.2, 1.0])
        with left:
            st.markdown("#### Optimisation result")
            st.success(
                f"Estimated optimum peak canyon temperature: {optimisation.optimum_temp_c:.2f} °C"
            )
            st.info(
                f"Baseline scenario: {optimisation.baseline_temp_c:.2f} °C · Estimated reduction: {optimisation.improvement_c:.2f} °C"
            )
            st.caption(f"Optimizer status: {optimisation.message}")

        with right:
            opt_df = pd.DataFrame(
                {
                    "Feature": FEATURES,
                    "Symbol": [PARAM_META[f]["symbol_ui"] for f in FEATURES],
                    "Optimised value": [
                        optimisation.optimum_params[f] for f in FEATURES
                    ],
                    "Unit": [PARAM_META[f]["unit"] for f in FEATURES],
                }
            )
            st.dataframe(opt_df, use_container_width=True, hide_index=True)

    with tabs[4]:
        st.markdown("### Interactive response explorer")
        left, right = st.columns([3.3, 1.3])
        with left:
            response_fig = build_response_plot(
                curve_df, x_feature=x_feature, baseline_x=control_values[x_feature]
            )
            st.plotly_chart(response_fig, use_container_width=True)
        with right:
            st.markdown("#### Current parameter state")
            control_df = pd.DataFrame(
                {
                    "Feature": FEATURES,
                    "Symbol": [PARAM_META[f]["symbol_ui"] for f in FEATURES],
                    "Value": [control_values[f] for f in FEATURES],
                    "Unit": [PARAM_META[f]["unit"] for f in FEATURES],
                    "Range": [
                        f"[{PARAM_BOUNDS[f][0]}, {PARAM_BOUNDS[f][1]}]"
                        for f in FEATURES
                    ],
                }
            )
            st.dataframe(control_df, use_container_width=True, hide_index=True)

            st.markdown("#### Model details")
            st.markdown(f"**Train samples:** {train_bundle.train_size}")
            st.markdown(f"**Test samples:** {train_bundle.test_size}")
            st.markdown("**Initial kernel**")
            st.code(train_bundle.kernel_before)
            st.markdown("**Optimised kernel**")
            st.code(train_bundle.kernel_after)

    with tabs[5]:
        render_summary(data_bundle, train_bundle, optimisation)
        st.markdown("### Interpretation notes")
        st.markdown(
            "- The response curve is a one-dimensional cross-section through the surrogate while the remaining variables are fixed.\n"
            "- The blue band shows an approximate 95% predictive interval from the Gaussian process.\n"
            "- The city sketch is intentionally schematic and designed for stakeholder communication.\n"
            "- A later extension can add an LLM assistant and PDF-based retrieval to explain results interactively."
        )


if __name__ == "__main__":
    build_ui()

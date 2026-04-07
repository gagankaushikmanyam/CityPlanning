import importlib
from typing import Dict, Optional

import pandas as pd
import streamlit as st

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import torch

    HF_AVAILABLE = True
except Exception:
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None
    torch = None
    HF_AVAILABLE = False


# -----------------------------------------------------------------------------
# Import your base app
# -----------------------------------------------------------------------------
base_app = importlib.import_module("app")


# -----------------------------------------------------------------------------
# Build live app state
# -----------------------------------------------------------------------------
def build_runtime_state(
    data_bundle,
    train_bundle,
    optimisation,
    control_values: Dict[str, float],
    x_feature: str,
    current_temp_c: float,
) -> Dict[str, object]:
    return {
        "sample_count": int(len(data_bundle.X)),
        "feature_count": int(len(base_app.FEATURES)),
        "features": list(base_app.FEATURES),
        "selected_x_feature": x_feature,
        "selected_x_symbol": base_app.PARAM_META[x_feature].get("symbol_ui", x_feature),
        "current_parameter_values": {k: float(v) for k, v in control_values.items()},
        "current_predicted_peak_temperature_c": float(current_temp_c),
        "optimised_peak_temperature_c": float(optimisation.optimum_temp_c),
        "baseline_peak_temperature_c": float(optimisation.baseline_temp_c),
        "optimised_reduction_c": float(optimisation.improvement_c),
        "optimised_parameters": {
            k: float(v) for k, v in optimisation.optimum_params.items()
        },
        "model_metrics": {
            "R2": float(train_bundle.metrics["R2"]),
            "RMSE_C": float(train_bundle.metrics["RMSE_C"]),
            "MAE_C": float(train_bundle.metrics["MAE_C"]),
        },
        "kernel_before": str(train_bundle.kernel_before),
        "kernel_after": str(train_bundle.kernel_after),
        "train_size": int(train_bundle.train_size),
        "test_size": int(train_bundle.test_size),
        "parameter_metadata": {
            k: {
                "label": base_app.PARAM_META[k]["label"],
                "symbol_ui": base_app.PARAM_META[k].get(
                    "symbol_ui", base_app.PARAM_META[k].get("symbol", k)
                ),
                "unit": base_app.PARAM_META[k]["unit"],
                "description": base_app.PARAM_META[k]["description"],
                "bounds": base_app.PARAM_BOUNDS[k],
            }
            for k in base_app.FEATURES
        },
    }


# -----------------------------------------------------------------------------
# Question parsing helpers
# -----------------------------------------------------------------------------
def _find_feature_by_question(question: str) -> Optional[str]:
    q = question.lower()

    for feature in base_app.FEATURES:
        meta = base_app.PARAM_META[feature]
        if feature.lower() in q:
            return feature
        if meta["label"].lower() in q:
            return feature
        symbol_ui = meta.get("symbol_ui", "").lower()
        if symbol_ui and symbol_ui in q:
            return feature

    aliases = {
        "wall albedo": "albedo_w",
        "canyon height": "Height_canyon",
        "canyon width": "Width_canyon",
        "roof width": "Width_roof",
        "ground vegetation": "fveg_G",
        "roof vegetation": "fveg_R",
        "tree height": "Height_tree",
        "tree radius": "Radius_tree",
        "crown radius": "Radius_tree",
    }

    for alias, feature in aliases.items():
        if alias in q:
            return feature

    return None


# -----------------------------------------------------------------------------
# Deterministic assistant
# -----------------------------------------------------------------------------
def answer_from_app_state(question: str, state: Dict[str, object]) -> str:
    q = question.lower().strip()
    metrics = state["model_metrics"]
    current_values = state["current_parameter_values"]
    opt_values = state["optimised_parameters"]
    param_meta = state["parameter_metadata"]

    if any(term in q for term in ["hello", "hi", "hey"]):
        return (
            "Hello. I can answer questions about the current scenario, parameter values, "
            "model metrics, optimisation results, and the Gaussian Process surrogate."
        )

    if any(
        term in q
        for term in ["r2", "r²", "rmse", "mae", "metric", "accuracy", "model quality"]
    ):
        return (
            f"Current surrogate metrics are: R² = {metrics['R2']:.4f}, "
            f"RMSE = {metrics['RMSE_C']:.3f} °C, and MAE = {metrics['MAE_C']:.3f} °C. "
            f"The model was trained on {state['train_size']} training samples and evaluated on "
            f"{state['test_size']} test samples."
        )

    if any(
        term in q
        for term in [
            "optim",
            "best temperature",
            "minimum temperature",
            "optimum",
            "best case",
        ]
    ):
        return (
            f"The current surrogate optimisation estimates an optimum peak canyon temperature of "
            f"{state['optimised_peak_temperature_c']:.2f} °C. The baseline scenario is "
            f"{state['baseline_peak_temperature_c']:.2f} °C, corresponding to an estimated reduction of "
            f"{state['optimised_reduction_c']:.2f} °C."
        )

    if any(
        term in q
        for term in [
            "current temperature",
            "predicted temperature",
            "current scenario",
            "current result",
        ]
    ):
        return (
            f"The currently selected scenario predicts a peak canyon temperature of "
            f"{state['current_predicted_peak_temperature_c']:.2f} °C. The selected x-axis feature is "
            f"{param_meta[state['selected_x_feature']]['label']} "
            f"({param_meta[state['selected_x_feature']]['symbol_ui']})."
        )

    if "kernel" in q or "gaussian process" in q or "gpr" in q or "surrogate" in q:
        return (
            f"The app uses a Gaussian Process surrogate. Initial kernel: {state['kernel_before']}. "
            f"Optimised kernel: {state['kernel_after']}."
        )

    if "all parameters" in q or "current parameters" in q:
        lines = []
        for feature in base_app.FEATURES:
            meta = param_meta[feature]
            value = current_values[feature]
            unit = meta["unit"]
            unit_txt = "" if unit == "-" else f" {unit}"
            lines.append(
                f"- {meta['label']} ({meta['symbol_ui']}): {value:.4f}{unit_txt}"
            )
        return "Current scenario parameters:\n" + "\n".join(lines)

    if (
        "optimised parameters" in q
        or "optimized parameters" in q
        or "optimal parameters" in q
    ):
        lines = []
        for feature in base_app.FEATURES:
            meta = param_meta[feature]
            value = opt_values[feature]
            unit = meta["unit"]
            unit_txt = "" if unit == "-" else f" {unit}"
            lines.append(
                f"- {meta['label']} ({meta['symbol_ui']}): {value:.4f}{unit_txt}"
            )
        return "Surrogate-optimised parameters:\n" + "\n".join(lines)

    feature = _find_feature_by_question(question)
    if feature is not None:
        meta = param_meta[feature]
        current_v = current_values[feature]
        opt_v = opt_values[feature]
        unit = meta["unit"]
        unit_txt = "" if unit == "-" else f" {unit}"
        return (
            f"For {meta['label']} ({meta['symbol_ui']}), the current scenario value is "
            f"{current_v:.4f}{unit_txt}. The surrogate-optimised value is "
            f"{opt_v:.4f}{unit_txt}. The feasible range used in the app is "
            f"[{meta['bounds'][0]}, {meta['bounds'][1]}]{unit_txt}. "
            f"{meta['description']}"
        )

    if (
        "what does this project do" in q
        or "project description" in q
        or "abstract" in q
    ):
        return base_app.PROJECT_ABSTRACT

    if "summary" in q:
        return (
            f"This dashboard uses a Gaussian Process surrogate to explore how urban design parameters affect "
            f"peak canyon temperature. The current scenario predicts {state['current_predicted_peak_temperature_c']:.2f} °C, "
            f"while the surrogate optimisation estimates a best-case value of "
            f"{state['optimised_peak_temperature_c']:.2f} °C."
        )

    return (
        "I can answer questions about the current app state, including parameter values, "
        "current predicted temperature, optimisation results, model metrics, and surrogate settings. "
        "Try asking things like: 'What is the current predicted temperature?', "
        "'What is the R² score?', 'What is the optimised canyon width?', "
        "or 'Show all current parameters.'"
    )


# -----------------------------------------------------------------------------
# Hugging Face explanation layer
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_explainer_model(model_name: str = "google/flan-t5-small"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


def explain_with_hf(
    factual_answer: str,
    question: str,
    model_name: str = "google/flan-t5-small",
    max_new_tokens: int = 220,
) -> str:
    if not HF_AVAILABLE:
        return factual_answer

    tokenizer, model = load_explainer_model(model_name)

    prompt = f"""
You are a scientific dashboard assistant.

Rewrite the following validated answer in a clear, natural, professional way.
Do not change any numbers, units, parameter names, symbols, or conclusions.
Do not add new facts.
Keep the answer concise but helpful.

User question:
{question}

Validated factual answer:
{factual_answer}

Improved explanation:
""".strip()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return text if text else factual_answer


# -----------------------------------------------------------------------------
# Assistant tab
# -----------------------------------------------------------------------------
def render_assistant_tab(state: Dict[str, object]) -> None:
    st.markdown("### App Assistant")
    st.markdown(
        "This assistant answers directly from the live app state. "
        "You can also turn on a Hugging Face explainer that rewrites the validated answer "
        "in a more natural and professional style without changing the numbers."
    )

    with st.expander("Assistant settings", expanded=False):
        use_hf_explainer = st.checkbox(
            "Use Hugging Face explainer for more natural answers",
            value=False,
        )
        hf_model_name = st.text_input(
            "Explainer model",
            value="google/flan-t5-small",
            disabled=not use_hf_explainer,
        )
        if use_hf_explainer and not HF_AVAILABLE:
            st.warning(
                "Transformers/PyTorch are not installed. Install them with: "
                "`pip install transformers torch`"
            )

    with st.expander("Current app state snapshot", expanded=False):
        st.json(state)

    if "app_assistant_messages" not in st.session_state:
        st.session_state.app_assistant_messages = [
            {
                "role": "assistant",
                "content": (
                    "Ask me about the current scenario, optimisation result, model metrics, "
                    "parameter values, or surrogate settings."
                ),
            }
        ]

    for msg in st.session_state.app_assistant_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask about the app results...")
    if prompt:
        st.session_state.app_assistant_messages.append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        factual_answer = answer_from_app_state(prompt, state)

        if use_hf_explainer and HF_AVAILABLE:
            with st.spinner("Improving explanation..."):
                final_answer = explain_with_hf(
                    factual_answer=factual_answer,
                    question=prompt,
                    model_name=hf_model_name,
                )
            final_answer += (
                "\n\nValidation: The explanation style was improved with Hugging Face, "
                "but the underlying facts came from the live app state."
            )
        else:
            final_answer = (
                factual_answer
                + "\n\nValidation: This answer was generated directly from the live app state."
            )

        with st.chat_message("assistant"):
            st.markdown(final_answer)

        st.session_state.app_assistant_messages.append(
            {"role": "assistant", "content": final_answer}
        )


# -----------------------------------------------------------------------------
# Main wrapper UI
# -----------------------------------------------------------------------------
def render_extended_ui() -> None:
    st.set_page_config(page_title="Urban Heat Surrogate Explorer", layout="wide")
    base_app.add_global_styles()

    st.markdown(
        """
        <div class="hero-card">
            <h1 style="margin-bottom:0.25rem;">Urban Heat Surrogate Explorer</h1>
            <div style="font-size:1.02rem; opacity:0.95; max-width:1000px;">
                Scientific dashboard for Gaussian-process-based urban heat analysis,
                scenario exploration, surrogate optimisation, and app-aware communication.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Data source settings", expanded=False):
        c1, c2 = st.columns(2)
        input_path = c1.text_input("Input CSV path", value=base_app.DEFAULT_INPUT_PATH)
        output_path = c2.text_input(
            "Output CSV path", value=base_app.DEFAULT_OUTPUT_PATH
        )
        st.caption(
            "The app uses 8 inputs only: albedo_w, Height_canyon, Width_canyon, Width_roof, "
            "fveg_G, fveg_R, Height_tree, Radius_tree. Extra columns such as albedo_r and alpha are ignored."
        )

    try:
        data_bundle = base_app.load_data(input_path, output_path)
        x_feature, control_values, test_size = base_app.build_sidebar(data_bundle.X)
        train_bundle = base_app.train_gp(
            data_bundle.X,
            data_bundle.y_celsius,
            test_size=test_size,
            random_state=base_app.RANDOM_STATE,
        )
        curve_df = base_app.predict_temperature_curve(
            model=train_bundle.model,
            control_values=control_values,
            x_feature=x_feature,
            n_points=260,
        )
        current_temp_c = float(
            train_bundle.model.predict(pd.DataFrame([control_values]))[0]
        )
        optimisation = base_app.optimise_surrogate(train_bundle.model, data_bundle.X)
    except Exception as exc:
        st.error(f"Application error: {exc}")
        st.stop()

    base_app.render_metrics(train_bundle, optimisation)

    state = build_runtime_state(
        data_bundle=data_bundle,
        train_bundle=train_bundle,
        optimisation=optimisation,
        control_values=control_values,
        x_feature=x_feature,
        current_temp_c=current_temp_c,
    )

    tabs = st.tabs(base_app.SECTION_ORDER + ["App Assistant"])

    with tabs[0]:
        base_app.render_project_description()
        parity_fig = base_app.build_parity_plot(
            train_bundle.model, data_bundle.X, data_bundle.y_celsius
        )
        st.plotly_chart(parity_fig, width="stretch")

    with tabs[1]:
        base_app.render_parameter_guide()

    with tabs[2]:
        st.markdown("### Scenario-based city visualisation")
        st.markdown(
            "This conceptual section turns the current parameter set into an imaginative urban canyon sketch."
        )
        city_fig = base_app.build_city_figure(control_values, current_temp_c)
        st.plotly_chart(city_fig, width="stretch")

    with tabs[3]:
        st.markdown("### Surrogate modelling and optimisation")
        st.markdown(base_app.OPTIMISATION_NOTES)
        base_app.render_equations()

        left, right = st.columns([1.2, 1.0])
        with left:
            st.markdown("#### Optimisation result")
            st.success(
                f"Estimated optimum peak canyon temperature: {optimisation.optimum_temp_c:.2f} °C"
            )
            st.info(
                f"Baseline scenario: {optimisation.baseline_temp_c:.2f} °C · "
                f"Estimated reduction: {optimisation.improvement_c:.2f} °C"
            )
            st.caption(f"Optimizer status: {optimisation.message}")

        with right:
            opt_df = pd.DataFrame(
                {
                    "Feature": base_app.FEATURES,
                    "Symbol": [
                        base_app.PARAM_META[f].get("symbol_ui", f)
                        for f in base_app.FEATURES
                    ],
                    "Optimised value": [
                        optimisation.optimum_params[f] for f in base_app.FEATURES
                    ],
                    "Unit": [base_app.PARAM_META[f]["unit"] for f in base_app.FEATURES],
                }
            )
            st.dataframe(opt_df, width="stretch", hide_index=True)

    with tabs[4]:
        st.markdown("### Interactive response explorer")
        left, right = st.columns([3.3, 1.3])

        with left:
            response_fig = base_app.build_response_plot(
                curve_df, x_feature=x_feature, baseline_x=control_values[x_feature]
            )
            st.plotly_chart(response_fig, width="stretch")

        with right:
            st.markdown("#### Current parameter state")
            control_df = pd.DataFrame(
                {
                    "Feature": base_app.FEATURES,
                    "Symbol": [
                        base_app.PARAM_META[f].get("symbol_ui", f)
                        for f in base_app.FEATURES
                    ],
                    "Value": [control_values[f] for f in base_app.FEATURES],
                    "Unit": [base_app.PARAM_META[f]["unit"] for f in base_app.FEATURES],
                    "Range": [
                        f"[{base_app.PARAM_BOUNDS[f][0]}, {base_app.PARAM_BOUNDS[f][1]}]"
                        for f in base_app.FEATURES
                    ],
                }
            )
            st.dataframe(control_df, width="stretch", hide_index=True)

            st.markdown("#### Model details")
            st.markdown(f"**Train samples:** {train_bundle.train_size}")
            st.markdown(f"**Test samples:** {train_bundle.test_size}")
            st.markdown("**Initial kernel**")
            st.code(train_bundle.kernel_before)
            st.markdown("**Optimised kernel**")
            st.code(train_bundle.kernel_after)

    with tabs[5]:
        base_app.render_summary(data_bundle, train_bundle, optimisation)
        st.markdown("### Interpretation notes")
        st.markdown(
            "- The response curve is a one-dimensional cross-section through the surrogate while the remaining variables are fixed.\n"
            "- Questions about model metrics and current parameter values are answered directly from the live app state.\n"
            "- The App Assistant is validation-first and can optionally rewrite answers in a more natural style using Hugging Face."
        )

    with tabs[6]:
        render_assistant_tab(state)


if __name__ == "__main__":
    render_extended_ui()

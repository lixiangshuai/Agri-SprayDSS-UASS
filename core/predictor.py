from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import xgboost as xgb

from .schema import (
    FEATURE_BY_KEY,
    MODEL_FEATURE_KEYS,
    MODEL_SPECS,
    ModelSpec,
    feature_label,
    unit_label,
)


@dataclass(frozen=True)
class PredictionOnlyResult:
    model_key: str
    prediction: float
    input_df: pd.DataFrame


@dataclass(frozen=True)
class PredictionResult:
    model_key: str
    prediction: float
    input_df: pd.DataFrame
    shap_values: object
    expected_value: float
    contribution_table: pd.DataFrame


@lru_cache(maxsize=2)
def load_model(model_key: str) -> xgb.XGBRegressor:
    spec = _get_model_spec(model_key)
    if not spec.model_path.exists():
        raise FileNotFoundError(f"Model file not found: {spec.model_path}")

    model = xgb.XGBRegressor()
    model.load_model(str(spec.model_path))
    return model


def _get_model_spec(model_key: str) -> ModelSpec:
    if model_key not in MODEL_SPECS:
        valid = ", ".join(sorted(MODEL_SPECS))
        raise KeyError(f"Unknown model '{model_key}'. Valid models: {valid}")
    return MODEL_SPECS[model_key]


def get_model_feature_names(model_key: str) -> List[str]:
    model = load_model(model_key)
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        names = model.get_booster().feature_names
    if names is None:
        names = MODEL_FEATURE_KEYS

    names = list(names)
    if len(names) != len(MODEL_FEATURE_KEYS):
        raise ValueError(
            f"Model '{model_key}' expects {len(names)} features, "
            f"but schema defines {len(MODEL_FEATURE_KEYS)}."
        )
    return names


def build_input_frame(model_key: str, values: Dict[str, float]) -> pd.DataFrame:
    missing = [feature_key for feature_key in MODEL_FEATURE_KEYS if feature_key not in values]
    if missing:
        raise KeyError(f"Missing input values: {', '.join(missing)}")

    model_feature_names = get_model_feature_names(model_key)
    row = {
        model_feature_name: float(values[feature_key])
        for model_feature_name, feature_key in zip(model_feature_names, MODEL_FEATURE_KEYS)
    }

    for model_feature_name, feature_key in zip(model_feature_names, MODEL_FEATURE_KEYS):
        feature = FEATURE_BY_KEY[feature_key]
        if feature.step.is_integer():
            row[model_feature_name] = int(round(row[model_feature_name]))

    return pd.DataFrame([row], columns=model_feature_names)


def predict_value(model_key: str, values: Dict[str, float]) -> PredictionOnlyResult:
    model = load_model(model_key)
    input_df = build_input_frame(model_key, values)
    prediction = float(model.predict(input_df)[0])
    return PredictionOnlyResult(
        model_key=model_key,
        prediction=prediction,
        input_df=input_df,
    )


def predict(model_key: str, values: Dict[str, float]) -> PredictionResult:
    model = load_model(model_key)
    input_df = build_input_frame(model_key, values)

    prediction = float(model.predict(input_df)[0])
    shap_values, expected_value = compute_shap_values(model, input_df)
    contribution_table = build_contribution_table(model_key, values, shap_values)

    return PredictionResult(
        model_key=model_key,
        prediction=prediction,
        input_df=input_df,
        shap_values=shap_values,
        expected_value=expected_value,
        contribution_table=contribution_table,
    )


def compute_shap_values(model: xgb.XGBRegressor, input_df: pd.DataFrame) -> Tuple[object, float]:
    """Compute exact Tree SHAP contributions for one current input row."""

    import shap

    booster = model.get_booster()
    dmatrix = xgb.DMatrix(input_df, feature_names=list(input_df.columns))
    kwargs = {}
    iteration_range = _get_iteration_range(model)
    if iteration_range is not None:
        kwargs["iteration_range"] = iteration_range

    contributions = booster.predict(dmatrix, pred_contribs=True, **kwargs)
    shap_values = contributions[:, :-1]
    expected_value = float(contributions[0, -1])
    explanation = shap.Explanation(
        values=shap_values,
        base_values=[expected_value],
        data=input_df.to_numpy(),
        feature_names=list(input_df.columns),
    )
    return explanation, expected_value


def _get_iteration_range(model: xgb.XGBRegressor) -> Optional[Tuple[int, int]]:
    try:
        best_iteration = model.best_iteration
    except AttributeError:
        best_iteration = None
    if best_iteration is None:
        best_iteration = model.get_booster().attr("best_iteration")
    if best_iteration is None:
        return None
    return (0, int(best_iteration) + 1)


def build_contribution_table(model_key: str, values: Dict[str, float], shap_values: object) -> pd.DataFrame:
    model_feature_names = get_model_feature_names(model_key)
    rows = []
    for index, feature_key in enumerate(MODEL_FEATURE_KEYS):
        feature = FEATURE_BY_KEY[feature_key]
        contribution = float(shap_values.values[0][index])
        rows.append(
            {
                "feature_key": feature_key,
                "feature": feature.label_en,
                "value": float(values[feature_key]),
                "unit": feature.unit,
                "model_feature": model_feature_names[index],
                "shap_value": contribution,
                "direction": "increases drift" if contribution > 0 else "reduces drift",
                "abs_shap_value": abs(contribution),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("abs_shap_value", ascending=False)
        .reset_index(drop=True)
    )


def make_force_plot(
    result: PredictionResult,
    language: str = "en",
    exclude_feature_keys: Iterable[str] = (),
):
    import shap

    plot_base_value, visible_values, display_input = _force_plot_payload(
        result=result,
        language=language,
        exclude_feature_keys=exclude_feature_keys,
    )
    return shap.force_plot(
        plot_base_value,
        visible_values,
        display_input,
    )


def make_force_plot_png(
    result: PredictionResult,
    language: str = "en",
    exclude_feature_keys: Iterable[str] = (),
    dpi: int = 300,
) -> bytes:
    import matplotlib.pyplot as plt
    import shap

    plot_base_value, visible_values, display_input = _force_plot_payload(
        result=result,
        language=language,
        exclude_feature_keys=exclude_feature_keys,
    )
    plt.close("all")
    shap.force_plot(
        plot_base_value,
        visible_values,
        display_input,
        matplotlib=True,
        show=False,
        figsize=(20, 3),
    )
    figure = plt.gcf()
    output = BytesIO()
    figure.savefig(output, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    output.seek(0)
    return output.getvalue()


def _force_plot_payload(
    result: PredictionResult,
    language: str = "en",
    exclude_feature_keys: Iterable[str] = (),
):
    excluded = set(exclude_feature_keys)
    visible_indices = [
        index
        for index, feature_key in enumerate(MODEL_FEATURE_KEYS)
        if feature_key not in excluded
    ]
    hidden_contribution = sum(
        float(result.shap_values.values[0][index])
        for index, feature_key in enumerate(MODEL_FEATURE_KEYS)
        if feature_key in excluded
    )
    plot_base_value = result.expected_value + hidden_contribution
    display_input = pd.Series(
        [
            result.input_df.iloc[0, index]
            for index in visible_indices
        ],
        index=[
            f"{feature_label(feature_key, language)} ({unit_label(feature_key, language)})"
            for index, feature_key in enumerate(MODEL_FEATURE_KEYS)
            if index in visible_indices
        ],
    )
    return plot_base_value, result.shap_values.values[0][visible_indices], display_input
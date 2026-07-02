from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


@dataclass(frozen=True)
class FeatureSpec:
    key: str
    model_order: int
    label_en: str
    label_zh: str
    unit: str
    recommended_range: Tuple[float, float]
    default: float
    step: float
    display_format: str
    controllable: bool = True


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label_en: str
    label_zh: str
    model_path: Path
    wind_speed_target: float
    altitude_target: float
    operating_speed_target: float
    temperature_empirical_range: Optional[Tuple[float, float]] = None


FEATURES: List[FeatureSpec] = [
    FeatureSpec(
        key="operating_altitude",
        model_order=0,
        label_en="Operating altitude",
        label_zh="作业高度",
        unit="m",
        recommended_range=(2.0, 6.0),
        default=3.0,
        step=0.01,
        display_format="%.2f",
    ),
    FeatureSpec(
        key="operating_speed",
        model_order=1,
        label_en="Operating speed",
        label_zh="作业速度",
        unit="m/s",
        recommended_range=(3.0, 6.0),
        default=8.0,
        step=0.01,
        display_format="%.2f",
    ),
    FeatureSpec(
        key="wind_speed",
        model_order=2,
        label_en="Wind speed",
        label_zh="风速",
        unit="m/s",
        recommended_range=(0.0, 7.0),
        default=2.0,
        step=0.01,
        display_format="%.2f",
    ),
    FeatureSpec(
        key="wind_deviation",
        model_order=3,
        label_en="Wind deviation",
        label_zh="风向偏角",
        unit="°",
        recommended_range=(0.0, 90.0),
        default=30.0,
        step=0.01,
        display_format="%.2f",
    ),
    FeatureSpec(
        key="temperature",
        model_order=4,
        label_en="Temperature",
        label_zh="温度",
        unit="℃",
        recommended_range=(10.0, 40.0),
        default=25.0,
        step=0.01,
        display_format="%.2f",
        controllable=False,
    ),
    FeatureSpec(
        key="relative_humidity",
        model_order=5,
        label_en="Relative humidity",
        label_zh="相对湿度",
        unit="%",
        recommended_range=(20.0, 80.0),
        default=60.0,
        step=0.01,
        display_format="%.2f",
        controllable=False,
    ),
    FeatureSpec(
        key="distance",
        model_order=6,
        label_en="Downwind distance",
        label_zh="下风向距离",
        unit="m",
        recommended_range=(1.0, 100.0),
        default=1.0,
        step=1.0,
        display_format="%.0f",
    ),
]

FEATURE_BY_KEY: Dict[str, FeatureSpec] = {feature.key: feature for feature in FEATURES}
MODEL_FEATURE_KEYS: List[str] = [
    feature.key for feature in sorted(FEATURES, key=lambda item: item.model_order)
]


MODEL_SPECS: Dict[str, ModelSpec] = {
    "fine": ModelSpec(
        key="fine",
        label_en="Fine droplet size",
        label_zh="细雾滴粒径",
        model_path=MODELS_DIR / "fine_droplet_xgb.ubj",
        wind_speed_target=2.2,
        altitude_target=2.92,
        operating_speed_target=4.65,
        temperature_empirical_range=(10.0, 40.0),
    ),
    "medium": ModelSpec(
        key="medium",
        label_en="Medium droplet size",
        label_zh="中等雾滴粒径",
        model_path=MODELS_DIR / "medium_droplet_xgb.ubj",
        wind_speed_target=3.78,
        altitude_target=3.34,
        operating_speed_target=5.09,
        temperature_empirical_range=(12.0, 23.0),
    ),
}


def feature_label(feature_key: str, language: str = "en") -> str:
    feature = FEATURE_BY_KEY[feature_key]
    return feature.label_zh if language == "zh" else feature.label_en


def unit_label(feature_key: str, language: str = "en") -> str:
    return FEATURE_BY_KEY[feature_key].unit


def model_label(model_key: str, language: str = "en") -> str:
    model = MODEL_SPECS[model_key]
    return model.label_zh if language == "zh" else model.label_en

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from .schema import FEATURE_BY_KEY, MODEL_SPECS


@dataclass(frozen=True)
class ThresholdRule:
    feature_key: str
    threshold: float
    operator: str
    direction_en: str
    direction_zh: str
    advice_en: str
    advice_zh: str


@dataclass(frozen=True)
class Recommendation:
    feature_key: str
    priority: int
    title_en: str
    title_zh: str
    detail_en: str
    detail_zh: str
    threshold: Optional[float] = None
    current_value: Optional[float] = None
    unit: str = ""
    operator: str = ""
    direction_en: str = ""
    direction_zh: str = ""


THRESHOLD_RULES: Dict[str, List[ThresholdRule]] = {
    "fine": [
        ThresholdRule(
            feature_key="operating_altitude",
            threshold=2.92,
            operator=">",
            direction_en="Higher operating altitude may increase drift.",
            direction_zh="作业高度偏高，可能促进飘移增加。",
            advice_en=(
                "Current parameters indicate drift risk; recommended action: reduce "
                "operating altitude to below 2.92 m."
            ),
            advice_zh="当前参数存在飘移风险，建议降低作业高度至 2.92 m 以下。",
        ),
        ThresholdRule(
            feature_key="operating_speed",
            threshold=4.65,
            operator=">",
            direction_en="Higher operating speed may increase drift.",
            direction_zh="作业速度偏高，可能促进飘移增加。",
            advice_en=(
                "Current parameters indicate drift risk; recommended action: reduce "
                "operating speed to below 4.65 m/s."
            ),
            advice_zh="当前参数存在飘移风险，建议降低作业速度至 4.65 m/s 以下。",
        ),
        ThresholdRule(
            feature_key="wind_speed",
            threshold=2.20,
            operator=">",
            direction_en="High wind speed clearly increases fine-droplet drift risk.",
            direction_zh="当前风速较高，细雾滴飘移风险明显增加。",
            advice_en=(
                "Current parameters indicate drift risk; recommended action: pause "
                "spraying or wait until wind speed falls below 2.20 m/s. If operation "
                "cannot be delayed, consider switching to a larger droplet size."
            ),
            advice_zh=(
                "当前参数存在飘移风险，建议暂停作业或等待风速降至 2.20 m/s 以下；"
                "若必须作业，可考虑更换为更大雾滴粒径以降低飘移风险。"
            ),
        ),
        ThresholdRule(
            feature_key="wind_deviation",
            threshold=20.59,
            operator="<",
            direction_en="A smaller wind deviation angle is associated with higher drift risk.",
            direction_zh="当前风向偏角较小，飘移风险较高。",
            advice_en=(
                "Current parameters indicate drift risk; recommended action: adjust the "
                "flight route direction or wait for wind direction changes so the wind "
                "deviation angle is not lower than 20.59°."
            ),
            advice_zh=(
                "当前参数存在飘移风险，建议调整航线方向或等待风向变化，"
                "使风向偏角不低于 20.59°。"
            ),
        ),
        ThresholdRule(
            feature_key="temperature",
            threshold=21.49,
            operator=">",
            direction_en="High temperature may accelerate droplet evaporation and diffusion.",
            direction_zh="温度较高，可能加速雾滴蒸发和扩散。",
            advice_en=(
                "Current parameters indicate drift risk; recommended action: avoid high "
                "temperature periods and wait for a cooler weather window. If operation "
                "cannot be delayed, consider switching to a larger droplet size."
            ),
            advice_zh=(
                "当前参数存在飘移风险，建议避开高温时段作业，等待更适宜天气窗口；"
                "若必须作业，可考虑更换为更大雾滴粒径。"
            ),
        ),
        ThresholdRule(
            feature_key="relative_humidity",
            threshold=55.51,
            operator="<",
            direction_en="Low relative humidity increases droplet evaporation risk.",
            direction_zh="空气湿度偏低，雾滴蒸发风险增加。",
            advice_en=(
                "Current parameters indicate drift risk; recommended action: wait until "
                "relative humidity is higher than 55.51%. If operation cannot be delayed, "
                "consider switching to a larger droplet size."
            ),
            advice_zh=(
                "当前参数存在飘移风险，建议在相对湿度高于 55.51% 时作业；"
                "若必须作业，可考虑更换为更大雾滴粒径。"
            ),
        ),
    ],
    "medium": [
        ThresholdRule(
            feature_key="operating_altitude",
            threshold=3.34,
            operator=">",
            direction_en="Higher operating altitude may increase drift risk.",
            direction_zh="作业高度偏高，可能增加飘移风险。",
            advice_en=(
                "Current parameters indicate drift risk; recommended action: reduce "
                "operating altitude to below 3.34 m."
            ),
            advice_zh="当前参数存在飘移风险，建议降低作业高度至 3.34 m 以下。",
        ),
        ThresholdRule(
            feature_key="operating_speed",
            threshold=5.09,
            operator=">",
            direction_en="Higher operating speed may increase drift.",
            direction_zh="作业速度偏高，可能促进飘移增加。",
            advice_en=(
                "Current parameters indicate drift risk; recommended action: reduce "
                "operating speed to below 5.09 m/s."
            ),
            advice_zh="当前参数存在飘移风险，建议降低作业速度至 5.09 m/s 以下。",
        ),
        ThresholdRule(
            feature_key="wind_speed",
            threshold=3.78,
            operator=">",
            direction_en="High wind speed increases medium-droplet drift risk.",
            direction_zh="当前风速较高，中等雾滴飘移风险增加。",
            advice_en=(
                "Current parameters indicate drift risk; recommended action: pause "
                "spraying or wait until wind speed falls below 3.78 m/s. If operation "
                "cannot be delayed, consider switching to a larger droplet size."
            ),
            advice_zh=(
                "当前参数存在飘移风险，建议暂停作业或等待风速降至 3.78 m/s 以下；"
                "若必须作业，可考虑更换为更大雾滴粒径以降低飘移风险。"
            ),
        ),
        ThresholdRule(
            feature_key="wind_deviation",
            threshold=20.46,
            operator="<",
            direction_en="A smaller wind deviation angle is associated with higher drift risk.",
            direction_zh="当前风向偏角较小，飘移风险较高。",
            advice_en=(
                "Current parameters indicate drift risk; recommended action: adjust the "
                "flight route direction or wait for wind direction changes so the wind "
                "deviation angle is not lower than 20.46°."
            ),
            advice_zh=(
                "当前参数存在飘移风险，建议调整航线方向或等待风向变化，"
                "使风向偏角不低于 20.46°。"
            ),
        ),
        ThresholdRule(
            feature_key="temperature",
            threshold=18.68,
            operator=">",
            direction_en="High temperature may accelerate droplet evaporation and diffusion.",
            direction_zh="温度较高，可能加速雾滴蒸发和扩散。",
            advice_en=(
                "Current parameters indicate drift risk; recommended action: avoid high "
                "temperature periods and wait for a cooler weather window. If operation "
                "cannot be delayed, consider switching to a larger droplet size."
            ),
            advice_zh=(
                "当前参数存在飘移风险，建议避开高温时段作业，等待更适宜天气窗口；"
                "若必须作业，可考虑更换为更大雾滴粒径。"
            ),
        ),
        ThresholdRule(
            feature_key="relative_humidity",
            threshold=54.05,
            operator="<",
            direction_en="Low relative humidity increases droplet evaporation risk.",
            direction_zh="空气湿度偏低，雾滴蒸发风险增加。",
            advice_en=(
                "Current parameters indicate drift risk; recommended action: wait until "
                "relative humidity is higher than 54.05%. If operation cannot be delayed, "
                "consider switching to a larger droplet size."
            ),
            advice_zh=(
                "当前参数存在飘移风险，建议在相对湿度高于 54.05% 时作业；"
                "若必须作业，可考虑更换为更大雾滴粒径。"
            ),
        ),
    ],
}


def build_recommendations(
    model_key: str,
    values: Dict[str, float],
    contribution_table: pd.DataFrame,
    max_items: int = 6,
) -> List[Recommendation]:
    del contribution_table
    spec = MODEL_SPECS[model_key]
    recommendations: List[Recommendation] = []

    for rule in THRESHOLD_RULES[model_key]:
        value = float(values[rule.feature_key])
        if not _is_risky(value, rule):
            continue

        feature = FEATURE_BY_KEY[rule.feature_key]
        recommendations.append(
            Recommendation(
                feature_key=rule.feature_key,
                priority=len(recommendations) + 1,
                title_en=f"{feature.label_en} exceeds the drift-reduction threshold",
                title_zh=f"{feature.label_zh}触发减飘阈值",
                detail_en=rule.advice_en,
                detail_zh=rule.advice_zh,
                threshold=rule.threshold,
                current_value=value,
                unit=feature.unit,
                operator=rule.operator,
                direction_en=rule.direction_en,
                direction_zh=rule.direction_zh,
            )
        )
        if len(recommendations) >= max_items:
            break

    if recommendations:
        return recommendations

    return [
        Recommendation(
            feature_key="overall",
            priority=1,
            title_en=f"No threshold risk detected for {spec.label_en}",
            title_zh=f"{spec.label_zh}未触发减飘阈值",
            detail_en=(
                "No current input parameter exceeds its model-specific drift-reduction "
                "threshold. Maintain the current plan and continue monitoring weather "
                "conditions before spraying."
            ),
            detail_zh=(
                "当前输入参数未触发该雾滴模型的单阈值减飘规则。建议保持当前作业方案，"
                "并在田间作业前继续监测风速、温度和相对湿度。"
            ),
        )
    ]


def _is_risky(value: float, rule: ThresholdRule) -> bool:
    if rule.operator == ">":
        return value > rule.threshold
    if rule.operator == "<":
        return value < rule.threshold
    raise ValueError(f"Unsupported threshold operator: {rule.operator}")

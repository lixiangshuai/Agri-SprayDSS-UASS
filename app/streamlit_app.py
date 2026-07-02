from __future__ import annotations

import base64
import html
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.predictor import make_force_plot, make_force_plot_png, predict, predict_value  # noqa: E402
from core.recommender import build_recommendations  # noqa: E402
from core.schema import FEATURES, FEATURE_BY_KEY, MODEL_SPECS, model_label, unit_label  # noqa: E402


APP_TITLE_EN = (
    "Agricultural Spray Decision Support System for Unmanned Aerial Spraying Systems "
    "(Agri-SprayDSS-UASS)"
)
APP_TITLE_ZH = "无人机航空施药系统农业喷雾决策支持系统（Agri-SprayDSS-UASS）"
HIDDEN_EXPLANATION_FEATURES = {"distance"}
TRANSFERABILITY_NOTE = (
    "Further data supplementation, iterative updating, and validation across different "
    "years, sites, and UASS platforms are still needed before this platform can be "
    "regarded as a generally transferable prediction tool."
)
BACKGROUND_IMAGE = ROOT / "app" / "assets" / "platform_background.jpg"
INSTITUTE_LOGO = ROOT / "app" / "assets" / "institute_logo.png"


TEXT = {
    "en": {
        "page_title": APP_TITLE_EN,
        "page_subtitle": "UASS spray drift prediction, SHAP explanation, and parameter guidance",
        "language": "Language",
        "droplet_size": "Droplet size",
        "input_panel": "Operation Parameters",
        "prediction": "Predicted drift rate",
        "model": "Model",
        "shap_force": "Current-input SHAP force plot",
        "download_shap": "Download SHAP force plot (PNG, 300 dpi)",
        "prepare_shap_download": "Generate SHAP force plot download",
        "run_shap": "Run SHAP explanation",
        "shap_pending": "SHAP is not computed on first load. Click Run SHAP explanation to generate the force plot, feature table, and SHAP chart.",
        "shap_not_run": "Not run",
        "base_value": "Base value",
        "contrib_table": "Feature contribution table",
        "recommendations": "Parameter recommendations",
        "recommendation_rule_title": "Single-threshold drift-reduction rule",
        "recommendation_rule_body": (
            "The platform selects the threshold set for the current droplet-size model, "
            "identifies parameters beyond the drift-risk threshold, and provides the "
            "threshold explanation, risk direction, and adjustment advice."
        ),
        "current_model": "Current model",
        "risk_parameters": "Risk parameters",
        "current_value": "Current value",
        "threshold": "Threshold",
        "risk_direction": "Risk direction",
        "positive": "Positive SHAP values increase the current prediction; negative values reduce it.",
        "distance_hidden_note": (
            "Downwind distance is used in model prediction and SHAP computation, but hidden "
            "from the force plot and feature list."
        ),
        "env_note_title": "Interpretation of SHAP force plots for temperature and relative humidity",
        "env_note_body": (
            "It should be noted that the effects of temperature and relative humidity on "
            "ground-measured drift deposition should be interpreted in relation to the response "
            "variable used in this study. When temperature exceeds the threshold, higher "
            "temperature accelerates the evaporation of fine droplets, rapidly reducing droplet "
            "diameter and mass. As a result, some droplets may evaporate or become very small "
            "residual particles before reaching the ground samplers, leading to a reduction in "
            "measured ground deposition, although the airborne transport risk may still increase. "
            "In contrast, when relative humidity exceeds the threshold, higher humidity suppresses "
            "droplet evaporation, allowing droplets to retain greater size and mass during "
            "transport. These droplets can therefore continue drifting downwind and eventually "
            "deposit on the ground, where they are captured by the samplers. Thus, under the "
            "ground-deposition-based observation system used in this study, higher relative "
            "humidity may appear to promote increased drift deposition."
        ),
        "run": "Run prediction",
        "range_warning": "Input range warning",
        "medium_temp_warning": (
            "Medium droplet model: temperature is outside the empirical 12-23 ℃ range. "
            "Treat the result as extrapolation."
        ),
        "distance_warning": (
            "Distance is above the 1-50 m recommended planning range. Keep this as an "
            "extrapolation/caution scenario unless it matches your validated dataset."
        ),
        "feature": "Feature",
        "value": "Value",
        "unit": "Unit",
        "shap": "SHAP contribution",
        "direction": "Direction",
    },
    "zh": {
        "page_title": APP_TITLE_ZH,
        "page_subtitle": "无人机喷雾飘移预测、SHAP 解释与参数建议",
        "language": "语言",
        "droplet_size": "雾滴粒径类型",
        "input_panel": "作业参数",
        "prediction": "预测喷雾飘移率",
        "model": "模型",
        "shap_force": "当前输入的 SHAP 力图",
        "download_shap": "下载 SHAP 力图（PNG，300 dpi）",
        "prepare_shap_download": "生成 SHAP 力图下载文件",
        "run_shap": "运行 SHAP 解释",
        "shap_pending": "首次打开不计算 SHAP。点击运行 SHAP 解释后，再生成力图、特征贡献表和 SHAP 图。",
        "shap_not_run": "未运行",
        "base_value": "Base value",
        "contrib_table": "特征贡献表",
        "recommendations": "参数优化建议",
        "recommendation_rule_title": "单阈值减飘规则",
        "recommendation_rule_body": (
            "平台根据当前雾滴粒径模型调用对应阈值规则，自动识别超过或低于风险阈值的参数，"
            "并输出阈值解释、风险影响方向和具体减飘调整建议。"
        ),
        "current_model": "当前模型",
        "risk_parameters": "风险参数",
        "current_value": "当前值",
        "threshold": "阈值",
        "risk_direction": "影响方向",
        "positive": "正向 SHAP 值会提高当前预测值，负向 SHAP 值会降低当前预测值。",
        "distance_hidden_note": "下风向距离参与模型预测和 SHAP 运算，但不在力图和特征列表中展示。",
        "env_note_title": "温度和相对湿度 SHAP 力图解释",
        "env_note_body": (
            "需要注意的是，温度和相对湿度对地面飘移沉积的影响应结合采样指标进行反向理解。"
            "当温度高于阈值时，较高温度会加速细雾滴蒸发，使雾滴粒径和质量迅速减小，"
            "部分雾滴可能在到达地面采样器之前已经蒸发或转化为更难被地面采集的细小残留颗粒。"
            "因此，虽然高温可能增强空气输运风险，但地面采样器记录到的沉积量可能减少。"
            "相反，当相对湿度高于阈值时，较高湿度会抑制雾滴蒸发，使雾滴在输运过程中保持较大的粒径和质量，"
            "从而能够在下风向持续飘移并最终沉降，被地面采样器截获。因此，在本研究的地面沉积观测体系下，"
            "高湿度可能表现为促进飘移沉积增加。"
        ),
        "run": "执行预测",
        "range_warning": "输入范围提示",
        "medium_temp_warning": "中等雾滴模型：温度超出 12-23 ℃ 经验范围，请将结果视为外推。",
        "distance_warning": "距离超出 1-50 m 推荐规划范围；除非该范围已在数据集中验证，否则请按外推场景谨慎使用。",
        "feature": "特征",
        "value": "取值",
        "unit": "单位",
        "shap": "SHAP 贡献",
        "direction": "作用方向",
    },
}


def main() -> None:
    st.set_page_config(
        page_title="Agri-SprayDSS-UASS",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_tech_background()

    with st.sidebar:
        language_choice = st.radio(
            "Language / 语言",
            options=["English", "中文"],
            index=0,
            horizontal=True,
        )
        language = "zh" if language_choice == "中文" else "en"
        t = TEXT[language]

        st.divider()
        st.subheader(t["input_panel"])
        model_key = _render_model_selector(language)
        input_values = _render_inputs(language)
        st.button(t["run"], type="primary", width="stretch")

    t = TEXT[language]
    st.markdown(
        (
            '<div class="app-hero-header">'
            f'{_institute_logo_html()}'
            '<div class="app-title-group">'
            f'<h1 class="app-title-main">{html.escape(t["page_title"])}</h1>'
            '<div class="app-subtitle" '
            'style="color:#06323d;font-size:1.14rem;line-height:1.42;'
            'font-weight:650;margin:0.15rem 0 0 0;'
            'text-shadow:0 1px 0 rgba(255,255,255,0.62);">'
            f'{html.escape(t["page_subtitle"])}</div>'
            '</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    warnings = _validate_inputs(model_key, input_values, language)
    if warnings:
        for warning in warnings:
            st.warning(warning)

    prediction_result = _cached_predict_value(model_key, _prediction_cache_key(input_values))
    recommendations = build_recommendations(
        model_key=model_key,
        values=input_values,
        contribution_table=pd.DataFrame(),
    )

    prediction_cache_key = _prediction_cache_key(input_values)
    shap_state_key = _shap_state_key(model_key, prediction_cache_key)
    shap_requested = bool(st.session_state.get(shap_state_key, False))
    shap_result = (
        _cached_predict(model_key, prediction_cache_key)
        if shap_requested
        else None
    )

    metric_cols = st.columns([1, 1, 1])
    metric_cols[0].metric(t["prediction"], f"{prediction_result.prediction:.4f} %")
    metric_cols[1].metric(t["model"], model_label(model_key, language))
    metric_cols[2].metric(
        t["base_value"],
        f"{shap_result.expected_value:.4f}" if shap_result is not None else t["shap_not_run"],
    )

    st.markdown(
        (
            '<div class="shap-note-block">'
            f'<div class="shap-note">{html.escape(t["positive"])}</div>'
            f'<div class="shap-note">{html.escape(t["distance_hidden_note"])}</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    st.subheader(t["shap_force"])
    if shap_result is None:
        st.info(t["shap_pending"])
        st.button(
            t["run_shap"],
            type="primary",
            width="stretch",
            on_click=_request_shap,
            args=(shap_state_key,),
        )
    else:
        _render_force_plot(shap_result, language, model_key)

    st.subheader(t["recommendations"])
    _render_recommendations(recommendations, language, model_key)

    if shap_result is not None:
        table_col, chart_col = st.columns([1, 1], gap="medium")
        with table_col:
            st.subheader(t["contrib_table"])
            st.dataframe(
                _display_contribution_table(shap_result.contribution_table, language),
                hide_index=True,
                width="stretch",
            )
        with chart_col:
            st.subheader("SHAP")
            _render_shap_bar(shap_result.contribution_table, language)

    st.markdown('<div class="bottom-section-spacer"></div>', unsafe_allow_html=True)
    _render_environment_note(language)
    _render_transferability_footer()


def _inject_tech_background() -> None:
    background_uri = _background_data_uri()
    st.markdown(
        """
        <style>
        @keyframes scanLine {
            0% { transform: translateX(-140%) skewX(-18deg); opacity: 0; }
            18% { opacity: 0.7; }
            55% { opacity: 0.18; }
            100% { transform: translateX(140%) skewX(-18deg); opacity: 0; }
        }
        .stApp {
            background:
                linear-gradient(90deg, rgba(7, 24, 34, 0.24) 0%, rgba(255, 255, 255, 0.06) 48%, rgba(7, 24, 34, 0.20) 100%),
                linear-gradient(180deg, rgba(255, 255, 255, 0.06) 0%, rgba(232, 249, 252, 0.22) 100%),
                url("__BACKGROUND_URI__");
            background-size: cover;
            background-position: center top;
            background-attachment: fixed;
            color: #10201d;
        }
        .stApp::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            background: radial-gradient(circle at 18% 12%, rgba(255, 255, 255, 0.26), transparent 30%);
            z-index: 0;
        }
        [data-testid="stAppViewContainer"] > .main,
        [data-testid="stSidebar"],
        [data-testid="stHeader"] {
            position: relative;
            z-index: 1;
        }
        [data-testid="stHeader"] {
            background: transparent;
        }
        [data-testid="stAppViewContainer"] .main .block-container {
            background: rgba(246, 252, 253, 0.82);
            border: 1px solid rgba(169, 224, 232, 0.64);
            border-radius: 10px;
            box-shadow: 0 24px 64px rgba(2, 18, 27, 0.28);
            backdrop-filter: blur(10px);
            padding-top: 1.75rem;
            padding-bottom: 2.05rem;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(249, 254, 255, 0.94), rgba(230, 248, 249, 0.88));
            border-right: 1px solid rgba(61, 149, 157, 0.30);
            box-shadow: 12px 0 34px rgba(2, 18, 27, 0.18);
            backdrop-filter: blur(10px);
        }
        [data-testid="stSidebar"] * {
            color: #061b22;
        }
        [data-testid="stSidebar"] [role="radiogroup"] {
            background: rgba(255, 255, 255, 0.58);
            border: 1px solid rgba(58, 137, 130, 0.16);
            border-radius: 8px;
            padding: 8px 10px;
        }
        [data-testid="stSidebar"] input {
            background: rgba(255, 255, 255, 0.98);
            color: #10201d;
            border: 1px solid rgba(58, 137, 130, 0.34);
            border-radius: 7px;
            box-shadow: inset 0 1px 2px rgba(31, 71, 58, 0.05);
        }
        [data-testid="stSidebar"] [data-testid="stNumberInput"] {
            margin-bottom: 0.2rem;
        }
        [data-testid="stSidebar"] [data-testid="stNumberInput"] button {
            display: none;
        }
        [data-testid="stSidebar"] [data-testid="stNumberInput"] input {
            padding-right: 0.75rem;
            min-height: 2.42rem;
        }
        [data-testid="stSidebar"] button {
            border-radius: 8px;
        }
        [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] {
            padding-top: 0.15rem;
            padding-bottom: 0.15rem;
        }
        div[data-testid="stMetric"] {
            background: rgba(252, 254, 255, 0.93);
            border: 1px solid rgba(67, 157, 166, 0.30);
            border-radius: 8px;
            padding: 14px 16px;
            box-shadow: 0 14px 34px rgba(2, 18, 27, 0.12);
            min-height: 86px;
        }
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #031922;
        }
        .app-hero-header {
            display: grid;
            grid-template-columns: auto minmax(0, 1fr);
            align-items: start;
            gap: 16px;
            margin-bottom: 1.05rem;
        }
        .app-title-group {
            min-width: 0;
        }
        .institute-logo-panel {
            justify-self: start;
            align-self: start;
            width: clamp(88px, 7.5vw, 124px);
            aspect-ratio: 1 / 1;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 8px;
            padding: 0;
            background: transparent;
            border: 0;
            box-shadow: none;
        }
        .institute-logo {
            display: block;
            width: 100%;
            height: auto;
            max-height: 100%;
            object-fit: contain;
            filter: drop-shadow(0 10px 18px rgba(2, 18, 27, 0.18));
        }
        @media (max-width: 980px) {
            .app-hero-header {
                grid-template-columns: 1fr;
                gap: 12px;
            }
            .institute-logo-panel {
                justify-self: start;
                width: 96px;
            }
        }
        h1 {
            max-width: none;
            white-space: nowrap !important;
            overflow: hidden;
            text-overflow: clip;
            font-size: clamp(1.16rem, 1.58vw, 1.75rem) !important;
            line-height: 1.18 !important;
            font-weight: 760 !important;
            letter-spacing: 0 !important;
            color: #03222e;
            margin: 0 0 0.25rem 0 !important;
            text-shadow: 0 1px 0 rgba(255, 255, 255, 0.64);
        }
        div[data-testid="stMarkdownContainer"] h1,
        div[data-testid="stMarkdownContainer"] h1 * {
            font-size: clamp(1.16rem, 1.58vw, 1.75rem) !important;
            line-height: 1.18 !important;
            white-space: nowrap !important;
            letter-spacing: 0 !important;
            color: #03222e !important;
        }
        h1::before,
        h3::before,
        .recommendation-title::before,
        .rule-kicker::before,
        .bottom-section-title::before {
            content: "";
            display: inline-block;
            width: 0.78em;
            height: 0.78em;
            margin-right: 0.44em;
            border-radius: 50%;
            vertical-align: -0.08em;
            background:
                radial-gradient(circle at 35% 35%, #ffffff 0 13%, transparent 14%),
                linear-gradient(135deg, #00a6b8 0%, #2f8f68 52%, #f0b45a 100%);
            box-shadow:
                0 0 0 2px rgba(255, 255, 255, 0.72),
                0 0 14px rgba(0, 166, 184, 0.28);
        }
        h3 {
            color: #03313b !important;
            font-weight: 760 !important;
            letter-spacing: 0 !important;
        }
        div[data-testid="stCaptionContainer"],
        div[data-testid="stMarkdownContainer"] p {
            color: #0a2d36;
        }
        .app-subtitle {
            color: #06323d;
            font-size: 1.08rem;
            line-height: 1.42;
            font-weight: 650;
            margin: 0.15rem 0 1.0rem 0;
            text-shadow: 0 1px 0 rgba(255, 255, 255, 0.62);
        }
        .shap-note-block {
            display: grid;
            gap: 0.28rem;
            margin: 0.55rem 0 0.95rem 0;
        }
        .shap-note {
            color: #06323d;
            font-size: 1.12rem;
            line-height: 1.45;
            font-weight: 640;
        }
        .shap-note::before {
            content: "";
            display: inline-block;
            width: 0.58em;
            height: 0.58em;
            margin-right: 0.48em;
            border-radius: 50%;
            vertical-align: 0.02em;
            background: linear-gradient(135deg, #00a6b8, #2f8f68);
            box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.66);
        }
        .app-title-main {
            max-width: none !important;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: clip;
            font-size: clamp(1.16rem, 1.58vw, 1.75rem);
            line-height: 1.18;
            font-weight: 760;
            letter-spacing: 0;
            color: #03222e;
            margin: 0 0 0.25rem 0;
        }
        .stDataFrame,
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 8px;
        }
        .stDataFrame,
        .stPlotlyChart,
        iframe {
            background: rgba(252, 254, 255, 0.93);
            border: 1px solid rgba(67, 157, 166, 0.28);
            border-radius: 8px;
            box-shadow: 0 14px 34px rgba(2, 18, 27, 0.10);
        }
        .stPlotlyChart {
            padding: 8px;
        }
        .recommendation-rule,
        .recommendation-card,
        .note-card {
            border: 1px solid rgba(91, 174, 168, 0.24);
            border-radius: 8px;
            background: rgba(252, 254, 255, 0.94);
            color: #041b23;
            padding: 17px 19px;
            margin: 0 0 14px 0;
            box-shadow: 0 14px 34px rgba(2, 18, 27, 0.12);
        }
        .recommendations-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 16px;
            align-items: stretch;
            margin-bottom: 1.05rem;
        }
        .recommendations-grid .recommendation-rule {
            grid-column: 1 / -1;
        }
        .recommendations-grid .recommendation-card {
            height: 100%;
            margin: 0;
        }
        @media (max-width: 900px) {
            .recommendations-grid {
                grid-template-columns: 1fr;
            }
        }
        .recommendation-rule {
            border-left: 4px solid #2f8f68;
        }
        .recommendation-card {
            border-left: 4px solid #d6902c;
        }
        .recommendation-title {
            font-weight: 700;
            font-size: 1.02rem;
            margin-bottom: 9px;
            color: #03313b;
        }
        .recommendation-detail {
            line-height: 1.58;
            font-size: 0.96rem;
            color: #0a2d36;
        }
        .recommendation-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin: 10px 0 12px 0;
        }
        .recommendation-pill {
            display: inline-flex;
            align-items: center;
            white-space: normal;
            border-radius: 999px;
            padding: 3px 9px;
            font-size: 0.78rem;
            font-weight: 650;
            background: rgba(31, 109, 76, 0.08);
            color: #164433;
            border: 1px solid rgba(31, 109, 76, 0.16);
        }
        .note-card .recommendation-detail {
            max-height: none;
            overflow-y: visible;
            padding-right: 4px;
        }
        .environment-note-card {
            margin-top: 1.05rem;
            margin-bottom: 0;
        }
        .transferability-footer {
            margin-top: 1.0rem;
            padding: 13px 16px;
            border-radius: 8px;
            border: 1px solid rgba(67, 157, 166, 0.26);
            background: rgba(252, 254, 255, 0.88);
            color: #06323d;
            font-size: 1.0rem;
            line-height: 1.52;
            font-weight: 640;
            box-shadow: 0 12px 30px rgba(2, 18, 27, 0.10);
        }
        .transferability-footer::before {
            content: "";
            display: inline-block;
            width: 0.62em;
            height: 0.62em;
            margin-right: 0.50em;
            border-radius: 50%;
            vertical-align: 0.02em;
            background: linear-gradient(135deg, #00a6b8, #2f8f68);
            box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.66);
        }
        .bottom-section-spacer {
            height: 1.65rem;
        }
        .bottom-section-title {
            font-weight: 760;
            font-size: 1.22rem;
            line-height: 1.25;
            margin: 0.2rem 0 0.65rem 0;
            color: #03313b;
        }
        .rule-kicker {
            font-size: 0.78rem;
            font-weight: 750;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #075e68;
            margin-bottom: 5px;
        }
        div[data-testid="stVerticalBlock"] {
            gap: 0.88rem;
        }
        .element-container {
            margin-bottom: 0.42rem;
        }
        h1, h2, h3 {
            margin-bottom: 0.52rem;
        }
</style>
        """
        .replace("__BACKGROUND_URI__", background_uri),
        unsafe_allow_html=True,
    )


def _background_data_uri() -> str:
    return _asset_data_uri(str(BACKGROUND_IMAGE), _image_mime_type(BACKGROUND_IMAGE))


def _institute_logo_html() -> str:
    logo_uri = _asset_data_uri(str(INSTITUTE_LOGO), "image/png")
    if not logo_uri:
        return ""
    return (
        '<div class="institute-logo-panel">'
        f'<img class="institute-logo" src="{logo_uri}" alt="Institute of Plant Protection logo">'
        '</div>'
    )


@st.cache_data(show_spinner=False)
def _asset_data_uri(path: str, mime: str) -> str:
    asset_path = Path(path)
    if not asset_path.exists():
        return ""
    data = base64.b64encode(asset_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"



def _image_mime_type(path: Path) -> str:
    return "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"


@st.cache_data(show_spinner=False)
def _shap_js() -> str:
    import shap

    return shap.getjs()


def _prediction_cache_key(values: dict[str, float]) -> tuple[tuple[str, float], ...]:
    return tuple((feature.key, float(values[feature.key])) for feature in FEATURES)


@st.cache_data(show_spinner=False)
def _cached_predict_value(model_key: str, values_items: tuple[tuple[str, float], ...]):
    return predict_value(model_key, dict(values_items))


@st.cache_data(show_spinner=False)
def _cached_predict(model_key: str, values_items: tuple[tuple[str, float], ...]):
    return predict(model_key, dict(values_items))


def _shap_state_key(model_key: str, values_items: tuple[tuple[str, float], ...]) -> str:
    input_signature = ",".join(f"{key}={value:.6f}" for key, value in values_items)
    return f"shap_requested::{model_key}::{input_signature}"


def _request_shap(state_key: str) -> None:
    st.session_state[state_key] = True


def _shap_png_state_key(result, language: str, model_key: str) -> str:
    input_signature = ",".join(f"{float(value):.6f}" for value in result.input_df.iloc[0].to_list())
    return f"shap_png::{model_key}::{language}::{input_signature}"


def _render_model_selector(language: str) -> str:
    labels = {key: model_label(key, language) for key in MODEL_SPECS}
    selected_label = st.radio(
        TEXT[language]["droplet_size"],
        options=list(labels.values()),
        index=0,
        horizontal=False,
    )
    return next(key for key, label in labels.items() if label == selected_label)


def _render_inputs(language: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for feature in FEATURES:
        min_value, max_value = feature.recommended_range
        input_max_value = max(float(max_value), float(feature.default))
        label = _input_label(feature.key, language)
        if feature.key == "distance":
            values[feature.key] = st.slider(
                label,
                min_value=float(min_value),
                max_value=input_max_value,
                value=float(feature.default),
                step=float(feature.step),
                format=feature.display_format,
            )
        else:
            values[feature.key] = st.number_input(
                label,
                min_value=float(min_value),
                max_value=input_max_value,
                value=float(feature.default),
                step=float(feature.step),
                format=feature.display_format,
            )
    return values


def _input_label(feature_key: str, language: str) -> str:
    feature = FEATURE_BY_KEY[feature_key]
    unit = unit_label(feature_key, language)
    if language == "zh":
        return f"{feature.label_zh} / {feature.label_en} ({unit})"
    return f"{feature.label_en} ({unit})"


def _validate_inputs(model_key: str, values: dict[str, float], language: str) -> list[str]:
    t = TEXT[language]
    warnings: list[str] = []
    spec = MODEL_SPECS[model_key]

    empirical = spec.temperature_empirical_range
    if model_key == "medium" and empirical is not None:
        temperature = values["temperature"]
        if temperature < empirical[0] or temperature > empirical[1]:
            warnings.append(t["medium_temp_warning"])

    if values["distance"] > 50:
        warnings.append(t["distance_warning"])

    return warnings


def _render_force_plot(result, language: str, model_key: str) -> None:
    force_plot = make_force_plot(
        result,
        language=language,
        exclude_feature_keys=HIDDEN_EXPLANATION_FEATURES,
    )
    force_html = (
        f"<head>{_shap_js()}</head>"
        "<body style='margin:0; zoom:1.08; overflow:hidden;'>"
        f"{force_plot.html()}"
        "</body>"
    )
    components.html(force_html, height=140, scrolling=False)

    png_state_key = _shap_png_state_key(result, language, model_key)
    if st.button(TEXT[language]["prepare_shap_download"], width="stretch"):
        st.session_state[png_state_key] = make_force_plot_png(
            result,
            language=language,
            exclude_feature_keys=HIDDEN_EXPLANATION_FEATURES,
            dpi=300,
        )

    if png_state_key in st.session_state:
        st.download_button(
            TEXT[language]["download_shap"],
            data=st.session_state[png_state_key],
            file_name=f"shap_force_plot_{model_key}_300dpi.png",
            mime="image/png",
            width="stretch",
        )


def _display_contribution_table(table: pd.DataFrame, language: str) -> pd.DataFrame:
    t = TEXT[language]
    display = _visible_contribution_table(table)
    if language == "zh":
        display["feature"] = display["feature_key"].map(
            lambda key: FEATURE_BY_KEY[key].label_zh
        )
        display["direction"] = display["shap_value"].map(
            lambda value: "增加飘移预测值" if value > 0 else "降低飘移预测值"
        )

    display["value"] = display.apply(
        lambda row: _format_feature_value(row["feature_key"], row["value"]),
        axis=1,
    )
    display["unit"] = display["feature_key"].map(lambda key: unit_label(key, language))
    display["shap_value"] = display["shap_value"].map(lambda value: f"{value:.6f}")

    return display.rename(
        columns={
            "feature": t["feature"],
            "value": t["value"],
            "unit": t["unit"],
            "shap_value": t["shap"],
            "direction": t["direction"],
        }
    )[[t["feature"], t["value"], t["unit"], t["shap"], t["direction"]]]


def _visible_contribution_table(table: pd.DataFrame) -> pd.DataFrame:
    return (
        table[~table["feature_key"].isin(HIDDEN_EXPLANATION_FEATURES)]
        .copy()
        .reset_index(drop=True)
    )


def _format_feature_value(feature_key: str, value: float) -> str:
    feature = FEATURE_BY_KEY[feature_key]
    if feature.step.is_integer():
        return f"{value:.0f}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _render_recommendations(recommendations, language: str, model_key: str) -> None:
    t = TEXT[language]
    risk_count = sum(1 for rec in recommendations if rec.feature_key != "overall")
    cards = [
        (
            '<div class="recommendation-rule">'
            f'<div class="rule-kicker">{html.escape(t["recommendation_rule_title"])}</div>'
            '<div class="recommendation-meta">'
            f'<span class="recommendation-pill">{html.escape(t["current_model"])}: '
            f'{html.escape(model_label(model_key, language))}</span>'
            f'<span class="recommendation-pill">{html.escape(t["risk_parameters"])}: {risk_count}</span>'
            '</div>'
            f'<div class="recommendation-detail">{html.escape(t["recommendation_rule_body"])}</div>'
            '</div>'
        )
    ]
    for rec in recommendations:
        title = rec.title_zh if language == "zh" else rec.title_en
        detail = rec.detail_zh if language == "zh" else rec.detail_en
        direction = rec.direction_zh if language == "zh" else rec.direction_en
        if rec.threshold is None or rec.current_value is None:
            meta_html = ""
        else:
            value_text = _format_feature_value(rec.feature_key, rec.current_value)
            threshold_text = f"{rec.threshold:.2f}"
            display_unit = unit_label(rec.feature_key, language)
            meta_html = (
                '<div class="recommendation-meta">'
                f'<span class="recommendation-pill">{html.escape(t["current_value"])}: '
                f'{html.escape(value_text)} {html.escape(display_unit)}</span>'
                f'<span class="recommendation-pill">{html.escape(t["threshold"])}: '
                f'{html.escape(rec.operator)} {html.escape(threshold_text)} {html.escape(display_unit)}</span>'
                f'<span class="recommendation-pill">{html.escape(t["risk_direction"])}: '
                f'{html.escape(direction)}</span>'
                '</div>'
            )
        cards.append(
            '<div class="recommendation-card">'
            f'<div class="recommendation-title">{rec.priority}. {html.escape(title)}</div>'
            f'{meta_html}'
            f'<div class="recommendation-detail">{html.escape(detail)}</div>'
            '</div>'
        )
    st.markdown(
        f'<div class="recommendations-grid">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


def _render_environment_note(language: str) -> None:
    t = TEXT[language]
    st.markdown(
        (
            '<div class="note-card environment-note-card">'
            f'<div class="recommendation-title">{html.escape(t["env_note_title"])}</div>'
            f'<div class="recommendation-detail">{html.escape(t["env_note_body"])}</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )


def _render_transferability_footer() -> None:
    st.markdown(
        f'<div class="transferability-footer">{html.escape(TRANSFERABILITY_NOTE)}</div>',
        unsafe_allow_html=True,
    )


def _render_shap_bar(table: pd.DataFrame, language: str) -> None:
    display = _visible_contribution_table(table)
    if language == "zh":
        display["feature"] = display["feature_key"].map(
            lambda key: FEATURE_BY_KEY[key].label_zh
        )

    fig = px.bar(
        display.sort_values("shap_value", ascending=True),
        x="shap_value",
        y="feature",
        orientation="h",
        labels={"shap_value": "SHAP", "feature": ""},
    )
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    st.plotly_chart(fig)


if __name__ == "__main__":
    main()

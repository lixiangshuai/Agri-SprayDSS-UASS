<p align="center">
  <img src="app/assets/institute_logo.png" alt="Institute logo" width="86">
</p>

<h1 align="center">Agricultural Spray Decision Support System for Unmanned Aerial Spraying Systems</h1>

<p align="center">
  <strong>Agri-SprayDSS-UASS</strong><br>
  Interpretable UASS spray-drift prediction, SHAP explanation, and field parameter guidance
</p>

<p align="center">
  <img src="app/assets/readme_badge_app.svg" alt="Streamlit app">
  <img src="app/assets/readme_badge_model.svg" alt="XGBoost models">
  <img src="app/assets/readme_badge_shap.svg" alt="On-demand SHAP">
  <img src="app/assets/readme_badge_guidance.svg" alt="Field guidance">
</p>

**Agri-SprayDSS-UASS** is a lightweight, interpretable decision-support platform for unmanned aerial spraying systems (UASS). It converts validated spray-drift prediction models into a field-facing web tool that can estimate drift deposition risk, explain the driving factors behind each prediction, and provide practical parameter-adjustment guidance for drift reduction.

The platform is designed for deployment rather than model training. At runtime, it follows a clean principle: **load trained models, predict the current input, explain the current input on demand, and generate one set of recommendations for the current operating scenario**.

## Highlights

- **Two-model deployment**: automatically loads the fine-droplet or medium-droplet XGBoost model according to the selected droplet-size type.
- **Fast first screen**: displays prediction and parameter recommendations immediately, while SHAP explanation is computed only after clicking **Run SHAP explanation**.
- **Current-input explainability**: SHAP values are calculated only for the user-provided parameter set, without random samples or test-set examples.
- **Prediction-faithful SHAP workflow**: downwind distance is retained in model prediction and SHAP computation, but hidden from the displayed force plot and feature list to keep the interpretation focused on actionable factors.
- **Publication-friendly export**: SHAP force plots can be generated and downloaded as 300 dpi PNG files on demand.
- **Decision-oriented recommendations**: droplet-size-specific threshold rules identify risk parameters and translate them into concise field-operation guidance.

## System Philosophy

Agri-SprayDSS-UASS separates offline model development from online decision support.

```text
Offline stage
  Model training -> model validation -> export trained XGBoost models as .ubj

Online platform stage
  Load .ubj model -> predict current input -> optional SHAP explanation -> threshold-based guidance
```

This architecture keeps the deployed platform stable, transparent, and easy to maintain. The web application does not retrain models, resample test data, or tune hyperparameters during operation.

## Project Structure

```text
Agri-SprayDSS-UASS/
  app/
    streamlit_app.py
    assets/
      institute_logo.png
      platform_background.jpg
      readme_badge_*.svg
  core/
    predictor.py
    recommender.py
    schema.py
  models/
    fine_droplet_xgb.ubj
    medium_droplet_xgb.ubj
    drift_xgb_metadata.json
  requirements.txt
  README.md
```

## Runtime Workflow

```text
Current operating and meteorological parameters
        |
        v
Select droplet-size model
        |
        v
Load trained XGBoost .ubj model
        |
        v
Predict spray drift deposition
        |
        +--> Generate threshold-based parameter recommendations
        |
        +--> Run SHAP explanation on demand
                  |
                  v
          Force plot + contribution table + SHAP bar chart + 300 dpi PNG export
```

## Model Strategy

- Fine droplet size uses `models/fine_droplet_xgb.ubj`.
- Medium droplet size uses `models/medium_droplet_xgb.ubj`.
- Droplet size is a model selector, not a numeric model feature.
- Models are loaded with `xgboost.XGBRegressor.load_model`.
- Lightweight prediction uses a prediction-only path, so SHAP is not imported or executed during the first page load.
- SHAP is computed only after user interaction and only for the current single-row input.
- SHAP values are obtained from XGBoost native Tree SHAP contribution output (`pred_contribs=True`).
- Force plots use default SHAP colors and unmodified contribution values.

## Recommendation Logic

The recommendation module applies droplet-size-specific single-threshold drift-reduction rules. For each current input, the platform reports:

- selected droplet-size model,
- risk parameters crossing model-specific thresholds,
- current value and threshold value,
- risk direction,
- concise drift-reduction advice.

For controllable operation parameters, the platform gives direct adjustment suggestions such as lowering operating altitude, lowering operating speed, or adjusting flight-route direction. For weather-related parameters, including wind speed, temperature, and relative humidity, it recommends waiting for a suitable weather window or using a larger droplet size when appropriate.

## Input Parameters

Numeric operation inputs accept values to two decimal places. Downwind distance uses integer steps.

| Input | Unit | Recommended range | Notes |
| --- | ---: | ---: | --- |
| Wind speed | m/s | 0-7 | Fine threshold: 2.20 m/s. Medium threshold: 3.78 m/s. |
| Wind deviation angle | ° | 0-90 | Fine threshold: not lower than 20.59°. Medium threshold: not lower than 20.46°. |
| Temperature | ℃ | 10-40 | Medium droplet model warns outside 12-23 ℃. |
| Relative humidity | % | 20-80 | Fine threshold: higher than 55.51%. Medium threshold: higher than 54.05%. |
| Operating altitude | m | 2-6 | Fine threshold: below 2.92 m. Medium threshold: below 3.34 m. |
| Operating speed | m/s | 3-8 | Fine threshold: below 4.65 m/s. Medium threshold: below 5.09 m/s. |
| Droplet size | μm | Fine or medium | Selects which trained `.ubj` model is loaded according to the fine- or medium-droplet size category. |
| Downwind distance | m | 1-100 | Default is 1 m; used in prediction and SHAP computation but hidden from displayed explanation views. |

## Interpretation Note

Temperature and relative humidity should be interpreted in relation to the ground-deposition response variable used by the study. Higher temperature may accelerate droplet evaporation and reduce measured ground deposition, even though airborne transport risk can still increase. Higher relative humidity may suppress evaporation, allowing droplets to retain size and mass during transport and remain detectable by downwind ground samplers.

## Run Locally

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app/streamlit_app.py
```

Then open:

```text
http://127.0.0.1:8501/
```

## Quick Core Check

Prediction-only check, without running SHAP:

```powershell
python -c "from core.predictor import predict_value; values={'operating_altitude':3.0,'operating_speed':8.0,'wind_speed':2.0,'wind_deviation':30.0,'temperature':25.0,'relative_humidity':60.0,'distance':1.0}; print(predict_value('fine', values).prediction); print(predict_value('medium', values).prediction)"
```

Full prediction with SHAP:

```powershell
python -c "from core.predictor import predict; values={'operating_altitude':3.0,'operating_speed':8.0,'wind_speed':2.0,'wind_deviation':30.0,'temperature':25.0,'relative_humidity':60.0,'distance':1.0}; r=predict('fine', values); print(r.prediction); print(r.expected_value); print(r.contribution_table.shape)"
```

## Transferability Statement

Further data supplementation, iterative updating, and validation across different years, sites, and UASS platforms are still needed before this platform can be regarded as a generally transferable prediction tool.
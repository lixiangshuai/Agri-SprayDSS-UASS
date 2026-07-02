# Agricultural Spray Decision Support System for Unmanned Aerial Spraying Systems (Agri-SprayDSS-UASS)

Agri-SprayDSS-UASS is a lightweight Streamlit platform for UASS spray drift prediction. The platform does not train models at runtime. It loads trained XGBoost `.ubj` models, predicts the current user input, explains that same input with SHAP, and generates drift-reduction recommendations from model-specific single-threshold rules.

## Project Structure

```text
Agri-SprayDSS-UASS/
  app/
    streamlit_app.py
    assets/
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

## Model Strategy

- Fine droplet size uses `models/fine_droplet_xgb.ubj`.
- Medium droplet size uses `models/medium_droplet_xgb.ubj`.
- Droplet size is a model selector, not a model input feature.
- The runtime uses `xgboost.XGBRegressor.load_model`.
- SHAP is computed for the current single-row input only, using XGBoost's native Tree SHAP contribution output (`pred_contribs=True`).
- SHAP force plots use the default SHAP colors and raw SHAP values, and can be downloaded as 300 dpi PNG files.
- Downwind distance remains in the model input and SHAP calculation, but is hidden from the force plot, feature contribution list, and recommendation list.
- Parameter recommendations use model-specific single-threshold drift-reduction rules. Fine and medium droplet models have separate thresholds for altitude, speed, wind speed, wind deviation angle, temperature, and relative humidity.
- Temperature and relative humidity guidance follows the evaporation mechanism: higher temperature promotes droplet evaporation, while higher relative humidity suppresses it.

## Run

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app/streamlit_app.py
```

## Quick Core Check

```powershell
python -c "from core.predictor import predict; values={'operating_altitude':3.0,'operating_speed':4.5,'wind_speed':2.0,'wind_deviation':30.0,'temperature':25.0,'relative_humidity':60.0,'distance':1.0}; print(predict('fine', values).prediction); print(predict('medium', values).prediction)"
```

## Input Ranges

Numeric operation inputs accept values to two decimal places. Downwind distance uses integer steps.

| Input | Unit | Recommended range | Notes |
| --- | ---: | ---: | --- |
| Wind speed | m/s | 0-7 | Fine threshold: 2.20 m/s. Medium threshold: 3.78 m/s. |
| Wind deviation / 风向偏角 | ° | 0-90 | Fine threshold: not lower than 20.59°. Medium threshold: not lower than 20.46°. |
| Temperature | ℃ | 10-40 | Medium droplet model warns outside 12-23℃. |
| Relative humidity | % | 20-80 | Fine threshold: higher than 55.51%. Medium threshold: higher than 54.05%. |
| Operating altitude | m | 2-6 | Fine threshold: <2.92 m. Medium threshold: <3.34 m. |
| Operating speed | m/s | 3-8 | Fine threshold: <4.65 m/s. Medium threshold: <5.09 m/s. |
| Droplet size | selector | Fine or medium | Selects which model is loaded. |
| Downwind distance | m | 1-100 | Default is 1 m; above 50 m is shown as a caution range. Hidden from explanation views while still used for prediction. |

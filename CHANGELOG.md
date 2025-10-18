# Changelog

All notable changes to the Virtual Diabetes Triage System.

---

## [v0.2] - 2024-12-XX

### Added

- Model comparison framework (LinearRegression, Ridge, RandomForest)
- High-risk patient flag at 75th percentile threshold
- Enhanced metrics: MAE, R², 5-fold cross-validation

### Changed

- **Selected RandomForestRegressor** as production model (best RMSE)
- **Removed StandardScaler** - dataset already pre-standardized
- Simplified Docker image (removed scaler artifact)
- Simplified Docker image (removed scaler artifact)

### Fixed

- Double standardization bug (sklearn diabetes dataset already scaled)

### Metrics

**Regression Performance**

| Metric  | v0.1   | v0.2         | Delta           |
| ------- | ------ | ------------ | --------------- |
| RMSE    | 53.853 | 53.680       | -0.173 (-0.32%) |
| MAE     | -      | 43.562       | -               |
| R²      | -      | 0.4561       | -               |
| CV RMSE | -      | 58.478 ±4.90 | -               |

**High-Risk Flag (75th percentile threshold)**

| Metric    | Value           |
| --------- | --------------- |
| Precision | 0.7391 (73.91%) |
| Recall    | 0.5667 (56.67%) |
| F1-Score  | 0.6415          |
| Threshold | 179.04          |
| Flagged   | 23/89 patients  |

### Technical Details

- **Model**: RandomForestRegressor (n_estimators=100, max_depth=5)
- **Threshold**: 75th percentile of predicted scores
- **Rationale**: Removed redundant preprocessing, systematic model selection captured non-linear patterns

---

## [v0.1] - 2024-12-16

### Added

- Baseline LinearRegression model with StandardScaler
- REST API (`/health`, `/predict`) with FastAPI
- Docker containerization
- CI/CD pipeline (GitHub Actions)
- Test suite (7/7 tests passing)

### Metrics

- **RMSE**: 53.853
- **Training time**: ~0.5s
- **Model size**: 2KB
- **API response**: <100ms

### Known Issues

- Double standardization (dataset pre-scaled) → Fixed in v0.2

---

## Notes

- Dataset: sklearn diabetes (442 samples, 10 features)
- Prediction range: 50-350 (higher = greater risk)
- Reproducibility: All seeds set to 42

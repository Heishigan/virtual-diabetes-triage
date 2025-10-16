# Changelog

All notable changes to the Virtual Diabetes Triage System.

## [v0.1] - 2024-12-16

### Added

- Baseline LinearRegression model with StandardScaler preprocessing
- REST API with `/health` and `/predict` endpoints using FastAPI
- Docker container with baked-in model artifacts
- CI/CD pipeline with GitHub Actions
- Comprehensive test suite (7 tests with full coverage)
- Project documentation (README, setup instructions)

### Technical Details

- **Model**: LinearRegression (scikit-learn)
- **Features**: 10 standardized diabetes indicators (age, sex, bmi, bp, s1-s6)
- **Dataset**: sklearn diabetes dataset (442 samples, already standardized)
- **API Framework**: FastAPI with Pydantic validation
- **Containerization**: Docker with Python 3.11-slim base

### Metrics

- **RMSE**: 53.853
- **Training time**: ~0.5s
- **Model size**: ~2KB
- **Docker image size**: ?
- **API response time**: <100ms
- **Test coverage**: 100% (7/7 tests passing)

### Notes

- Features are pre-standardized in the sklearn dataset
- Prediction range: 50-350 (higher = greater disease progression risk)
- All dependencies pinned for reproducibility

### Known Limitations

- Uses proxy dataset (sklearn diabetes) instead of real EHR data
- Model is baseline - no hyperparameter tuning
- Features require pre-standardization before API call

---

## [Unreleased]

### Planned for v0.2

- Improved model algorithm (Ridge/RandomForest)
- Enhanced preprocessing pipeline
- Performance metrics comparison
- Reduced RMSE by 5%+

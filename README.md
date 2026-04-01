# MLOPS LAB05 - Experiment Tracking Labs

labs covering ML experiment tracking using three tools: Python Logging, MLflow, and Weights & Biases.

## Lab 1: Python Logging

**File:** `Logging_Labs/Starter.ipynb`

Covers Python's built-in `logging` library — the foundation before using ML-specific tools. Topics include five log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL), custom loggers, exception logging with tracebacks, file logging, and using multiple handlers (console + file).

## Lab 2: MLflow

### Lab 2a — MLflow Basics (`Mlflow_Labs/Lab1/`)

**Datasets:** Breast Cancer, California Housing

Introduces MLflow experiment tracking through multiple files:
- **starter.ipynb** — `mlflow.autolog()` vs manual logging with `log_param()`, `log_metric()`, `log_model()`, plus loading saved models
- **linear_regression.ipynb/.py** — ElasticNet with manual metric tracking (RMSE, MAE, R2). The `.py` version accepts command line args for running multiple experiments
- **serving.ipynb** — Train, log, serve model as REST API, and make HTTP predictions
- **serving.py** — Six different ways to manage pip requirements when logging models

**Results:** Breast Cancer classification — 96.50% accuracy, 97.21% F1 score

### Lab 2b — End-to-End Pipeline (`Mlflow_Labs/Lab2/`)

**Dataset:** Wine Classification (178 samples, 13 features, 3 classes)

Full ML pipeline in one notebook covering: data loading → train/val/test split → Random Forest baseline → MLflow Model Registry → promote to Production → XGBoost hyperparameter tuning with Hyperopt (10 Bayesian optimization trials) → promote best model → final evaluation → model serving as REST API.

**Results:** Baseline RF: 91.67% → Tuned XGBoost: 97.22% → Final: 100% accuracy

## Lab 3: Weights & Biases

### Lab 3a — XGBoost Tracking (`W&B/Lab1.ipynb`)

**Dataset:** Breast Cancer (binary classification)

XGBoost training with W&B integration: `wandb.init()`, `wandb.config.update()`, `wandb.log()`, confusion matrix generation, and summary metrics.

**Result:** 95.91% accuracy

### Lab 3b — CNN with Advanced Logging (`W&B/Lab2.ipynb`)

**Dataset:** Fashion MNIST (10 clothing categories, 28x28 images)

CNN training (Conv2D → MaxPool → Dropout → Dense) with per-epoch metric logging, confusion matrix, sample prediction visualization (true vs predicted labels), and model artifact saving.

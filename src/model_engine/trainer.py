"""
Model engine for AutoML Studio.

Trains multiple ML models with Optuna hyperparameter optimization
and logs all experiments to MLflow.
"""

import logging
from types import SimpleNamespace
from typing import Any

import mlflow
import numpy as np
import optuna
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

from src.utils.artifact_manager import ArtifactManager

# Prevent NumPy underflow from crashing Optuna
np.seterr(all="warn")


class ModelTrainer:

    def __init__(
        self,
        config: SimpleNamespace,
        logger: logging.Logger,
        artifact_manager: ArtifactManager,
    ) -> None:
        self.config = config
        self.logger = logger
        self.artifact_manager = artifact_manager

    def _get_model_registry(self, task_type: str) -> dict:

        self.logger.info(f"Building model registry for task: {task_type}")

        if task_type == "classification":
            registry = {
                "logistic_regression": LogisticRegression,
                "random_forest": RandomForestClassifier,
                "gradient_boosting": GradientBoostingClassifier,
                "xgboost": XGBClassifier,
                "lightgbm": LGBMClassifier,
            }
        elif task_type == "regression":
            registry = {
                "linear_regression": Ridge,
                "random_forest": RandomForestRegressor,
                "gradient_boosting": GradientBoostingRegressor,
                "xgboost": XGBRegressor,
                "lightgbm": LGBMRegressor,
            }
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

        enabled_models = self.config.models.enabled
        registry = {
            name: cls
            for name, cls in registry.items()
            if name in enabled_models
        }

        self.logger.info(f"Enabled models: {list(registry.keys())}")
        return registry

    def _get_search_space(self, trial, model_name: str) -> dict:

        hp = self.config.models.hyperparameters

        if model_name == "logistic_regression":
            return {
                "C": trial.suggest_float(
                    "C", hp.logistic_regression.C[0],
                    hp.logistic_regression.C[1], log=True,
                ),
                "max_iter": trial.suggest_int(
                    "max_iter", hp.logistic_regression.max_iter[0],
                    hp.logistic_regression.max_iter[1],
                ),
            }

        elif model_name == "random_forest":
            return {
                "n_estimators": trial.suggest_int(
                    "n_estimators", hp.random_forest.n_estimators[0],
                    hp.random_forest.n_estimators[1],
                ),
                "max_depth": trial.suggest_int(
                    "max_depth", hp.random_forest.max_depth[0],
                    hp.random_forest.max_depth[1],
                ),
                "min_samples_split": trial.suggest_int(
                    "min_samples_split", hp.random_forest.min_samples_split[0],
                    hp.random_forest.min_samples_split[1],
                ),
            }

        elif model_name == "gradient_boosting":
            return {
                "n_estimators": trial.suggest_int(
                    "n_estimators", hp.gradient_boosting.n_estimators[0],
                    hp.gradient_boosting.n_estimators[1],
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    hp.gradient_boosting.learning_rate[0],
                    hp.gradient_boosting.learning_rate[1],
                    log=True,
                ),
                "max_depth": trial.suggest_int(
                    "max_depth", hp.gradient_boosting.max_depth[0],
                    hp.gradient_boosting.max_depth[1],
                ),
            }

        elif model_name == "xgboost":
            return {
                "n_estimators": trial.suggest_int(
                    "n_estimators", hp.xgboost.n_estimators[0],
                    hp.xgboost.n_estimators[1],
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    hp.xgboost.learning_rate[0],
                    hp.xgboost.learning_rate[1],
                    log=True,
                ),
                "max_depth": trial.suggest_int(
                    "max_depth", hp.xgboost.max_depth[0],
                    hp.xgboost.max_depth[1],
                ),
            }

        elif model_name == "lightgbm":
            return {
                "n_estimators": trial.suggest_int(
                    "n_estimators", hp.lightgbm.n_estimators[0],
                    hp.lightgbm.n_estimators[1],
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    hp.lightgbm.learning_rate[0],
                    hp.lightgbm.learning_rate[1],
                    log=True,
                ),
                "max_depth": trial.suggest_int(
                    "max_depth", hp.lightgbm.max_depth[0],
                    hp.lightgbm.max_depth[1],
                ),
            }

        else:
            raise ValueError(f"No search space defined for model: {model_name}")

    def _objective(
        self,
        trial,
        model_name: str,
        model_class: Any,
        X_train,
        X_test,
        y_train,
        y_test,
        task_type: str,
    ) -> float:

        params = self._get_search_space(trial, model_name)

        model = model_class(**params)
        model.fit(X_train, y_train)

        if task_type == "classification":
            if hasattr(model, "predict_proba"):
                preds = model.predict_proba(X_test)[:, 1]
                score = roc_auc_score(y_test, preds)
            else:
                preds = model.predict(X_test)
                score = accuracy_score(y_test, preds)
        else:
            preds = model.predict(X_test)
            score = r2_score(y_test, preds)

        with mlflow.start_run(
            run_name=f"{model_name}_trial_{trial.number}",
            nested=True,
        ):
            mlflow.log_params(params)
            mlflow.log_metric("score", score)

        return score

    def _train_single_model(
        self,
        model_name: str,
        model_class: Any,
        X_train,
        X_test,
        y_train,
        y_test,
        task_type: str,
        experiment_id: str,
    ) -> tuple[Any, float]:

        self.logger.info(f"Training model: {model_name}")
        n_trials = self.config.models.n_trials

        mlflow.set_experiment(self.config.mlflow.experiment_name)

        with mlflow.start_run(run_name=model_name):

            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(
                    multivariate=True,
                    group=True,
                    seed=42,
                ),
            )

            study.optimize(
                lambda trial: self._objective(
                    trial,
                    model_name,
                    model_class,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    task_type,
                ),
                n_trials=n_trials,
                catch=(FloatingPointError,),
            )

            best_params = study.best_params
            best_score = study.best_value

            self.logger.info(f"{model_name} best score: {best_score:.4f}")
            self.logger.info(f"{model_name} best params: {best_params}")

            mlflow.log_params(best_params)
            mlflow.log_metric("best_score", best_score)

            best_model = model_class(**best_params)
            best_model.fit(X_train, y_train)

            self.artifact_manager.save_model(
                best_model,
                model_name,
                experiment_id,
            )

        return best_model, best_score

    def run(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        task_type: str,
        experiment_id: str,
    ) -> tuple[Any, str, float]:

        self.logger.info("--- Model Training Started ---")

        registry = self._get_model_registry(task_type)

        if not registry:
            raise ValueError("No models enabled in configuration.")

        best_model = None
        best_model_name = None
        best_score = -np.inf

        for model_name, model_class in registry.items():

            self.logger.info(f"Starting training for: {model_name}")

            model, score = self._train_single_model(
                model_name,
                model_class,
                X_train,
                X_test,
                y_train,
                y_test,
                task_type,
                experiment_id,
            )

            self.logger.info(f"{model_name} completed — score: {score:.6f}")

            if score > best_score:
                best_score = score
                best_model = model
                best_model_name = model_name

        if best_model is None:
            raise RuntimeError(
                "Model training completed but no best model was selected."
            )

        self.artifact_manager.save_model(
            best_model,
            "best_model",
            experiment_id,
        )

        self.logger.info(
            f"Best model: {best_model_name} | Score: {best_score:.6f}"
        )

        self.logger.info("--- Model Training Complete ---")

        return best_model, best_model_name, best_score
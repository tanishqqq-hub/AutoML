"""
Evaluation module for AutoML Studio.

Computes evaluation metrics for all trained models,
generates a leaderboard, and logs results to MLflow.
"""

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from src.utils.artifact_manager import ArtifactManager


class Evaluator:
    """
    Evaluates all trained models and generates a performance leaderboard.

    Parameters
    ----------
    config : SimpleNamespace
        Loaded configuration object.
    logger : logging.Logger
        Configured logger instance.
    artifact_manager : ArtifactManager
        Artifact persistence manager.
    """

    def __init__(
        self,
        config: SimpleNamespace,
        logger: logging.Logger,
        artifact_manager: ArtifactManager,
    ) -> None:
        self.config = config
        self.logger = logger
        self.artifact_manager = artifact_manager

    def _compute_metrics(
        self,
        model: Any,
        X_test,
        y_test,
        task_type: str,
    ) -> dict:
        """
        Compute evaluation metrics for a trained model.

        Parameters
        ----------
        model : Any
            Trained ML model.
        X_test : array-like
            Test feature matrix.
        y_test : array-like
            True labels.
        task_type : str
            "classification" or "regression".

        Returns
        -------
        dict
            Dictionary containing computed metrics.

        Raises
        ------
        ValueError
            If task_type is unsupported.
        """
        self.logger.info("Computing evaluation metrics.")

        if task_type == "classification":
            y_pred = model.predict(X_test)
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(
                    y_test, y_pred, average="weighted", zero_division=0
                ),
                "recall": recall_score(
                    y_test, y_pred, average="weighted", zero_division=0
                ),
                "f1": f1_score(
                    y_test, y_pred, average="weighted", zero_division=0
                ),
            }
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)
                n_classes = len(np.unique(y_test))

                if n_classes == 2:
                    # Binary classification — use probability of positive class
                    metrics["roc_auc"] = roc_auc_score(
                        y_test,
                        y_prob[:, 1],
                    )
                else:
                    # Multiclass — use OVR averaging
                    metrics["roc_auc"] = roc_auc_score(
                        y_test,
                        y_prob,
                        multi_class="ovr",
                        average="weighted",
                    )
            else:
                metrics["roc_auc"] = np.nan

        elif task_type == "regression":
            y_pred = model.predict(X_test)
            metrics = {
                # Measures how far predictions are from true values. Lower is better.
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                # Average magnitude of errors, direction ignored. Lower is better.
                "mae": mean_absolute_error(y_test, y_pred),
                # Proportion of variance explained by the model.
                # 1.0 = perfect, 0.0 = predicts mean, negative = worse than mean.
                "r2": r2_score(y_test, y_pred),
            }

        else:
            raise ValueError(f"Unsupported task type: {task_type}")

        self.logger.info(f"Computed metrics: {metrics}")
        return metrics

    def _log_metrics_to_mlflow(
        self, metrics: dict, model_name: str
    ) -> None:
        """
        Log evaluation metrics to MLflow.

        Parameters
        ----------
        metrics : dict
            Dictionary of computed metrics.
        model_name : str
            Name of the evaluated model.
        """
        self.logger.info(
            f"Logging evaluation metrics for model: {model_name}"
        )

        mlflow.set_tag("evaluated_model", model_name)

        for metric_name, metric_value in metrics.items():
            if metric_value is not None and metric_name != "model_name":
                mlflow.log_metric(metric_name, float(metric_value))

        self.logger.info("Metrics successfully logged to MLflow.")

    def _save_leaderboard(
        self,
        all_metrics: list[dict],
        experiment_id: str,
        primary_metric: str = None,
    ) -> Path:
        """
        Save model leaderboard sorted by primary metric.

        Parameters
        ----------
        all_metrics : list[dict]
            List of metric dictionaries for each model.
        experiment_id : str
            Current experiment identifier.
        primary_metric : str, optional
            Metric to rank by. Falls back to config if not provided.

        Returns
        -------
        Path
            Path to saved leaderboard CSV.

        Raises
        ------
        ValueError
            If primary metric is not found in computed metrics.
        """
        self.logger.info("Generating model leaderboard.")

        if primary_metric is None:
            primary_metric = self.config.evaluation.primary_metric

        df = pd.DataFrame(all_metrics)

        if primary_metric not in df.columns:
            raise ValueError(
                f"Primary metric '{primary_metric}' not found "
                f"in computed metrics."
            )

        df = df.sort_values(primary_metric, ascending=False)

        experiment_dir = (
            Path(self.config.paths.experiments_dir) / experiment_id
        )
        experiment_dir.mkdir(parents=True, exist_ok=True)

        leaderboard_path = experiment_dir / "leaderboard.csv"
        df.to_csv(leaderboard_path, index=False)

        self.logger.info(f"Leaderboard saved to: {leaderboard_path}")
        self.logger.info(f"\n{df.to_string(index=False)}")

        return leaderboard_path

    def run(
        self,
        best_model: Any,
        best_model_name: str,
        X_test,
        y_test,
        task_type: str,
        experiment_id: str,
    ) -> dict:
        """
        Evaluate all trained models and generate leaderboard.

        Parameters
        ----------
        best_model : Any
            Best trained model object.
        best_model_name : str
            Name of the best model.
        X_test : array-like
            Test feature matrix.
        y_test : array-like
            Test labels.
        task_type : str
            "classification" or "regression".
        experiment_id : str
            Current experiment identifier.

        Returns
        -------
        dict
            Metrics dictionary for the best model.

        Raises
        ------
        ValueError
            If no trained models are found for evaluation.
        """
        self.logger.info("--- Model Evaluation Started ---")

        all_metrics = []

        trained_models = self.artifact_manager.list_models(experiment_id)

        if not trained_models:
            raise ValueError(
                "No trained models found for evaluation."
            )

        for model_name in trained_models:
            model = self.artifact_manager.load_model(
                model_name, experiment_id
            )
            self.logger.info(f"Evaluating model: {model_name}")

            metrics = self._compute_metrics(
                model, X_test, y_test, task_type
            )
            metrics["model_name"] = model_name
            all_metrics.append(metrics)

            self._log_metrics_to_mlflow(metrics, model_name)

        # Select ranking metric with priority fallback
        available_metrics = [
            k for k in all_metrics[0].keys() if k != "model_name"
        ]

        metric_priority = (
            ["roc_auc", "f1", "accuracy"]
            if task_type == "classification"
            else ["r2", "rmse"]
        )

        selected_metric = next(
            (m for m in metric_priority if m in available_metrics),
            None,
        )

        if selected_metric is None:
            raise ValueError("No valid metric found for leaderboard ranking.")

        self.logger.info(f"Leaderboard ranking metric: {selected_metric}")

        # Pass selected_metric directly — do NOT mutate config
        leaderboard_path = self._save_leaderboard(
            all_metrics, experiment_id, selected_metric
        )
        self.logger.info(f"Leaderboard generated at: {leaderboard_path}")

        # ── Optuna-selected best model metrics ──────────────────────
        # This is the model the trainer picked based on validation score.
        # Artifact best_model.pkl corresponds to this model.
        best_metrics = next(
            m for m in all_metrics
            if m["model_name"] == best_model_name
        )

        # ── Test-set best model ───────────────────────────────────────
        # Re-select the winner purely by test set performance.
        # This may differ from Optuna's choice when the validation fold
        # was not fully representative of the test distribution.
        # We do NOT re-save best_model.pkl — this is informational only.
        test_best = max(
            all_metrics,
            key=lambda m: m.get(selected_metric, 0.0)
        )
        test_best_model_name  = test_best["model_name"]
        test_best_score       = test_best.get(selected_metric, 0.0)
        test_best_metric_name = selected_metric

        self.logger.info(
            f"Optuna selected : {best_model_name} "
            f"(test {selected_metric}={best_metrics.get(selected_metric, 'N/A'):.4f})"
        )
        self.logger.info(
            f"Test set best   : {test_best_model_name} "
            f"({selected_metric}={test_best_score:.4f})"
        )

        if test_best_model_name != best_model_name:
            self.logger.warning(
                f"Model selection discrepancy detected. "
                f"Optuna chose '{best_model_name}' by validation score, "
                f"but '{test_best_model_name}' performs better on the test set. "
                f"Consider re-running with more Optuna trials for better generalization."
            )

        # Attach test-set comparison info to best_metrics dict
        # so the pipeline can surface it to the API and frontend.
        best_metrics["_test_best_model_name"]  = test_best_model_name
        best_metrics["_test_best_score"]       = float(test_best_score)
        best_metrics["_test_best_metric_name"] = test_best_metric_name
        best_metrics["_optuna_selected"]       = best_model_name

        self.artifact_manager.save_metrics(best_metrics, experiment_id)

        self.logger.info(
            f"Best model metrics ({best_model_name}): {best_metrics}"
        )
        self.logger.info("--- Model Evaluation Complete ---")

        return best_metrics
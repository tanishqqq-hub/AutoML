"""
Background training worker for AutoML Studio.

Runs TrainingPipeline in a daemon thread so FastAPI returns
experiment_id immediately. Progress is estimated via a ticker
thread that increments the store while pipeline.run() blocks.

Why not call modules directly?
-------------------------------
TrainingPipeline.__init__ wires config, logger, and artifact_manager
into every module. Re-doing that in the worker would duplicate
that logic and break every time a module signature changes.
The correct approach is: instantiate TrainingPipeline, call run().
"""

import threading
import time
import traceback
import logging
from typing import Any

from api.job_store import experiment_store
from src.pipeline.train_pipeline import TrainingPipeline


# ---------------------------------------------------------------------------
# Log capture — mirrors pipeline logs into the experiment store
# ---------------------------------------------------------------------------

class StoreLogHandler(logging.Handler):
    """
    Routes pipeline log records into ExperimentStore so the
    /status endpoint can stream them to React.

    Parameters
    ----------
    experiment_id : str
        Target experiment log buffer.
    """

    def __init__(self, experiment_id: str) -> None:
        super().__init__(level=logging.INFO)
        self.experiment_id = experiment_id
        self.setFormatter(
            logging.Formatter("%(levelname)s — %(message)s")
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            experiment_store.append_log(
                self.experiment_id,
                self.format(record),
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Progress ticker — increments progress while pipeline.run() blocks
# ---------------------------------------------------------------------------

# Estimated seconds spent in each stage based on observed pipeline runs.
# These are used to distribute the 0→95 progress range across stages.
# The final 95→100 jump happens when run() returns successfully.
_STAGE_SCHEDULE = [
    (10,  "Ingesting data..."),
    (25,  "Validating dataset..."),
    (40,  "Preprocessing features..."),
    (85,  "Training models — this may take a while..."),
    (95,  "Evaluating results..."),
]


class ProgressTicker(threading.Thread):
    """
    Daemon thread that advances experiment progress through
    estimated stage milestones while the pipeline runs.

    Stops when stop() is called (on pipeline completion or failure).

    Parameters
    ----------
    experiment_id : str
        Store key to update.
    total_seconds : float
        Estimated total pipeline duration. Used to pace stage transitions.
    """

    def __init__(self, experiment_id: str, total_seconds: float = 60.0) -> None:
        super().__init__(daemon=True, name=f"ticker-{experiment_id}")
        self.experiment_id = experiment_id
        self.total_seconds = total_seconds
        self._stop_event = threading.Event()

    def stop(self) -> None:
        """Signal the ticker to stop on next iteration."""
        self._stop_event.set()

    def run(self) -> None:
        """
        Walk through stage milestones, sleeping proportionally
        between each one until stopped.
        """
        eid = self.experiment_id
        prev_progress = 0

        for target_progress, stage_label in _STAGE_SCHEDULE:
            if self._stop_event.is_set():
                return

            experiment_store.set_stage_label(eid, stage_label, target_progress)

            # Sleep proportional to how much of the timeline this stage covers
            stage_fraction = (target_progress - prev_progress) / 95.0
            sleep_duration = self.total_seconds * stage_fraction

            # Wake up every 0.5s to check for stop signal
            slept = 0.0
            while slept < sleep_duration:
                if self._stop_event.is_set():
                    return
                time.sleep(0.5)
                slept += 0.5

            prev_progress = target_progress


# ---------------------------------------------------------------------------
# Leaderboard loader
# ---------------------------------------------------------------------------

def _load_leaderboard(experiment_id: str, config: Any) -> list:
    """
    Read the leaderboard CSV saved by the evaluator.

    Parameters
    ----------
    experiment_id : str
    config : SimpleNamespace

    Returns
    -------
    list[dict]
        Each dict has 'model_name' and 'metrics' keys.
    """
    import pandas as pd
    from pathlib import Path

    leaderboard_path = (
        Path(config.paths.experiments_dir) / experiment_id / "leaderboard.csv"
    )

    if not leaderboard_path.exists():
        return []

    try:
        df = pd.read_csv(leaderboard_path)
        records = []
        for _, row in df.iterrows():
            metrics = {
                col: float(row[col])
                for col in df.columns
                if col != "model_name"
            }
            records.append({
                "model_name": row["model_name"],
                "metrics": metrics,
            })
        return records
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Worker entry point
# ---------------------------------------------------------------------------

def run_training_job(
    experiment_id: str,
    config: Any,
    logger: Any,
) -> None:
    """
    Run TrainingPipeline in a background thread.

    Attaches a StoreLogHandler to capture all pipeline logs,
    starts a ProgressTicker for estimated progress updates,
    then calls pipeline.run() which handles all module
    instantiation internally.

    Parameters
    ----------
    experiment_id : str
        Registered experiment ID to update in the store.
    config : SimpleNamespace
        Deep-copied runtime config for this run.
    logger : logging.Logger
        API-level logger.
    """
    # Attach log handler to root so all pipeline module loggers are captured
    store_handler = StoreLogHandler(experiment_id)
    root_logger = logging.getLogger()
    root_logger.addHandler(store_handler)

    # Start progress ticker with 60s estimate
    # (adjusts automatically — stops as soon as pipeline finishes)
    ticker = ProgressTicker(experiment_id, total_seconds=60.0)
    ticker.start()

    start_time = time.time()

    try:
        logger.info(f"Worker started for experiment: {experiment_id}")

        # TrainingPipeline handles all module instantiation internally.
        # It wires config → logger → artifact_manager → each module.
        # We override experiment_id after init so artifacts land in the
        # correct directory (store registered this ID, not pipeline's timestamp).
        pipeline = TrainingPipeline(config=config)
        pipeline.experiment_id = experiment_id

        best_metrics = pipeline.run()

        duration = round(time.time() - start_time, 2)

        # Stop ticker before writing complete state
        ticker.stop()
        ticker.join(timeout=2.0)

        leaderboard = _load_leaderboard(experiment_id, config)

        # Determine best score using priority order that matches the
        # evaluator's primary_metric config (roc_auc for classification,
        # r2 for regression, then fallback to first available metric).
        PRIORITY_METRICS = ["roc_auc", "r2", "r2_score", "accuracy", "f1"]
        best_score = 0.0
        metrics_only = {
            k: v for k, v in best_metrics.items() if k != "model_name"
        }
        for pm in PRIORITY_METRICS:
            if pm in metrics_only:
                best_score = float(metrics_only[pm])
                break
        else:
            # Fallback — take first available metric
            best_score = float(next(iter(metrics_only.values()), 0.0))

        # Extract private comparison fields added by evaluator
        # These start with _ and are not part of the public metrics
        test_best_model_name  = best_metrics.pop("_test_best_model_name",  None)
        test_best_score       = best_metrics.pop("_test_best_score",       None)
        test_best_metric_name = best_metrics.pop("_test_best_metric_name", None)
        optuna_selected       = best_metrics.pop("_optuna_selected",       None)

        result = {
            "experiment_id":    experiment_id,
            "best_model_name":  best_metrics.get("model_name", "unknown"),
            "best_score":       best_score,
            "best_metrics": {
                k: float(v)
                for k, v in best_metrics.items()
                if k != "model_name"
            },
            "leaderboard":      leaderboard,
            "task_type":        getattr(config.data, "task_type", "classification"),
            "duration_seconds": duration,
            # Model selection comparison — surfaced to frontend
            "optuna_selected":        optuna_selected or best_metrics.get("model_name"),
            "test_best_model_name":   test_best_model_name,
            "test_best_score":        float(test_best_score) if test_best_score is not None else None,
            "test_best_metric_name":  test_best_metric_name,
            "selection_mismatch":     (
                test_best_model_name is not None and
                test_best_model_name != best_metrics.get("model_name")
            ),
        }

        experiment_store.set_complete(experiment_id, result)
        logger.info(
            f"Experiment {experiment_id} complete in {duration}s — "
            f"best: {result['best_model_name']} ({result['best_score']:.4f})"
        )

    except Exception as exc:
        ticker.stop()
        error_message = str(exc)
        tb = traceback.format_exc()
        logger.error(f"Experiment {experiment_id} failed: {error_message}\n{tb}")
        experiment_store.set_failed(experiment_id, error_message)
        experiment_store.append_log(experiment_id, f"ERROR — {error_message}")

    finally:
        root_logger.removeHandler(store_handler)


# ---------------------------------------------------------------------------
# Thread launcher
# ---------------------------------------------------------------------------

def launch_training_thread(
    experiment_id: str,
    config: Any,
    logger: Any,
) -> threading.Thread:
    """
    Spawn and start a daemon thread running run_training_job.

    Parameters
    ----------
    experiment_id : str
    config : SimpleNamespace
    logger : logging.Logger

    Returns
    -------
    threading.Thread
        The started thread.
    """
    thread = threading.Thread(
        target=run_training_job,
        args=(experiment_id, config, logger),
        daemon=True,
        name=f"training-{experiment_id}",
    )
    thread.start()
    logger.info(f"Launched training thread: {thread.name}")
    return thread
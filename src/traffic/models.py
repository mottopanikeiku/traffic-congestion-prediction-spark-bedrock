"""Train and evaluate the three Spark ML classifiers required by the spec.

Models:
* Logistic Regression — linear baseline
* Decision Tree Classifier — single tree, interpretable
* Random Forest Classifier — bagged-tree ensemble (tuned to outperform
  the single tree; optionally cross-validated)

For each model we train on the training split, score the test split, and
record:
* training time
* accuracy, weighted precision, weighted recall, weighted F1
* confusion matrix (numpy)
* feature importances (where the model exposes them)
* hyperparameters actually used (after CV if applicable) — surfaced in
  the SageMaker tracking payload.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from pyspark.ml import Estimator
from pyspark.ml.classification import (
    DecisionTreeClassifier,
    LogisticRegression,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import DataFrame

from traffic.config import MODEL_CFG, MODELS_DIR, RESULTS_DIR

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    name: str
    train_time_s: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: List[List[int]] = field(default_factory=list)
    feature_importances: List[float] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "train_time_s": round(self.train_time_s, 3),
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "confusion_matrix": self.confusion_matrix,
            "feature_importances": [round(x, 4) for x in self.feature_importances],
            "parameters": self.parameters,
        }


def _evaluate(predictions: DataFrame) -> Dict[str, float]:
    metrics = {}
    for metric in ("accuracy", "weightedPrecision", "weightedRecall", "f1"):
        evaluator = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName=metric
        )
        metrics[metric] = float(evaluator.evaluate(predictions))
    return metrics


def _confusion_matrix(predictions: DataFrame, n_classes: int = 3) -> List[List[int]]:
    rows = (
        predictions.groupBy("label", "prediction")
        .count()
        .collect()
    )
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for r in rows:
        cm[int(r["label"])][int(r["prediction"])] = int(r["count"])
    return cm.tolist()


def _params_of(model) -> Dict[str, Any]:
    """Extract the hyperparameters actually used by a fitted model."""
    out: Dict[str, Any] = {}
    cls_name = type(model).__name__
    for getter in (
        "getOrDefault", "getMaxIter", "getRegParam", "getMaxDepth",
        "getMaxBins", "getNumTrees", "getSubsamplingRate",
        "getFeatureSubsetStrategy", "getElasticNetParam",
    ):
        if not hasattr(model, getter):
            continue
        try:
            if getter == "getOrDefault":
                continue
            val = getattr(model, getter)()
            out[getter[3:].lower()] = val
        except Exception:
            continue
    out["class"] = cls_name
    return out


def _train_one(
    name: str, estimator: Estimator, train: DataFrame, test: DataFrame,
    param_grid: List = None, num_folds: int = 0,
) -> ModelResult:
    logger.info(
        "Training %s%s ...",
        name, f" with {num_folds}-fold CV" if num_folds and param_grid else "",
    )
    t0 = time.time()
    if num_folds and param_grid:
        cv = CrossValidator(
            estimator=estimator,
            estimatorParamMaps=param_grid,
            evaluator=MulticlassClassificationEvaluator(
                labelCol="label", predictionCol="prediction", metricName="f1"
            ),
            numFolds=num_folds,
            parallelism=2,
            seed=MODEL_CFG.seed,
        )
        cv_model = cv.fit(train)
        model = cv_model.bestModel
    else:
        model = estimator.fit(train)
    train_time = time.time() - t0

    predictions = model.transform(test)
    metrics = _evaluate(predictions)
    cm = _confusion_matrix(predictions)

    importances: List[float] = []
    if hasattr(model, "featureImportances"):
        try:
            importances = list(model.featureImportances.toArray().tolist())
        except Exception:  # pragma: no cover
            importances = []

    # Persist the fitted Spark model.
    out_path = MODELS_DIR / name.lower().replace(" ", "_")
    model.write().overwrite().save(str(out_path))

    return ModelResult(
        name=name,
        train_time_s=train_time,
        accuracy=metrics["accuracy"],
        precision=metrics["weightedPrecision"],
        recall=metrics["weightedRecall"],
        f1=metrics["f1"],
        confusion_matrix=cm,
        feature_importances=importances,
        parameters=_params_of(model),
    )


def _build_estimators_and_grids() -> List[Tuple[str, Estimator, List]]:
    """Return (name, estimator, paramGrid) triples for the three classifiers."""
    cfg = MODEL_CFG

    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=cfg.lr_max_iter,
        regParam=cfg.lr_reg_param,
        elasticNetParam=cfg.lr_elastic_net_param,
    )
    lr_grid = (
        ParamGridBuilder()
        .addGrid(lr.regParam, [0.0, 0.01, 0.1])
        .addGrid(lr.elasticNetParam, [0.0, 0.5])
        .build()
    )

    dt = DecisionTreeClassifier(
        featuresCol="features",
        labelCol="label",
        maxDepth=cfg.dt_max_depth,
        maxBins=cfg.dt_max_bins,
        seed=cfg.seed,
    )
    dt_grid = (
        ParamGridBuilder()
        .addGrid(dt.maxDepth, [6, 8, 10, 12])
        .build()
    )

    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        numTrees=cfg.rf_num_trees,
        maxDepth=cfg.rf_max_depth,
        maxBins=cfg.rf_max_bins,
        subsamplingRate=cfg.rf_subsampling_rate,
        featureSubsetStrategy=cfg.rf_feature_subset_strategy,
        seed=cfg.seed,
    )
    rf_grid = (
        ParamGridBuilder()
        .addGrid(rf.numTrees, [60, 100])
        .addGrid(rf.maxDepth, [12, 15])
        .build()
    )

    return [
        ("LogisticRegression", lr, lr_grid),
        ("DecisionTree", dt, dt_grid),
        ("RandomForest", rf, rf_grid),
    ]


def train_all(train: DataFrame, test: DataFrame, use_cv: bool = None) -> List[ModelResult]:
    cfg = MODEL_CFG
    if use_cv is None:
        use_cv = cfg.use_cv
    folds = cfg.cv_num_folds if use_cv else 0
    results: List[ModelResult] = []
    for name, estimator, grid in _build_estimators_and_grids():
        result = _train_one(
            name, estimator, train, test, grid if folds else None, folds,
        )
        results.append(result)
        # Persist incrementally so an interrupted run still leaves partial
        # artifacts on disk.
        save_results(results)
    return results


def save_results(results: List[ModelResult], out_dir: Path = RESULTS_DIR) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = [r.to_dict() for r in results]
    target = out_dir / "model_results.json"
    with target.open("w") as fh:
        json.dump(payload, fh, indent=2)
    # Also write a human-readable summary CSV
    import csv
    with (out_dir / "model_results.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model", "accuracy", "precision", "recall", "f1", "train_time_s"])
        for r in results:
            writer.writerow([
                r.name,
                f"{r.accuracy:.4f}",
                f"{r.precision:.4f}",
                f"{r.recall:.4f}",
                f"{r.f1:.4f}",
                f"{r.train_time_s:.3f}",
            ])
    return target

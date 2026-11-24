import argparse
import errno
import hashlib
import io
import json
import logging
import math
import platform
import tarfile
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
LOG = logging.getLogger(__name__)
FAILURE = "is_failure"
LEVELS = {"info", "warn", "error"}
METRICS = ("roc_auc", "average_precision", "precision", "recall", "f1", "tp", "fp", "fn", "tn")


def load_data(source, features):
    """Read daily JSONL partitions directly from the source archive."""
    required = {"ts", "node_id", "level"} | (set(features) - {"hour"})
    frames = []
    with tarfile.open(source, "r:gz") as archive:
        partitions = sorted(
            (
                member for member in archive.getmembers()
                if member.isfile()
                and PurePosixPath(member.name).match("raw/telemetry/dt=*/telemetry.jsonl")
            ),
            key=lambda member: member.name,
        )
        if not partitions:
            raise ValueError("No raw/telemetry/dt=*/telemetry.jsonl partitions found")
        if len({member.name for member in partitions}) != len(partitions):
            raise ValueError("Duplicate partition paths in the input archive")
        for member in partitions:
            with archive.extractfile(member) as partition:
                batch = pd.read_json(
                    io.BytesIO(partition.read()), lines=True,
                    dtype={"node_id": "string"}, convert_dates=False,
                )
            missing = required - set(batch.columns)
            if batch.empty or missing:
                raise ValueError(f"Invalid partition {member.name}. Missing columns: {sorted(missing)}")
            frames.append(batch)
    data = pd.concat(frames, ignore_index=True)
    LOG.info("Loaded %d records from %d daily partitions", len(data), len(partitions))
    return data


def preprocess(data, config):
    """Validate records, prepare features and assign each node to one split."""
    features = config["features"]
    clean = data.drop_duplicates().reset_index(drop=True)
    clean["ts"] = pd.to_datetime(clean["ts"], format="ISO8601", utc=True)
    clean["node_id"] = clean["node_id"].str.strip()
    clean["level"] = clean["level"].astype("string").str.strip().str.lower()
    if clean["ts"].isna().any() or clean["node_id"].fillna("").eq("").any():
        raise ValueError("Missing timestamp or node ID")
    if not clean["level"].isin(LEVELS).all():
        raise ValueError("Unexpected telemetry level")
    clean["hour"] = clean["ts"].dt.hour
    for feature in features:
        clean[feature] = pd.to_numeric(clean[feature], errors="coerce")
    if not np.isfinite(clean[features].to_numpy(dtype=float)).all():
        raise ValueError("Features contain missing or non-finite numeric values")
    clean[FAILURE] = clean["level"].eq("error").astype("int8")

    x, y, groups = clean[features], clean[FAILURE], clean["node_id"]
    holdout = StratifiedGroupKFold(config["test_folds"], shuffle=True, random_state=config["seed"])
    validation = StratifiedGroupKFold(
        config["validation_folds"], shuffle=True, random_state=config["seed"]
    )
    rest, test = next(holdout.split(x, y, groups))
    train, val = next(validation.split(x.iloc[rest], y.iloc[rest], groups.iloc[rest]))
    clean["split"] = ""
    for split, rows in {"train": rest[train], "validation": rest[val], "test": test}.items():
        clean.loc[rows, "split"] = split
        if clean.loc[rows, FAILURE].nunique() != 2:
            raise ValueError(f"Both classes must be represented in {split}")
    if (clean["split"].eq("").any()
            or clean.groupby("node_id")["split"].nunique().ne(1).any()):
        raise RuntimeError("Invalid split assignment")

    LOG.info("Prepared %d records, removed %d duplicates, found %d failures",
             len(clean), len(data) - len(clean), clean[FAILURE].sum())
    LOG.info("Split sizes: %s", clean["split"].value_counts().to_dict())
    return clean


def train_model(train, config):
    """Fit both models using only the training split."""
    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(random_state=config["seed"], **config["logistic_regression"]),
        ),
        "xgboost": XGBClassifier(random_state=config["seed"], **config["xgboost"]),
    }
    for name, model in models.items():
        LOG.info("Training %s on %d records", name, len(train))
        model.fit(train[config["features"]], train[FAILURE])
    return models


def evaluate(y_true, scores, threshold):
    """Compute ranking and classification metrics on the held-out labels."""
    if (scores.shape != (len(y_true),) or not np.isfinite(scores).all()
            or not ((scores >= 0) & (scores <= 1)).all()):
        raise RuntimeError("Invalid model probabilities")
    predicted = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }
    if tp + tn + fp + fn != len(y_true) or not all(
        math.isfinite(metrics[key]) and 0 <= metrics[key] <= 1 for key in METRICS[:5]
    ):
        raise RuntimeError("Invalid evaluation metrics")
    return metrics


def select_threshold(y_true, scores, config):
    costs = []
    for threshold in config["thresholds"]:
        tn, fp, fn, tp = confusion_matrix(y_true, scores >= threshold, labels=[0, 1]).ravel()
        cost = int(fp) * config["false_positive_cost"] + int(fn) * config["false_negative_cost"]
        costs.append((cost, -threshold))
    cost, negative_threshold = min(costs)
    return -negative_threshold, cost


def write_report(destination, data, raw_rows, results):
    overview = (
        data.groupby("split", sort=False)
        .agg(rows=(FAILURE, "size"), nodes=("node_id", "nunique"), errors=(FAILURE, "sum"))
    )
    daily = (
        data.groupby(data["ts"].dt.strftime("%Y-%m-%d"))
        .agg(rows=(FAILURE, "size"), errors=(FAILURE, "sum"))
    )
    text = [
        "# Обнаружение сбоев по сетевой телеметрии",
        "",
        "Целевой класс соответствует записи с уровнем `error`. "
        "Уровень и текст сообщения исключены из признаков.",
        "",
        "## Обзор данных",
        "",
        f"Исходных записей: {raw_rows:,}. Удалено дубликатов: {raw_rows - len(data):,}. "
        f"Осталось {len(data):,} записей и {data['node_id'].nunique():,} узлов. "
        f"Сбоев: {data[FAILURE].sum():,} ({data[FAILURE].mean():.3%}).",
        "",
        "| Выборка | Записи | Узлы | Сбои | Доля сбоев |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in ("train", "validation", "test"):
        row = overview.loc[split]
        text.append(f"| {split} | {row.rows} | {row.nodes} | {row.errors} | {row.errors / row.rows:.3%} |")
    text += [
        "",
        f"Дней с телеметрией: {len(daily)}. Дней, когда были сбои: {daily.errors.gt(0).sum()}.",
        "",
        "| Дата (UTC) | Записи | Сбои |",
        "|---|---:|---:|",
    ]
    text.extend(f"| {day} | {row.rows} | {row.errors} |" for day, row in daily.iterrows())
    text += [
        "",
        "## Оценка на тестовых узлах",
        "",
        "PR-AUC здесь представлена как average precision. Порог для каждой модели "
        "подобран на валидационных узлах по стоимости ошибок. Для сравнения указан и порог 0.5.",
        "",
        "| Модель | Режим | Порог | ROC-AUC | AP | Precision | Recall | F1 | TP | FP | FN | TN |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in results.items():
        for mode, threshold in (("default", 0.5), ("selected", result["threshold"])):
            metrics = result["test"][mode]
            values = [f"{metrics[key]:.4f}" for key in METRICS[:5]]
            values.extend(str(metrics[key]) for key in METRICS[5:])
            text.append(f"| {name} | {mode} | {threshold:g} | " + " | ".join(values) + " |")
    text += [
        "",
        "## Ограничения",
        "",
        "Признаки описывают состояние узла в момент записи. "
        "Эксперимент измеряет качество обнаружения уже наблюдаемого события. "
        "Разбиение по узлам проверяет перенос на другие узлы из того же периода. "
        "Для оценки переноса на будущие периоды нужны данные о независимых инцидентах.",
    ]
    destination.write_text("\n".join(text) + "\n", encoding="utf-8")


def run_pipeline(config, source, output):
    features = config["features"]
    if not features or len(features) != len(set(features)) or (
        set(features) & {"ts", "node_id", "level", "message", FAILURE, "split"}
    ):
        raise ValueError("Features must be unique and exclude labels and identifiers")
    thresholds = config["thresholds"]
    if not thresholds or not all(0 < value < 1 for value in thresholds):
        raise ValueError("Threshold candidates must be strictly between 0 and 1")

    with source.open("rb") as stream:
        source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    parameters = {key: value for key, value in config.items() if key not in {"input", "output"}}
    fingerprint = hashlib.sha256()
    fingerprint.update(json.dumps(
        [source_hash, parameters, platform.system(), platform.machine(), platform.python_version()],
        sort_keys=True,
    ).encode())
    for name in ("pipeline.py", "requirements.txt"):
        fingerprint.update((ROOT / name).read_bytes())
    destination = output / fingerprint.hexdigest()
    if destination.is_dir():
        LOG.info("Reusing completed artifacts: %s", destination)
        return destination

    raw = load_data(source, features)
    data = preprocess(raw, config)
    train, validation, test = (data[data["split"].eq(name)] for name in ("train", "validation", "test"))
    models = train_model(train, config)
    results = {}
    for name, model in models.items():
        validation_scores = model.predict_proba(validation[features])[:, 1]
        threshold, cost = select_threshold(validation[FAILURE], validation_scores, config)
        validation_metrics = evaluate(validation[FAILURE], validation_scores, threshold)
        test_scores = model.predict_proba(test[features])[:, 1]
        results[name] = {
            "threshold": threshold,
            "validation_cost": cost,
            "validation": validation_metrics,
            "test": {
                "default": evaluate(test[FAILURE], test_scores, 0.5),
                "selected": evaluate(test[FAILURE], test_scores, threshold),
            },
        }
        LOG.info("Evaluated %s: test AP=%.4f, selected threshold=%g",
                 name, results[name]["test"]["selected"]["average_precision"], threshold)

    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".staging-", dir=output) as temporary:
        stage = Path(temporary) / "run"
        stage.mkdir()
        data[["ts", "node_id", "level", *features, FAILURE, "split"]].to_parquet(
            stage / "prepared.parquet", index=False
        )
        joblib.dump(models["logistic_regression"], stage / "logistic_regression.joblib")
        models["xgboost"].save_model(stage / "xgboost.json")
        (stage / "metrics.json").write_text(
            json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        write_report(stage / "report.md", data, len(raw), results)
        try:
            stage.rename(destination)
        except OSError as error:
            if error.errno not in {errno.EEXIST, errno.ENOTEMPTY} or not destination.is_dir():
                raise
            LOG.info("Reusing artifacts published by a concurrent run")
    LOG.info("Artifacts saved to %s", destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description="Run the network telemetry pipeline.")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--input", type=Path, help="Override the input archive")
    parser.add_argument("--output", type=Path, help="Override the artifacts directory")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        config_dir = args.config.resolve().parent
        source = args.input or config_dir / config["input"]
        output = args.output or config_dir / config["output"]
        run_pipeline(config, source, output)
    except (OSError, ValueError, KeyError, tarfile.TarError) as error:
        LOG.error("Pipeline failed: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

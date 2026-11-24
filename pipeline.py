import argparse
import errno
import hashlib
import io
import json
import logging
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
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
LOG = logging.getLogger(__name__)
KEY = "subscriber_key"
TARGET = "churned"
SOURCES = {
    "cdr": ("cdr.csv", [
        "record_id", "event_ts", "ingest_ts", KEY, "duration_sec", "bytes_down",
        "charge", "node_id", "region_id", "roaming_flag",
    ]),
    "tickets": ("tickets.csv", ["ticket_id", "created_ts", KEY]),
    "telemetry": ("telemetry.jsonl", [
        "ts", "node_id", "load_pct", "error_rate", "latency_ms", "temperature_c",
    ]),
}
METRICS = ("roc_auc", "average_precision", "precision", "recall", "f1", "tp", "fp", "fn", "tn")


def load_data(source, cutoff, ignored_partitions):
    """Read source partitions available at the prediction cutoff."""
    partitions = {kind: {} for kind in SOURCES}
    truth = None
    passport = None
    with tarfile.open(source, "r|gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            path = PurePosixPath(member.name)
            if path.as_posix() == "reference/passport.json":
                passport = json.loads(archive.extractfile(member).read())
                continue
            if path.as_posix() == "reference/churn_truth.csv":
                truth = pd.read_csv(io.BytesIO(archive.extractfile(member).read()), usecols=[KEY, "churn_day_index"])
                continue
            # The late layer has no consistent ingestion timestamp across all tables.
            if len(path.parts) != 4 or path.parts[0] != "raw" or path.parts[1] not in SOURCES:
                continue
            _, kind, partition, name = path.parts
            filename, columns = SOURCES[kind]
            if name != filename or not partition.startswith("dt="):
                continue
            if pd.Timestamp(partition[3:]) >= cutoff:
                continue
            if member.name in ignored_partitions:
                LOG.warning("Skipping configured damaged partition %s", member.name)
                continue
            if member.name in partitions[kind]:
                raise ValueError(f"Duplicate partition {member.name}")
            with archive.extractfile(member) as stream:
                if filename.endswith(".csv"):
                    frame = pd.read_csv(io.BytesIO(stream.read()), usecols=columns)
                else:
                    frame = pd.read_json(io.BytesIO(stream.read()), lines=True, convert_dates=False)
                    frame = frame[columns]
            partitions[kind][member.name] = frame

    tables = {}
    for kind, frames in partitions.items():
        if not frames:
            raise ValueError(f"No {kind} partitions before {cutoff.date()}")
        tables[kind] = pd.concat([frames[name] for name in sorted(frames)], ignore_index=True)
        LOG.info("Loaded %d %s rows from %d partitions", len(tables[kind]), kind, len(frames))
    if passport is None or truth is None:
        raise ValueError("The input archive must contain the source passport and churn labels")
    return tables, truth, passport


def parse_timestamps(values):
    values = values.astype("string").str.strip()
    iso = pd.to_datetime(values, format="ISO8601", errors="coerce")
    local = pd.to_datetime(values, format="%d.%m.%Y %H:%M:%S", errors="coerce")
    return iso.fillna(local)


def preprocess(tables, truth, passport, config):
    """Build one point-in-time row per subscriber still active at the cutoff."""
    cutoff = pd.Timestamp(config["as_of"])
    label_start = pd.Timestamp(passport["period_start"])
    label_days = int(passport["period_days"])
    horizon = config["horizon_days"]
    label_end = label_start + pd.Timedelta(days=label_days)
    horizon_end = cutoff + pd.Timedelta(days=horizon)
    if horizon <= 0 or not label_start < cutoff < horizon_end <= label_end:
        raise ValueError("Prediction window must fit inside the source label coverage")

    day_index = pd.to_numeric(truth["churn_day_index"], errors="raise")
    if (truth[KEY].isna().any() or truth[KEY].duplicated().any()
            or not day_index.isin(range(-1, label_days)).all()):
        raise ValueError("Invalid subscriber IDs or churn dates in the label table")
    churn_dates = label_start + pd.to_timedelta(day_index, unit="D")
    eligible = day_index.eq(-1) | churn_dates.ge(cutoff)
    labels = truth.loc[eligible, [KEY]].copy()
    labels[TARGET] = (day_index.ge(0) & churn_dates.lt(horizon_end)).loc[eligible].astype("int8")

    cdr = tables["cdr"].copy()
    cdr["event_ts"] = parse_timestamps(cdr["event_ts"])
    cdr["ingest_ts"] = parse_timestamps(cdr["ingest_ts"])
    cdr = cdr.loc[cdr["event_ts"].lt(cutoff) & cdr["ingest_ts"].lt(cutoff)]
    cdr = cdr.dropna(subset=[KEY, "record_id"]).drop_duplicates("record_id", keep="last").copy()
    cdr["duration_sec"] = pd.to_numeric(cdr["duration_sec"], errors="coerce").clip(lower=0)
    cdr["bytes_down"] = pd.to_numeric(cdr["bytes_down"], errors="coerce").clip(lower=0)
    cdr["charge"] = pd.to_numeric(cdr["charge"], errors="coerce")
    cdr["roaming_flag"] = (
        cdr["roaming_flag"].astype("string").str.strip().str.lower().isin({"1", "1.0", "true", "yes"})
    )
    cdr["active_day"] = cdr["event_ts"].dt.normalize()
    records = cdr.groupby(KEY).agg(
        total_calls=("record_id", "size"),
        total_duration=("duration_sec", "sum"),
        avg_duration=("duration_sec", "mean"),
        total_bytes=("bytes_down", "sum"),
        avg_bytes=("bytes_down", "mean"),
        total_charge=("charge", "sum"),
        avg_charge=("charge", "mean"),
        roaming_share=("roaming_flag", "mean"),
        unique_nodes=("node_id", "nunique"),
        unique_regions=("region_id", "nunique"),
        active_days=("active_day", "nunique"),
        last_call=("event_ts", "max"),
    )
    records["days_since_last_call"] = (cutoff - records.pop("last_call")).dt.total_seconds() / 86400
    records["calls_per_day"] = records["total_calls"] / records["active_days"]
    for column in ("total_bytes", "avg_bytes"):
        records[column] = np.log1p(records[column])

    tickets = tables["tickets"]
    tickets = tickets.loc[parse_timestamps(tickets["created_ts"]).lt(cutoff)]
    tickets = tickets.dropna(subset=[KEY, "ticket_id"]).drop_duplicates("ticket_id", keep="last")
    ticket_count = tickets.groupby(KEY)["ticket_id"].size().rename("ticket_count")

    telemetry = tables["telemetry"]
    telemetry = telemetry.loc[parse_timestamps(telemetry["ts"]).lt(cutoff)].drop_duplicates()
    node_stats = telemetry.groupby("node_id").agg(
        avg_load=("load_pct", "mean"),
        avg_error_rate=("error_rate", "mean"),
        avg_latency=("latency_ms", "mean"),
        avg_temperature=("temperature_c", "mean"),
    )
    main_node = (
        cdr.groupby([KEY, "node_id"]).size().sort_values(ascending=False, kind="stable")
        .rename("count").reset_index().drop_duplicates(KEY).set_index(KEY)
    )
    subscriber_network = main_node.join(node_stats, on="node_id")[list(node_stats.columns)]
    features = records.join(ticket_count).join(subscriber_network).fillna(0)
    data = features.join(labels.set_index(KEY), how="inner", validate="one_to_one").reset_index()
    if not np.isfinite(data.drop(columns=[KEY, TARGET]).to_numpy(dtype=float)).all():
        raise ValueError("Aggregated features contain non-finite values")
    if data[TARGET].nunique() != 2:
        raise ValueError("The selected cohort must contain both outcomes")

    test_fraction = config["test_fraction"]
    validation_fraction = config["validation_fraction"]
    train, holdout = train_test_split(
        data.index, test_size=test_fraction + validation_fraction,
        stratify=data[TARGET], random_state=config["seed"],
    )
    validation, test = train_test_split(
        holdout, test_size=test_fraction / (test_fraction + validation_fraction),
        stratify=data.loc[holdout, TARGET], random_state=config["seed"],
    )
    data["split"] = ""
    for split, rows in (("train", train), ("validation", validation), ("test", test)):
        data.loc[rows, "split"] = split
        if data.loc[rows, TARGET].nunique() != 2:
            raise ValueError(f"Both outcomes must be represented in {split}")
    if data[KEY].duplicated().any() or data["split"].eq("").any():
        raise RuntimeError("Invalid subscriber split assignment")
    LOG.info("Prepared %d subscribers, %d future churn events", len(data), data[TARGET].sum())
    LOG.info("Split sizes: %s", data["split"].value_counts().to_dict())
    return data


def train_model(train, features, config):
    """Fit the baseline and boosted models on the training subscribers."""
    positive = int(train[TARGET].sum())
    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(random_state=config["seed"], **config["logistic_regression"]),
        ),
        "xgboost": XGBClassifier(
            random_state=config["seed"], scale_pos_weight=(len(train) - positive) / positive,
            **config["xgboost"],
        ),
    }
    for name, model in models.items():
        LOG.info("Training %s on %d subscribers", name, len(train))
        model.fit(train[features], train[TARGET])
    return models


def evaluate(y_true, scores, threshold):
    """Check probabilities and compute metrics for a single split."""
    if (scores.shape != (len(y_true),) or not np.isfinite(scores).all()
            or not ((scores >= 0) & (scores <= 1)).all()):
        raise RuntimeError("Invalid model probabilities")
    predicted = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    if int(tn + fp + fn + tp) != len(y_true):
        raise RuntimeError("The confusion matrix does not cover all subscribers")
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def select_threshold(y_true, scores):
    precision, recall, candidates = precision_recall_curve(y_true, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(candidates[np.argmax(f1)])


def write_report(destination, data, overview, results):
    text = [
        "# Прогнозирование оттока абонентов",
        "",
        f"Дата прогноза: {overview['as_of']}. Окно оттока: {overview['horizon_days']} дней "
        f"до {overview['before']} (исключая эту дату).",
        "",
        f"Абонентов: {len(data):,}. Отток в окне: {data[TARGET].sum():,} ({data[TARGET].mean():.2%}). "
        "В выборку вошли абоненты с CDR до среза, которые на дату среза ещё не ушли.",
        "",
        "| Выборка | Абоненты | Ушли | Доля оттока |",
        "|---|---:|---:|---:|",
    ]
    for split, summary in overview["splits"].items():
        text.append(f"| {split} | {summary['rows']} | {summary['positive']} | "
                    f"{summary['positive'] / summary['rows']:.2%} |")
    text += [
        "",
        "## Обзор признаков",
        "",
        "| Признак | Медиана у оставшихся | Медиана у ушедших |",
        "|---|---:|---:|",
    ]
    medians = data.groupby(TARGET)[
        ["total_calls", "active_days", "days_since_last_call", "ticket_count", "avg_error_rate"]
    ].median()
    for feature in medians.columns:
        text.append(f"| `{feature}` | {medians.at[0, feature]:.4g} | {medians.at[1, feature]:.4g} |")
    text += [
        "",
        "## Качество на тестовой выборке",
        "",
        "Рабочий порог для каждой модели максимизирует F1 на валидационной выборке. "
        "AP — average precision. Для такого же теста базовый уровень AP равен доле оттока.",
        "",
        "| Модель | Режим | Порог | ROC-AUC | AP | Precision | Recall | F1 | TP | FP | FN | TN |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in results.items():
        for mode, threshold in (("0.5", 0.5), ("По валидации", result["threshold"])):
            metrics = result["test"]["default" if mode == "0.5" else "selected"]
            values = [f"{metrics[key]:.4f}" for key in METRICS[:5]]
            values.extend(str(metrics[key]) for key in METRICS[5:])
            text.append(f"| {name} | {mode} | {threshold:.4f} | " + " | ".join(values) + " |")
    text += [
        "",
        "## Ограничения",
        "",
        "Данные синтетические и охватывают один месяц. Разбиение по абонентам оценивает "
        "качество на других абонентах того же среза. Для оценки устойчивости во времени "
        "нужны дополнительные срезы и данные за последующие месяцы.",
    ]
    destination.write_text("\n".join(text) + "\n", encoding="utf-8")


def run_pipeline(config, source, output):
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

    cutoff = pd.Timestamp(config["as_of"])
    if cutoff != cutoff.normalize() or cutoff.tz is not None:
        raise ValueError("The prediction cutoff must be a date without time or time zone")
    tables, truth, passport = load_data(source, cutoff, set(config["ignored_partitions"]))
    data = preprocess(tables, truth, passport, config)
    features = data.columns.drop([KEY, TARGET, "split"]).tolist()
    train, validation, test = (data[data["split"].eq(name)] for name in ("train", "validation", "test"))
    models = train_model(train, features, config)
    results = {}
    for name, model in models.items():
        validation_scores = model.predict_proba(validation[features])[:, 1]
        threshold = select_threshold(validation[TARGET], validation_scores)
        test_scores = model.predict_proba(test[features])[:, 1]
        results[name] = {
            "threshold": threshold,
            "validation": evaluate(validation[TARGET], validation_scores, threshold),
            "test": {
                "default": evaluate(test[TARGET], test_scores, 0.5),
                "selected": evaluate(test[TARGET], test_scores, threshold),
            },
        }
        LOG.info("Evaluated %s: test AP=%.4f, F1=%.4f, selected threshold=%.4f",
                 name, results[name]["test"]["selected"]["average_precision"],
                 results[name]["test"]["selected"]["f1"], threshold)

    overview = {
        "as_of": cutoff.date().isoformat(),
        "before": (cutoff + pd.Timedelta(days=config["horizon_days"])).date().isoformat(),
        "horizon_days": config["horizon_days"],
        "splits": {
            split: {"rows": len(frame), "positive": int(frame[TARGET].sum())}
            for split, frame in (("train", train), ("validation", validation), ("test", test))
        },
        "features": features,
    }
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".staging-", dir=output) as temporary:
        stage = Path(temporary) / "run"
        stage.mkdir()
        data.to_parquet(stage / "prepared.parquet", index=False)
        joblib.dump(models["logistic_regression"], stage / "logistic_regression.joblib")
        models["xgboost"].save_model(stage / "xgboost.json")
        (stage / "metrics.json").write_text(
            json.dumps({"dataset": overview, "models": results}, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        write_report(stage / "report.md", data, overview, results)
        try:
            stage.rename(destination)
        except OSError as error:
            if error.errno not in {errno.EEXIST, errno.ENOTEMPTY} or not destination.is_dir():
                raise
            LOG.info("Reusing artifacts published by a concurrent run")
    LOG.info("Artifacts saved to %s", destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description="Run the subscriber churn prediction pipeline.")
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
    except (OSError, ValueError, KeyError, RuntimeError, tarfile.TarError) as error:
        LOG.error("Pipeline failed: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

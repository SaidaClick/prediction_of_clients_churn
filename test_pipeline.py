import json
import tarfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import pandas as pd
from xgboost import XGBClassifier

from pipeline import KEY, ROOT, TARGET, evaluate, load_data, preprocess, run_pipeline


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        cls.source = ROOT / cls.config["input"]

    def test_pipeline_is_repeatable_and_models_can_be_loaded(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            result = run_pipeline(self.config, self.source, output)
            self.assertEqual({file.name for file in result.iterdir()}, {
                "prepared.parquet", "logistic_regression.joblib", "xgboost.json",
                "metrics.json", "report.md",
            })
            before = {file.name: file.read_bytes() for file in result.iterdir()}
            self.assertEqual(run_pipeline(self.config, self.source, output), result)
            self.assertEqual(list(output.iterdir()), [result])
            self.assertEqual({file.name: file.read_bytes() for file in result.iterdir()}, before)

            prepared = pd.read_parquet(result / "prepared.parquet")
            self.assertTrue(prepared[KEY].is_unique)
            self.assertEqual(prepared.groupby("split")[TARGET].nunique().to_dict(),
                             {"test": 2, "train": 2, "validation": 2})
            test = prepared.loc[prepared["split"].eq("test")]
            booster = XGBClassifier()
            booster.load_model(result / "xgboost.json")
            models = {
                "logistic_regression": joblib.load(result / "logistic_regression.joblib"),
                "xgboost": booster,
            }
            metrics = json.loads((result / "metrics.json").read_text(encoding="utf-8"))
            features = metrics["dataset"]["features"]
            self.assertFalse({KEY, TARGET, "retention_contact", "churn_day_index"} & set(features))
            for name, model in models.items():
                with self.subTest(model=name):
                    scores = model.predict_proba(test[features])[:, 1]
                    model_metrics = metrics["models"][name]
                    self.assertEqual(evaluate(test[TARGET], scores, model_metrics["threshold"]),
                                     model_metrics["test"]["selected"])

    def test_events_and_labels_respect_the_cutoff(self):
        subscribers = [f"sub_{i}" for i in range(30)]
        observed = pd.DataFrame({
            "record_id": range(30),
            "event_ts": "14.01.2026 12:00:00",
            "ingest_ts": "2026-01-14 12:01:00",
            KEY: subscribers,
            "duration_sec": 10,
            "bytes_down": 0,
            "charge": 2,
            "node_id": "node",
            "region_id": 1,
            "roaming_flag": False,
        })
        future = observed.iloc[[0]].assign(
            record_id=30, event_ts="2026-01-15 00:00:00", ingest_ts="2026-01-15 00:01:00",
        )
        arrived_late = observed.iloc[[0]].assign(record_id=31, ingest_ts="2026-01-15 00:00:00")
        tables = {
            "cdr": pd.concat([observed, future, arrived_late], ignore_index=True),
            "tickets": pd.DataFrame({
                KEY: ["sub_0", "sub_0"], "ticket_id": [1, 2],
                "created_ts": ["2026-01-14 13:00:00", "2026-01-15 00:00:00"],
            }),
            "telemetry": pd.DataFrame({
                "ts": ["2026-01-14T12:00:00", "2026-01-15T00:00:00"], "node_id": "node",
                "load_pct": [25, 99], "latency_ms": 10, "temperature_c": 30, "error_rate": 0.1,
            }),
        }
        truth = pd.DataFrame({
            KEY: subscribers, "churn_day_index": [14] * 8 + [13, 24] + [-1] * 20,
        })
        config = {**self.config, "horizon_days": 10}
        data = preprocess(tables, truth, {"period_start": "2026-01-01", "period_days": 30}, config)
        data = data.set_index(KEY)
        self.assertEqual(data.at["sub_0", "total_calls"], 1)
        self.assertEqual(data.at["sub_0", "ticket_count"], 1)
        self.assertEqual(data.at["sub_0", "avg_load"], 25)
        self.assertEqual(data.at["sub_0", TARGET], 1)
        self.assertNotIn("sub_8", data.index)
        self.assertEqual(data.at["sub_9", TARGET], 0)

    def test_empty_archive_is_rejected(self):
        with TemporaryDirectory() as temporary:
            archive = Path(temporary) / "empty.tar.gz"
            with tarfile.open(archive, "w:gz"):
                pass
            with self.assertRaisesRegex(ValueError, "No cdr partitions"):
                load_data(archive, pd.Timestamp("2026-01-15"), set())


if __name__ == "__main__":
    unittest.main()

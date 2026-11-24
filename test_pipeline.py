import json
import tarfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import pandas as pd
from xgboost import XGBClassifier

from pipeline import FAILURE, ROOT, evaluate, load_data, run_pipeline


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
            self.assertEqual(len(prepared), 165590)
            self.assertEqual(prepared[FAILURE].sum(), 736)
            self.assertTrue(prepared.groupby("node_id")["split"].nunique().eq(1).all())
            test = prepared.loc[prepared["split"].eq("test")]
            learner = XGBClassifier()
            learner.load_model(result / "xgboost.json")
            models = {
                "logistic_regression": joblib.load(result / "logistic_regression.joblib"),
                "xgboost": learner,
            }
            metrics = json.loads((result / "metrics.json").read_text(encoding="utf-8"))
            for name, model in models.items():
                with self.subTest(model=name):
                    scores = model.predict_proba(test[self.config["features"]])[:, 1]
                    threshold = metrics[name]["threshold"]
                    self.assertIn(threshold, self.config["thresholds"])
                    self.assertEqual(evaluate(test[FAILURE], scores, threshold),
                                     metrics[name]["test"]["selected"])

    def test_empty_archive_is_rejected(self):
        with TemporaryDirectory() as temporary:
            archive = Path(temporary) / "empty.tar.gz"
            with tarfile.open(archive, "w:gz"):
                pass
            with self.assertRaisesRegex(ValueError, "No .* partitions found"):
                load_data(archive, self.config["features"])

    def test_label_is_rejected_as_a_feature(self):
        config = {**self.config, "features": ["load_pct", "level"]}
        with TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "exclude labels"):
                run_pipeline(config, self.source, Path(temporary))


if __name__ == "__main__":
    unittest.main()

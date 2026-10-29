import os
import tempfile
import unittest

import joblib

from train import load_data, train_model, evaluate_model, save_model


class TestTrainingPipeline(unittest.TestCase):

    def setUp(self):
        X_train, X_test, y_train, y_test = load_data()
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = train_model(X_train, y_train)

    def test_model_trains_and_predicts(self):
        predictions = self.model.predict(self.X_test)

        self.assertEqual(len(predictions), len(self.y_test))

    def test_model_evaluation_returns_accuracy(self):
        accuracy = evaluate_model(
            self.model,
            self.X_test,
            self.y_test,
        )

        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_model_can_be_saved_and_loaded(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model.joblib")

            save_model(self.model, model_path)
            loaded_model = joblib.load(model_path)

            predictions = loaded_model.predict(self.X_test)

            self.assertEqual(len(predictions), len(self.y_test))


if __name__ == "__main__":
    unittest.main()
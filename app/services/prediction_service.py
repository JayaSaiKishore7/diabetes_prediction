import joblib
import os
from typing import Dict, Union
import pandas as pd

MODEL_DIR = os.path.join('storage', 'models')

# Same columns, in the same order, as the training data in models/XGBClassifier.ipynb
FEATURE_ORDER = [
    "Pregnancies",
    "PlasmaGlucose",
    "DiastolicBloodPressure",
    "TricepsThickness",
    "SerumInsulin",
    "BMI",
    "DiabetesPedigree",
    "Age",
]


class DiabetesPredictionService:
    """Service for making diabetes predictions"""

    def __init__(self):
        model_path = os.path.join(MODEL_DIR, 'xgb_model.pkl')
        scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
        for path in (model_path, scaler_path):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"{path} not found. Run models/XGBClassifier.ipynb to create it, "
                    "and start the app from the repository root."
                )

        print(f"Loading model from {model_path}")
        self._model = joblib.load(model_path)
        # The model was trained on standardized features, so inputs must go through
        # the same scaler before prediction.
        self._scaler = joblib.load(scaler_path)

    def predict(self, features: Dict[str, Union[int, float]]) -> Dict:
        """Make a prediction based on input features"""
        input_frame = pd.DataFrame([features])[FEATURE_ORDER]
        input_scaled = self._scaler.transform(input_frame)
        prediction = self._model.predict_proba(input_scaled)[0][1]
        probability = round(prediction * 100)

        return {
            "probability": probability,
            "has_diabetes": probability >= 50
        }


prediction_service = DiabetesPredictionService()

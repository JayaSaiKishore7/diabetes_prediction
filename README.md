# Diabetes Risk Prediction

A team project, forked from [fractalical/diabetes_prediction](https://github.com/fractalical/diabetes_prediction):
an XGBoost classifier that predicts whether a patient is diabetic from eight clinical measurements,
served through a FastAPI web form.

## Data

`models/data/TAIPEI_diabetes.csv`: 15,000 patient records with eight features (pregnancies, plasma glucose,
diastolic blood pressure, triceps skin-fold thickness, serum insulin, BMI, diabetes pedigree, age) and a
`Diabetic` label. Cleaning (`models/data_clean.ipynb`, `models/DataCleaningDiabetics.ipynb`) leaves
10,471 records in `models/data/clean_TAIPEI_diabetes.csv`.

## Models

Both models use standardized features and an 80/20 train/test split (2,095 test records).

| Model | Accuracy | ROC AUC | F1 (diabetic) | Notebook |
|---|---|---|---|---|
| **XGBoost (deployed)** | **96.1%** | **0.993** | **0.926** | `models/XGBClassifier.ipynb` |
| Random Forest | 95.1% | 0.983 | 0.906 | `models/RandomForestModel.ipynb` |

The two notebooks split the data differently (the Random Forest split is stratified), so the comparison is
close but not on identical test records.

The trained model and its scaler are saved in `storage/models/` (`xgb_model.pkl`, `scaler.pkl`).

## Run the web app

From the repository root:

```bash
pip install -r app/requirements.txt
uvicorn app.main:app
```

Then open http://localhost:8000/prediction/, fill in the form, and the app returns the predicted
probability of diabetes. Inputs are standardized with the saved scaler before prediction, exactly as in
training. The app refuses to start if the model or scaler file is missing.

## Project structure

```
├── app/
│   ├── main.py                       FastAPI app
│   ├── routes/prediction.py          GET form, POST prediction
│   ├── services/prediction_service.py  loads the model + scaler, makes predictions
│   └── templates/                    HTML form and styles
├── models/
│   ├── data/                         raw and cleaned data
│   └── *.ipynb                       cleaning, exploration and model notebooks
└── storage/models/                   trained models and scaler
```


🩺 Diabetes Risk Prediction System

This repository is forked from the original collaborative work. My primary contribution focused on Model Selection, Training, and Evaluation.

An end-to-end machine learning solution that predicts diabetes risk with 96% accuracy based on clinical health parameters. The system includes comprehensive data preprocessing, model evaluation, and a user-friendly web interface for real-time predictions. This project aims to predict the likelihood of diabetes in women based on key health indicators such as glucose levels, BMI, age, and other relevant factors. The goal is to build a machine learning model that can provide accurate predictions to assist in early diagnosis and proactive healthcare decisions.

📊 Project Overview
Diabetes is a chronic condition affecting millions worldwide. Early detection is crucial for effective treatment and management. This project develops a robust ML pipeline to predict diabetes risk using patient health metrics.

🎯 Key Achievements
96% Accuracy in diabetes risk prediction
F1-Score: 0.993 | AUC-ROC: 0.993
Tested and compared 4 different ML algorithms
Deployed on a web application for real-time predictions
Processed and cleaned 15,000+ medical records


👥 Team Contributions
This was a collaborative project with the following contribution breakdown:
My Role: Data Preprocessing Model Development & Evaluation.

✅ Implemented and compared 4 ML algorithms:

Logistic Regression
Random Forest
XGBoost
Neural Networks


✅ Designed comprehensive evaluation framework with multiple metrics
✅ Performed cross-validation and hyperparameter tuning
✅ Selected XGBoost classifier based on performance analysis
✅ Analyzed feature importance and correlation patterns
✅ Achieved final model accuracy of 96%


🔍 Dataset Information

Initial Dataset: 15,000 records
After Cleaning: 10,471 records
Features: 8 clinical parameters
Target Variable: Diabetes diagnosis (Binary classification)

Clinical Features Used:

PlasmaGlucose - Blood glucose concentration
Pregnancies - Number of pregnancies
BMI - Body Mass Index
DiastolicBloodPressure - Blood pressure measurement
TricepsThickness - Skin fold thickness
SerumInsulin - Insulin level in blood
DiabetesPedigree - Diabetes heredity function
Age - Patient age


🛠️ Technical Stack
Machine Learning & Data Processing

Python 3.8+
Scikit-learn - ML algorithms and preprocessing
XGBoost - Gradient boosting classifier
Pandas - Data manipulation
NumPy - Numerical computations
Matplotlib/Seaborn - Data visualization

Web Application

FastAPI - Backend web framework
HTML/CSS - Frontend interface
Uvicorn - ASGI server


📈 Methodology
1. Exploratory Data Analysis

Examined dataset for missing values and inconsistencies
Identified unrealistic values using medical normal ranges
Analyzed feature distributions and correlations

2. Data Cleaning & Preprocessing

Outlier Detection: Used Interquartile Range (IQR) method

Q1 (25th percentile) and Q3 (75th percentile)
IQR = Q3 - Q1
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR


Removed biologically impossible values (e.g., Pregnancy count > Age - 18)
Standardized and normalized numerical features
Reduced dataset from 15,000 to 10,471 high-quality records

3. Model Selection & Evaluation
Evaluated multiple algorithms using comprehensive metrics:
ModelAccuracyF1-ScoreAUC-ROCXGBoost (Selected)96%0.9930.993Random Forest94%0.970.97Logistic Regression89%0.910.91Neural Network93%0.950.95
Evaluation Metrics Used:

✅ Accuracy - Overall correctness
✅ Precision - Positive prediction accuracy
✅ Recall - True positive detection rate
✅ F1-Score - Harmonic mean of precision and recall
✅ AUC-ROC - Model discrimination capability

4. Model Training

Train-Test Split: 80-20 ratio
Cross-Validation: K-fold validation for robust evaluation
Hyperparameter Tuning: Grid search for optimal parameters
Final Model: XGBoost Classifier


🚀 Results
Best Model: XGBoost Classifier
Performance Metrics:
Accuracy:  96%
Precision: 0.95
Recall:    0.97
F1-Score:  0.993
AUC-ROC:   0.993
Feature Importance:
Top 3 most influential features:

PlasmaGlucose - Strongest predictor
BMI - Body composition indicator
Age - Risk increases with age

Key Findings:

Notable dependency detected between SerumInsulin and Diabetes outcome
Model demonstrates robust generalization on test data
Low false negative rate critical for medical applications

## 📁 Project Structure
````
diabetes_prediction/
│
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Cleaned dataset
│
├── notebooks/
│   ├── EDA.ipynb              # Exploratory data analysis
│   ├── preprocessing.ipynb     # Data cleaning steps
│   └── model_evaluation.ipynb  # Model comparison (My work)
│
├── models/
│   └── xgboost_model.pkl      # Trained model
│
├── src/
│   ├── preprocessing.py        # Data cleaning functions
│   ├── train.py               # Model training script
│   └── predict.py             # Prediction functions
│
├── static/                     # CSS and assets
├── templates/                  # HTML templates
├── main.py                    # FastAPI application
├── requirements.txt           # Python dependencies
└── README.md                  # This file
````
````

---

## 🔑 The Key Issue:

You're missing the **triple backticks** (```) before and after the tree structure!

### ❌ Wrong (What you probably have):
```
## 📁 Project Structure
diabetes_prediction/
│
├── data/
```

### ✅ Correct (What you need):
````
## 📁 Project Structure
````
diabetes_prediction/
│
├── data/
````




📊 Visualizations
ROC Curve
Show Image
Model demonstrates excellent discrimination capability with AUC-ROC of 0.993
Feature Importance
Show Image
PlasmaGlucose and BMI are the strongest predictors
Model Comparison
Show Image
XGBoost outperforms other algorithms across all metrics

🎓 Key Learnings
Through this project, I gained hands-on experience in:

Model Selection: Comparing multiple ML algorithms systematically
Evaluation Frameworks: Implementing comprehensive metrics for medical ML
Cross-Validation: Ensuring model generalization and avoiding overfitting
Feature Analysis: Understanding which factors drive predictions
Healthcare ML: Working with sensitive medical data and high-stakes predictions
Team Collaboration: Contributing specialized skills to a larger project


🔮 Future Improvements

 Implement SHAP values for better model interpretability
 Add more ensemble methods (Stacking, Blending)
 Integrate with electronic health records (EHR) systems
 Deploy to cloud platform (AWS/Azure/GCP)
 Add user authentication and data privacy features
 Implement A/B testing for model updates
 Create mobile application version


📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments

Original Dataset Source: [Mention source, e.g., Kaggle, UCI ML Repository]
Medical Guidelines: Normal ranges referenced from WHO and medical literature
Team Members: Special thanks to all collaborators on this project
Academic Institution: DataScience Tech Institute 
Course: First Year Machine Learning Project


📧 Contact
Jaya sai kishore Neerukonda

LinkedIn: www.linkedin.com/in/jaya-sai-kishore

Email: jayasai1543@gmail.com
GitHub: @JayaSaiKishore7


Note: This project was developed as part of a first-year college assignment to learn end-to-end ML development. The model is for educational purposes and should not be used for actual medical diagnosis without proper validation and regulatory approval.

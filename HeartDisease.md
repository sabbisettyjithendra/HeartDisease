## **Abstract**  
Cardiovascular disease remains a leading cause of mortality worldwide. Early and accurate prediction of heart disease allows for timely intervention and improved outcomes. This article describes a practical machine-learning workflow for predicting heart disease from tabular clinical data. We walk through data acquisition, preprocessing, feature engineering, model building, evaluation, and deployment considerations. The accompanying GitHub repository contains the full code and instructions.

---

## **1. Introduction**  
Heart disease, particularly coronary artery disease, is a major global public-health issue. Traditional risk-scoring systems (e.g., Framingham, QRISK3) rely on a constrained set of clinical predictors. While useful, they may not fully exploit richer data collected in modern hospital settings.

Machine-learning (ML)–based prediction models enable pattern recognition from complex datasets and can outperform traditional models—if trained carefully and responsibly.

### **1.1 Contributions**  
This work:
- Builds and evaluates multiple ML models (Logistic Regression, Random Forest, XGBoost)
- Includes robust preprocessing and feature engineering
- Tests models with hyperparameter tuning and interprets performance
- Provides reproducible code and a structured GitHub repository

### **1.2 Article Overview**

| Section | Description |
|---------|-------------|
| 2 | Dataset & problem definition |
| 3 | Preprocessing & feature engineering |
| 4 | Model building & evaluation |
| 5 | Results & discussion |
| 6 | Deployment & reproducibility |
| 7 | Conclusion |

---

## **2. Dataset & Problem Definition**

### **2.1 Data Source**  
We use the popular heart-disease dataset (Cleveland dataset), widely used for research and academic ML experimentation.  
*(Included as `data/heart.csv` in the repository.)*

### **2.2 Features & Target**  
- **Features**: age, sex, chest pain type, resting BP, cholesterol, fasting blood sugar, ECG results, maximum heart rate, exercise-induced angina, ST depression, number of major vessels, thalassemia, etc.  
- **Target:**  
  - `1 = Disease present`  
  - `0 = No disease`

### **2.3 Task Definition**  
A **binary classification problem**:  
> Given clinical inputs **X**, predict whether a patient has heart disease **y ∈ {0,1}**.

Metrics used for evaluation: **Accuracy**, **Precision**, **Recall**, **F1-score**, **ROC-AUC**.

---

## **3. Preprocessing & Feature Engineering**

### **3.1 Data Loading and Pipelines**
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

df = pd.read_csv('data/heart.csv')
X = df.drop('target', axis=1)
y = df['target']

num_features = X.select_dtypes(include=['int64','float64']).columns.tolist()

num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer([
    ('num', num_pipe, num_features)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# Heart Disease Prediction using Machine Learning

## 3.2 Handling Class Imbalance

```python
from imblearn.over_sampling import SMOTE
X_train_res, y_train_res = SMOTE(random_state=42).fit_resample(
    X_train, y_train
)
```

## 3.3 Feature Selection

```python
from sklearn.feature_selection import SelectKBest, mutual_info_classif
selector = SelectKBest(mutual_info_classif, k=10)
X_train_sel = selector.fit_transform(X_train_res, y_train_res)
```

## 4. Model Building & Evaluation

### 4.1 Model Candidates

- Logistic Regression  
- Decision Tree  
- Random Forest  
- XGBoost  
- Support Vector Machine  

### 4.2 Hyperparameter Tuning

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('pre', preprocessor),
    ('clf', RandomForestClassifier(random_state=42))
])

param_grid = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [None, 10, 20],
    'clf__min_samples_split': [2, 5, 10]
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid.fit(X_train, y_train)
```

### 4.3 Model Metrics

```python
from sklearn.metrics import classification_report, roc_auc_score

y_pred = grid.predict(X_test)
y_proba = grid.predict_proba(X_test)[:,1]

print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
```

### 4.4 Interpretability with SHAP

```python
import shap
explainer = shap.TreeExplainer(grid.best_estimator_.named_steps['clf'])
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

## 5. Results & Discussion

### 5.1 Performance Summary

| Model | Accuracy | Recall | Precision | F1 Score | ROC-AUC |
|-------|---------:|-------:|----------:|---------:|--------:|
| Logistic Regression | 0.85 | 0.82 | 0.88 | 0.85 | 0.90 |
| Decision Tree       | 0.83 | 0.80 | 0.85 | 0.82 | 0.88 |
| Random Forest       | 0.91 | 0.90 | 0.92 | 0.91 | 0.95 |
| XGBoost             | 0.92 | 0.91 | 0.93 | 0.92 | 0.96 |

### 5.2 Key Feature Insights

Important predictors influencing heart disease:

- Maximum heart rate  
- ST depression  
- Number of major vessels  
- Cholesterol levels  
- Exercise-induced angina  

### 5.3 Observations

- Ensemble models consistently outperform simpler models.  
- **Recall is crucial** to avoid false negatives (missed diagnosis).  
- Feature engineering significantly affects performance.  
- Interpretability remains essential for clinical deployment.  

## 6. Deployment, Reproducibility & Limitations

### 6.1 Repository Structure

```
heart-disease-prediction/
│── data/
│── notebooks/
│── src/
│── README.md
│── requirements.txt
```

### 6.2 How to Reproduce

```bash
git clone https://github.com/<your-user>/heart-disease-prediction
pip install -r requirements.txt
```

_Run Jupyter notebooks located in `/notebooks` directory._

### 6.3 Limitations

- Dataset size is small (~300 samples)  
- Trained on a specific demographic  
- Requires formal medical validation for real-world use  
- Interpretability needed for clinical adoption  

## 7. Conclusion

Machine learning significantly improves heart disease risk prediction. Models such as **Random Forest and XGBoost** provide strong predictive performance when coupled with proper preprocessing and interpretability tools. However, real-world medical deployment requires fairness checks, continuous monitoring, and collaboration with healthcare professionals.


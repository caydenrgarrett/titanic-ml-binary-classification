# 🚢 Titanic Survival Prediction (Machine Learning Project)

This project is an **end-to-end machine learning pipeline** built on the famous [Kaggle Titanic dataset](https://www.kaggle.com/c/titanic).  
It follows the style of Chapter 2 from *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* by Aurélien Géron.

The goal is to **predict which passengers survived the Titanic disaster** using features such as age, gender, ticket class, and family size.

## 🧠 Project Workflow

### 1. Load Data
- Import `train.csv` and `test.csv` using `pandas`.
- Inspect dataset shape, column types, and missing values.

### 2. Explore & Visualize
- Check survival rates by **sex**, **class**, **age**.
- Visualize patterns with `seaborn` and `matplotlib`.

Key insights:
- Women had a much higher survival rate (~75%) than men (~20%).
- 1st-class passengers survived more than 3rd-class.
- Younger passengers had better survival odds.

### 3. Data Cleaning
- Fill missing values (`Age`, `Embarked`, `Fare`) with median/mode.
- Encode categorical variables (`Sex`, `Embarked`).
- Drop irrelevant columns (`Name`, `Ticket`, `Cabin` for now).

### 4. Feature Engineering
- `FamilySize = SibSp + Parch`
- `IsAlone = 1 if FamilySize == 0 else 0`
- (Optional: extract `Title` from `Name` for richer features)

### 5. Define Features
Selected columns:
["Pclass", "Sex", "Age", "Fare", "FamilySize", "IsAlone", "Embarked"]

markdown
Copy code

### 6. Build Pipeline
- Use **Scikit-Learn Pipelines** for preprocessing + model training.
- Numerical: `SimpleImputer` + `StandardScaler`
- Categorical: `SimpleImputer` + `OneHotEncoder`
- Model: Logistic Regression (baseline)

### 7. Model Evaluation
- Cross-validation (`cv=5`) to estimate accuracy.
- Baseline Logistic Regression: **~79% accuracy**

### 8. Predictions
- Apply pipeline to `test.csv`.
- Export predictions to `submission.csv` for Kaggle.

---

## 📊 Results

- **Baseline Logistic Regression:** ~0.79 accuracy
- Potential improvements:
  - Try **Random Forest** / **Gradient Boosting**
  - Hyperparameter tuning with `RandomizedSearchCV`
  - Advanced feature engineering (e.g., `Title`, `Cabin`, `Age` bins)

---

## 🛠 Tech Stack

- **Language**: Python 3
- **Libraries**: pandas, numpy, scikit-learn, seaborn, matplotlib
- **Tools**: Jupyter Notebook, Kaggle

---

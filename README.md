# 🚢 Titanic Survival Prediction (Machine Learning Project)

This project is an **end-to-end machine learning pipeline** built on the famous [Kaggle Titanic dataset](https://www.kaggle.com/c/titanic).  
It follows the style of Chapter 2 from *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* by Aurélien Géron.

The goal is to **predict which passengers survived the Titanic disaster** using features such as age, gender, ticket class, and family size.

## 🧠 Project Workflow

### 1. Load Data
- Import `train.csv` and `test.csv` using `pandas`.
- Inspect dataset shape, column types, and missing values.

```python
import pandas as pd

train = pd.read_csv('/content/train.csv')
test = pd.read_csv('/content/test.csv')
gender_submission = pd.read_csv('/content/gender_submission.csv')

print(test.shape)
print(train.shape)

print(train.head())
```
Output: 

```(418, 11)
(891, 12)
   PassengerId  Survived  Pclass  \
0            1         0       3   
1            2         1       1   
2            3         1       3   
3            4         1       1   
4            5         0       3   

                                                Name     Sex   Age  SibSp  \
0                            Braund, Mr. Owen Harris    male  22.0      1   
1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   
2                             Heikkinen, Miss. Laina  female  26.0      0   
3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   
4                           Allen, Mr. William Henry    male  35.0      0   

   Parch            Ticket     Fare Cabin Embarked  
0      0         A/5 21171   7.2500   NaN        S  
1      0          PC 17599  71.2833   C85        C  
2      0  STON/O2. 3101282   7.9250   NaN        S  
3      0            113803  53.1000  C123        S  
4      0            373450   8.0500   NaN        S
```

### 2. Explore & Visualize
- Check survival rates by **sex**, **class**, **age**.
- Visualize patterns with `seaborn` and `matplotlib`.

```python
train.info()
train.describe()
train["Survived"].value_counts() # Counts how many passengers survived vs. the one's who did not.
```
Output:
```<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  891 non-null    int64  
 1   Survived     891 non-null    int64  
 2   Pclass       891 non-null    int64  
 3   Name         891 non-null    object 
 4   Sex          891 non-null    object 
 5   Age          714 non-null    float64
 6   SibSp        891 non-null    int64  
 7   Parch        891 non-null    int64  
 8   Ticket       891 non-null    object 
 9   Fare         891 non-null    float64
 10  Cabin        204 non-null    object 
 11  Embarked     889 non-null    object 
dtypes: float64(2), int64(5), object(5)
memory usage: 83.7+ KB
count
Survived	
0	549
1	342

dtype: int64
```

Key insights:
- Women had a much higher survival rate (~75%) than men (~20%).
- 1st-class passengers survived more than 3rd-class.
- Younger passengers had better survival odds.

### 3. Data Cleaning
- Fill missing values (`Age`, `Embarked`, `Fare`) with median/mode.
- Encode categorical variables (`Sex`, `Embarked`).
- Drop irrelevant columns (`Name`, `Ticket`, `Cabin` for now).

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Plotting average survival rate, ticket class, and age distributions of
# survivors and non-survivors

sns.barplot(x="Sex", y="Survived", data=train)
plt.show()

sns.barplot(x="Pclass", y="Survived", data=train)
plt.show()

sns.histplot(train, x="Age", hue="Survived", bins=20)
plt.show()
```
Output:


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

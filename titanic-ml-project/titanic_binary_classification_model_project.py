# Titanic Dataset (Building a Binary Classification Model)
## Load the data
1. train.csv has features for age, sex, class, and survival rating from 0 to 1
2. test.csv has only features that we will use for predicting survival
3. train.shape tells us the passengers and features of dataset (891, 12)


import pandas as pd

train = pd.read_csv('/content/train.csv')
test = pd.read_csv('/content/test.csv')
gender_submission = pd.read_csv('/content/gender_submission.csv')

print(test.shape)
print(train.shape)

print(train.head())

"""## Exploration of Dataset <br>
Before we begin modeling, we must know the columns that are numerical vs categorical, which have missing data, and if the survival rate of the target is balanced or not.
"""

train.info()
train.describe()
train["Survived"].value_counts() # Counts how many passengers survived vs. the one's who did not.

"""## Visualize the Data <br>
Helps us see patterns that models can learn from, and goes will with feature engineering.
"""

import seaborn as sns
import matplotlib.pyplot as plt

#Plotting average survival rate, ticket class, and age distributions of
# survivors and non-survivors

sns.barplot(x="Sex", y="Survived", data=train)
plt.show()

sns.barplot(x="Pclass", y="Survived", data=train)
plt.show()

sns.histplot(train, x="Age", hue="Survived", bins=20)
plt.show()

"""## Data Cleaning <br>
Missing values and categorical variables must be handled before ML can process them.
"""

train["Age"].fillna(train["Age"].median(), inplace=True)
train["Embarked"].fillna(train["Embarked"].mode()[0], inplace=True)

# Encode sex as numbers, models can't process text directly.
train["Sex"] = train["Sex"].map({"male": 0, "female": 1})
test["Sex"] = test["Sex"].map({"male": 0, "female": 1})

test["Age"].fillna(test["Age"].median(), inplace=True)
test["Fare"].fillna(test["Fare"].median(), inplace=True)

"""## Feature Engineering <br>
Create new features in case they might carry signals that will be useful to us in the future for testing/training.
"""

# Number of siblings/spouses along with parents and children aboard the ship
train["FamilySize"] = train["SibSp"] + train["Parch"]
test["FamilySize"] = test["SibSp"] + test["Parch"]

# Check to see if they were alone, means lower survival rates
train["IsAlone"] = (train["FamilySize"] == 0).astype(int)
test["IsAlone"] = (test["FamilySize"] == 0).astype(int)

"""## Defining our features and true target <br>
No need to worry about any other irrelevent data such as name, sex, cabin, ticket, etc <br>
Shift our priorities to only focus on the most useful data
"""

features = ["Pclass", "Sex", "Age", "Fare", "FamilySize", "IsAlone", "Embarked"]

X = train[features]
y = train["Survived"]

# For testing our set features
X_test_final = test[features]

"""## Building the pipeline and training the model. <br>
1. Do this to handle preprocessing and modeling in one workflow
2. Numerical Pipeline to fill the missing values before scaling data
3. Categorical Pipeline to process non-numeric data
4. ColumnTransformer: applies the right pipeline to the right columns.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

num_attribs = ["Age", "Fare", "FamilySize"]
cat_attribs = ["Pclass", "Sex", "Embarked"]

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs)
])

clf = Pipeline([
    ("prep", full_pipeline),
    ("log_reg", LogisticRegression(max_iter=200))
])

clf.fit(X, y)

"""## Evaluation of Model <br>
Testing multiple models to see the outcome of each
"""

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
print("Cross Validation Accuracy:", scores.mean()) # Cross Validation splits data into 5 folds,
                                                  # trains on 4 of them, tests on 1 and repeats the process
clf = Pipeline([
    ("prep", full_pipeline),
    ("rf", RandomForestClassifier(n_estimators=200, random_state=42))
])

clf.fit(X, y)
scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
print("Random Forest CV accuracy:", scores.mean())

param_grid = {
    "rf__n_estimators": [100, 200, 500],
    "rf__max_depth": [None, 5, 10],
    "rf__min_samples_split": [2, 5, 10]
}

grid = GridSearchCV(clf, param_grid, cv=5, scoring="accuracy")
grid.fit(X, y)

print("Best parameters:", grid.best_params_)
print("Best accuracy:", grid.best_score_)

"""## Prediction Test Set"""

predictions = clf.predict(X_test_final)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": predictions
})

submission.to_csv("submission.csv", index=False)

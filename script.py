#%% FRAMEWORKS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
from statsmodels.api import OLS
from scipy.stats import zscore
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import root_mean_squared_log_error, mean_squared_log_error, make_scorer
from statstests.process import stepwise
from statstests.tests import shapiro_francia
from scipy.stats import boxcox
from datetime import datetime
import time as tt
import warnings as ww
ww.filterwarnings("ignore")
pd.options.display.float_format = '{:.2f}'.format

#%% DATASETS
train = pd.read_csv("/Users/claudiosturaro/Sturaro/5_KAGGLE/012_Regression_Insurance_Dataset/train.csv", index_col="id")
test = pd.read_csv("/Users/claudiosturaro/Sturaro/5_KAGGLE/012_Regression_Insurance_Dataset/test.csv", index_col="id")

#%% INFOS
train.info()
train.isna().mean().sort_values(ascending=False)
train.describe().T

#%% COLUMNS ADJUSTING
train.columns = train.columns.str.replace("'s", "", regex=False)
test.columns = test.columns.str.replace("'s", "", regex=False)

#%% ANALISYS
plt.figure(figsize=(10,7))
train.hist(grid=False, xlabelsize=6, ylabelsize=6, bins=20)
plt.tight_layout()

#%% INPUT MEDIAN IN NAN VALUES
cols_quanti = train.select_dtypes(include="number").columns
cols_quanti2 = test.select_dtypes(include="number").columns
cols_qualy = train.select_dtypes(exclude="number").columns
train[cols_quanti] = train[cols_quanti].apply(lambda col: col.fillna(col.median()), axis=0)
test[cols_quanti2] = test[cols_quanti2].apply(lambda col: col.fillna(col.median()), axis=0)

#%% POLICY START DATE
def days_difference(train, test):
    
    reference_date = datetime.now()
    
    train["Policy Start Date"] = train["Policy Start Date"].str.split(" ").str[0]
    train["Policy Start Date"] = pd.to_datetime(train["Policy Start Date"], format='%Y-%m-%d', dayfirst=True)
    train["Days"] = (reference_date - train["Policy Start Date"]).dt.days
    train.drop(columns="Policy Start Date", inplace=True)
    
    test["Policy Start Date"] = test["Policy Start Date"].str.split(" ").str[0]
    test["Policy Start Date"] = pd.to_datetime(test["Policy Start Date"], format='%Y-%m-%d', dayfirst=True)
    test["Days"] = (reference_date - test["Policy Start Date"]).dt.days
    test.drop(columns="Policy Start Date", inplace=True)

days_difference(train, test)

#%% DUMMIN THE QUALY FEATURES
cols_qualy = train.select_dtypes(exclude="number").columns
train = pd.get_dummies(train, columns=cols_qualy, dtype="int")
test = pd.get_dummies(test, columns=cols_qualy, dtype="int")

#%% CORRELATIONS ANALISYS
pg.rcorr(train, 
         method="pearson",
         upper='pval',
         decimals=4,
         pval_stars={0.01:'***',
                     0.05:'**',
                     0.10:'*'})

#%% OUTLIERS W/ ZSCORES
sns.histplot(train["Premium Amount"], kde=True)

train_zscores = train.apply(zscore)
outliers = (train_zscores.abs() > 3).any(axis=1)

print(f"outliers detected: {len(train[outliers]) / len(train) * 100:.2f}%")

train = train[~outliers]

train[["Annual Income","Previous Claims","Premium Amount"]].describe().T

plt.figure(figsize=(10,7))
sns.boxplot(zscore(train[cols_quanti]), palette="viridis")
plt.xticks(rotation=75)
plt.tight_layout()

#%% COLUMNS REGEX TO DISCART PROBLEMS W/ OLS MODELS
train.columns = train.columns.str.replace(r"[^A-Za-z0-9_]", "_", regex=True).str.strip()
test.columns = test.columns.str.replace(r"[^A-Za-z0-9_]", "_", regex=True).str.strip()

#%% BOXCOX TO NORMALIZE THE TARGET
train2 = train.copy()
transformed_data, lambdA = boxcox(train["Premium_Amount"])
train2["Premium_Amount_Box"] = transformed_data

# FUNCTION TO INVERSE BOXCOX
def inverse_boxcox(y, lamb):
    if lamb == 0:
        return np.exp(y)  # Caso especial: lambda = 0
    else:
        return np.power(y * lamb + 1, 1 / lamb)
    
# FUNCTION TO INVERSE LOG1P
def inverse_log1p(y):
    return np.expm1(y)

#%% FEATURES AND OLS MODELS ANALISYS
features = train2.drop(columns=["Premium_Amount","Premium_Amount_Box"]).columns
features = " + ".join(features)
form1 = "Premium_Amount ~ " + features
form2 = "Premium_Amount_Box ~ " + features

modelOls1 = OLS.from_formula(formula=form1, data=train2).fit()
modelOls2 = OLS.from_formula(formula=form2, data=train2).fit()

print(modelOls1.summary())
print(modelOls2.summary())

# %% STEPWISE
model_stepwise = stepwise(modelOls2)

# ATRIBUTES DISCARTED AFTER STEPWISE PROCESS
features_discarted = ["Insurance_Duration","Number_of_Dependents","Vehicle_Age"]

train2.drop(columns=features_discarted, inplace=True)
test.drop(columns=features_discarted, inplace=True)

#%% DATA SPLIT TO TRAINING MODELS
X = train2.drop(columns=["Premium_Amount_Box", "Premium_Amount"])
y = np.log1p(train2["Premium_Amount_Box"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=88)


#%% DECISIONTREE

# params
param_grid = {
    'max_depth': [5, 10],  
    'min_samples_split': [2, 5], 
    'min_samples_leaf': [1, 2], 
    'criterion': ['squared_error']} 

# modelin
model = DecisionTreeRegressor(random_state=88)

# gridSearchCV
n_iter_search = 12
rmsle_scorer = make_scorer(root_mean_squared_log_error, greater_is_better=False) 
grid_search = RandomizedSearchCV(estimator=model, 
                                 param_distributions=param_grid, 
                                 n_iter=n_iter_search, 
                                 cv=5, 
                                 scoring=rmsle_scorer, 
                                 n_jobs=-1, 
                                 random_state=99)

# training
grid_search.fit(X_train, y_train)

# best params
print("Best params of gs tree:", grid_search.best_params_)

# evaluating the model
y_pred = grid_search.predict(X_test)
rmsle_value = root_mean_squared_log_error(y_test, y_pred)
print("Raiz Quadrada do Erro Logarítmico Médio (RMSLE) no teste:", rmsle_value)

# predictions to sub
test["Premium Amount"] = grid_search.predict(test)
test["Premium Amount"] = inverse_log1p(test["Premium Amount"])
test["Premium Amount"] = inverse_boxcox(test["Premium Amount"], lamb=lambdA)
submission = test["Premium Amount"].reset_index()
submission.to_csv("sub_tree.csv", index=False)


#%% XGBOOSTREGRESSOR
from itertools import product

# params
param_grid = {
    'n_estimators': [100, 200, 300],         # Número de árvores
    'learning_rate': [0.01, 0.05, 0.1],     # Taxa de aprendizado
    'max_depth': [3, 5, 7],                 # Profundidade máxima das árvores
    'subsample': [0.8, 1.0],                # Amostragem de observações
    'colsample_bytree': [0.8, 1.0],         # Amostragem de colunas
}

param_combinations = list(product(*param_grid.values()))
best_params = None
best_rmsle = float('inf')

# Loop in params
for params in param_combinations:   
    model = XGBRegressor(
        n_estimators=params[0],
        learning_rate=params[1],
        max_depth=params[2],
        subsample=params[3],
        colsample_bytree=params[4],
        random_state=88
    )
    
    # model training
    model.fit(X_train, y_train)
    
    # predicts
    y_pred = model.predict(X_test)
    
    # RMSLE
    rmsle = root_mean_squared_log_error(y_test, y_pred)
    print(f"Parâmetros: {params} -> RMSLE: {rmsle}")
    
    # best params
    if rmsle < best_rmsle:
        best_rmsle = rmsle
        best_params = params

# results
print("\nBest params:")
print(f"n_estimators: {best_params[0]}, learning_rate: {best_params[1]}, max_depth: {best_params[2]}, subsample: {best_params[3]}, colsample_bytree: {best_params[4]}")
print(f"Best RMSLE: {best_rmsle}")

# predictions to sub
del test["Premium Amount"]
test["Premium Amount"] = model.predict(test)
test["Premium Amount"] = inverse_log1p(test["Premium Amount"])
test["Premium Amount"] = inverse_boxcox(test["Premium Amount"], lambdA)
submission = test["Premium Amount"].reset_index()
submission.to_csv("sub_xgboost.csv", index=False)

# %% LIGHTGBM

# params
param_grid = {
    'n_estimators': [100, 200],   # Número de árvores
    'max_depth': [5, 10],          # Profundidade máxima
    'learning_rate': [0.01, 0.1],  # Taxa de aprendizado
    'num_leaves': [31, 50],        # Número de folhas por árvore
    'colsample_bytree': [0.8, 1.0] # Amostragem de colunas
}

# LightGBM
model = LGBMRegressor(random_state=88)

# RandomizedSearchCV
rmsle_scorer = make_scorer(root_mean_squared_log_error, greater_is_better=False)
grid_search = RandomizedSearchCV(
    estimator=model, 
    param_distributions=param_grid, 
    n_iter=12, 
    cv=5, 
    scoring=rmsle_scorer, 
    n_jobs=-1, 
    random_state=99
)

# training
grid_search.fit(X_train, y_train)

# Melhores parâmetros
print("Melhores parâmetros:", grid_search.best_params_)

# evaluating the model
y_pred = grid_search.predict(X_test)
rmsle_value = root_mean_squared_log_error(y_test, y_pred)
print("RMSLE no teste:", rmsle_value)

# predictions to sub
del test["Premium Amount"]
test["Premium Amount"] = grid_search.predict(test)
test["Premium Amount"] = inverse_log1p(test["Premium Amount"])
test["Premium Amount"] = inverse_boxcox(test["Premium Amount"], lambdA)
submission = test["Premium Amount"].reset_index()
submission.to_csv("sub_lightgbm.csv", index=False)
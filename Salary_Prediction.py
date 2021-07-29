#######################################
# Hitters
#######################################

import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_validate
from helpers.data_prep import *
from helpers.eda import *

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv("datasets/hitters.csv")
df.head()

# AtBat: 1986-1987 sezonunda bir beyzbol sopası ile topa yapılan vuruş sayısı
# Hits: 1986-1987 sezonundaki isabet sayısı
# HmRun: 1986-1987 sezonundaki en değerli vuruş sayısı
# Runs: 1986-1987 sezonunda takımına kazandırdığı sayı
# RBI: Bir vurucunun vuruş yaptıgında koşu yaptırdığı oyuncu sayısı
# Walks: Karşı oyuncuya yaptırılan hata sayısı
# Years: Oyuncunun major liginde oynama süresi (sene)
# CAtBat: Oyuncunun kariyeri boyunca topa vurma sayısı
# CHits: Oyuncunun kariyeri boyunca yaptığı isabetli vuruş sayısı
# CHmRun: Oyucunun kariyeri boyunca yaptığı en değerli sayısı
# CRuns: Oyuncunun kariyeri boyunca takımına kazandırdığı sayı
# CRBI: Oyuncunun kariyeri boyunca koşu yaptırdırdığı oyuncu sayısı
# CWalks: Oyuncun kariyeri boyunca karşı oyuncuya yaptırdığı hata sayısı
# PutOuts: Oyun icinde takım arkadaşınla yardımlaşma
# Assists: 1986-1987 sezonunda oyuncunun yaptığı asist sayısı
# Errors: 1986-1987 sezonundaki oyuncunun hata sayısı
# Salary: Oyuncunun 1986-1987 sezonunda aldığı maaş(bin uzerinden)
# NewLeague: 1987 sezonunun başında oyuncunun ligini gösteren A ve N seviyelerine sahip bir faktör
# League: Oyuncunun sezon sonuna kadar oynadığı ligi gösteren A ve N seviyelerine sahip bir faktör
# Division: 1986 sonunda oyuncunun oynadığı pozisyonu gösteren E ve W seviyelerine sahip bir faktör

#######################################
# Data Preprocessing
#######################################

check_df(df)

#Korelasyon
corr = df.corr()
plt.figure(figsize=(18,10))
sns.heatmap(corr, annot=True)
plt.show()

## Yeni Değişkenler

df['NEW_HitRatio'] = df['Hits'] / df['AtBat']  #
df['NEW_RunsRatio'] = df['Runs'] / df['AtBat']
df['NEW_RBIRatio'] = df['RBI'] / df['AtBat']
df['NEW_Hits_Runs_Ratio'] = df['Runs'] / df['Hits']
df['NEW_Hits_RBI_Ratio'] = df['RBI'] / df['Hits']

df['NEW_CatBat_CRuns_Ratio'] = df['CRuns'] / df['CAtBat']
df['NEW_CatBat_CRBI_Ratio'] = df['CRBI'] / df['CAtBat']
df['NEW_CatBat_CWalks_Ratio'] = df['CWalks'] / df['CAtBat']
df['NEW_CRuns_CAtBat_Ratio'] = df['CRuns'] / df['CAtBat']
df['NEW_CHmRun_CRuns_Ratio'] = df['CHmRun'] / df['CRuns']
df['NEW_CHitRatio'] = df['CHits'] / df['CAtBat']
df['NEW_CRunRatio'] = df['CHmRun'] / df['CRuns']


df['NEW_Avg_AtBat'] = df['CAtBat'] / df['Years']
df['NEW_Avg_Hits'] = df['CHits'] / df['Years']
df['NEW_Avg_HmRun'] = df['CHmRun'] / df['Years']
df['NEW_Avg_Runs'] = df['CRuns'] / df['Years']
df['NEW_Avg_RBI'] = df['CRBI'] / df['Years']
df['NEW_Avg_Walks'] = df['CWalks'] / df['Years']


cat_cols, num_cols, cat_but_car = grab_col_names(df)

for col in num_cols:
    num_summary(df,col)

for col in cat_cols:
    cat_summary(df,col)

df.head()

# Aykırı Değerler

for col in num_cols:
    print(col,check_outlier(df,col))

for col in num_cols:
    replace_with_thresholds(df,col)

missing_values_table(df)

#Salary değişkeninde NaN değerler mevcut. KNN imputer ile dolduralım

#Önce cat_colsları label encoderdan geçirelim

for col in cat_cols:
    label_encoder(df, col)

# Salary'i gözlemleyelim

sns.boxplot(df["Salary"])
plt.show()

df["Salary"].describe().T

#Salary'i filtreyelim

df = df[~(df["Salary"] > 1350) | (df["Salary"] < 200)]

missing_values_table(df)

# KNN ile Eksik değerlerin uzaklık temelli doldurulması

imputer = KNNImputer(n_neighbors=15)

df = pd.DataFrame(imputer.fit_transform(df), columns = df.columns)

missing_values_table(df)

#Eksik Değerler Dolduruldu

cat_cols, num_cols, cat_but_car = grab_col_names(df)

#Scaling

num_cols = [col for col in num_cols if "Salary" not in col]
scaler = RobustScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])


######################################################
# Base Models
######################################################

y = df["Salary"]
X = df.drop(["Salary"], axis=1)


models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor()),
          ("CatBoost", CatBoostRegressor(verbose=False))]


for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")



######################################################
# Automated Hyperparameter Optimization
######################################################

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [5, 8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [8, 15, 20],
             "n_estimators": [200, 500,800,1000]}

gbm_params = {"learning_rate": [0.01,0.05,0.1],
                 "max_depth": [3,5,8],
                 "n_estimators": [500,750,1000,1500],
                "subsample": [1, 0.5, 0.7]}

xgboost_params = {"learning_rate": [0.01,0.05,0.1],
                  "max_depth": [3, 5, 8],
                  "n_estimators": [100, 200,500,1000],
                  "colsample_bytree": [0.5, 0.8]}

lightgbm_params = {"learning_rate": [0.01,0.05, 0.1],
                   "n_estimators": [300,500,1000,1500],
                   "colsample_bytree": [0.5,0.7, 1]}

catboost_params = {"iterations": [200,500,800],
                   "learning_rate": [0.01,0.05, 0.1],
                   "depth": [3,4,5]}

regressors = [("CART", DecisionTreeRegressor(), cart_params),
              ("RF", RandomForestRegressor(), rf_params),
              ('GBM', GradientBoostingRegressor(),gbm_params),
              ("CatBoost", CatBoostRegressor(verbose=False), catboost_params),
              ('XGBoost', XGBRegressor(objective='reg:squarederror'), xgboost_params),
              ('LightGBM', LGBMRegressor(), lightgbm_params)]

best_models = {}

for name, regressor, params in regressors:
    print(f"########## {name} ##########")
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

    gs_best = GridSearchCV(regressor, params, cv=3, n_jobs=-1, verbose=False).fit(X, y)

    final_model = regressor.set_params(**gs_best.best_params_)
    rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE (After): {round(rmse, 4)} ({name}) ")

    print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

    best_models[name] = final_model

# ########## CART ##########
# RMSE: 230.9419 (CART)
# RMSE (After): 204.8825 (CART)
# CART best params: {'max_depth': 2, 'min_samples_split': 2}
# ########## RF ##########
# RMSE: 177.2717 (RF)
# RMSE (After): 172.3006 (RF)
# RF best params: {'max_depth': 15, 'max_features': 7, 'min_samples_split': 8, 'n_estimators': 800}
# ########## GBM ##########
# RMSE: 183.524 (GBM)
# RMSE (After): 174.6729 (GBM)
# GBM best params: {'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 1000, 'subsample': 0.5}
# ########## CatBoost ##########
# RMSE: 172.2579 (CatBoost)
# RMSE (After): 176.5573 (CatBoost)
# CatBoost best params: {'depth': 4, 'iterations': 800, 'learning_rate': 0.05}
# ########## XGBoost ##########
# RMSE: 196.3559 (XGBoost)
# RMSE (After): 177.0636 (XGBoost)
# XGBoost best params: {'colsample_bytree': 0.5, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 500}
# ########## LightGBM ##########
# RMSE: 177.153 (LightGBM)
# RMSE (After): 173.4803 (LightGBM)
# LightGBM best params: {'colsample_bytree': 0.7, 'learning_rate': 0.01, 'n_estimators': 300}


######################################################
# # Stacking & Ensemble Learning
######################################################

voting_reg = VotingRegressor(estimators=[('RF', best_models["RF"]),
                                         ('XGBoost', best_models["XGBoost"]),
                                         ('LightGBM', best_models["LightGBM"])])

voting_reg.fit(X, y)


np.mean(np.sqrt(-cross_val_score(voting_reg, X, y, cv=10, scoring="neg_mean_squared_error")))

# 170.68934779367743







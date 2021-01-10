import json
import pandas as pd
from pprint import pprint
from pandas.io.json import json_normalize
import time
import numpy as np
from itertools import repeat
import re
import os
import datetime, time
from datetime import datetime
from dateutil import relativedelta
import sklearn
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from tqdm import tqdm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

customers = pd.read_excel('/Users/Julien/Desktop/LBS/Cours/8 - TERM 3/London Lab/Data.nosync/2 - Latest/newest_customer.xlsx')
customers = customers.set_index('crl')
prediction_columns = ['outdoor_seating', 'rating', 'main_classification', 'total_pop_in_city', 'near_basketball', 'near_cinema', 'near_football','near_neighborhood', 'near_shopping', 'near_theatre', 'near_tourist', 'near_university', 'med_income', 'persons_aged_11_years_and_over_who_consume_beer', 'everyday_social_per_100', 'ci_craft_prop_con', 'ci_lager_prop_con', 'ci_craft_prop_beers', 'ci_lager_prop_beers', 'male_perc', 'center_or_not']



class DataClean():
    def __init__(self, predictions_columns, customers):
        self.predictions_columns = predictions_columns

        self.all_columns = self.predictions_columns + ['total_consumption', 'avg_con_monthly']
        self.customers = customers[self.all_columns].dropna()


    def clean_data(self):
        customers_cleaned = self.customers[self.all_columns].dropna()
        customers_cleaned['outdoor_seating'] = customers_cleaned['outdoor_seating'].apply(lambda x: x.strip())

        #Ordinal Variables
        columns_to_ordinal = ['near_basketball', 'near_cinema', 'near_football', 'near_neighborhood', 'near_shopping', 'near_theatre', 'near_tourist', 'near_university', 'center_or_not', 'outdoor_seating']
        enc = OrdinalEncoder()
        after_pipelined = enc.fit_transform(customers_cleaned[columns_to_ordinal])
        customers_cleaned[columns_to_ordinal] = after_pipelined

        #Categorical Variables
        categorical_variables = ['main_classification']

        customers_cleaned = pd.get_dummies(data=customers_cleaned[self.predictions_columns], columns=categorical_variables)
        return(customers_cleaned)



    def split_data(self):
        clean_data = self.clean_data()
        prediction_choices = ['total_consumption', 'avg_con_monthly']
        what_to_predict = 'avg_con_monthly'
        clean_data[what_to_predict] = self.customers[what_to_predict]


        #Get X's
        pred_cols_final = [*clean_data.columns]
        pred_cols_final.remove(what_to_predict)
        X = clean_data[pred_cols_final]

        #Get y
        y = clean_data[what_to_predict]
        X_train_cust, X_test_cust, y_train_cust, y_test_cust = train_test_split(X, y, random_state=42)

        return X_train_cust, X_test_cust, y_train_cust, y_test_cust


class AdaModel():
    def __init__(self, DataClean):
        self.DataClean = DataClean


    def model(self):
        X_train_cust, X_test_cust, y_train_cust, y_test_cust = self.DataClean.split_data()
        #Building model
        ada = AdaBoostRegressor(base_estimator = DecisionTreeRegressor(max_depth = 5), n_estimators = 200, learning_rate=0.1)
        ada.fit(X_train_cust, y_train_cust)
        return(ada)


    def optimize_model(self):

        ada = self.model()
        rate_range = np.arange(0.05, 0.1, 0.002)
        estimators_range = np.arange(50, 300, 50)
        param_grid = [{'learning_rate': rate_range, 'n_estimators': estimators_range}]


        gs = GridSearchCV(estimator=ada, param_grid=param_grid, cv=10, refit=True)
        gs.fit(X_train_cust, y_train_cust)

        return(gs)

    def evaluate_model(self, optimized_model):
        X_train_cust, X_test_cust, y_train_cust, y_test_cust = self.DataClean.split_data()
        mse_ada_cv = mean_squared_error(y_test_cust, optimized_model.predict(X_test_cust))

        return(mse_ada_cv)

    def feature_importances(self, optimized_model):

        try:
            feature_importance = optimized_model.feature_importances_
        except:
            feature_importance = optimized_model.best_estimator_.feature_importances_

        prediction_columns = [*self.DataClean.clean_data().columns]
        list_features = [*zip(prediction_columns, feature_importance)]

        db_features = pd.DataFrame(list_features, columns = ['feature', 'importance']).sort_values(by = "importance", ascending = False)
        return(db_features)

    def plot_feature_importance(self, optimized_model):
        fig = plt.figure(figsize=(15, 15))
        feature_importance = self.feature_importances(optimized_model)
        sns.set(rc={'figure.figsize':(11.7,8.27)})
        fig = sns.barplot(y="feature", x="importance", data=feature_importance, color = (18.37/100,51.02/100,30.61/100))
        _, ylabels = plt.yticks()

        fig.set_yticklabels(ylabels, size=15)

        plt.title('Feature importance for beer consumption', fontsize = 20)
        plt.xlabel('Importance', fontsize=15)
        plt.ylabel('', fontsize=16)

        plt.savefig('feature_importance.png', bbox_inches = 'tight')

DataClean_test = DataClean(prediction_columns, customers)
ada_model_test = AdaModel(DataClean_test)
model = ada_model_test.model()

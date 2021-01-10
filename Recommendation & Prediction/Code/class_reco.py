import dask.dataframe as dd
import json
import pandas as pd
from pprint import pprint
from pandas.io.json import json_normalize
import time
import numpy as np
from itertools import repeat
import datetime, time
from datetime import datetime
from dateutil import relativedelta
import sklearn
import seaborn as sns
from tslearn.utils import to_time_series
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tslearn.clustering import TimeSeriesKMeans
from pathlib import Path
from surprise import Reader, Dataset, Trainset
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
import gower
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
import datetime, time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer



material = pd.read_csv('/Users/Julien/Desktop/LBS/Cours/8 - TERM 3/London Lab/Data.nosync/Original_Data/material.csv')
material['material_id'] = material['material_id'].apply(lambda x: x.lower())
material = material.set_index('material_id')

l = ['avg_con_hourly', 'avg_con_daily', 'avg_con_monthly', 'months', 'total_consumption', 'center_or_not', 'classification_new', 'total_pop_in_province', 'total_pop_in_city','male_pop', 'female_pop', 'male_perc', 'female_perc', 'never_social_per_100', 'everyday_social_per_100', 'persons.aged.11.years.and.over.who.consume.wine', 'persons.aged.11.years.and.over..who.consume.beer', 'more.of.half.a.liter.of.wine.a.day', 'he.she.consumes.wine.more.rarely', 'he.she.consumes.beer.only.seasonally', 'he.she.consumes.beer.more.rarely', 'he.she.consumes.beer.everyday', 'X1.2.glasses.of.wine.a.day', 'sd', 'med_income', 'avg_income', 'ci_craft', 'ci_lager', 'ci_specialties', 'ci_craft_prop_beers', 'ci_lager_prop_beers', 'ci_specialties_prop_beers', 'ci_craft_prop_con', 'ci_lager_prop_con', 'ci_specialties_prop_con', 'number_of_brands', 'near_basketball', 'near_cinema', 'near_football', 'near_neighborhood', 'near_shopping', 'near_theatre', 'near_tourist', 'near_university', 'real_name', 'google_address', 'google_classification', 'rating', 'keyword_1', 'keyword_2', 'keyword_3', 'outdoor_seating']

customers = pd.read_excel('/Users/Julien/Desktop/LBS/Cours/8 - TERM 3/London Lab/Data.nosync/2 - Latest/newest_customer.xlsx')
customers = customers.set_index('crl')
customers = customers.drop(columns = 'Unnamed: 128')


beer_columns = [x for x in [*customers.columns] if x.startswith('b9')]
prediction_columns = ['outdoor_seating', 'rating', 'main_classification', 'total_pop_in_city', 'near_basketball', 'near_cinema', 'near_football','near_neighborhood', 'near_shopping', 'near_theatre', 'near_tourist', 'near_university', 'med_income', 'persons_aged_11_years_and_over_who_consume_beer', 'everyday_social_per_100', 'ci_craft_prop_con', 'ci_lager_prop_con', 'ci_craft_prop_beers', 'ci_lager_prop_beers', 'male_perc', 'center_or_not']
prediction_columns_new_outlet = [x for x in prediction_columns if x.startswith('ci') == False]

class DataCleanReco():
    def __init__(self, predictions_columns, customers):
        self.predictions_columns = predictions_columns

        self.all_columns = self.predictions_columns + ['total_consumption', 'avg_con_monthly']
        # self.customers = customers[self.all_columns].dropna()
        self.customers = customers[self.all_columns]

        self.customers_original = customers

    def test_for_cat(self, df):
        cat_var = []
        num_var = []
        prediction_columns = [*df.columns]
        for x in prediction_columns:
            if customers[x].dtype == object:
                cat_var.append(x)
            else:
                num_var.append(x)
        return(cat_var, num_var)

    def fill_missing_values(self):
        imp_num = SimpleImputer(strategy='mean')
        imp_cat = SimpleImputer(strategy='most_frequent')

        df = self.customers
        cat_var, num_var = self.test_for_cat(df)


        full_pipeline = ColumnTransformer([("num", imp_num, num_var), ("cat", imp_cat, cat_var)])
        df_new = pd.DataFrame(full_pipeline.fit_transform(df))
        df_new.columns = num_var + cat_var
        df_new.index = df.index
        return(df_new)


    def clean_data(self):
        customers_cleaned = self.fill_missing_values()
        # customers_cleaned = self.customers[self.all_columns].dropna()
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


    # def clean_data_v2(self):
    #     imp_num = SimpleImputer(strategy='mean')
    #     imp_cat = SimpleImputer(strategy='mean')
    #
    #     self.customers['rating'] = imp_mean.fit_transform(pd.DataFrame(self.customers['rating']).values)
    #
    #     customers_cleaned = self.customers[self.all_columns].dropna()
    #     customers_cleaned['outdoor_seating'] = customers_cleaned['outdoor_seating'].apply(lambda x: x.strip())
    #
    #     #Ordinal Variables
    #     columns_to_ordinal = ['near_basketball', 'near_cinema', 'near_football', 'near_neighborhood', 'near_shopping', 'near_theatre', 'near_tourist', 'near_university', 'center_or_not', 'outdoor_seating']
    #     enc = OrdinalEncoder()
    #     after_pipelined = enc.fit_transform(customers_cleaned[columns_to_ordinal])
    #     customers_cleaned[columns_to_ordinal] = after_pipelined
    #
    #     #Categorical Variables
    #     categorical_variables = ['main_classification']
    #
    #     customers_cleaned = pd.get_dummies(data=customers_cleaned[self.predictions_columns], columns=categorical_variables)
    #     return(customers_cleaned)



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

class RecoSystem():
    def __init__(self, DataCleanReco, n_neighbors, n_beers):
        self.DataCleanReco = DataCleanReco
        self.data = self.DataCleanReco.clean_data()
        self.customers_original = DataCleanReco.customers_original
        self.beer_cols = [*filter(lambda string: string.startswith("b9"), self.customers_original.columns)]
        self.beer_cols_recommended = [x+'_reco' for x in self.beer_cols]
        self.n_neighbors = n_neighbors
        self.n_beers = n_beers

    def get_gower_matrix(self):
        distances = gower.gower_matrix(self.data)
        distances = pd.DataFrame(distances, index=self.data.index)
        distances.columns = distances.index
        distances=distances.replace(0, 1000)
        return(distances)

    def find_n_neighbours(self, n):
        df = self.get_gower_matrix()
        order = np.argsort(df.values, axis=1)[:, :n]
        df = df.apply(lambda x: pd.Series(x.sort_values(ascending=True)
               .iloc[:n].index,
              index=['top{}'.format(i) for i in range(1, n+1)]), axis=1)
        return df

    def get_reco_db(self, crl):
        x = crl
        neighbors = self.find_n_neighbours(self.n_neighbors)
        df = neighbors.loc[x].values
        df_cust = self.customers_original[self.customers_original.index.isin(df)][self.beer_cols]
        df_real = self.customers_original[self.customers_original.index.isin([x])][self.beer_cols]
        df_cust.columns = self.beer_cols_recommended
        totals = df_cust.sum(axis=0)
        totals_mean = df_cust.mean(axis=0)
        totals_real = df_real.sum(axis=0)

        n = self.n_beers
        sorted_recommended = totals_mean.sort_values(ascending=False)[:n]
        sorted_recommended_values = totals_mean.sort_values(ascending=False)[:n].values
        sorted_real = totals_real.sort_values(ascending=False)[:n]
        sorted_real_values = totals_real.sort_values(ascending=False)[:n].values
        both_compared = df = pd.DataFrame(columns=['real', 'recommended', 'real_values', 'recommended_values'])
        both_compared['real'] = sorted_real.index
        both_compared['recommended'] = sorted_recommended.index
        both_compared['recommended_2'] = both_compared['recommended'].apply(lambda x: x.split('_')[0])
        both_compared['real_values'] = sorted_real_values
        both_compared['recommended_values'] = sorted_recommended_values
        both = pd.concat([totals_mean, totals_real], axis = 0)
        both_3 = pd.DataFrame(both)
        both_3 = both_3.reset_index()
        both_3['index'] = both_3['index'].apply(lambda x: x.split('_')[0])
        both_2 = both_3.groupby('index').max().reset_index()
        both_2.columns = both_2.columns.astype(str)
        both_2 = both_2.rename({'0' : 'value'})
        both_2.columns = ['beer', 'value']

        best_combination = both.sort_values(ascending=False)[:n]
        both_compared['optimal_combination'] = best_combination.index
        dict_1 = dict(zip(both_compared.real,both_compared.real_values))
        dict_2 = dict(zip(both_compared.recommended,both_compared.recommended_values))
        z = {**dict_1, **dict_2}
        both_compared['optimal_values'] = both_compared['optimal_combination'].apply(lambda x: z[x])

        both_compared['real_name'] = both_compared['real'].apply(lambda x: material.loc[x.split('_')[0]]['material'])
        both_compared['recommended_name'] = both_compared['recommended'].apply(lambda x: material.loc[x.split('_')[0]]['material'])
        both_compared['optimal_combination_name'] = both_compared['optimal_combination'].apply(lambda x: material.loc[x.split('_')[0]]['material'])
        return(both_compared)

    def get_recommended_beers(self, crl):
        reco_db = self.get_reco_db_v2(crl)
        reco_db['optimal_combination_stripped'] = reco_db['optimal_combination'].apply(lambda x: x.split('_')[0])
        reco_db['optimal_combination_names'] = reco_db['optimal_combination_stripped'].apply(lambda x: material.loc[x]['material'])
        return(reco_db['optimal_combination_names'])

    def my_func(self, db):
        index = [*db.index]
        index_filtered = [*filter(lambda string: string.endswith("reco"),index)]
        index_filtered_split = [x.split('_')[0] for x in index_filtered]
        for x in index_filtered:
            db_row = db.loc[x]
            value_reco = db_row['value']
            real = x.split('_')[0]
            real_value = db.loc[real]['value']
            if real_value != float(0):
                db = db.drop(x)
        return(db)

    def get_reco_db_v2(self, crl):
        x = crl
        neighbors = self.find_n_neighbours(self.n_neighbors)
        df = neighbors.loc[x].values
        df_cust = self.customers_original[self.customers_original.index.isin(df)][self.beer_cols]
        df_real = self.customers_original[self.customers_original.index.isin([x])][self.beer_cols]
        df_cust.columns = self.beer_cols_recommended
        totals = df_cust.sum(axis=0)
        totals_mean = df_cust.mean(axis=0)
        totals_real = df_real.sum(axis=0)

        n = self.n_beers #Top n beers
        sorted_recommended = totals_mean.sort_values(ascending=False)
        sorted_recommended_values = totals_mean.sort_values(ascending=False).values
        sorted_real = totals_real.sort_values(ascending=False)
        sorted_real_values = totals_real.sort_values(ascending=False).values

        sorted_recommended_n = totals_mean.sort_values(ascending=False)[:n]
        sorted_recommended_values_n = totals_mean.sort_values(ascending=False)[:n].values
        sorted_real_n = totals_real.sort_values(ascending=False)[:n]
        sorted_real_values_n = totals_real.sort_values(ascending=False)[:n].values


        concat_all = pd.concat([sorted_recommended, sorted_real], axis = 0)

        concat_all = pd.DataFrame(concat_all)
        concat_all.columns = ['value']
        concat_all_v2 = self.my_func(concat_all)
        concat_all_v2 = concat_all_v2.sort_values(ascending = False, by = 'value')


        both_compared = df = pd.DataFrame(columns=['real', 'recommended', 'real_values', 'recommended_values'])
        both_compared['real'] = sorted_real_n.index
        both_compared['recommended'] = sorted_recommended_n.index
        # both_compared['recommended_2'] = both_compared['recommended'].apply(lambda x: x.split('_')[0])
        both_compared['real_values'] = sorted_real_values_n
        both_compared['recommended_values'] = sorted_recommended_values_n
        # # print(both_compared)
        both = pd.concat([totals_mean, totals_real], axis = 0)

        best_combination = concat_all_v2[:n]
        both_compared['optimal_combination'] = best_combination.index
        dict_1 = dict(zip(both_compared.real,both_compared.real_values))
        dict_2 = dict(zip(both_compared.recommended,both_compared.recommended_values))
        z = {**dict_1, **dict_2}
        both_compared['optimal_values'] = both_compared['optimal_combination'].apply(lambda x: z[x])
        both_compared['real_name'] = both_compared['real'].apply(lambda x: material.loc[x.split('_')[0]]['material'])
        both_compared['recommended_name'] = both_compared['recommended'].apply(lambda x: material.loc[x.split('_')[0]]['material'])
        both_compared['optimal_combination_name'] = both_compared['optimal_combination'].apply(lambda x: material.loc[x.split('_')[0]]['material'])
        return(both_compared)

    def get_similar_outlets_db(self, crl):
        db_clean = self.DataCleanReco.customers_original
        outlet_neighbors = self.find_n_neighbours(10).loc[crl]
        outlet_neighbors = outlet_neighbors.reset_index()
        outlet_neighbors.columns = ['index', 'crl']
        outlet_neighbors = outlet_neighbors.set_index('crl')
        similar_outlets = outlet_neighbors.join(db_clean, how = 'left')
        return(similar_outlets)

class RecoSystemNewOutlet():
    def __init__(self, DataCleanReco, n_neighbors, n_beers):
        self.DataCleanReco = DataCleanReco
        self.data = self.DataCleanReco.clean_data()
        self.customers_original = DataCleanReco.customers_original
        self.beer_cols = [*filter(lambda string: string.startswith("b9"), self.customers_original.columns)]
        self.beer_cols_recommended = [x+'_reco' for x in self.beer_cols]
        self.n_neighbors = n_neighbors
        self.n_beers = n_beers

    def my_func(self, db):
        index = [*db.index]
        index_filtered = [*filter(lambda string: string.endswith("reco"),index)]
        index_filtered_split = [x.split('_')[0] for x in index_filtered]
        for x in index_filtered:
            db_row = db.loc[x]
            value_reco = db_row['value']
            real = x.split('_')[0]
            real_value = db.loc[real]['value']
            if real_value != float(0):
                db = db.drop(x)
        return(db)

    def get_reco_db_new_outlet(self, crl):
        x = crl
        neighbors = self.find_n_neighbours(self.n_neighbors)
        df = neighbors.loc[x].values
        df_cust = self.customers_original[self.customers_original.index.isin(df)][self.beer_cols]
        df_cust.columns = self.beer_cols_recommended
        totals = df_cust.sum(axis=0)
        totals_mean = df_cust.mean(axis=0)
        n = self.n_beers #Top n beers
        sorted_recommended = totals_mean.sort_values(ascending=False)
        sorted_recommended_values = totals_mean.sort_values(ascending=False).values
        sorted_recommended_n = totals_mean.sort_values(ascending=False)[:n]
        sorted_recommended_values_n = totals_mean.sort_values(ascending=False)[:n].values
        sorted_recommended = pd.DataFrame(sorted_recommended).reset_index()
        sorted_recommended.columns = ['beer', 'value']
        concat_all_v2 = sorted_recommended
        concat_all_v2 = concat_all_v2.sort_values(ascending = False, by = 'value')
        both_compared = df = pd.DataFrame(columns=['recommended', 'recommended_values'])
        both_compared['recommended'] = sorted_recommended_n.index
        both_compared['recommended_values'] = sorted_recommended_values_n
        dict_2 = dict(zip(both_compared.recommended,both_compared.recommended_values))
        both_compared['recommended_name'] = both_compared['recommended'].apply(lambda x: material.loc[x.split('_')[0]]['material'])
        return(both_compared)

    def get_gower_matrix(self):
        distances = gower.gower_matrix(self.data)
        distances = pd.DataFrame(distances, index=self.data.index)
        distances.columns = distances.index
        distances=distances.replace(0, 1000)
        return(distances)

    def find_n_neighbours(self, n):
        df = self.get_gower_matrix()
        order = np.argsort(df.values, axis=1)[:, :n]
        df = df.apply(lambda x: pd.Series(x.sort_values(ascending=True)
               .iloc[:n].index,
              index=['top{}'.format(i) for i in range(1, n+1)]), axis=1)
        return df

my_class = DataCleanReco(prediction_columns_new_outlet, customers)
my_next_class = RecoSystem(my_class, 10, 5)

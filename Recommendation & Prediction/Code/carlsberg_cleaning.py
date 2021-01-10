import dask.dataframe as dd
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
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pathlib import Path
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (20, 20)
sns.set(rc={'figure.figsize':(15,10)})




material = pd.read_csv('/Users/Julien/Desktop/LBS/Cours/8 - TERM 3/London Lab/Data.nosync/Original_Data/material.csv')
material = material.set_index("material_id")
customers_latest = pd.read_csv('/Users/Julien/Desktop/LBS/Cours/8 - TERM 3/London Lab/Data.nosync/2 - Latest/customers_latest.csv')
customers_latest = customers_latest.rename(columns={"crl": "CRL"})

def parser(file):
    data_cleaned = pd.DataFrame()
    with open(file) as f:
        data = json.load(f)
        max_length_list = [len(data[x]) for x in data.keys()]
        inital_list = [max(data[x][0]) for x in data.keys()]
        if len(max_length_list)>0:
            argmax = np.argmax(max_length_list)
            keys = list(data.keys())
            argmax = keys[argmax]
            tot_consumption = []
            for x in data.keys():
                df_temp = data[x]
                b = map(lambda x: x[1], df_temp) #Old Version
                b = [x[1] for x in df_temp]
                consumption = [*b]
                tot_consumption.append(consumption)
            # date_col = [*map(lambda x: pd.to_datetime(str(x[0])[:-3], unit = 's').strftime('%Y-%m-%d'), data[argmax])]
            date_col = [*map(lambda x: pd.to_datetime(str(x[0])[:-3], unit = 's').strftime('%Y-%m-%d %H:%M:%S %Z'), data[argmax])]

            # t = pd.to_datetime(test, unit = "s")
            # t = t.strftime('%Y-%m-%d')
            series = [*map(lambda x: pd.Series(x), tot_consumption)]
            data_cleaned = pd.concat(series, join='outer', axis=1)
            data_cleaned.columns = list(data.keys())
            data_cleaned['date'] = date_col
        else:
            data_cleaned = pd.DataFrame()
            #print("no data")

    return(data_cleaned)


destdir = Path('/Users/Julien/Desktop/LBS/Cours/8 - TERM 3/London Lab/Data.nosync/Original_Data/db')
files = [p for p in destdir.iterdir() if p.is_file()]
files = [*map(str, files)]
new_files = [*filter(lambda string: string.endswith(".json"), files)]
test = '/Users/Julien/Desktop/LBS/Cours/8 - TERM 3/London Lab/Data.nosync/CSV/CRL2920085759.csv'


destdir_csv = Path('/Users/Julien/Desktop/LBS/Cours/8 - TERM 3/London Lab/Data.nosync/CSV')
files_csv = [p for p in destdir_csv.iterdir() if p.is_file()]
files_csv = [*map(str, files_csv)]
final = [*filter(lambda string: string.split('/')[10].startswith("CRL"), files_csv)]


def id_parser(string):
    pattern = '[/w-]+?(?=/.)'
    pattern_2 = '^//(.+//)*(.+)/.(.+)$'
    end_path = os.path.split(string)
    id = end_path[1].split('.')[0]
    return(id)

ids = [*map(lambda string: id_parser(string), new_files)]

def json_to_csv():
    for file in new_files:
        df = parser(file)
        id = id_parser(file)
        df.to_csv('{}.csv'.format(id))

# material = material.set_index("material_id")
def agg(new_files):
    today = datetime.today()
    export_df = pd.DataFrame()
    beer_df = pd.DataFrame()
    for file in new_files:
        df = parser(file)

        #print(df.info())
        id = id_parser(file)
        cols = [*df.columns]
        if cols:
            df['date'] = pd.to_datetime(df['date'])
            cols_1 = [*filter(lambda string: string.startswith("B"), cols)]
            cols_2 = [*map(lambda col: col.rstrip(), cols_1)]
            type = [*map(lambda col: material.loc[col.rstrip()]['subgroup'], cols_2)]
            type_dict = dict(zip(cols_2, type))

            #All beers
            all_beers = pd.DataFrame(df[cols_1].agg(np.sum)).transpose()

            #Aggregate Data
            df_melted = pd.melt(df, id_vars = 'date', value_vars = cols_2)


            #print(all_beers)
            dict_apply = df_melted.apply(
            lambda row: type_dict[row['variable']], axis=1)

            df_melted["beer_type"] = dict_apply

            # print(len(df_melted))
            min_date = min(df['date'])
            difference = relativedelta.relativedelta(today, min_date)

            n_months = max(difference.months, 1)

            total_con = df_melted['value'].sum()
            average_con = df_melted['value'].mean()
            con_per_month = total_con/n_months
            standard_dev = np.std(df_melted['value'])

            all_beers = all_beers.apply(lambda x: x/n_months)
            s = np.sum(all_beers,axis=1)


            pivot_types = df_melted.pivot_table(values="value", index="beer_type", aggfunc=[np.sum])
            pivot_df = pd.DataFrame(pivot_types.transpose())
            pivot_df['CRL'] = id
            pivot_df['total_consumption'] = total_con
            pivot_df['average_consumption'] = average_con
            pivot_df['con_per_month'] = con_per_month
            pivot_df['sd'] = standard_dev

            beer_df = pd.concat([beer_df, all_beers], ignore_index = True)
            export_df = pd.concat([export_df, pivot_df], ignore_index = True)



            test = pd.concat([export_df, beer_df], axis=1, sort=False)
            test = test.fillna(0)
    return(test)

def agg_v2():
    destdir_csv = Path('/Users/Julien/Desktop/LBS/Cours/8 - TERM 3/London Lab/Data.nosync/CSV')
    files_csv = [p for p in destdir_csv.iterdir() if p.is_file()]
    files_csv = [*map(str, files_csv)]
    final = [*filter(lambda string: id_parser(string).startswith("CRL"), files_csv)]

    today = datetime.today()
    export_df = pd.DataFrame()
    beer_df = pd.DataFrame()
    for file in final:
        df = pd.read_csv(file)

        #print(df.info())
        id = id_parser(file)
        cols = [*df.columns]
        cols.remove('Unnamed: 0')
        if cols:
            df['date'] = pd.to_datetime(df['date'])
            cols_1 = [*filter(lambda string: string.startswith("B"), cols)]
            cols_2 = [*map(lambda col: col.rstrip(), cols_1)]
            type = [*map(lambda col: material.loc[col.rstrip()]['subgroup'], cols_2)]
            type_dict = dict(zip(cols_2, type))

            #All beers
            all_beers = pd.DataFrame(df[cols_1].agg(np.sum)).transpose()
            all_beers_2 = pd.DataFrame(df[cols_1].agg(np.mean)).transpose()
            all_beers_3 = pd.DataFrame(df[cols_1])
            print(all_beers_3)
            #print(np.mean(all_beers_2.transpose()))
            print('next')
            #Aggregate Data
            df_melted = pd.melt(df, id_vars = 'date', value_vars = cols_2)

            no_nas = df_melted.dropna()


            #print(all_beers)
            dict_apply = df_melted.apply(
            lambda row: type_dict[row['variable']], axis=1)

            df_melted["beer_type"] = dict_apply

            # print(len(df_melted))
            min_date = min(df['date'])

            max_date = max(df['date'])
            num_months = (max_date. year - min_date. year) * 12 + (max_date. month - min_date. month)
            difference = relativedelta.relativedelta(max_date, min_date)

            n_months = max(difference.months, 1)

            total_con = df_melted['value'].sum()
            average_con = df_melted['value'].mean()
            con_per_month = total_con/num_months
            standard_dev = np.std(df_melted['value'])

            all_beers = all_beers.apply(lambda x: x/num_months)
            s = np.sum(all_beers,axis=1)


            pivot_types = df_melted.pivot_table(values="value", index="beer_type", aggfunc=[np.sum])
            pivot_df = pd.DataFrame(pivot_types.transpose())
            pivot_df['CRL'] = id
            pivot_df['total_consumption'] = total_con
            pivot_df['average_consumption'] = average_con
            pivot_df['months'] = num_months
            pivot_df['con_per_month'] = con_per_month
            pivot_df['sd'] = standard_dev

            beer_df = pd.concat([beer_df, all_beers], ignore_index = True)
            export_df = pd.concat([export_df, pivot_df], ignore_index = True)



            test = pd.concat([export_df, beer_df], axis=1, sort=False)
            test = test.fillna(0)
    return(test)

def clr_cols(new_files):
    today = datetime.today()
    total_df = pd.DataFrame()
    #beer_df = pd.DataFrame()
    #clrs = [*map(lambda x: id_parser(x), new_files)]
    clrs = []
    for file in new_files:
        df = parser(file)

        id = id_parser(file)
        cols = [*df.columns]
        if cols:
            clrs.append(id)
            df['date'] = pd.to_datetime(df['date'])
            cols_1 = [*filter(lambda string: string.startswith("B"), cols)]
            cols_2 = [*map(lambda col: col.rstrip(), cols_1)]
            type = [*map(lambda col: material.loc[col.rstrip()]['subgroup'], cols_2)]
            type_dict = dict(zip(cols_2, type))

            #All beers
            all_beers = pd.DataFrame(df[cols_1].agg(np.sum, axis = "columns"))
            df_func = pd.DataFrame(df[cols_1].apply(lambda x: np.mean(x), axis=1))
            df_func['date'] = df['date']
            df_func = df_func.set_index('date')
            total_df = pd.concat([total_df, df_func], axis=1, ignore_index = False)
            # #df_func = df_func.set_index('date_index')
            # total_df_temp = pd.concat([df_temp, df_func])
            # #total_df = total_df.join(df_func, how='outer')
            # total_df = total_df.merge(total_df_temp, how = "outer", on = "date")
    total_df.columns = clrs
    total_df = total_df.fillna(0)
    return(total_df)

def clr_cols_v2():
    today = datetime.today()
    total_df = pd.DataFrame()
    clrs = []
    for file in final:
        df = pd.read_csv(file)
        id = id_parser(file)
        cols = [*df.columns]
        cols.remove('Unnamed: 0')
        if cols:
            clrs.append(id)
            df['date'] = pd.to_datetime(df['date'])
            cols_1 = [*filter(lambda string: string.startswith("B"), cols)]
            cols_2 = [*map(lambda col: col.rstrip(), cols_1)]
            type = [*map(lambda col: material.loc[col.rstrip()]['subgroup'], cols_2)]
            type_dict = dict(zip(cols_2, type))
            all_beers = pd.DataFrame(df[cols_1].agg(np.sum, axis = "columns"))
            df_func = pd.DataFrame(df[cols_1].apply(lambda x: np.mean(x), axis=1))
            df_func['date'] = df['date']
            df_func = df_func.set_index('date')
            total_df = pd.concat([total_df, df_func], axis=1, ignore_index = False)
    total_df.columns = clrs
    total_df = total_df.fillna(0)
    return(total_df)



destdir_csv = Path('/Users/Julien/Desktop/LBS/Cours/8 - TERM 3/London Lab/Data.nosync/CSV')
files_csv = [p for p in destdir_csv.iterdir() if p.is_file()]
files_csv = [*map(str, files_csv)]
new_files_csv = [*filter(lambda string: id_parser(string).startswith("CRL"), files_csv)]


def parser_concat_hourly():
    destdir_csv = Path('/Users/Julien/Desktop/LBS/Cours/8 - TERM 3/London Lab/Data.nosync/CSV')
    files_csv = [p for p in destdir_csv.iterdir() if p.is_file()]
    files_csv = [*map(str, files_csv)]
    new_files_csv = [*filter(lambda string: id_parser(string).startswith("CRL"), files_csv)]

    final_df = pd.DataFrame()
    for dir in new_files_csv:
        df =  pd.read_csv(dir)
        cols = [* df.columns]
        cols.remove('Unnamed: 0')
        cols = [*filter(lambda string: string.startswith("B"), cols)]
        #cols = [*map(lambda col: col.rstrip(), cols)]
        if cols:
            cols.append('date')
            df['date'] = pd.to_datetime(df['date'])
            df['date_index'] = pd.to_datetime(df['date'])
            df.set_index('date_index')
            df.index = pd.to_datetime(df['date_index'])
            agg = df.resample('H').sum()
            agg['date'] = agg.index.values
            agg['CRL'] = id_parser(dir)
            cols.append('CRL')


            #print(agg)
            final_df = pd.concat([final_df, agg[cols]], ignore_index=True)
        #df_melted = pd.melt(final_df, id_vars = ['date'], value_vars = cols)
    return(final_df)

def get_hourly_barchart():
    final_df = parser_concat_hourly()
    final_df_2 = final_df.set_index('date')
    cols_final = [*final_df.columns]
    cols_final_beers = [x for x in cols_final if x.startswith('B9') == True]
    test = final_df_2[cols_final_beers].mean(axis = 1, skipna=True)
    test = pd.DataFrame(test)
    test.columns = test.columns.astype(str)
    test.index = test.index.hour
    me = test.groupby('date').mean()
    fig, ax = plt.subplots( nrows=1, ncols=1 )
    ax.bar(me.index, me['0'], color = (18.37/100,51.02/100,30.61/100))
    ax.tick_params(axis='both', which='major', labelsize=30)
    ax.set_title('Average beer consumption by hour of day', fontsize = 35)
    fig.savefig('consumption_by_hour.png')

def get_hourly_barchart_sum():
    final_df = parser_concat_hourly()
    final_df_2 = final_df.set_index('date')
    cols_final = [*final_df.columns]
    cols_final_beers = [x for x in cols_final if x.startswith('B9') == True]
    test = final_df_2[cols_final_beers].mean(axis = 1, skipna=True)
    test = pd.DataFrame(test)
    test.columns = test.columns.astype(str)
    test.index = test.index.hour
    me = test.groupby('date').sum()
    fig, ax = plt.subplots( nrows=1, ncols=1 )
    ax.bar(me.index, me['0'], color = (18.37/100,51.02/100,30.61/100))
    ax.tick_params(axis='both', which='major', labelsize=30)
    ax.set_title('Total beer consumption by hour of day', fontsize = 35)
    fig.savefig('total_consumption_by_hour.png')

def parser_concat_weekly():
    destdir_csv = Path('/Users/Julien/Desktop/LBS/Cours/8 - TERM 3/London Lab/Data.nosync/CSV')
    files_csv = [p for p in destdir_csv.iterdir() if p.is_file()]
    files_csv = [*map(str, files_csv)]
    new_files_csv = [*filter(lambda string: id_parser(string).startswith("CRL"), files_csv)]

    final_df = pd.DataFrame()
    for dir in new_files_csv:
        df =  pd.read_csv(dir)
        cols = [* df.columns]
        cols.remove('Unnamed: 0')
        cols = [*filter(lambda string: string.startswith("B"), cols)]
        #cols = [*map(lambda col: col.rstrip(), cols)]
        if cols:
            cols.append('date')
            df['date'] = pd.to_datetime(df['date'])
            df['date_index'] = pd.to_datetime(df['date'])
            df.set_index('date_index')
            df.index = pd.to_datetime(df['date_index'])
            agg = df.resample('W').sum()
            agg['date'] = agg.index.values
            agg['CRL'] = id_parser(dir)
            cols.append('CRL')


            #print(agg)
            final_df = pd.concat([final_df, agg[cols]], ignore_index=True)
        #df_melted = pd.melt(final_df, id_vars = ['date'], value_vars = cols)
    return(final_df)

def parser_concat_daily():

    destdir_csv = Path('/Users/Julien/Desktop/LBS/Cours/8 - TERM 3/London Lab/Data.nosync/CSV')
    files_csv = [p for p in destdir_csv.iterdir() if p.is_file()]
    files_csv = [*map(str, files_csv)]
    new_files_csv = [*filter(lambda string: id_parser(string).startswith("CRL"), files_csv)]

    final_df = pd.DataFrame()
    for dir in new_files_csv:
        df =  pd.read_csv(dir)
        cols = [* df.columns]
        cols.remove('Unnamed: 0')
        cols = [*filter(lambda string: string.startswith("B"), cols)]
        #cols = [*map(lambda col: col.rstrip(), cols)]
        if cols:
            cols.append('date')
            df['date'] = pd.to_datetime(df['date'])
            df['date_index'] = pd.to_datetime(df['date'])
            df.set_index('date_index')
            df.index = pd.to_datetime(df['date_index'])
            agg = df.resample('D').sum()
            agg['date'] = agg.index.values
            agg['CRL'] = id_parser(dir)
            cols.append('CRL')


            #print(agg)
            final_df = pd.concat([final_df, agg[cols]], ignore_index=True)
        #df_melted = pd.melt(final_df, id_vars = ['date'], value_vars = cols)
    return(final_df)

def final_csv():
    a=parser_concat_hourly()
    cols = [* a.columns]
    cols = [*filter(lambda string: string.startswith("B"), cols)]
    test = pd.melt(a, id_vars = ['date', 'CRL'], value_vars = cols)

    type_2 = [*map(lambda col: material.loc[col.rstrip()]['material'], cols)]
    type_dict_2 = dict(zip(cols, type_2))

    dict_apply_2 = test.apply(
    lambda row: type_dict_2[row['variable']], axis=1)
    test["beer_name"] = dict_apply_2
    test.to_csv('final_csv_hourly.csv')
    return(test)

def time_series_agg_long_percentile(): #Weekly long database for beers over certain percentile (percentile is done by summing all consumption for beers)
    a=parser_concat_weekly()
    cols = a.columns
    #print(cols)
    cols_1 = [*filter(lambda string: string.startswith("B"), cols)]
    cols_2 = [*map(lambda col: col.rstrip(), cols_1)]
    #print(cols_2)
    df_melted = pd.melt(a, id_vars = 'date', value_vars = cols_2)
    df_melted = df_melted.fillna(0)

    type = [*map(lambda col: material.loc[col.rstrip()]['subgroup'], cols_2)]
    type_dict = dict(zip(cols_2, type))

    type_2 = [*map(lambda col: material.loc[col.rstrip()]['material'], cols_2)]
    type_dict_2 = dict(zip(cols_2, type_2))

    dict_apply = df_melted.apply(
    lambda row: type_dict[row['variable']], axis=1)
    df_melted["beer_type"] = dict_apply

    dict_apply_2 = df_melted.apply(
    lambda row: type_dict_2[row['variable']], axis=1)
    df_melted["beer_name"] = dict_apply_2

    #print(df_melted.info())
    grouped = df_melted.groupby("variable").sum()
    grouped = grouped.reset_index()
    p = np.percentile(grouped['value'], 60)
    grouped_filtered = grouped[grouped['value'] > p]
    #print(grouped_filtered.info())
    grouped_filtered_list = [*grouped_filtered['variable']]
    grouped_filtered_beers = df_melted[df_melted['variable'].isin(grouped_filtered_list)]

    return(grouped_filtered_beers)

def time_series_weekly_sum_by_beer(): #Weekly long database for all beers with beer type
    a=parser_concat_weekly()
    cols = a.columns
    #print(cols)
    cols_1 = [*filter(lambda string: string.startswith("B"), cols)]
    cols_2 = [*map(lambda col: col.rstrip(), cols_1)]
    df_melted = pd.melt(a, id_vars = 'date', value_vars = cols_2)
    df_melted = df_melted.fillna(0)
    type = [*map(lambda col: material.loc[col.rstrip()]['subgroup'], cols_2)]
    type_dict = dict(zip(cols_2, type))

    type_2 = [*map(lambda col: material.loc[col.rstrip()]['material'], cols_2)]
    type_dict_2 = dict(zip(cols_2, type))

    dict_apply = df_melted.apply(
    lambda row: type_dict[row['variable']], axis=1)

    df_melted["beer_type"] = dict_apply
    return(df_melted)

def cum_sum_time_series_by_beer(): #Long database with cumulative sum by beer
    df = time_series_weekly_sum_by_beer()
    df2 = df.groupby(by=['variable','date']).sum().groupby(level=[0]).cumsum()
    df2 = df2.reset_index()
    cols = set(df['variable'].values)
    cols_1 = [*filter(lambda string: string.startswith("B"), cols)]
    cols_2 = [*map(lambda col: col.rstrip(), cols_1)]

    type_2 = [*map(lambda col: material.loc[col.rstrip()]['material'], cols_2)]
    type_dict_2 = dict(zip(cols_2, type_2))

    dict_apply_2 = df2.apply(
    lambda row: type_dict_2[row['variable']], axis=1)

    df2["beer_name"] = dict_apply_2
    # sns.set(rc={'figure.figsize':(11.7,8.27)})
    #
    # ax = sns.lineplot(x="date", y="value", hue="beer_name", data=df2, ci=False)
    return(df2)



ax = sns.lineplot(x="date", y="value", hue="variable", data=cum_sum, ci=False)




beer_type_dict = pd.Series(material['subgroup'].values,index=material.index).to_dict()

final_csv = pd.read_csv('/Users/Julien/Desktop/LBS/Cours/8 - TERM 3/London Lab/Data.nosync/CSV/final_csv.csv')
test = final_csv[final_csv['CRL'] == 'CRL3000082685']
test_2 = test[test['variable'] == "B9LRI"]

final_csv = final_csv.dropna()


cols = [*final_csv.columns]
cols_v2 = [*filter(lambda string: string.startswith("Unn")==False, cols)]
final_csv = final_csv[cols_v2]
variables_to_use = set([s.rstrip() for s in np.unique(final_csv['variable'])])
final_csv = final_csv[final_csv['variable'].isin(variables_to_use)]
final_csv['beer_type'] = final_csv['variable'].map(beer_type_dict)

cols = [*customers_latest.columns]
cols_v2 = [*filter(lambda string: string.startswith("b9")==False, cols)]
cols_v3 = [*filter(lambda string: string.startswith("Unn")==False, cols_v2)]

merged = pd.merge(final_csv, customers_latest[cols_v3], on = 'CRL', how = 'left')
merged.to_csv('merged_final_customers_latest_no_nas.csv')

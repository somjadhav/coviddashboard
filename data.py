import requests
import pandas as pd
import numpy as np
from openpyxl import load_workbook
import datetime

'''
def get_data():

    url = "https://dshs.texas.gov/coronavirus/TexasCOVID19DailyCountyCaseCountData.xlsx"
    file = requests.get(url)
    open('datafile.xlsx','wb').write(file.content)

    df = pd.DataFrame(pd.read_excel('datafile.xlsx'))

    print(df.head())
'''

def get_data():
    df = pd.read_csv('datafile.csv')

    df = df.drop(['Cases 03-04-2020', 'Cases 03-05-2020'], axis=1)
    df_dates = []
    for val in list(df.columns)[1:]:
        time_str = val[6:]
        time = datetime.datetime.strptime(time_str, '%m-%d-%Y')
        df_dates.append(time)

    df = df.transpose()
    county_names = df.iloc[0].to_list()
    df.columns = county_names
    df = df.iloc[1:]
    df.insert(0, 'Date', df_dates)
    df.reset_index(inplace=True, drop=True)

    return df
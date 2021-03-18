import requests
import pandas as pd
import numpy as np
from openpyxl import load_workbook

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

    print(df.head())

    df = df.drop(['Cases 03-04-2020', 'Cases 03-05-2020'], axis=1)

    return df
import pandas as pd
import numpy as np
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from data import get_data

app = dash.Dash(__name__)

df = get_data()
df.reset_index(inplace=True)

options_dict = {}


# App layout
app.layout = html.Div([

    html.H1('Texas Corovirus Cases Dashboard', style={'text-align':'center'}),

    dcc.Dropdown(id='select-county',
            options = [
                
            ])
])

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

options_list = []
for index, row in df.iterrows():
    county_dict = {"label":row['County Name'], "value":row['County Name']}
    options_list.append(county_dict)



# App layout
app.layout = html.Div([

    html.H1('Texas Corovirus Cases Dashboard', style={'text-align':'center'}),

    dcc.Dropdown(id='select-county',
            options = options_list,
            multi=False,
            value="Total",
            style={"width":"50%"}
            ),
    
    html.Br(),

    dcc.Graph(id='cum_cases_graph', figure={}),
    
    html.Br(),

    dcc.Graph(id='new_cases_graph', figure={})
  
])

@app.callback(
    [Output(component_id='cum_cases_graph', component_property='figure'),
    Output(component_id='new_cases_graph', component_property='figure')],
    [Input(component_id='select-county', component_property='value')]
)
def update_graph(county_selected):

    df_copy = df.copy()
    df_copy = df_copy[df_copy['County Name'] == county_selected]

    df_new_cases = df_copy.copy()
    

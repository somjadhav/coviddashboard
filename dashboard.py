import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from data import get_data
from models import *
from LSTM import *

# create instance of Dash app
app = dash.Dash(__name__)

# get data
df = get_data()

# create dropdown list of Counties
options_list = []
options_list.append({"label":"Texas", "value":"Total"})
for name in list(df.columns)[1:len(list(df.columns))-1]:
    county_dict = {"label":str(name), "value":str(name)}
    options_list.append(county_dict)

# get predictions for every county
preds_dict = {}    


# App layout
app.layout = html.Div([

    html.H1('Texas Coronavirus Cases Dashboard', style={'text-align':'center'}),

    html.Div([
        html.Div([
            dcc.Dropdown(id='select-county',
                options = options_list,
                multi=False,
                value="Total",
                style={'width':'65%'}
            )   

        ], style={"width":"30%", "display":"inline-block", "align-items":"center", "justify-content":"center"}),

        html.Div([
            dcc.Dropdown(id='graph-type',
                options = [
                    {"label":"Cumulative Cases", "value":"Cumulative"},
                    {"label":"New Cases", "value":"New"}
                ],
                multi=False,
                value="Cumulative",
                style={'width':'70%'}
            )

        ], style={"width":"30%", "display":"inline-block", "align-items":"center", "justify-content":"center"}),

        html.Div([
            dcc.Dropdown(id='pred-type',
                options = [
                    #{"label":"All", "value":"All"},
                    {"label":"None", "value":"None"},
                    {"label":"LSTM", "value":"LSTM"}
                ],
                multi=False,
                value="None",
                style={'width':'65%'}
            )
        ], style={"width":"30%", "display":"inline-block", "align-items":"center", "justify-content":"center"})
    ], style={"display":"flex", "align-items":"center", "justify-content":"center"}),

    
    html.Br(),

    dcc.Graph(id='cases_graph', figure={})
  
])

# callbacks
@app.callback(
    Output(component_id='cases_graph', component_property='figure'),
    [Input(component_id='select-county', component_property='value'),
    Input(component_id='graph-type', component_property='value'),
    Input(component_id='pred-type', component_property='value')]
)
def update_graph(county_selected, graph_type, pred_type):
    
    if county_selected not in preds_dict.keys():
        preds_dict[county_selected] = get_all_preds(df, county_selected)

    diff = df[county_selected].diff().fillna(df[county_selected])
    df['Diff'] = diff
    preds = preds_dict[county_selected]

    if county_selected == 'Total':
        title_str_1 = "Cumulative Cases for Texas"
        title_str_2 = "Daily New Cases for Texas"
        

    else:
        title_str_1 = "Cumulative Cases for " + county_selected + " County"
        title_str_2 = "New Cases for " + county_selected + " County"

    if graph_type == "Cumulative":

        fig = px.line(data_frame=df, x='Date', y=county_selected, 
            title=title_str_1,
            labels={'Date':"Date", county_selected:"# of Cases"},
            hover_data=[county_selected]
        )

        if pred_type == 'LSTM':
            
            preds_graph = go.Scatter(
                x=preds['Date'],
                y=preds['LSTM'],
                mode='lines',
                showlegend=False,
                name="Forecast"
            )

            fig.add_trace(preds_graph)   
    
    else:
        fig = px.line(data_frame=df, x='Date', y='Diff',
            title=title_str_2,
            labels={'Date':"Date", 'Diff':"# of Cases"},
            hover_data=['Diff']
        )

        if pred_type == 'LSTM':
            
            preds_diff = preds['LSTM'].diff().fillna(preds['LSTM'])
            preds_diff.iloc[0] = df['Diff'].iloc[-1]

            preds_graph = go.Scatter(
                x=preds['Date'],
                y=preds_diff,
                mode='lines',
                showlegend=False,
                name="Forecast"
            )  

            fig.add_trace(preds_graph)

    fig.update_xaxes(
        rangeslider_visible=False,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(label="All", step="all")
            ])
        )
    )

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)


    

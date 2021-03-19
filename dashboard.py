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

options_list = []
for name in list(df.columns)[1:]:
    county_dict = {"label":str(name), "value":str(name)}
    options_list.append(county_dict)



# App layout
app.layout = html.Div([

    html.H1('Texas Coronavirus Cases Dashboard', style={'text-align':'center'}),

    dcc.Dropdown(id='select-county',
            options = options_list,
            multi=False,
            value="Total",
            style={"width":"35%"}
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
    df_copy['Diff'] = df_copy[county_selected].diff().fillna(df_copy[county_selected])
    df_copy

    if county_selected == 'Total':
        title_str_1 = "Cumulative Cases for Texas"
        title_str_2 = "Daily New Cases for Texas"
        

    else:
        title_str_1 = "Cumulative Cases for " + county_selected + " County"
        title_str_2 = "New Cases for " + county_selected + " County"

    fig1 = px.line(data_frame=df, x='Date', y=county_selected, 
        title=title_str_1,
        labels={'Date':"Date", county_selected:"# of Cases"},
        hover_data=[county_selected]
    )

    fig1.update_xaxes(
        rangeslider_visible=True,
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
    
    fig2 = px.line(data_frame=df_copy, x='Date', y='Diff',
        title=title_str_2,
        labels={'Date':"Date", 'Diff':"# of Cases"},
        hover_data=['Diff']
    )

    fig2.update_xaxes(
        rangeslider_visible=True,
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

    return fig1, fig2

if __name__ == '__main__':
    app.run_server(debug=True)


    

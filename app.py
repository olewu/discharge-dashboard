from dash import dcc, html, Input, Output, Dash, callback
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

import pandas as pd
from datetime import date, timedelta, datetime

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

prop_file_sm = 'data/catchment_properties/smaakraft/smaakraft_prop_and_clim.csv'
df_sm = pd.read_csv(prop_file_sm)

prop_file_nve = 'data/catchment_properties/nve/_prop_and_clim.csv'
df_nve = pd.read_csv(prop_file_nve)

init_back = 2

startdate = date(2023,11,14)

app.layout = html.Div([
    html.Div([

        html.Div([
            dcc.RadioItems(
                ['Smaakraft', 'NVE'],
                'Smaakraft',
                id='catch-type',
                labelStyle={'display': 'inline-block', 'marginTop': '5px'},
            )
        ], style={'display': 'block'}),
        html.Div([
            dcc.Graph(
                id='map',
                clickData={'points': [{'hovertext': 'bjoergum'}]}
            )
        ],),
    ], style={'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '3vw', 'margin-top': '3vw','width': '35%','height': '120%'}
    ),
    html.Div([
        html.Div([
            dcc.Graph(id='forecast'),
        ], style={'display': 'block'}),

        html.Div(
            children=[
                dcc.Slider(
                    -init_back,
                    0,
                    step=None,
                    id='init-slider',
                    value=0,
                    marks={str(ii-init_back): str(date.strftime('%Y-%m-%d')) for ii,date in enumerate(pd.date_range(startdate - pd.Timedelta(days=init_back),startdate))}
                ),
                html.P('Initialization Date',style={'horizontal-align':'center'}),
            ],
        ),
    ],style={'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '3vw', 'margin-top': '3vw','width': '55%','height': '30%'})
])



@callback(
    Output('map', 'figure'),
    Input('catch-type', 'value'),)
def update_graph(catch_type):
    if catch_type.lower() == 'smaakraft':
        df = df_sm
    elif catch_type.lower() == 'nve':
        df = df_nve
    else:
        df = df_sm

    fig = px.scatter_mapbox(df, lat="latitude", lon="longitude",     color="mean_annual_prec", size="mean_wet_days",
                color_continuous_scale="Viridis_r", size_max=15, zoom=4, labels = {'mean_annual_prec':'P_ann'},
                hover_name="stat_id", hover_data=[],mapbox_style='carto-positron')# open-street-map
    fig.update_layout(clickmode='event+select')

    return fig


def fc_timeseries(init, df_discharge, df_met, catchment, df_ens=None, df_metens=None, local_param_sim=None, obs_disch=None):
    
    plt_lim = pd.Timestamp(startdate - timedelta(days=31))
    df_red = df_discharge[df_discharge.date <= pd.Timestamp(init)]
    df_fc = df_discharge[df_discharge.date >= pd.Timestamp(init)]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=df_red.date, y=df_red.q_sim_mm, name='historical',),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=df_fc.date, y=df_fc.q_sim_mm, name='10d forecast'),
        secondary_y=False,
    )

    if df_ens is not None:
        fig.add_traces(
            [
                go.Scatter(
                    name='21d Forecast Median',
                    x=df_ens.date,
                    y=df_ens.p50,
                    mode='lines',
                    line=dict(color='rgb(31, 119, 180)'),
                ),
                go.Scatter(
                    name='90th Percentile',
                    x=df_ens.date,
                    y=df_ens.p90,
                    mode='lines',
                    marker=dict(color="#444"),
                    line=dict(width=0),
                    showlegend=False
                ),
                go.Scatter(
                    name='10th Percentile',
                    x=df_ens.date,
                    y=df_ens.p10,
                    marker=dict(color="#444"),
                    line=dict(width=0),
                    mode='lines',
                    fillcolor='rgba(68, 68, 68, 0.2)',
                    fill='tonexty',
                    showlegend=False
                ),
                go.Scatter(
                    name='75th Percentile',
                    x=df_ens.date,
                    y=df_ens.p75,
                    mode='lines',
                    marker=dict(color="#444"),
                    line=dict(width=0),
                    showlegend=False
                ),
                go.Scatter(
                    name='25th Percentile',
                    x=df_ens.date,
                    y=df_ens.p25,
                    marker=dict(color="#444"),
                    line=dict(width=0),
                    mode='lines',
                    fillcolor='rgba(68, 68, 68, 0.2)',
                    fill='tonexty',
                    showlegend=False
                )
            ],
        )

    fig.add_trace(
        px.bar(df_met, x='date', y='prec',opacity=1).data[0],
        secondary_y=True,
    )

    if df_metens is not None:
        fig.add_trace(
            px.bar(df_metens, x=df_metens.date, y=df_metens.prec_p50,opacity=1).data[0],
            secondary_y=True,
        )

    if local_param_sim is not None:
        fig.add_trace(
            go.Scatter(x=local_param_sim.date, y=local_param_sim.q_sim_mm, name='simulated (local param)',),
            secondary_y=False,
        )

    if obs_disch is not None:
            fig.add_trace(
                go.Scatter(x=obs_disch.date, y=obs_disch.Discharge_mmday, name='observed (NVE)',),
                secondary_y=False,
            )

    fig.update_xaxes(range=[plt_lim,pd.Timestamp(startdate + timedelta(days=22))])
    if obs_disch is not None:
        fig.update_layout(
            title_text = 'Simulated runoff at {0:s}'.format(obs_disch.stationName.to_list()[0])
        )
    else:
        fig.update_layout(
            title_text = 'Simulated runoff at {0:s}'.format(catchment)
        )
    fig.update_layout(
        hovermode = 'x'
    )
    fig.update_yaxes(title_text='Discharge [mm/day]', secondary_y=False)
    fig.update_yaxes(title_text='Precipitation [mm/day]', secondary_y=True)
    fig.update_yaxes(autorange='reversed', secondary_y=True)
            
    return fig


@callback(
    Output('forecast', 'figure'),
    Input('map', 'clickData'),
    Input('init-slider', 'value'),
    Input('catch-type','value'),)
def update_x_timeseries(clickData, init_subtr, catch):

    if clickData:
        catchment = clickData['points'][0]['hovertext']

        if datetime.now().hour < 10:
            init = startdate + timedelta(days=init_subtr-1)
        else:
            init = startdate + timedelta(days=init_subtr)
        
        # init = date.fromisoformat('2023-10-05')

        df = pd.read_csv('data/discharge_forecast/{3:s}/daily_{0:0>2d}-{1:0>2d}-{2:0>2d}T06:00:00Z.csv'.format(init.year,init.month,init.day,catch))
        df_met = pd.read_csv('data/forecast_input/{3:s}/fc_init_{0:0>2d}-{1:0>2d}-{2:0>2d}T06:00:00Z_merge_sn_2020-01-01.csv'.format(init.year,init.month,init.day,catch))
    
        if catch.lower() == 'nve':
            df_local_param_sim = pd.read_csv('data/discharge_forecast/{3:s}/daily_local_{0:0>2d}-{1:0>2d}-{2:0>2d}T06:00:00Z.csv'.format(init.year,init.month,init.day,catch))
            df_local_param_sim.date = pd.to_datetime(df_local_param_sim.date)
            local_param_sim = df_local_param_sim[df_local_param_sim.stat_id == catchment]

            drange_obs = pd.date_range(startdate - timedelta(days=62),startdate)
            obs_filelist = ['data/sildre_nve/{1:s}/nve_{0:s}.csv'.format(drsn.strftime('%Y-%m-%d'),catch) for drsn in drange_obs]

            df_obs_disch = pd.concat((pd.read_csv(f) for f in obs_filelist)).reset_index()
            df_obs_disch.date = pd.to_datetime(df_obs_disch.date)
            obs_disch = df_obs_disch[df_obs_disch.catchname == catchment]

        else:
            local_param_sim = None
            obs_disch = None

        df_ens = pd.read_csv('data/discharge_forecast/{3:s}/daily21d_{0:0>2d}-{1:0>2d}-{2:0>2d}_empiricalpercentiles.csv'.format(init.year,init.month,init.day,catch))
        df_ens.date = pd.to_datetime(df_ens.date)
        df_ens_sel = df_ens[df_ens.stat_id == catchment]

        df_metens_ = pd.read_csv('data/ens_forecast_input/{3:s}/fc_init_{0:0>2d}-{1:0>2d}-{2:0>2d}.csv'.format(init.year,init.month,init.day,catch))
        df_metens_.date = pd.to_datetime(df_metens_.date)
        df_metens_ = df_metens_[df_metens_.date > datetime.combine(init + timedelta(days=10),datetime.min.time())]
        df_metens = df_metens_.copy()[df_metens_.catchname == catchment]
        df_metens.loc[:,'prec_p50'] = np.percentile(df_metens[['prec_{0:d}'.format(i) for i in range(31)]],50,axis=1)
            

        df.date = pd.to_datetime(df.date)
        df_met.date = pd.to_datetime(df_met.date)

        df_sel = df[df.stat_id == catchment]
        df_met_sel = df_met[df_met.catchname == catchment]

        catchment_name = catchment.capitalize()
        return fc_timeseries(init,df_sel, df_met_sel, catchment_name, df_ens_sel, df_metens, local_param_sim, obs_disch) # TODO: include obeserved dishcarge!

    else:
        print('did not receive clickData')

if __name__ == '__main__':
    app.run(debug=True)
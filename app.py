# Copyright 2023 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import json

from dash import Dash, html, dcc, Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.express as px
import pandas as pd
import numpy as np

from data import DataSet


DATASET_NAMES = [
    {'label': 'Titanic Survival', 'value': 'titanic'},
    {'label': 'Scene', 'value': 'scene'},
]

GRAPH_FONT_SIZE = 14
DWAVE_PRIMARY_COLORS = ['#2a7de1', '#f37820']


# Global Variables

# Define global variables that are used to store information about the feature
# selection datasets.  Each variable is a dict keyed on the dataset name.

# The dict of `DataSet` instances is prepopulated because it is accessed by
# multiple callbacks during app loading.
datasets = {d['value']: DataSet(d['value']) for d in DATASET_NAMES}

# The following dicts are lazily populated because they are only needed by a
# single callback, so a race condition is not possible.
redundancy_data = {}
feature_plot_dfs = {}

# End Global Variables


app = Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div(children=[
    html.Div(
        children=[
            html.H1(children='Feature Selection',
                    className='header-title'),
            html.P(children="A constrained quadratic model for feature selection "
                   "using the Leap hybrid solver service",
                   className='header-description'),
        ],
        className='header',
    ),

    html.Div(
        children=[

            html.Div(id='input=div', children=[

                html.Div(children='Dataset',
                         className='menu-title'),
                dcc.Dropdown(
                    options=DATASET_NAMES,
                    value='titanic',
                    searchable=False,
                    clearable=False,
                    id='data-dropdown'),

                html.Div(id='solve-input-div'),
            ], style={'marginLeft': '20px', 'marginRight': '100px'}),

            html.Div(children=[
                dcc.Checklist(
                    ['Show redundancy'], [],
                    id='redundancy_check',
                ),
                html.Div(children=[
                    dcc.Graph(id='feature-graph', style={'flex': 1}, responsive=True),
                    html.Div(id='score-div', style={'flex-basis': '230px'}),
                ], style={'display': 'flex'}),
            ], style={'flex': 1}),
            dcc.Store(id='feature-solution'),

            dcc.Store(id='feature-score'),

        ],
        style={'display': 'flex'})
])


@app.callback(
    Output('solve-input-div', 'children'),
    Input('data-dropdown', 'value'))
def create_input_div(data_key):
    """Create layout div for the inputs."""
    data = datasets[data_key]
    
    children = [
        html.Div(children=[
            html.Div(children=[
                html.Div(children='Number of features',
                         className='menu-title'),
                dcc.Slider(1, data.n, 1, value=data.default_k,
                           tooltip={'always_visible': data.n > 15, 'placement': 'bottom'},
                           marks={i: str(i) for i in (1, data.n)} if data.n > 15 else {},
                           id='num-features-slider'),
            ], className='menu'),

            html.Div(children=[
                html.Div(children='Penalty for redundancy',
                         className='menu-title'),
                dcc.Slider(0, 1.0, value=data.default_redundancy_penalty,
                           tooltip={},
                           id='redundancy-slider'),
            ], className='menu', style={'width': '300px'}),

            html.Div(children=[
                html.Button('Solve', id='solve-button',
                            className='button'),
            ], className='menu'),
            html.Div(
                style={'position': 'relative', 'top':'30px'},
                children=[
                    html.Img(src="assets/D-Wave logo color.png", width=300),
                ],
            ),
        ]),

        # Visual indicator that solve is running.  It is actually a placeholder
        # to display some output.  One way to disable output is with
        # `display:none` in the style.  However, that prevents the "margin"
        # attributes from having an effect, which can be useful for controlling
        # where the loading indicator shows up.
        dcc.Loading(id='loading-solve',
                    color=DWAVE_PRIMARY_COLORS[0],
                    children=html.Div(id='loading-solve-output',
                                      style={'marginTop': '45px'})),

    ]
    return children


@app.callback(
    Output('feature-graph', 'figure'),
    Input('feature-graph', 'hoverData'),
    Input('redundancy_check', 'value'),
    Input('feature-solution', 'data'),
    Input('data-dropdown', 'value'))
def update_figure(hover_data, redundancy_check, feature_solution_data, data_key):
    """Update the main feature bar plot."""
    data = datasets[data_key]
    if data_key not in feature_plot_dfs:
        redundancy_data[data_key] = data.get_redundancy()
        feature_plot_dfs[data_key] = pd.DataFrame({
            'Feature': data.X.columns,
            'Feature Relevance': data.get_relevance()
        })

    df = feature_plot_dfs[data_key]

    color = None
    hover_cols = {'Feature Relevance': False}
    if hover_data and redundancy_check:
        idx = hover_data['points'][0]['pointIndex']
        # Protect against case where the last hovered point was from a larger data set.
        if idx < data.n:
            # Note: this modifies global data and so is not compatible with use
            # of the demo in a multi-user environment.  One alternative is to
            # make a copy of the DataFrame prior to modifying the redundancy
            # column.
            df['Redundancy'] = redundancy_data[data_key][idx]
            color = 'Redundancy'
            hover_cols['Redundancy'] = False

    opacity = 1.0
    mlw = 0
    feature_solution = None
    if feature_solution_data:
        solution_dataset, solution = json.loads(feature_solution_data)
        if solution_dataset == data_key:
            feature_solution = solution
    if feature_solution:
        opacity = np.repeat(0.2, len(df))
        opacity[feature_solution] = 1.0
        if data.n < 100:
            mlw = np.repeat(0, len(df))
            mlw[feature_solution] = 3

    # Custom D-Wave theme color scale.  Alternatively, use #008C82 for the
    # middle color to darken the green
    color_scale = ['#074C91', '#2A7DE1', '#17BEBB', '#FFA143', '#F37820']
    fig = px.bar(df, x="Feature", y="Feature Relevance", color=color, range_color=[0,1], opacity=opacity,
                 hover_data=hover_cols, color_continuous_scale=color_scale,
                 color_discrete_sequence=DWAVE_PRIMARY_COLORS)
    fig.update_traces(marker_line_color='black', marker_line_width=mlw)
    fig.update_layout(margin=dict(t=20))
    fig.update_layout(font=dict(size=GRAPH_FONT_SIZE))

    # Modify axis labels:
    fig.update_layout(yaxis_title='Feature Relevance to Outcome')
    if data_key == 'titanic':
        fig.update_layout(xaxis_title='Passenger Features')
    elif data_key == 'scene':
        fig.update_layout(xaxis_title='Color and Texture Features in Image')

    # Disable zooming because hover callback will reset it:
    fig.layout.xaxis.fixedrange = True
    fig.layout.yaxis.fixedrange = True

    # Adjust spacing between the two figures:
    # fig.update_layout(margin=dict(r=30))

    # Disable hover info:
    # fig.update_traces(hoverinfo='none', hovertemplate=None)

    return fig


@app.callback(
    Output('feature-solution', 'data'),
    Output('feature-score', 'data'),
    Output('loading-solve-output', 'children'),
    Input('solve-button', 'n_clicks'),
    State('redundancy-slider', 'value'),
    State('num-features-slider', 'value'),
    State('data-dropdown', 'value'),
    prevent_initial_call=True)
def on_solve_clicked(btn, redund_value, num_features, data_key):
    """Run feature selection when the solve button is clicked."""
    if not btn:
        raise PreventUpdate
    data = datasets[data_key]
    print('solving...')
    solution = data.solve_cqm(num_features, 1.0 - redund_value)
    # For testing:
    # solution = np.random.choice(np.size(data.X, 1), num_features, replace=False)
    solution = [int(i) for i in solution] # Avoid issues with json and int64
    print('solution:', solution)
    score = data.score_indices_cv(solution)
    return json.dumps((data_key, solution)), json.dumps((data_key,score)), ''


@app.callback(
    Output('score-div', 'children'),
    Input('feature-score', 'data'),
    Input('data-dropdown', 'value'),
    prevent_initial_call=False)
def update_score_figure(feature_score_data, data_key):
    """Update the plot of feature scores."""
    score = 0.0
    if feature_score_data:
        feature_score_dataset, score_ = json.loads(feature_score_data)
        if feature_score_dataset == data_key:
            score = score_
    print('score:', score)
    data = datasets[data_key]

    df_scores = pd.DataFrame({
        'Features': ['All', 'Selected'],
        'Classifier Accuracy': [data.baseline_cv_score, score]
    })

    # Swap color order so that the blue color in the feature graph corresponds
    # to the blue color for selected features in the score graph.
    fig = px.bar(df_scores, x="Features", y="Classifier Accuracy", color='Features',
                 color_discrete_sequence=DWAVE_PRIMARY_COLORS[1::-1])
    fig.update_layout(legend=dict(
        yanchor='bottom',
        y=1.03,
        xanchor='right',
        x=1
    ))
    fig.update_xaxes(visible=False, showticklabels=False)
    fig.update_yaxes(range=data.score_range)
    fig.update_layout(font=dict(size=GRAPH_FONT_SIZE))
    # Decrease bottom margin to bring text description closer:
    fig.update_layout(margin=dict(b=30))
    # Disable zooming:
    fig.layout.xaxis.fixedrange = True
    fig.layout.yaxis.fixedrange = True

    children=[
        dcc.Graph(
            id='score-graph',
            figure=fig,
            config={'displayModeBar': False},
        ),
        html.Div(children='Classifier accuracy as measured using a random '
                 'forest classifier with 3-fold cross-validation'),

    ]

    return children


if __name__ == '__main__':
    # Set dev_tools_ui=False or debug=False to disable the dev tools UI
    app.run_server(debug=True, dev_tools_ui=False)

from dash import Dash, dcc, html, dash_table, Input, Output, callback
import plotly.express as px
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import ThemeChangerAIO, template_from_url
from dash import Dash, html ,dcc
import pandas as pd
from datetime import datetime
import pandas as pd
import pytz
import plotly.graph_objects as go
from keras.models import load_model
import pymongo
import plotly.offline as pyo
import numpy as np
from sklearn.preprocessing import MinMaxScaler



clinet=pymongo.MongoClient("mongodb+srv://admin:H1IbFaRWkbw4nLpL@metatradeer.v87htex.mongodb.net/test")
db=clinet["metatrader"]
data=db['metatrader5']
list_c=list(data.find())
rates_frame=pd.DataFrame(list_c)

#rates_frame=rates_frame.set_index('time')
rates_frame=rates_frame.drop(columns=['_id'])

df = rates_frame



# stylesheet with the .dbc class

dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc_css])

header = html.H4(
    "Historical Volatility", className="bg-primary text-white p-2 mb-2 text-center"
)

table = dash_table.DataTable(
    id="table",
    columns=[{"name": i, "id": i, "deletable": True} for i in df.columns],
    data=df.to_dict("records"),
    page_size=10,
    editable=True,
    cell_selectable=True,
    filter_action="native",
    sort_action="native",
    style_table={"overflowX": "auto"},
    row_selectable="multi",
)


"""
MODEL_PATH = 'histo_model_f.h5'
model = load_model(MODEL_PATH)
training_set = rates_frame.iloc[:, 1:2].values

sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)


X_test = []
for i in range(700 , 1251):
 X_test.append(training_set_scaled[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
pred=model.predict(X_test)

data=rates_frame[700:1251]
data['train']=training_set_scaled[700:1251]
data['pred']=pred






tab1 = dbc.Tab([dcc.Graph(id="line-chart" , figure={'data':[ {'x':df.time,'y':df.close}]}   ) , ], label="Close Volatility"  )
tab2 = dbc.Tab([dcc.Graph(id="line-chart2" , figure={'data':[ {'x':data.time,'y':data.pred}, {'x':data.time,'y':data.train}   ]}   ) , ], label="Predict Close Volatility ( LSTM)")
tab3 = dbc.Tab([table], label="Table", className="p-4")


"""
tab1 = dbc.Tab([dcc.Graph(id="line-chart" , figure={'data':[ {'x':df.time,'y':df.close}]}   ) , ], label="Close Volatility"  )
tab3 = dbc.Tab([table], label="Table", className="p-4")
tabs = dbc.Card(dbc.Tabs([tab1,tab3]))
app.layout = dbc.Container(
    [
        header,
        dbc.Row(
            [
                dbc.Col(
                    [

                        # When running this app locally, un-comment this line:
                         ThemeChangerAIO(aio_id="theme")
                    ],
                    width=1,
                ),
                dbc.Col([tabs], width=8)
                ,
            ]
        ),
    ],
    fluid=True,
    className="dbc dbc-row-selectable",
)






if __name__ == "__main__":
    app.run_server(debug=False,host='0.0.0.0')
from sklearn.linear_model import LinearRegressionAdd commentMore actions
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse

with st.sidebar:
    file = st.file_uploader("Upload Data File", type = 'CSV')
model = LinearRegression()
if file:
    t1,t2 = st.tabs(["Train", "Inference"])
    with t1:
        co1,co2 = st.columns(2)
        with co1:
            df = pd.read_csv(file, index_col = None)
            st.dataframe(df)
        with co2:
            k = st.multiselect("Select Features", ['TV', 'Radio', 'Newspaper'], max_selections = 2)
            if k:
                y = np.array(df['Sales'])
                X = np.array([df[k[i]].values for i in range(len(k))])
                a = np.random.permutation(list(range(len(X[0]))))
                X_train = np.array([X[i, a[:-30]] for i in range(len(k))])
                X_test = np.array([X[i, a[-30:]] for i in range(len(k))])
                y_train = y[a[:-30]]
                y_test = y[a[-30:]]
                X_train1 = np.array([[X_train[s,i] for s in range(len(k))] for i in range(len(X_train[0]))])
                model.fit(X_train1, y_train)
                y_pred = model.predict(np.array([[X_test[s,i] for s in range(len(k))] for i in range(len(X_test[0]))]))
                st.info(f'Model trained, MAE: {round(mae(y_test, y_pred),3)}, MSE : {round(mse(y_test, y_pred,),3)}')
                if len(k) == 1:
                    fig,ax = plt.subplots()
                    ax.set_xlabel(k[0])
                    ax.set_ylabel('Sales')
                    ax.plot([X.min(), X.max()], [model.predict(X.min().reshape(-1,1)), model.predict(X.max().reshape(-1,1))], color = 'red')
                    ax.scatter(X,y)
                    st.pyplot(fig)
                elif len(k) == 2:
                    x1 = np.linspace(X[0].min(), X[0].max(), 100)
                    y1 = np.linspace(X[1].min(), X[1].max(), 100)
                    xx, yy = np.meshgrid(x1, y1)
                    xy = np.c_[xx.ravel(), yy.ravel()]
                    z = model.predict(xy)
                    z = z.reshape(xx.shape)

                    fig = go.Figure(data = [go.Scatter3d(x = X[0], y = X[1], z = y, mode = 'markers'),
                                            go.Surface(x = x1, y = y1, z = z)])
                    fig.update_layout(scene={'xaxis_title':k[0], 'yaxis_title':k[1], 'zaxis_title':'Sales'})
                    st.plotly_chart(fig)
    with t2:
        X = []
        if len(k) == 1:
            X.append(st.number_input(k[0], min_value = 0.00))
        elif len(k) == 2:
            c1,c2 = st.columns(2)
            with c1:
                X.append(st.number_input(k[0], min_value = 0.00))
            with c2:
                X.append(st.number_input(k[1], min_value = 0.00))
        if X:
            if sum(X) == 0:
                st.warning(f'Please input:{", ".join(k)}')
            else:
                X = np.array(X).reshape(1,-1)
                y_pred = model.predict(X)
                st.success(f"Prediction: {y_pred}$")

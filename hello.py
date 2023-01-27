import numpy as np
#import matplotlib.colors as mcolors
#import matplotlib.pyplot as plt
import openpyxl as op
import pandas as pd  # pip install pandas openpyxl
import plotly.express as px  # pip install plotly-express
#import plotly.figure_factory as ff
import streamlit as st  # pip install streamlit
import streamlit.components.v1 as components
#import altair as alt
import numpy as np
#import os
from pathlib import Path
import gc
import os

st.set_page_config(
    page_title = "Marketplaces",
    page_icon = ":phone:",
    initial_sidebar_state="expanded",
)

#df1 = pd.read_csv('list/petshop-28-12-2022-26-01-2023.csv', )

dtypes = {
    "SKU": "int32",
    "Продавец": "str",
    "Продаж": "int32",
    "Выручка30": "float32"
}

for feature in [f'f_{i}' for i in range(300)]:
    dtypes[feature] = "float32"


train = pd.read_csv("list/petshop-28-12-2022-26-01-2023.csv", error_bad_lines=False , dtype=dtypes, sep = ',' )

st.text( f"Total len: {len(train)}")
st.dataframe(train.head())


def sla_category(val):

    if val <= 1000000:
        return '1mi'
    elif val <= 5000000:
        return '5mi'
    elif val <= 1000000:
        return '10mi'
    elif val <= 50000000:
        return '50mi'
    else:
        return 'неизвестно'

train['filter'] = train['Выручка30'].apply(sla_category)
df8 = train.loc[train['filter'].isin(['10mi', '50mi'])]
#df8.

df10 = df8[['Продавец',
            'Выручка30',
            'filter',
            '2022-12-28',
            '2022-12-29',
            '2022-12-30',
            '2022-12-31',
            '2023-01-01',
            '2023-01-02',
            '2023-01-03',
            '2023-01-04',
            '2023-01-05',
            '2023-01-06',
            '2023-01-07',
            '2023-01-08',
            '2023-01-09',
            '2023-01-10',
            '2023-01-11',
            '2023-01-12',
            '2023-01-13',
            '2023-01-14',
            '2023-01-15',
            '2023-01-16',
            '2023-01-17',
            '2023-01-18',
            '2023-01-19',
            '2023-01-20',
            '2023-01-21',
            '2023-01-22',
            '2023-01-23',
            '2023-01-24',
            '2023-01-25',
            '2023-01-26',
            'Продаж',
            ]]



st.dataframe(df10)




#st.text(f"Heatmap:")
#df10 = pd.pivot_table(df10, index='date', columns='hour', values='ticket number', aggfunc='count')
#df10.fillna(0, inplace=True)
#pivot.sort_values(by='total', ascending=False, inplace=True)

#piv =  px.imshow(df10)
#st.plotly_chart(piv, theme=None)

#df10.sort_values(by='Выручка30', ascending=True, inplace=True)


#with st.expander("Pivot"):

#    pivot = pd.pivot_table(df10, index='Продавец', values='filter',aggfunc='count')
#    pivot.sort_values(by='filter', ascending=True, inplace=True)
#    pivot = pivot.style.format('{:.0}')\
#    .format('{:.0f}', subset=['filter'])\
#    .background_gradient(cmap='ocean_r', subset=['filter'])
#    st.dataframe(pivot)

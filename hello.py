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
#import seaborn as sns
#import matplotlib.pyplot as plt
#from matplotlib import colors

st.set_page_config(
    page_title = "Marketplaces",
    page_icon = ":phone:",
    initial_sidebar_state="expanded",
)

dtypes = {
    "SKU": "int32",
    "Продавец": "str",
    "Продаж": "int32",
    "Выручка30": "float32"
}

for feature in [f'f_{i}' for i in range(300)]:
    dtypes[feature] = "float32"

train = pd.read_csv("list/petshop-28-12-2022-26-01-2023.csv" , dtype=dtypes, sep = ',' )

st.text( f"Total len: {len(train)}")
st.text( f"Total len sellers non empty: {train['Продавец'].count()}")
st.dataframe(train.head())

with st.expander("Category"):
    ticket_by_subtopics = (
    train.groupby(by=["Категория"]).count()[["SKU"]].sort_values(by="SKU")
    )
    fig_ticket_by_subtopics = px.bar(
    ticket_by_subtopics,
    x=ticket_by_subtopics.index,
    y="SKU",
    text="SKU",
    orientation="v",
    title="<b>sellers by category</b>",
    color_discrete_sequence=["#0083B8"] * len(ticket_by_subtopics),
    template="plotly_white",
    )

    fig_ticket_by_subtopics.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=(dict(showgrid=False))
    )

    st.plotly_chart(fig_ticket_by_subtopics, use_container_width=True)

df = train[['Продавец',
            'Выручка30',
            'Упущенная выручка',
            'Коммент.',
            'Дней с продажами',
            'Наличие',
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

df = df.dropna()
df = df.groupby(['Продавец']).sum()
st.text( f"Total len sellers non empty > 5 000 000:")

def sla_category(val):
    if val >= 5000000:
        return '5mi'
    else:
        return 'неизвестно'

df['filter'] = df['Выручка30'].apply(sla_category)

df1 = df.loc[df['filter'].isin(['5mi'])]
df1.sort_values(by='Выручка30', ascending=False, inplace=True)

st.dataframe(df1.style.highlight_max(color = 'lightgreen', axis=0) , 2000, 1000)
#cm = sns.light_palette("#79C", as_cmap=True)
#st.dataframe(df1.style.background_gradient(cm), 2000, 1000)

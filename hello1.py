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
    "Выручка(' . 30 . ' дней)": "float32"
}

for feature in [f'f_{i}' for i in range(300)]:
    dtypes[feature] = "float32"


train = pd.read_csv("list/petshop-28-12-2022-26-01-2023.csv", dtype=dtypes, sep = ';' )

st.text( f"Total CSI index: {len(train)}")

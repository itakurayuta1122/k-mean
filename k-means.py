import pandas as pd
import streamlit as st

dataset = pd.read_csv("k-means.csv")
dataset = dataset.drop('Unnamed:0', axis=1)
st.write("クラスタリング前",dataset)

data = dataset.sample(frac=0.95, random_state=786).reset_index(drop=True)
data_unseen = dataset.drop(data.index).reset_index(drop=True)

from pycaret.clustering import *
data_clust = setup(data, normalize = True, 
                   ignore_features = ['MouseID'],
                   session_id = 123)

kmeans = create_model('kmeans',num_clusters = 5 )

kmean_results = assign_model(kmeans)

st.write("クラスタリング後",kmean_results)
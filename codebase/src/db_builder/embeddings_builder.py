import numpy as np
import pandas as pd
from openai import OpenAI
from scipy.spatial.distance import cosine


def read_csv(file_name):
    df = pd.read_csv(file_name)
    return df

def combine_data(df):
    restaurants = df['Restaurant']
    reviews = df['Review']
    combined_series = 'restaurant name: ' + restaurants + ' Review: ' + reviews
    return combined_series

def build_client():
    # api_key = ""
    client = OpenAI(
        api_key=api_key
    )  # get API key from platform.openai.com
    return client

def text_to_vectors(text_inputs):
    client = build_client()
    MODEL = "text-embedding-3-small"
    res = client.embeddings.create(
        input=text_inputs, model=MODEL
    )
    vectors = res.data
    np_embeddings = [np.array(vector.embedding) for vector in vectors]
    return np_embeddings

def build_embeddings_df_from(text_inputs):
    embeddings = text_to_vectors(text_inputs)
    df = pd.DataFrame(text_inputs, columns = ['text'])
    df = df.assign(embedding = embeddings)
    return df

def build_in_batches_from(combined_series, batch_size = 30):
    dfs = []
    cleaned_df = combined_series.dropna()
    for i in range(0, len(cleaned_df), batch_size):
        selected_df = cleaned_df.iloc[i:i+batch_size]
        embeddings_df = build_embeddings_df_from(selected_df.to_list())
        dfs.append(embeddings_df)
    combined_embedded = pd.concat(dfs, axis = 0)
    return combined_embedded
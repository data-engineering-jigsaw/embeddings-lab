import numpy as np
import pandas as pd
from openai import OpenAI
from scipy.spatial.distance import cosine


def read_csv(file_name):
    df = pd.read_csv(file_name)
    return df

def combine_data(df):
    pass

def build_client():
    # api_key = ""
    client = OpenAI(
        api_key=api_key
    )  # get API key from platform.openai.com
    return client

def text_to_vectors(text_inputs):
    pass

def build_embeddings_df_from(text_inputs):
    pass

def build_in_batches_from(combined_series, batch_size = 30):
    pass
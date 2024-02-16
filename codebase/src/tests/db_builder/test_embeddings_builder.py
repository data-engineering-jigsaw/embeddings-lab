import pandas as pd

from db_builder.embeddings_builder import (build_embeddings_df_from,
                                           build_in_batches_from, combine_data,
                                           text_to_vectors)


def test_combine_data():
    columns = ['Restaurant', 'Reviewer', 'Review', 'Rating', 'Metadata', 'Time',
       'Pictures', '7514']
    data = [
        {'Restaurant': 'Beyond Flavours', 'Reviewer': 'Rusha Chakraborty', 'Review': 'The ambience was good, food was quite good .', 'Rating': 5.0, 'Metadata': '1 Review , 2 Followers', 'Time': '5/25/2019 15:54', 'Pictures': 0, '7514': 2447.0},
        {'Restaurant': 'Deli 9 Bistro', 'Reviewer': 'Rusha Chakraborty', 'Review': 'The best part about this place was the ambiance', 'Rating': 5.0, 'Metadata': '1 Review , 2 Followers', 'Time': '5/25/2019 15:54', 'Pictures': 0, '7514': 2447.0},
    ]
    df = pd.DataFrame(data = data, columns = columns)
    combined_series = combine_data(df)
    assert combined_series.to_list() == ['restaurant name: Beyond Flavours Review: The ambience was good, food was quite good .',
       'restaurant name: Deli 9 Bistro Review: The best part about this place was the ambiance']
    
def test_text_to_vectors():
    text_inputs = ['restaurant name: Beyond Flavours Review: The ambience was good, food was quite good .',
       'restaurant name: Deli 9 Bistro Review: The best part about this place was the ambiance']
    vectors = text_to_vectors(text_inputs)
    assert len(vectors) == 2
    assert vectors[0][0] == -0.04781287536025047

def test_build_embeddings_df():
    text_inputs = ['restaurant name: Beyond Flavours Review: The ambience was good, food was quite good .',
       'restaurant name: Deli 9 Bistro Review: The best part about this place was the ambiance']
    df = build_embeddings_df_from(text_inputs)
    assert list(df.columns) == ['text', 'embedding']

def test_build_in_batches_from():
    df = pd.read_csv('./db_builder/source_data/reviews.csv')[:300]
    combined_series = combine_data(df)
    vector_df = build_in_batches_from(combined_series, batch_size = 30)
    assert len(vector_df) == 300
    assert vector_df.columns.to_list() == ['text', 'embedding']
    assert vector_df.embedding.iloc[0].shape == (1536,)
    

    
    # build_in_batches_from(db)
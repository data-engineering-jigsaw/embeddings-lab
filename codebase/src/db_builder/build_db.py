from embeddings_builder import *

file_name = './source_data/reviews.csv'
df = read_csv(file_name)
text_series = combine_data(df)
combined_embedded = build_in_batches_from(text_series, batch_size = 30)
combined_embedded.to_parquet('../database.parquet')

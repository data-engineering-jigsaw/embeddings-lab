import pandas as pd
from openai import OpenAI
from scipy.spatial.distance import cosine

from db_builder.embeddings_builder import text_to_vector_df


def build_context_from_distances_to(question, db_path):
    pass

def generate_prompt(question, context):
    return f'''Provide a 2-3 sentence answer to the question based on the following sources. Be original, concise, accurate, and helpful.

    query: {question},

    context:
    {context}'''

def get_answer(prompt, api_key):
    COMPLETIONS_MODEL = "gpt-4-0125-preview"
    COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 300,
    "model": COMPLETIONS_MODEL}
    client = OpenAI(
    api_key=api_key,
    )
    response = client.chat.completions.create(
                model=COMPLETIONS_MODEL,
                messages = [{"role": "system", "content": "Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\n"},
                {"role": "user", "content": prompt}]
            )
    return response

def question_and_answer(question, db_path):
    pass
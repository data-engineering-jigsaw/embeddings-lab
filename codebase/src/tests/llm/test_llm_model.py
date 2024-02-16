import pandas as pd
from openai import OpenAI
from scipy.spatial.distance import cosine

from llm.llm_model import *

def test_build_context_from_distances_to():
    question = "What is a good brunch place to go to"
    db_path = './tests/database.parquet'
    context = build_context_from_distances_to(question, db_path)
    assert context == "restaurant name: Mazzo - Marriott Executive Apartments Review: Best place in the city special for the brunch\nNice food,staff was very nice and friendly\nI really enjoyed my Sunday in this place\n\nAmazing ambience that anyone will fall in love\n\nrestaurant name: The Fisherman's Wharf Review: Good for a nice Sunday brunch. The buffet for two is 1500 INR, and worth the deal, I'd say. The ambience is also great and you do get a lot of seating options. Would recommend.\n\nrestaurant name: Mazzo - Marriott Executive Apartments Review: This place is amazing. We went for Sunday brunch and the food was really good .View and the ambience was awesome. Would definitely recommend for those who want to have lots of food and relax and have a perfect weekend."

def test_generate_prompt():
    question = "What is a good brunch place to go to"
    context = "restaurant name: Mazzo - Marriott Executive Apartments Review: Best place in the city special for the brunch."
    prompt = generate_prompt(question, context)
    assert prompt == 'Provide a 2-3 sentence answer to the question based on the following sources. Be original, concise, accurate, and helpful.\n\n    query: What is a good brunch place to go to,\n\n    context:\n    restaurant name: Mazzo - Marriott Executive Apartments Review: Best place in the city special for the brunch.'

def test_question_and_answer():
    question = "What is a good brunch place to go to"
    db_path = './tests/database.parquet'
    answer = question_and_answer(question, db_path)
    assert type(answer) == str
    assert 'Mazzo' in answer
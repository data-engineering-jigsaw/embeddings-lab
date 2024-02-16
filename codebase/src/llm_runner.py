from llm.llm_model import question_and_answer

question = "What is a good brunch place to go to"
db_path = './database.parquet'
print(question_and_answer(question, db_path = db_path))


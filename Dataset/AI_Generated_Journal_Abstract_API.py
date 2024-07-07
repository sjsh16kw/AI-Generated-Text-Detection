import os
from openai import OpenAI
import openai
import pandas as pd
file_path = 'abs.csv'

OPENAI_API_KEY = "Enter_Your_API_Key"
client = OpenAI(api_key=OPENAI_API_KEY)
df = pd.read_csv('intro.csv')

def generate_abstract(intro_text):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"You are an expert academic writer. Write abstract of following Introduction of journal (Write in 1 paragraph, about 120 words) : \n\n introduction : \"{intro_text}\""}
        ],
    )
    return response.choices[0].message.content

abstracts = []
cnt = 0
for index, row in df.iterrows():
    cnt += 1
    introduction_text = row[1]
    #print(introduction_text)
    abstracts.append(generate_abstract(introduction_text))

    if (cnt % 100) == 0:
        abstract_df = pd.DataFrame({"Abstract": abstracts})
        abstract_df.to_csv(file_path, index=False)

abstract_df = pd.DataFrame({"Abstract": abstracts})
abstract_df.to_csv(file_path, index=False)


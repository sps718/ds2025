"""
A demonstration of how you could use a created vector store to augment prompts (RAG)

docs: https://platform.openai.com/storage/vector_stores
example code: https://colab.research.google.com/drive/1nUe0ExifMTibktn9MKJYAMUAV8HBgs6n?usp=sharing

note the new packages we're using, you may need to run...
pip install v
"""

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

import os
from openai import OpenAI

# load in your chroma db
lecture_db = Chroma(persist_directory="./lecture_index", embedding_function=OpenAIEmbeddings())

# set retrieval policy
lecture_retrieve = lecture_db.as_retriever(search_kwargs={"k": 3})

# get relevant documents according to request
message = "When did we go over the meaning of life?"
retrieved_docs = lecture_retrieve.get_relevant_documents(message)

context = ""
for doc in retrieved_docs:
    context += f"\nTITLE:{doc.metadata["title"]}\nPAGE:{doc.metadata["page"]}\nCONTENT:\n{doc.page_content}"

print(context)

# augment prompt with retrieved documents
rag_prompt = (
    "ONLY look through the 'LECTURE NOTES' section to inform your answer to a query. Respond with title & page number when applicable.'\n"
    f"LECTURE NOTES:\n{context}\n\n"
    f"QUERY:{message}"
)

# access openai api
client = OpenAI()

# get response from 4o-mini
resp = client.responses.create(
    model='gpt-4o-mini',
    input=rag_prompt
)

print(rag_prompt)
print(resp.output_text)

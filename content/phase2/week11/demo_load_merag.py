"""
A demonstration of how you could save a folder of pdfs into a vector store

docs1: https://python.langchain.com/docs/how_to/document_loader_pdf/
docs2: https://python.langchain.com/docs/integrations/document_loaders/pypdfloader/
example code: https://colab.research.google.com/drive/1tqKnZHjud38LcCkwtBP0VM0_MXCwFLlf#scrollTo=NLyGj4uS2v33

pip install pypdf
"""

from langchain_community.document_loaders.text import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

import os

# set up api keys...
#os.environ["OPENAI_API_KEY"] = "..."
#os.environ["OPENAI_PROJECT"] = "proj_fHRnVJY0Oyfm1ufG1sffxa6W"

# I need to grab all files from a folder...
files = os.listdir("farukh_chat_logs/")

# And for each file
replies = []
for f in files:
    path = f'farukh_chat_logs/{f}'
    print(f"load lecture file {f}")

    # convert it into a langchain document...
    loader = TextLoader(path)

    # and save this document into a list...
    for log in loader.load():
        print(log)
        replies.append(log)

# Take this list of documents and save it into a persistent Chroma store...
solutions_db = Chroma.from_documents(documents=replies, embedding=OpenAIEmbeddings(), persist_directory="./chat_index")

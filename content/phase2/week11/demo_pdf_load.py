"""
A demonstration of how you could save a folder of pdfs into a vector store

docs1: https://python.langchain.com/docs/how_to/document_loader_pdf/
docs2: https://python.langchain.com/docs/integrations/document_loaders/pypdfloader/
example code: https://colab.research.google.com/drive/1tqKnZHjud38LcCkwtBP0VM0_MXCwFLlf#scrollTo=NLyGj4uS2v33

pip install pypdf
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

import os

# set up api keys...
#os.environ["OPENAI_API_KEY"] = "..."
#os.environ["OPENAI_PROJECT"] = "proj_fHRnVJY0Oyfm1ufG1sffxa6W"

# I need to grab all files from a folder...

# And for each file

# convert it into a langchain document...

# and save this document into a list...

# Take this list of documents and save it into a persistent Chroma store...

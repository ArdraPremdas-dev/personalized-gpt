import os
import sys

import openai
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma


os.environ["OPENAI_API_KEY"] = "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"

query = sys.argv[1]

loader = TextLoader("data/data.txt")
index = VectorstoreIndexCreator().from_loaders([loader])

print(index.query(query,llm=ChatOpenAI()))
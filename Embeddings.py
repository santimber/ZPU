
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone


# load document
directory = '/Users/Boo/Coding/AI/Langchain/ZPU/platform _rules.pdf'

def load_doc(directory):
    loader = PyPDFLoader(directory)
    data = loader.load()
    return data


data = load_doc(directory)
total_character_count = 0
for document in data:
    total_character_count += len(document.page_content)

print(f'There are {total_character_count} characters in your document')

# Chunk your data up into smaller documents


def split_docs(data, chunk_size=800, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    text = text_splitter.split_documents(data)
    return text


texts = split_docs(data)
print(f'Now you have {len(texts)} documents')

# Create embeddings
embeddings = OpenAIEmbeddings()

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_API_ENV
)
index_name = "zpu-bot"
vectorstore = Pinecone.from_texts(
    [t.page_content for t in texts], embeddings, index_name=index_name)

# Search for similar products

def get_similiar_docs(query, k=3, score=False):
    similar_docs = vectorstore.similarity_search(query, k=k)
    return similar_docs

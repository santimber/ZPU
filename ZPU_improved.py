from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

import tiktoken
import pprint

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory, ConversationSummaryBufferMemory, ConversationBufferMemory

from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import streamlit as st
from streamlit_chat import message
from utils import *


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone


# load document
directory = 'platform _rules.pdf'

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
    st.write("api_key:", st.secrets["PINECONE_API_KEY"]),
    st.write("environment:", st.secrets["PINECONE_API_ENV"])
)
index_name = "zpu-bot"
vectorstore = pinecone.Index("zpu-bot")

# Search for similar products

def get_similiar_docs(query, k=3, score=False):
    similar_docs = vectorstore.similarity_search(query, k=k)
    return similar_docs


# pretty printing
pp = pprint.PrettyPrinter(indent=4)

# load keys
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
os.environ['PINECONE_API_KEY'] = st.secrets['PINECONE_API_KEY']
os.environ['PINECONE_API_ENV'] = st.secrets['PINECONE_API_ENV']

# prompting
template = """
You are an exceptional partner support chatbot that politely and consisely answers questions.

You know the following context information.

Answer to the following question from a partner. Use only information from the previous context information.

If the answer is not contained within the text below, say 'I don't know
{context}
Question: 

Answer:"""

system_message_prompt = SystemMessagePromptTemplate.from_template(
    template=template)

human_template = "{input}"
human_message_prompt = HumanMessagePromptTemplate.from_template(
    human_template)

prompt_template = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt])

# setting up a chain
def load_chain():
    llm = OpenAI(temperature=0)
    prompt = prompt_template
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain

chain = load_chain()

# setting up streamlit

st.set_page_config(page_title="PaCa chatbot Demo", page_icon=":robot:")
st.header("PaCA Platform rules Bot")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

def get_text():
    input_text = st.text_input("You: ", "Hello, how are you?", key="input")
    return input_text

user_input = get_text()

if user_input:
    docs = get_similiar_docs(user_input)
    output = chain.run(input_documents=docs, question=user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
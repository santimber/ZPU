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
from langchain.chains.conversation.memory import ConversationBufferWindowMemory, ConversationSummaryBufferMemory

from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import streamlit as st
from streamlit_chat import message
from utils import *

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

Question: 

Answer:"""

system_message_prompt = SystemMessagePromptTemplate.from_template(
    template=template)

human_template = "{input}"
human_message_prompt = HumanMessagePromptTemplate.from_template(
    human_template)

prompt_template = ChatPromptTemplate.from_messages(
    [system_message_prompt, MessagesPlaceholder(variable_name="history"), human_message_prompt])

# Setting the LLM and Chain
def load_chain():
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    chain =  ConversationChain(memory=ConversationSummaryBufferMemory(llm=llm, max_token_limit=650,
                                                                     return_messages=True), prompt=prompt_template, llm=llm, verbose=True)
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
    input_text = st.text_input("You: ", "Hello, I have some questions about the platform rules", key="input")
    return input_text

user_input = get_text()

if user_input:
    output = chain.run(input=user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")

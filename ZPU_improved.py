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

# Building the chatbot
llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo-0613')

st.subheader("PaCa Chatbot with memory")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=650,
                                                                     return_messages=True)
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

# Creating user interface
st.title("PaCa Chatbot")
...
response_container = st.container()
textcontainer = st.container()
...
with textcontainer:
    query = st.text_input("Query: ", key="input")
    ...
with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i],
                        is_user=True, key=str(i) + '_user')


# Setting the LLM and Chain
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key="")

conversation = ConversationChain(
    memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)


def query_refiner(conversation, query):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response['choices'][0]['text']


def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(input_em, top_k=2, includeMetadata=True)
    return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']


def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: " + \
            st.session_state['responses'][i+1] + "\n"
    return conversation_string

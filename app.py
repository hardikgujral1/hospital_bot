import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM 
from transformers import pipeline
import torch
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline

app=FastAPI()
# Define the persist directory (make sure it exists)
persist_directory = "persist_directory"
os.makedirs(persist_directory, exist_ok=True)

# Initialize SentenceTransformerEmbeddings with a pre-trained model
embeddings = SentenceTransformerEmbeddings(model_name="multi-qa-mpnet-base-dot-v1")

# Load the Chroma vector database
db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# Specify the checkpoint for the language model
checkpoint = "MBZUAI/LaMini-Flan-T5-783M"

# Initialize the tokenizer and base model for text generation
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,
    device_map="auto",
    torch_dtype=torch.float32
)
pipe = pipeline(
    'text2text-generation',
    model=base_model,
    tokenizer=tokenizer,
    max_length=512,
    do_sample=True,
    temperature=0.3,
    top_p=0.95
)

# Initialize a local language model pipeline
local_llm = HuggingFacePipeline(pipeline=pipe)

# Create a RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=local_llm,
    chain_type='stuff',
    retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 2}),
    return_source_documents=True,
)

# Function to interact with the model
def chat_with_model(query):
    response = qa_chain({"query": query})
    return response['result']

# Prompt the user for a query and display the response
@app.get('/bot')
def response(query:str):
    # input_query = input("Enter your query (or type 'exit' to quit): ")
    # if input_query.lower() == 'exit':
    #     exit()
    llm_response = chat_with_model( query)
    print("Response:", llm_response)
    return JSONResponse(content={"response":llm_response})

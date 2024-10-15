from google.colab import drive
drive.mount("/content/drive")


!pip install langchain sentence-transformers chromadb llama-cpp-python langchain_community pypdf

langchain-pipeline framework similar to the tensorflow or sklearn for llms
sentence-transformer-transforms sentences to vectors
chromadb-vector database used to store , index and search across embeddings.
llama-cpp-python- running llms
langchain_community- extra additional features.

##Importing libraries

from langchain_community.document_loaders import PyPDFDirectoryLoader #pdf loading
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA,LLMChain
import os


##Import the document


NOTE: Its directory loader so it looks for the folder not the file.

loader=PyPDFDirectoryLoader("/content/drive/MyDrive/medchat/data")
docs=loader.load()

len(docs) #number of pages.Each page is considered as one document.

docs[5]


text_splitter=RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=50)
chunks=text_splitter.split_documents(docs)


len(chunks) #total text is divided into a group of 300 characters

chunks[300]

##Embeddings creation

os.environ["HUGGINGFACEHUB_API_TOKEN"]="api_key_insert_here"

embeddings=SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
#downloading the model and loading it into the ram.


##VECTOR STORE CREATION

vectorstore=Chroma.from_documents(chunks,embeddings)
#search types:
#keyword search
#vector search
#hybrid search

query="Who is at risk of heart disease?"
search_results=vectorstore.similarity_search(query)
print(search_results)

retriever=vectorstore.as_retriever(search_kwargs={'k':5})

retriever.get_relevant_documents(query)

##LLM Model Loading

llm=LlamaCpp(
    model_path="/content/drive/MyDrive/medchat/BioMistral-7B.Q4_K_S.gguf",
    temperature=0.2,
    max_tokens=2048,
    top_p=1
)
#temperature:creativity.
#here temeperature is lower because we need more accurate results(on a range of 0 to 2)
#top_p:nucleus sampling
#choses a subset of tokens to sample based on their probabilities.




##LLM and Retreiver and Query to generate final response

template="""
<|context|>
You are a medical assistant that follows the instructions and provides accurate response based on the query and the context provided.
Be truthful and give direct answers
</s>
<|user|>
{query}
</s>
<|assistant|>
"""






from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate

prompt=ChatPromptTemplate.from_template(template)


rag_chain=(
    {"context":retreiver,"query":RunnablePassthrough()}
    |prompt
    |llm
    |StrOutputParser()
)

response=rag_chain.invoke(query)
response


#to make this an interactive chat we need to make this a loop
import sys
while True:
  user_input=input(f"Input Query:")
  if user_input =='exit':
    print("Exiting")
    sys.exit()
  if user_input=="":
    continue
  result=rag_chain.invoke(user_input)
  print(result)







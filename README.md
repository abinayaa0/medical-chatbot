# Medical Assistant Chatbot with LangChain and LLaMA

This project demonstrates how to build a medical assistant chatbot using **LangChain**, **LLaMA**, and **ChromaDB**. The chatbot answers medical-related queries based on a set of PDF documents by combining document retrieval with large language models (LLMs).

## Features
- Load and process multiple PDFs from a directory.
- Split documents into manageable chunks for better querying.
- Use embeddings from the **Sentence Transformer** to index and search documents.
- Retrieve relevant information from the vector store based on a user query.
- Use **LLaMA** to generate answers based on the retrieved context and user queries.
- A simple interactive chat interface that loops, allowing the user to query the chatbot continuously.

## Prerequisites
- **Google Colab** or a local environment with the necessary packages installed.
- A set of PDF files that the model will use to answer queries.
- Hugging Face API Token (for downloading sentence-transformer embeddings).

## Required Libraries
- **LangChain**: A framework for building LLM-based applications.
- **Sentence Transformers**: To convert sentences into vector embeddings.
- **ChromaDB**: A vector database used to store, index, and search embeddings.
- **LlamaCpp**: To run LLMs on your local machine or Colab.

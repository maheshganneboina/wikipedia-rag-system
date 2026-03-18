Wikipedia RAG System

An end-to-end Retrieval-Augmented Generation (RAG) system built using Wikipedia data, enabling grounded question answering using vector search and LLMs.
📌 Overview

This project builds a Retrieval-Augmented Generation (RAG) system using Wikipedia data.
It allows users to ask questions and receive answers grounded in retrieved knowledge from a local dataset, instead of relying only on the language model.


🧠 Problem Statement

Large Language Models (LLMs) often:

hallucinate answer
lack access to real-time or domain-specific data


This project solves that by:

retrieving relevant context from a knowledge base
augmenting the prompt with that context
generating accurate, grounded answers


🏗️ Architecture

Wikipedia XML Dump
        ↓
extract_wiki.py
        ↓
Cleaned JSON (wiki_articles.json)
        ↓
build_db.py
        ↓
Embeddings (MiniLM)
        ↓
Chroma Vector Database
        ↓
ask.py
        ↓
User Query → Retrieval → Augmentation → LLM → Answer


🔄 RAG Flow

1. Retrieval
Convert query into embedding
Search vector DB (Chroma)
Fetch top-k relevant chunks

2. Augmentation
Combine retrieved chunks
Build prompt with context + question

3. Generation
Send prompt to LLM
Generate answer using only retrieved context


🛠️ Tech Stack

Python
Sentence Transformers (all-MiniLM-L6-v2)
ChromaDB (vector database)
OpenAI API (for generation)
Wikipedia XML dump (data source)


📂 Project Structure

![Image 3-17-26 at 11 32 PM](https://github.com/user-attachments/assets/fb9eb745-d107-4413-b825-8f5e0c7bdcf9)



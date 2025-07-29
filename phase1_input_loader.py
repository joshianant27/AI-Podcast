#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install langchain')


# In[5]:


get_ipython().system('pip install -U langchain-community')


# In[7]:


get_ipython().system('pip install pypdf newspaper3k transformers accelerate torch')


# In[9]:


get_ipython().system('pip install lxml_html_clean')


# In[10]:


import os
from langchain.document_loaders import PyPDFLoader, TextLoader
from newspaper import Article
from typing import List


# In[12]:


def load_text_file(file_path: str) -> str:
    loader = TextLoader(file_path)
    documents = loader.load()
    return "\n".join([doc.page_content for doc in documents])

def load_pdf_file(file_path: str) -> str:
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return "\n".join([doc.page_content for doc in documents])

def load_url_article(url: str) -> str:
    article = Article(url)
    article.download()
    article.parse()
    return article.text

def load_input(source: str) -> str:
    if source.startswith("http"):
        return load_url_article(source)
    elif source.endswith(".pdf"):
        return load_pdf_file(source)
    elif source.endswith(".txt"):
        return load_text_file(source)
    else:
        raise ValueError("Unsupported format")


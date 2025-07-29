#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install ipywidgets notebook tqdm --upgrade')
get_ipython().system('jupyter nbextension enable --py widgetsnbextension')


# In[2]:


get_ipython().system('pip install huggingface_hub')


# In[3]:


get_ipython().system('pip install tiktoken protobuf==3.20.3')


# In[4]:


import torch


# In[5]:


import transformers


# In[6]:


from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM


# In[7]:


from huggingface_hub import InferenceClient


# In[ ]:


class Summarizer:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = 0 if device == "cuda" else -1

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.summarizer = pipeline("summarization", model=self.model, tokenizer=self.tokenizer, device=self.device)

    def chunk_text(self, text: str, max_tokens=600, overlap=100) -> list:
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = words[i:i+max_tokens]
            chunks.append(" ".join(chunk))
            i += max_tokens - overlap
        return chunks

    def summarize_chunk(self, chunk: str) -> str:
        try:
            result = self.summarizer(
                chunk,
                max_length=150,
                min_length=40,
                truncation=True,
                do_sample=False
            )
            return result[0]['summary_text'].strip()
        except Exception as e:
            print(f"❌ Error summarizing chunk: {e}")
            return ""

    def summarize(self, text: str, recursive=True) -> str:
        chunks = self.chunk_text(text)
        summaries = [self.summarize_chunk(chunk) for chunk in chunks if chunk.strip()]
        merged_summary = "\n".join(summaries)

        if recursive and len(summaries) > 3:
            print("🔁 Recursive summarization...")
            return self.summarize_chunk(merged_summary)
        else:
            return merged_summary


#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer


# In[2]:


import torch


# In[3]:


class ConversationGenerator:
    def __init__(self, model_name="HuggingFaceH4/zephyr-7b-beta"):
        print("⏳ Loading conversation model...")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16 if self.device == "cuda" else torch.float32)
        self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=0 if self.device == "cuda" else -1)

        print("✅ Zephyr model loaded!")

    def generate_conversation(self, summary_text: str, max_turns: int = 6) -> str:
        prompt = f"""
You are writing a podcast dialogue between two hosts: Alex and Jordan.

Topic:
\"\"\"{summary_text}\"\"\"

Alex is curious and asks questions.
Jordan is thoughtful and insightful.

Write a natural, flowing conversation. Limit to {max_turns} exchanges.

Alex:
"""

        result = self.generator(prompt, max_new_tokens=500, do_sample=True, temperature=0.7, top_p=0.9)[0]["generated_text"]
        return result


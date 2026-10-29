from retrieval import hybrid_search
from embeddings import client
from config import CHAT_MODEL
import openai
from usage import record
import json
from config import MAX_TOOL_ROUNDS, COLLECTION_NAME


#system prompt
SYSTEM_PROMPT = ("To answer the provided question use the search tool provided. "
"If information gathered by the search tool is not enough, search again with a more specific query."
" Answer only based on the documents retrieved with the search tool."
" If the answer is not in the documents, you do not know it, even if you believe you know it. "
"Cite every document you used, by number, e.g. [1][3]."
" Do NOT fabricate answers or use any prior or outside knowledge other than the provided documents. "
"If the retrieved documents contain conflicting information about the same fact, you MUST report all conflicting versions"
" explicitly and NEVER silently pick one. State: 'Source [X] says A, but source [Y] says B - these conflict.'"
" The documents may contain errors. Your job is to report what the documents say, not decide which is correct.")


# 1. Define a list of callable tools for the model
tools = [
    {
        "type": "function",
        "function": { "name": "search_documents",
        "description": "Search the knowledge base for information",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Search query",
                },
            },
            "required": ["question"], },
        },
    },
]

max_iterations = MAX_TOOL_ROUNDS

def search_documents(question, start, collection_name=COLLECTION_NAME):
   hits = hybrid_search(question, collection_name=collection_name)
   if hits is None:
      context = "Search failed"
      return context, []
   context = build_context(hits, start)
   return context, hits
   

def build_context(hits, offset):
    context = []
    for i, hit in enumerate(hits, start = offset):
      chunk = f"[{i}] Title: {hit.payload.get('title', hit.payload.get('name', ''))}\n {hit.payload['text']}"
      context.append(chunk)
    return "\n\n".join(context)
                                

def answer(question, collection_name=COLLECTION_NAME):

    messages = [
    {"role": "system", "content": SYSTEM_PROMPT},  
    {"role": "user", "content": question},   
    ]
    all_hits = []
    for i in range(max_iterations):
          try:  
                resp = client.chat.completions.create(model=CHAT_MODEL, messages=messages, reasoning_effort="none", tools=tools) 
                if resp.choices[0].finish_reason == "tool_calls":
                  record(resp.usage)
                  response = resp.choices[0].message
                  tool_calls = resp.choices[0].message.tool_calls[0]
                  tool_call_id = tool_calls.id 
                  question = json.loads(tool_calls.function.arguments)["question"]  
                  context_string, hits = search_documents(question, len(all_hits) + 1, collection_name=collection_name) 
                  all_hits.extend(hits)
                  messages.append(response)
                  messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": context_string})  
                elif resp.choices[0].finish_reason == "stop":
                  break
          except openai.OpenAIError as e:
            print(str(e))
            return("The answer could not be generated. Please try again.", all_hits)
    if resp.choices[0].finish_reason != "stop":
      try:
        resp1 = client.chat.completions.create(model=CHAT_MODEL, messages=messages, reasoning_effort="none")
        answer_text = resp1.choices[0].message.content
        record(resp1.usage)
        return answer_text, all_hits
      except openai.OpenAIError as e:
         print(str(e))
         return("The answer could not be generated. Please try again.", all_hits)
    answer_text = resp.choices[0].message.content
    record(resp.usage)
    return answer_text, all_hits

if __name__ == "__main__":
    while True:
      q = input("Question: ")
      if not q:
        break 
      text, pts = answer(q)
      print(text)
      for i, point in enumerate(pts, start = 1):
         print(f"[{i}], [{point.payload.get('title', point.payload.get('name', ''))}]")
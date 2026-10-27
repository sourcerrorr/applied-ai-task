from retrieval import search_qdrant
from embeddings import client
from config import CHAT_MODEL
import openai
from usage import record


#system prompt
SYSTEM_PROMPT = ("Use ONLY the provided documents. Do not use any prior or outside knowledge. "
                "If the answer is not in the documents, you do not know it , "
                "even if you believe you know it. Cite every document you used, by number, e.g. [1][3],"
                " IF the answer is not in the PROVIDED DOCUMENTS, state that the information is insufficient,"
                " and do NOT fabricate answer or use any prior or outside knowledge other than the provided documents.")

def build_context(hits):
    context = []
    for i, hit in enumerate(hits, start =1):
      chunk = f"[{i}] Title: {hit.payload['title']}\n {hit.payload['text']}"
      context.append(chunk)
    return "\n\n".join(context)
                                

def answer(question):
    points  = search_qdrant(question) 
    if points is None:
       return ("Couldn't retrieve documents, please try again.", [])
               
    context = build_context(points)
    user_msg = f"Retrieved information: {context}\n\nQuestion: {question}"

    messages = [
    {"role": "system", "content": SYSTEM_PROMPT},  
    {"role": "user", "content": user_msg},   
    ]
    try:  
      resp = client.chat.completions.create(model=CHAT_MODEL, messages=messages, reasoning_effort="none")   
      answer_text = resp.choices[0].message.content
      record(resp.usage)
    except openai.OpenAIError as e:
       print(str(e))
       return("The answer could not be generated. Please try again.", points)

    return answer_text, points

if __name__ == "__main__":
    while True:
      q = input("Question: ")
      if not q:
        break 
      text, pts = answer(q)
      print(text)
      for i, point in enumerate(pts, start = 1):
         print(f"[{i}], [{point.payload['title']}]")
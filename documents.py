import uuid
import datetime
from embeddings import embed_texts
import os
from pathlib import Path
from vectordb import upsert_points
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue, FilterSelector
from pypdf import PdfReader
from docx import Document
from vectordb import client
from config import COLLECTION_NAME
import hashlib
from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_LEN


def extract_txt(file):

    file.seek(0)
    file_content = file.read().decode("utf-8")
    line_count = file_content.count("\n") + 1
    metadata = {"file_type": "txt", "line_count": line_count}
    return file_content, metadata

def extract_pdf(file):

    file.seek(0)
    reader = PdfReader(file)
    pages = reader.pages
    page_number =len(pages)
    content = []
    for page in pages:
        page_content = page.extract_text()
        content.append(page_content)
    full_content = "\n\n".join(content)
    metadata = {"file_type": "pdf", "page_count": page_number}

    return full_content, metadata


def extract_docx(file):

    file.seek(0)
    doc = Document(file)
    pars = doc.paragraphs
    par_n = len(pars)
    content = []
    for par in pars:
        par_content = par.text
        content.append(par_content)
    full_content = "\n\n".join(content)
    metadata = {"file_type": "docx", "paragraph_count": par_n}

    return full_content, metadata

    

handlers = {
    ".txt": extract_txt,
    ".pdf": extract_pdf,
    ".docx": extract_docx,
}

def extract_text(file, filename):
    
    file_type = os.path.splitext(filename)[1]
    if file_type not in handlers:
        raise ValueError("Unsupported file type: " + file_type)
    file_content = handlers[file_type](file)

    return file_content

def chunk_text(text, strategy):
    
    chunk_size = CHUNK_SIZE
    overlap = CHUNK_OVERLAP
    chunks = []
    text = text.replace("\r\n", "\n")

    if strategy == "paragraph":
       chunks = [c.strip() for c in text.split("\n\n") if len(c.strip()) >= MIN_CHUNK_LEN]
       if len(chunks) <= 1 or max(len(c) for c in chunks) > chunk_size * 2:
           chunks = []
           for i in range(0, len(text), chunk_size - overlap):
               chunk = text[i : i + chunk_size]
               if len(chunk) >= MIN_CHUNK_LEN:
                   chunks.append(chunk)
    elif strategy == "fixed":
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i : i + chunk_size]
            if len(chunk)>=MIN_CHUNK_LEN:
                chunks.append(chunk)
    return chunks


def process_document(file, filename, strategy="paragraph"):

    doc_id = str(uuid.uuid4())
    text, metadata = extract_text(file, filename)
    content_hash= hashlib.md5(text.encode()).hexdigest()
    
    dupes, _ = client.scroll(
             collection_name=COLLECTION_NAME,
             scroll_filter=Filter(
                 must=[FieldCondition(key="content_hash", match=MatchValue(value=content_hash))]
             ),
             with_payload=True,
             limit=1000
          )   
    if dupes:
        old_doc_id = dupes[0].payload["doc_id"]
        delete_document(old_doc_id)
    chunks = chunk_text(text, strategy)
    vectors = embed_texts(chunks)
    
    points = []
    for i, (vector, chunk) in enumerate(zip(vectors, chunks)):
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "source": "upload",
                "doc_id": doc_id,
                "name": filename,
                "upload_date": datetime.datetime.now().isoformat(),
                "text": chunk,
                **metadata,
                "content_hash": content_hash
            }
        ))
    upsert_points(points)
    return doc_id
def delete_document(doc_id):
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=FilterSelector(
            filter=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            )
        )
    )   

def list_documents():
    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[FieldCondition(key="source", match=MatchValue(value="upload"))]
        ),
        with_payload=True,
        limit=1000
     )
    docs = {}
    for point in points:
        did = point.payload["doc_id"]
        if did not in docs:
            docs[did] = {
                "doc_id": did,
                "name": point.payload["name"],
                "upload_date": point.payload["upload_date"],
            }
    return list(docs.values())


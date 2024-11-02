import os
import tqdm

from spacy.lang.en import English

pdf_path = "nutrition_handbook.pdf"

if not os.path.exists(pdf_path):
  print("File doesn't exist.")
import fitz #PyMuPDF

def text_formatter(text: str) -> str:
    cleaned_text = text.replace("\n", " ").strip() 

    return cleaned_text

def open_and_read_pdf(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    pages_and_texts = []
    for page_number, page in tqdm(enumerate(doc)):
        text = page.get_text()
        text = text_formatter(text)
        pages_and_texts.append({"page_number": page_number -41,
                               "page_char_count": len(text),
                               "page_word_count": len(text.split(" ")),
                               "page_sentence_count_raw": len(text.split(". ")),
                               "page_token_count": len(text) /4, 
                               "text": text})
    return pages_and_texts

pages_and_texts = open_and_read_pdf(pdf_path=pdf_path)
pages_and_texts[:2]                    
import random 

random.sample(pages_and_texts, k=3)
import pandas as pd

df = pd.DataFrame(pages_and_texts)
df.head(10) 
df.describe().round(2)

nlp = English()

#sentencizer pipeline
nlp.add_pipe("sentencizer")

for item in tqdm(pages_and_texts):
    item["sentences"] = list(nlp(item["text"]).sents)
    
    item["sentences"] = [str(sentence) for sentence in item["sentences"]]
    
    item["page_sentence_count_spacy"] = len(item["sentences"])
random.sample(pages_and_texts, k=1)

#chunking 
num_sentence_chunk_size = 10 # this equates to ~ 287 tokens per chunk

def split_list(input_list: list, 
               slice_size: int,
               overlap: int = 5) -> list[list[str]]:
    """Split a list into chunks with overlap"""
    chunks = []
    for i in range(0, len(input_list), slice_size - overlap):
        chunk = input_list[i:i + slice_size]
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
    return chunks

for item in tqdm(pages_and_texts):
    item["sentence_chunks"] = split_list(
        input_list=item["sentences"],
        slice_size=num_sentence_chunk_size,
        overlap=5  # 5 sentence overlap
    )
    item["num_chunks"] = len(item["sentence_chunks"])
df = pd.DataFrame(pages_and_texts)
df.describe().round(2)
import re

#splitting each chunk into its own item (for references)
pages_and_chunks = []
for item in tqdm(pages_and_texts):
    for sentence_chunk in item["sentence_chunks"]:
        chunk_dict = {}
        chunk_dict["page_number"] = item["page_number"]
        
        joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
        joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk) 
        chunk_dict["sentence_chunk"] = joined_sentence_chunk

        #stats about the chunk
        chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
        chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
        chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4 
        
        pages_and_chunks.append(chunk_dict)

len(pages_and_chunks)
df = pd.DataFrame(pages_and_chunks)
df.describe().round(2)
#remove smallest chunks
min_token_length = 30
pages_and_chunks_over_min_token_len = df[df["chunk_token_count"] > min_token_length].to_dict(orient="records")
pages_and_chunks_over_min_token_len[:2]
### Preprocessing done, now onto embeddings
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", #Can use AI server for this
                                      device="cpu")

#Test with AI server
embedding_model.to("cpu")

for item in tqdm(pages_and_chunks_over_min_token_len):
    item["embedding"] = embedding_model.encode(item["sentence_chunk"])
#Save to file
text_chunks_and_embeddings_df = pd.DataFrame(pages_and_chunks_over_min_token_len)
embeddings_df_save_path = "text_chunks_and_embeddings_df.csv"
text_chunks_and_embeddings_df.to_csv(embeddings_df_save_path, index=False)
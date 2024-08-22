import gdown
import zipfile

file_id = '15nZSasyaTPjJs5jXLE-wA34T0Ju-tjIO' 
destination = 'text_chunks_and_embeddings_df.csv'

url = f"https://drive.google.com/uc?id={file_id}"
gdown.download(url, destination, quiet=False)

with zipfile.ZipFile(destination, 'r') as zip_ref:
    zip_ref.extractall('downloads/embeddings')




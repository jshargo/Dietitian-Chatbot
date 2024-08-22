import gdown
import zipfile

file_id = '15nZSasyaTPjJs5jXLE-wA34T0Ju-tjIO' 
download_path = 'text_chunks_and_embeddings_df.csv'

url = f"https://drive.google.com/uc?id={file_id}"
gdown.download(url, download_path, quiet=False)

with zipfile.ZipFile(download_path, 'r') as ziphandler:
    ziphandler.extractall('dataset_folder')




from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from dotenv import load_dotenv
import os
import pickle

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv("API_KEY")


def list_csv_files(directory):
    """List all .csv files in the given directory."""
    return [f for f in os.listdir(directory) if f.endswith('.csv')]


while True:
    print("---Step-1: Select a dataset to embed---")
    csv_files = list_csv_files('./data')
    if not csv_files:
        print("No .csv files found in the ./data folder.")
        break

    for i, file in enumerate(csv_files, start=1):
        print(f"{i}. {file}")
    print("0. Exit")

    try:
        choice = int(
            input("Enter the number of the file you want to embed (or 0 to exit): "))
        if choice == 0:
            break
        selected_file = csv_files[choice - 1]
    except (ValueError, IndexError):
        print("Invalid selection. Please try again.")
        continue

    print(f"Processing file: {selected_file}")

    if selected_file.startswith('product_'):
        content_columns = ["保養保健", "產品名稱", "商品說明"]
        metadata_columns = ["產品分類", "商品編號",
                            "商品相關分類", "原價", "折扣價", "網址", "品牌"]
    else:
        content_columns = ["SeasonType", "Human", "Assistant"]
        metadata_columns = ["Age",
                            "Gender", "Property", "Protype"]

    loader = CSVLoader(file_path=f'./data/{selected_file}',
                       encoding='utf-8',
                       autodetect_encoding=True,
                       content_columns=content_columns,
                       metadata_columns=metadata_columns)

    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=100
    )

    doc_splits = text_splitter.split_documents(docs)

    print("---Step-2: Initialize the Embedding ---")

    embeddings = NVIDIAEmbeddings(
        model="nvidia/llama-3.2-nv-embedqa-1b-v2",
        api_key=API_KEY,
        truncate="END",
    )

    bm25_retriever = BM25Retriever.from_documents(doc_splits)
    faiss_vectorstore = FAISS.from_documents(doc_splits, embeddings)

    print("---Step-3: Saving the Embedding ---")

    base_name = os.path.splitext(selected_file)[0]
    with open(f"./stores/{base_name}_bm25_retriever.pkl", "wb") as f:
        pickle.dump(bm25_retriever, f)

    faiss_vectorstore.save_local(f"./stores/{base_name}_vectors")

    print(f"---Step-4: Done processing {selected_file} ---")

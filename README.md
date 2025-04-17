# Install Packages
先下 Common 的 pip install，再依照環境安裝 faiss，
要調用 GPU 的話，需要 WSL2 + Ubuntu + CUDA 模擬 Linux 環境，否則只能用 CPU。

## Common
pip install -U langchain_community langchain-nvidia-ai-endpoints rank_bm25 unstructured[all-docs] langchainhub langchain langgraph pandas gradio gradio_modal nvidia-riva-client

### Linux
pip install -U faiss-gpu

### Windows
pip install -U faiss-cpu

# Env Settings
修改 .env 貼上 <YOUR_API_KEY>

# Run
```
python app.py
```

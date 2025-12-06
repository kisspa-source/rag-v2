# Local RAG Setup Guide (로컬 RAG 구축 가이드)

## 1. Environment Setup (환경 설정)

To set up a robust local RAG (Retrieval-Augmented Generation) system, we recommend using **Ollama** for model inference and **LangChain** for orchestration.

### 1.1 Prerequisites
*   OS: macOS (Ventura or later) or Linux (Ubuntu 22.04 LTS)
*   Python 3.10+
*   RAM: Min 16GB (32GB+ recommended for 70B models)

### 1.2 Installation

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull Embedding & LLM Models
ollama pull qwen2:7b
ollama pull mxbai-embed-large

# 3. Create Python Virtual Environment
python3 -m venv venv
source venv/bin/activate
pip install langchain langchain-community chromadb
```

## 2. RAG Pipeline Configuration

로컬 RAG 파이프라인은 보안성이 뛰어나지만, 검색 정확도(Retrieval Accuracy)를 높이기 위해 정교한 튜닝이 필요합니다.

### 2.1 Chunking Strategy
Document splitting is crucial.
*   **Fixed-size chunking**: Simple but may break context.
*   **RecursiveCharacterTextSplitter**: Respects sentence structure (Recommended).
*   **Semantic Chunking**: Splits based on meaning changes (Advanced).

> **Tip**: 한글 문서의 경우 `chunk_size=500~800`, `chunk_overlap=50~100` 정도가 적당합니다.

### 2.2 Vector Database
For local lightweight setups, **ChromaDB** or **FAISS** are popular.
ChromaDB saves raw data and embeddings in `./chroma_db`, making it persistent across restarts.

## 3. TroubleShooting (문제 해결)

| Issue | Possible Cause | Solution |
|-------|----------------|----------|
| Low TPS | CPU bottleneck | Enable GPU/Metal acceleration in Ollama |
| Hallucination | Poor Retrieval | Use Hybrid Search (BM25 + Vector) |
| OOM (Out of Memory) | Context window too large | Reduce `context_count` or quantized model |

---
*Created by RAG Automation Script*

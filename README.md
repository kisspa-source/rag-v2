# 🤖 Local RAG Chatbot (Ollama + LangChain)

이 프로젝트는 **Ollama**와 **LangChain**을 활용하여 로컬 환경(내 PC)에서 안전하게 동작하는 **RAG (검색 증강 생성) 챗봇**입니다. 인터넷 연결 없이도 내 문서를 기반으로 답변하며, 개인정보 유출 걱정 없이 사용할 수 있습니다.

## ✨ 주요 기능

### 1. 핵심 기능 (MVP)
*   **로컬 LLM 구동**: Ollama를 통해 `qwen2:7b`, `llama3.1:8b` 등 고성능 오픈소스 모델을 로컬에서 실행합니다.
*   **문서 기반 답변**: PDF, Markdown, Text 파일을 업로드하면 내용을 이해하고 답변합니다.
*   **출처 표시**: 답변이 문서의 어느 페이지에서 인용되었는지 정확히 표시합니다.
*   **채팅 UI**: Streamlit 기반의 깔끔하고 직관적인 채팅 인터페이스를 제공합니다.

### 2. 고급 기능 (Advanced)
*   **다양한 문서 지원**: **Word(.docx), Excel(.xlsx), PowerPoint(.pptx)** 파일도 지원합니다.
*   **OCR (광학 문자 인식)**: 스캔된 PDF나 이미지 위주의 문서에서도 텍스트를 자동 추출합니다 (Tesseract 사용).
*   **고급 검색 엔진**:
    *   **Hybrid Search**: 키워드 검색(BM25)과 의미 검색(Vector)을 결합하여 정확도를 높였습니다.
    *   **Rerank (재순위화)**: 검색된 결과 중 가장 관련성 높은 내용을 정밀하게 다시 골라냅니다.
    *   **RRF (Reciprocal Rank Fusion)**: 여러 검색 알고리즘의 결과를 최적으로 병합합니다.

---

## 🛠️ 사전 요구 사항 (Prerequisites)

*   **Python**: 3.10 이상 버전
*   **Ollama**: 최신 버전 설치 필요 ([공식 홈페이지](https://ollama.com/))
*   **RAM**: 최소 8GB (16GB 이상 권장)
*   **Disk**: 모델 및 인덱스 저장을 위한 여유 공간 (약 10GB+)

---

## 🚀 설치 및 실행 가이드 (OS별)

### 🍎 macOS

1.  **Python 설치** (Homebrew 사용 권장)
    ```bash
    brew install python
    ```

2.  **필수 시스템 도구 설치** (OCR 및 PDF 처리를 위해 필요)
    ```bash
    brew install tesseract poppler
    ```

3.  **프로젝트 클론 및 이동**
    ```bash
    git clone <repository-url>
    cd rag-v2
    ```

4.  **가상환경 생성 및 활성화**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

5.  **패키지 설치**
    ```bash
    pip install -r requirements.txt
    ```

6.  **Ollama 모델 다운로드**
    ```bash
    # Ollama 앱을 실행한 상태에서 터미널에 입력
    ollama pull qwen2:7b
    ```

7.  **앱 실행**
    ```bash
    streamlit run app.py
    ```

---

### 🪟 Windows

1.  **Python 설치**
    *   [Python 공식 홈페이지](https://www.python.org/downloads/)에서 Python 3.10 이상을 다운로드하여 설치합니다.
    *   설치 시 **"Add Python to PATH"** 옵션을 반드시 체크하세요.

2.  **필수 시스템 도구 설치**
    *   **Tesseract OCR**: [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)에서 설치 파일을 다운로드하여 설치합니다.
    *   **Poppler**: [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases/)에서 최신 릴리즈를 다운로드하여 압축을 풀고, `bin` 폴더 경로를 시스템 환경 변수 `PATH`에 추가합니다.

3.  **프로젝트 클론 및 이동** (PowerShell 또는 CMD)
    ```powershell
    git clone <repository-url>
    cd rag-v2
    ```

4.  **가상환경 생성 및 활성화**
    ```powershell
    python -m venv venv
    .\venv\Scripts\activate
    ```

5.  **패키지 설치**
    ```powershell
    pip install -r requirements.txt
    ```

6.  **Ollama 모델 다운로드**
    ```powershell
    # Ollama 설치 후 실행된 상태에서 입력
    ollama pull qwen2:7b
    ```

7.  **앱 실행**
    ```powershell
    streamlit run app.py
    ```

---

### 🐧 Linux (Ubuntu/Debian 기준)

1.  **Python 및 시스템 도구 설치**
    ```bash
    sudo apt update
    sudo apt install python3 python3-venv python3-pip tesseract-ocr poppler-utils
    ```

2.  **프로젝트 클론 및 이동**
    ```bash
    git clone <repository-url>
    cd rag-v2
    ```

3.  **가상환경 생성 및 활성화**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

4.  **패키지 설치**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Ollama 설치 및 모델 다운로드**
    ```bash
    # Ollama 설치 스크립트
    curl -fsSL https://ollama.com/install.sh | sh
    
    # 모델 다운로드
    ollama pull qwen2:7b
    ```

6.  **앱 실행**
    ```bash
    streamlit run app.py
    ```

---

## ⚙️ 설정 (Configuration)

`config.yaml` 파일을 수정하여 RAG 시스템의 동작을 제어할 수 있습니다.

```yaml
rag:
  chunk_size: 800        # 문서를 자르는 크기 (토큰 단위)
  chunk_overlap: 100     # 잘린 문서 간 겹치는 부분
  top_k: 5               # 검색할 문서 개수
  
  # 고급 검색 설정
  search_type: "hybrid_rrf"  # hybrid_weighted (가중평균) 또는 hybrid_rrf (순위병합)
  use_rerank: true           # 재순위화 사용 여부 (정확도↑ 속도↓)

llm:
  model_name: "qwen2:7b" # 사용할 Ollama 모델
  temperature: 0.3       # 창의성 조절 (낮을수록 사실적)
```

## ❓ 문제 해결 (Troubleshooting)

*   **Ollama 연결 실패**: `ollama serve` 명령어로 백그라운드 서비스를 실행하거나, Ollama 앱이 켜져 있는지 확인하세요.
*   **OCR 오류**: `tesseract` 또는 `poppler`가 제대로 설치되지 않았을 수 있습니다. OS별 설치 가이드를 다시 확인하세요.
*   **메모리 부족**: `config.yaml`에서 `use_rerank: false`로 변경하고 `context_count`를 줄이세요.
*   **속도 저하**: 처음 실행 시 임베딩 모델 및 Reranker 모델 다운로드로 인해 시간이 걸릴 수 있습니다. 이후에는 빨라집니다.

---

## 🛡️ 보안 및 데이터

*   모든 데이터는 **로컬 PC**의 `chroma_db/` 폴더에 저장됩니다.
*   외부 서버로 데이터가 전송되지 않습니다.
*   로그 파일(`logs/`)에 개인정보가 포함되지 않도록 마스킹 처리가 적용되어 있습니다.

# 구현 계획 - 로컬 RAG 챗봇 (Ollama)

## 목표 설명
사용자의 Windows PC에서 완전히 구동되는 로컬 RAG (검색 증강 생성) 챗봇을 구축합니다. 이 시스템은 **Ollama**를 LLM으로, **LangChain**을 오케스트레이션 도구로, 그리고 로컬 벡터 데이터베이스를 사용하여 문서를 검색합니다.

## 사용자 검토 필요 사항
> [!IMPORTANT]
> **Ollama 필수**: 사용자의 PC에 [Ollama](https://ollama.com/)가 설치되어 있고 실행 중이어야 합니다.
> **모델 선택**: 로컬 하드웨어에서 속도와 성능의 균형을 위해 `llama3` 또는 `mistral` 모델을 시작으로 권장합니다.
> **다국어 지원**: PDF가 다양한 언어(한/영/불/독)로 제공되므로, 임베딩 모델과 LLM 프롬프트가 이를 처리할 수 있어야 합니다.

## 제안 아키텍처 (초안)

### 1. 프로그램 환경
*   **OS**: Windows (사용자 현재 OS)
*   **언어**: Python 3.10+
*   **핵심 프레임워크**: [LangChain](https://python.langchain.com/) (RAG 로직용)
*   **사용자 인터페이스**: [Streamlit](https://streamlit.io/) (깔끔한 웹 기반 채팅 인터페이스용)
*   **LLM 제공자**: Ollama (로컬 포트 11434에서 실행)
*   **임베딩**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (다국어 지원 모델, HuggingFace)
*   **검색 전략**: **Hybrid Search** (BM25 + Vector) + **Rerank** + **Query Expansion** (질의 확장).
*   **Reranker**: `BAAI/bge-reranker-v2-m3` (Cross-Encoder).
*   **Query Expansion**: 사용자의 질문을 다양한 관점으로 3~5개 생성하여 검색 확률을 높임 (Multi-Query).

### 2. 권장 데이터베이스 (Vector Store)
**주요 권장 사항: ChromaDB**
*   **이유**: 오픈 소스이며, 파일 기반 데이터베이스로 로컬에서 실행됩니다 (별도 서버 설정 불필요). LangChain과의 연동성이 매우 뛰어납니다. 데이터를 쉽게 영구 저장할 수 있어 앱을 재시작할 때마다 문서를 다시 인덱싱할 필요가 없습니다.

**대안: FAISS**
*   **이유**: 매우 빠르지만, 일반적으로 메모리 내에서 실행됩니다. 영구 저장을 위해서는 인덱스를 수동으로 저장/로드해야 합니다. 매우 큰 정적 데이터셋에는 좋지만, 이 사용 사례에는 ChromaDB가 개발자 친화적입니다.

## 제안 변경 사항

### [프로젝트 구조]
#### [NEW] [requirements.txt](file:///C:/Users/이은수/withai/rag-v2/requirements.txt)
*   `langchain`, `langchain-community`, `streamlit`, `chromadb`, `sentence-transformers`, `pypdf`, `rank_bm25` (키워드 검색용), `openpyxl` (Excel 읽기용).

#### [NEW] [app.py](file:///C:/Users/이은수/withai/rag-v2/app.py)
*   메인 Streamlit 애플리케이션.
*   **기능**:
    *   채팅 인터페이스 (이전 대화 기억 - Chat History).
    *   답변 시 **출처(Source) 및 페이지 번호** 명시.
    *   사이드바: PDF 업로드 및 인덱싱 상태 표시.

#### [NEW] [rag_engine.py](file:///C:/Users/이은수/withai/rag-v2/rag_engine.py)
*   RAG 로직 캡슐화:
    *   문서 로드 (PyPDFLoader, UnstructuredExcelLoader 등 사용).
    *   **텍스트 분할 (Chunking)**:
        *   `RecursiveCharacterTextSplitter` 사용.
        *   **Chunk Size**: 1000자 (문맥 유지에 적절한 크기).
        *   **Chunk Overlap**: 200자 (문단 간 맥락이 끊기지 않도록 중복 허용).
    *   Vector DB 생성/로드 (Chroma).
    *   **고급 검색 파이프라인 (Academic Best Practices)**:
        *   **Query Routing**: 질문이 검색이 필요한지(RAG) 단순 대화인지(LLM Only) 분류.
        *   **Query Expansion (Multi-Query)**: 질문을 3개로 변형하여 다양한 키워드로 검색 (Recall 향상).
        *   **Hybrid Search**: BM25 + Vector 결합.
        *   **Rerank**: Cross-Encoder로 정밀 재순위화.
    *   Ollama 쿼리 (System Prompt: "Context가 어떤 언어이든 답변은 항상 한국어로 하라. 답변 끝에 출처를 표기하라.").

## 검증 계획
### 자동화 테스트
*   `rag_engine.py`에 대한 단위 테스트를 통해 문서가 올바르게 인덱싱되고 검색되는지 확인합니다.

### 수동 검증
1.  Ollama 실행 (`ollama serve`).
2.  `streamlit run app.py` 실행.
3.  샘플 PDF 업로드.
4.  PDF 내용과 관련된 질문 수행.
5.  답변이 정확하고 출처를 인용하는지 확인.

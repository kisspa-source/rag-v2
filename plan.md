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
*   **임베딩**: `BAAI/bge-m3` (다국어 성능 우수).
*   **검색 전략**: **Hybrid Search** (BM25 + Vector).
*   **Rerank**: **Optional** (기본 OFF, 필요 시 `BAAI/bge-reranker-v2-m3` 또는 경량 모델 사용).
*   **Query Expansion**: **Smart Expansion** (명사 중심 짧은 쿼리만 적용, '정의/차이' 등 키워드 포함 시 제외).
*   **Query Routing**: **Rule-based Only** (속도 최적화).
*   **Context Limit**: LLM 입력 청크 최대 5개로 제한 (속도/정확도 균형).
*   **Hybrid Search Fusion**: **Reciprocal Rank Fusion (RRF)**로 BM25와 Vector 검색 결과 병합 (연구 기반 Best Practice).

### 2. 권장 데이터베이스 (Vector Store)
**주요 권장 사항: ChromaDB**
*   **이유**: 오픈 소스이며, 파일 기반 데이터베이스로 로컬에서 실행됩니다 (별도 서버 설정 불필요). LangChain과의 연동성이 매우 뛰어납니다. 데이터를 쉽게 영구 저장할 수 있어 앱을 재시작할 때마다 문서를 다시 인덱싱할 필요가 없습니다.

**대안: FAISS**
*   **이유**: 매우 빠르지만, 일반적으로 메모리 내에서 실행됩니다. 영구 저장을 위해서는 인덱스를 수동으로 저장/로드해야 합니다. 매우 큰 정적 데이터셋에는 좋지만, 이 사용 사례에는 ChromaDB가 개발자 친화적입니다.

## 제안 변경 사항

### [프로젝트 구조]
#### [NEW] [requirements.txt](file:///C:/Users/이은수/withai/rag-v2/requirements.txt)
*   `langchain`, `langchain-community`, `streamlit`, `chromadb`, `sentence-transformers`, `pypdf`, `rank_bm25`, `openpyxl`, `python-docx`, `python-pptx`, `markdown`, `pytesseract`, `pillow`.

#### [NEW] [app.py](file:///C:/Users/이은수/withai/rag-v2/app.py)
*   메인 Streamlit 애플리케이션.
*   **기능**:
    *   답변 시 **출처(Source) 및 페이지 번호** 명시 (예: `sample.pdf p.11, p.12` 형식으로 문단 하단 표시).
    *   **UI 안정성**:
        *   **비동기 처리**: `threading` 사용 시 UI 락(Lock) 및 버튼 비활성화(`disabled=True`) 처리.
        *   **모델 캐싱**: `@st.cache_resource` 사용하여 임베딩/LLM 모델 인스턴스 재로딩 방지.
        *   인덱싱 진행률 `session_state` 관리.
        *   DB 로딩 중 `st.stop()`으로 불필요한 Rerun 방지.
        *   파싱 실패 페이지 목록 표시 (OCR Fallback 옵션).
    *   사이드바:
        *   **모델 선택**: Ollama 모델 선택.
        *   PDF 업로드 및 인덱싱 상태 표시.
    *   **관리자 페이지 (Admin/Settings)**:
        *   **RAG 설정**: Chunk Size, Overlap, Hybrid Search Weight (Alpha) 조절.
        *   **LLM 설정**: 모델별 System Prompt 편집 및 저장, Temperature/Top-P 설정.
        *   **파일 관리**: 인덱싱된 파일 목록 조회 및 삭제 (Vector DB에서 제거).
        *   **관리자 비밀번호**, 파일 암호화 옵션.
        *   **API 인증**: ChromaDB 접근 시 Token 기반 인증 (프로덕션 배포 시 필수).
    *   **UX 강화**:
        *   업로드 후 **샘플 질문 자동 추천** (문서 목차 기반).
        *   답변 내 키워드 **하이라이트** 표시.
        *   최근 대화 기록 자동 저장.

#### [NEW] [rag_engine.py](file:///C:/Users/이은수/withai/rag-v2/rag_engine.py)
*   RAG 로직 캡슐화:
    *   **문서 로딩 (Multi-format Support)**:
        *   PDF, Excel, Word (.docx), PowerPoint (.pptx), Markdown (.md), Text (.txt) 지원.
        *   ZIP 파일 내부 자동 분석.
        *   이미지 단독 파일 OCR 처리.
    *   **오류 처리 (Robust Error Handling)**:
        *   OCR 실패 시 Raw Image Snippet 저장.
        *   인코딩 깨짐 감지 및 자동 재처리.
        *   페이지 단위 파싱 오류 로깅 및 UI 표시.
        *   업로드 실패 시 사용자 친화적 메시지.
    *   **텍스트 전처리 (Robust Cleaning)**:
        *   **공백 정규화**: 다중 공백/줄바꿈을 단일 공백으로 치환.
        *   **헤더/푸터 제거**: 페이지 상/하단 텍스트 Vector 비교로 반복 패턴 자동 제거.
        *   **한글 자모 복구**: `unicodedata` 정규화(NFC).
        *   **표 구조화**: 표 데이터를 텍스트로 평탄화(Flatten).
        *   **중복 제거**: 내용이 유사한 Chunk 제거 (Deduplication).
    *   **텍스트 분할 (Advanced Chunking)**:
        *   **전략**: **Semantic-Aware Adaptive Chunking** (일반: 500~700, 기술/API: 800~1000).
        *   **Overlap**: 150 (문맥 보존 강화).
        *   **Context Enrichment**: 각 Chunk에 문서 제목, 섹션 메타데이터 추가 (검색 품질 향상).
    *   Vector DB 생성/로드 (Chroma).
        *   **Incremental Indexing (점진적 인덱싱)**:
            *   파일 해시 기반 변경 감지.
            *   변경된 페이지/섹션만 재인덱싱.
            *   Stale Chunk 자동 삭제 및 Merge.
        *   **버전 관리**: 파일명/해시 기반 Collection 분리 (이전 데이터 오염 방지).
        *   **최적화**: `HNSW` 파라미터 튜닝 (`M=32`, `ef_construction=200`, `ef=50`), `cosine` 거리 사용.
        *   **메타데이터**: 파일명, 페이지 번호, 시작/끝 토큰 인덱스 저장.
    *   **고급 검색 파이프라인 (Research-based Optimization)**:
        *   **Query Preprocessing**:
            *   **Query Rewriting**: LLM으로 질문 명확화 및 개선 (Optional).
            *   **Query Decomposition**: 복잡한 질문을 하위 질문으로 분해 (Optional).
        *   **Query Routing**: 단순 키워드 기반 Rule 매칭 (속도 보장).
        *   **Query Expansion**: 명사 위주 짧은 질문에만 적용 (동사/특정 키워드 제외).
        *   **Hybrid Search**: BM25 + Vector + **RRF Fusion**.
        *   **Context Filtering**: 질문 키워드 기반 Relevance 필터링, 제목/소제목 가중치 조정.
        *   **Rerank**: 옵션으로 제공 (검색 결과 3배 초과치 후 재순위화).
    *   Ollama 쿼리 (System Prompt 최적화, Context Max 5).
    *   **Prompt Engineering (연구 기반)**:
        *   **Contextual Instructions**: LLM에게 제공된 Context만 사용하도록 명시적 지시.
        *   **Hallucination Prevention**: 답변 불가 시 "모르겠다"고 답변하도록 설정.
    *   **설정 관리 (Configuration)**:
        *   `config.yaml` 또는 DB를 통해 사용자 정의 설정(Chunk, Prompt, Params) 영구 저장.
        *   **동적 파이프라인**: 설정 변경 시 RAG 파이프라인(Splitter, Retriever) 즉시 반영.
    *   **모델 변경 지원**: `st.session_state` 활용하여 모델 객체 캐싱.
    *   **상세 로깅**:
        *   질문, 답변, 소요 시간.
        *   유사도 점수, BM25 점수, Rerank 점수.
        *   Query Expansion 적용 여부, Routing 규칙.
        *   단계별 Latency (검색, LLM 생성 등).

## 검증 계획
### 자동화 테스트
*   `rag_engine.py`에 대한 단위 테스트를 통해 문서가 올바르게 인덱싱되고 검색되는지 확인합니다.

### 수동 검증
1.  Ollama 실행 (`ollama serve`).
2.  `streamlit run app.py` 실행.
3.  샘플 PDF 업로드.
4.  PDF 내용과 관련된 질문 수행.
5.  답변이 정확하고 출처를 인용하는지 확인.

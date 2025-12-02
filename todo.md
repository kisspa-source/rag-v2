# Local RAG Chatbot 개발 작업 목록 (Detailed Checklist)

## 0. 프로젝트 초기화 및 구조 설계 (Project Initialization)
- [ ] **디렉토리 구조 생성**
    - [ ] `rag-v2/` (루트 디렉토리)
    - [ ] `rag-v2/data/` (PDF/Excel 파일 저장소)
    - [ ] `rag-v2/chroma_db/` (Vector DB 저장소)
    - [ ] `rag-v2/src/` (소스 코드 분리)
        - [ ] `rag-v2/src/utils.py` (로깅, 헬퍼 함수)
        - [ ] `rag-v2/src/rag_engine.py` (RAG 코어 로직)
        - [ ] `rag-v2/src/ui.py` (UI 컴포넌트)
- [ ] **환경 설정 파일**
    - [ ] `.gitignore` 생성 (`venv/`, `__pycache__/`, `.env`, `chroma_db/`, `data/` 등)
    - [ ] `requirements.txt` 작성 및 설치 (langchain, streamlit, chromadb, sentence-transformers, pypdf, rank_bm25, openpyxl, unstructured, pytesseract)
    - [ ] `config.py` 또는 `.env` 생성 (모델명, 청크 사이즈 등 상수 관리)

## 1. 환경 설정 (Environment Setup)
- [ ] **Python 가상환경**
    - [ ] 가상환경 생성 및 활성화
    - [ ] pip 업그레이드
- [ ] **Ollama 및 모델 준비**
    - [ ] Ollama 서비스 실행 확인
    - [ ] LLM 모델 Pull (`llama3.1:8b` 또는 `qwen2:7b`)
    - [ ] 임베딩 모델 캐싱 확인 (`BAAI/bge-m3`)

## 2. RAG 엔진 코어 구현 (src/rag_engine.py)
### 2.1 문서 처리 파이프라인
- [ ] **Document Loader**
    - [ ] `PyPDFLoader`로 PDF 로드 기능 구현
    - [ ] [Fallback] PDF 파싱 실패 시 OCR (Tesseract) 처리 로직 추가
    - [ ] `UnstructuredExcelLoader` (또는 pandas)로 Excel 로드 기능 구현
- [ ] **Text Preprocessing (Robust Cleaning)**
    - [ ] 공백 정규화 (다중 공백/줄바꿈 -> 단일 공백)
    - [ ] 헤더/푸터 자동 제거 (Vector 유사도 기반 반복 패턴 감지)
    - [ ] 한글 자모 복구 (`unicodedata.normalize('NFC')`)
    - [ ] 표 데이터 평탄화 (Flatten to Markdown/Text)
- [ ] **Text Splitter**
    - [ ] **Adaptive Chunking** 전략 구현
        - [ ] 일반 문서: 500~700 tokens
        - [ ] 기술/API 문서: 800~1000 tokens
        - [ ] Overlap: 150 tokens
    - [ ] **Chunk Deduplication**: 중복 청크 제거 로직

### 2.2 벡터 저장소 및 검색기
- [ ] **Embedding**
    - [ ] `BAAI/bge-m3` 모델 로드 및 캐싱 (`@st.cache_resource` 활용)
- [ ] **Vector Store (ChromaDB)**
    - [ ] **HNSW 파라미터 튜닝**: `M=32`, `ef_construction=200`, `ef=50`, `space="cosine"`
    - [ ] **버전 관리**: 파일명/해시 기반 Collection 분리
    - [ ] 메타데이터 저장 (파일명, 페이지, 토큰 인덱스)
- [ ] **Retrievers**
    - [ ] **BM25Retriever**: 키워드 검색용 인덱스 생성
    - [ ] **ChromaRetriever**: Vector 검색용
    - [ ] **EnsembleRetriever**: Hybrid Search (BM25 + Vector)

### 2.3 고급 RAG 로직 (Advanced Features)
- [ ] **Query Routing (Rule-based)**
    - [ ] 단순 키워드 매칭으로 검색 필요 여부 판단 (속도 최적화)
- [ ] **Query Expansion (Smart Expansion)**
    - [ ] 명사 중심의 짧은 질문(20자 이하)에만 적용
    - [ ] '정의', '차이' 등 특정 키워드 포함 시 제외
- [ ] **Reranking (Optional)**
    - [ ] 필요 시 `BAAI/bge-reranker-v2-m3` 적용 (옵션 처리)
- [ ] **Context Management**
    - [ ] LLM 입력 Context Chunk 최대 5개로 제한

### 2.4 LLM 응답 생성
- [ ] **Prompt Engineering**
    - [ ] 시스템 프롬프트 최적화 (한국어 답변, 출처 명시 유도)
- [ ] **Chain 구성**
    - [ ] LCEL 체인 구성 (History 포함)

## 3. 사용자 인터페이스 구현 (app.py)
- [ ] **Session State 및 캐싱**
    - [ ] `@st.cache_resource`: 임베딩/LLM 모델 인스턴스 전역 캐싱
    - [ ] `st.session_state`: 대화 기록, 인덱싱 진행률, 처리 상태 관리
- [ ] **사이드바 (설정 및 업로드)**
    - [ ] **비동기 인덱싱**: `threading` + `st.status` 활용 (UI 락 방지)
    - [ ] 모델 선택 드롭다운 (Ollama 설치 모델 연동)
    - [ ] 파싱 실패한 페이지 목록 표시
- [ ] **메인 채팅 화면**
    - [ ] 대화 내용 렌더링
    - [ ] 답변 스트리밍 출력
    - [ ] **출처 표시 개선**: `파일명 p.페이지번호` 형식으로 문단 하단에 링크처럼 표시 (동일 페이지 병합)

## 4. 로깅 및 디버깅 (Logging & Debugging)
- [ ] **상세 로깅**
    - [ ] 질문, 답변, 소요 시간 기록
    - [ ] 검색된 Chunk 정보, 유사도/BM25/Rerank 점수 기록
    - [ ] Query Expansion/Routing 적용 여부 기록

## 5. 최종 테스트 및 배포
- [ ] **단위 테스트**
    - [ ] 전처리/청킹 로직 테스트
    - [ ] 검색 정확도 테스트
- [ ] **통합 테스트**
    - [ ] UI 비동기 동작 및 에러 핸들링 테스트

# Local RAG Chatbot 개발 작업 목록 (Detailed Checklist)

## 0. 프로젝트 초기화 및 구조 설계 (Project Initialization)
- [ ] **디렉토리 구조 생성**
    - [ ] `rag/` (루트 디렉토리)
    - [ ] `rag/data/` (PDF/Excel 파일 저장소)
    - [ ] `rag/chroma_db/` (Vector DB 저장소)
    - [ ] `rag/src/` (소스 코드 분리)
        - [ ] `rag/src/utils.py` (로깅, 헬퍼 함수)
        - [ ] `rag/src/rag_engine.py` (RAG 코어 로직)
        - [ ] `rag/src/ui.py` (UI 컴포넌트)
- [ ] **환경 설정 파일**
    - [ ] `.gitignore` 생성 (`venv/`, `__pycache__/`, `.env`, `chroma_db/`, `data/` 등)
    - [ ] `requirements.txt` 작성 및 설치
    - [ ] `config.py` 또는 `.env` 생성 (모델명, 청크 사이즈 등 상수 관리)

## 1. 환경 설정 (Environment Setup)
- [ ] **Python 가상환경**
    - [ ] 가상환경 생성 및 활성화
    - [ ] pip 업그레이드
- [ ] **Ollama 및 모델 준비**
    - [ ] Ollama 서비스 실행 확인
    - [ ] LLM 모델 Pull (`llama3` 또는 `mistral`)
    - [ ] 임베딩 모델 캐싱 확인 (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`)

## 2. RAG 엔진 코어 구현 (src/rag_engine.py)
### 2.1 문서 처리 파이프라인
- [ ] **Document Loader**
    - [ ] `PyPDFLoader`로 PDF 로드 기능 구현
    - [ ] `UnstructuredExcelLoader` (또는 pandas)로 Excel 로드 기능 구현
    - [ ] [예외처리] 암호화된 파일이나 깨진 파일 처리
- [ ] **Text Splitter**
    - [ ] `RecursiveCharacterTextSplitter` 초기화
    - [ ] Config에서 Chunk Size(1000), Overlap(200) 불러오기
    - [ ] 분할된 청크의 메타데이터(페이지 번호, 시트명, 파일명) 보존 확인

### 2.2 벡터 저장소 및 검색기
- [ ] **Embedding**
    - [ ] HuggingFaceEmbeddings 초기화 (device: cpu/cuda 자동 감지)
- [ ] **Vector Store (ChromaDB)**
    - [ ] PersistentClient 설정 (데이터 영구 저장)
    - [ ] 컬렉션 생성 및 `get_or_create` 로직 구현
    - [ ] 문서 추가(`add_documents`) 함수 구현
- [ ] **Retrievers**
    - [ ] **BM25Retriever**: 문서 청크 기반 BM25 인덱스 생성 (키워드 검색용)
    - [ ] **ChromaRetriever**: 유사도 기반 검색기 생성 (k=10)
    - [ ] **EnsembleRetriever**: BM25(0.5) + Chroma(0.5) 결합

### 2.3 고급 RAG 로직 (Advanced Features)
- [ ] **Query Routing (질문 분류)**
    - [ ] 프롬프트 템플릿 작성: 질문이 "데이터 검색 필요"인지 "일상 대화"인지 분류
    - [ ] JsonOutputParser 등을 사용하여 구조화된 출력 파싱
- [ ] **Query Expansion (Multi-Query)**
    - [ ] 프롬프트 템플릿 작성: 원본 질문을 기반으로 유사 질문 3개 생성
    - [ ] 생성된 3개 질문으로 각각 검색 후 결과 중복 제거(Union) 로직 구현
- [ ] **Reranking (재순위화)**
    - [ ] Cross-Encoder 모델(`BAAI/bge-reranker-v2-m3`) 로드
    - [ ] 검색된 후보군(Candidate Docs)과 질문을 쌍으로 입력하여 점수 계산
    - [ ] 상위 N개(예: 5개) 필터링 함수 구현

### 2.4 LLM 응답 생성
- [ ] **Prompt Engineering**
    - [ ] 시스템 프롬프트: "당신은 유능한 비서입니다. 주어진 Context를 바탕으로 한국어로 답변하세요."
    - [ ] 답변 형식을 지정하여 출처(Source, Page/Sheet)가 명확히 드러나도록 유도
- [ ] **Chain 구성**
    - [ ] `RunnablePassthrough` 등을 활용한 LCEL(LangChain Expression Language) 체인 구성
    - [ ] History(대화 기록)를 프롬프트에 주입하는 로직 추가

## 3. 사용자 인터페이스 구현 (app.py)
- [ ] **Session State 관리**
    - [ ] `messages`: 대화 기록 저장
    - [ ] `rag_chain`: 초기화된 RAG 체인 객체 저장 (매번 로드하지 않도록)
    - [ ] `processing`: 처리 중 상태 플래그
- [ ] **사이드바 (설정 및 업로드)**
    - [ ] 파일 업로더 (`st.file_uploader`, accept_multiple_files=True, type=['pdf', 'xlsx', 'xls'])
    - [ ] "문서 처리 시작" 버튼 및 진행률 표시 (`st.spinner`, `st.progress`)
    - [ ] 처리 완료 시 "인덱싱 완료" 토스트 메시지
- [ ] **메인 채팅 화면**
    - [ ] 이전 대화 내용 렌더링 (User: 우측, AI: 좌측)
    - [ ] `st.chat_input` 입력 처리
    - [ ] **답변 스트리밍**: `st.write_stream` 또는 콜백을 이용한 실시간 출력
    - [ ] **출처 아코디언**: 답변 아래에 `st.expander("참고 문서 확인")`로 검색된 문서 원문 표시

## 4. 로깅 및 디버깅 (Logging & Debugging)
- [ ] **로거 설정**
    - [ ] `logging` 모듈 설정 (Console 출력)
    - [ ] 주요 단계(검색 된 문서 수, Rerank 점수, LLM 입력 프롬프트) 로깅
- [ ] **디버그 모드 (UI 옵션)**
    - [ ] 사이드바에 "디버그 정보 보기" 체크박스 추가
    - [ ] 체크 시 검색된 문서(Raw Text)와 Rerank 점수를 화면에 표시

## 5. 최종 테스트 및 배포
- [ ] **단위 테스트**
    - [ ] PDF/Excel 로드 테스트 (한글/영문)
    - [ ] 검색 정확도 테스트 (의도한 문서가 상위에 오는지)
- [ ] **통합 테스트**
    - [ ] 전체 시나리오 수행 (업로드 -> 질문 -> 답변 -> 출처 확인)
- [ ] **README 작성**
    - [ ] 설치 및 실행 방법 문서화

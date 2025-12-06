# 구현 계획 - 로컬 RAG 챗봇 (Ollama)

## 목표 설명
사용자의 Windows PC에서 완전히 구동되는 로컬 RAG (검색 증강 생성) 챗봇을 구축합니다. 이 시스템은 **Ollama**를 LLM으로, **LangChain**을 오케스트레이션 도구로, 그리고 로컬 벡터 데이터베이스를 사용하여 문서를 검색합니다.

## 사용자 검토 필요 사항
> [!IMPORTANT]
> **Ollama 필수**: 사용자의 PC에 [Ollama](https://ollama.com/)가 설치되어 있고 실행 중이어야 합니다.
> **모델 선택**: 로컬 하드웨어에서 속도와 성능의 균형을 위해 **`qwen2:7b`** (권장, 경량) 또는 `llama3.1:8b` 모델을 권장합니다.
> **다국어 지원**: PDF가 다양한 언어(한/영/불/독)로 제공되므로, 임베딩 모델과 LLM 프롬프트가 이를 처리할 수 있어야 합니다.
> **하드웨어 권장**: 최소 16GB RAM (M3/M2 Mac 또는 동급 Windows PC). 메모리 부족 시 Rerank 비활성화 권장.

## 제안 아키텍처 (초안)

### 1. 프로그램 환경
*   **OS**: Windows (사용자 현재 OS)
*   **언어**: Python 3.10+
*   **핵심 프레임워크**: [LangChain](https://python.langchain.com/) (RAG 로직용)
*   **사용자 인터페이스**: [Streamlit](https://streamlit.io/) (깔끔한 웹 기반 채팅 인터페이스용)
*   **LLM 제공자**: Ollama (로컬 포트 11434에서 실행)
*   **임베딩**: `BAAI/bge-m3` (다국어 성능 우수).
*   **검색 전략**: **Hybrid Search** (BM25 + Vector).
*   **Context Limit**: LLM 입력 청크 **기본 3개** (16GB 환경), 최대 5개로 제한 (속도/정확도/메모리 균형).

### 2. Ollama 설치 및 모델 준비

#### 모델 다운로드
실행 전 아래 명령어로 필요한 모델을 다운로드합니다:
```bash
ollama pull qwen2:7b
ollama pull llama3.1:8b
```

#### 모델 선택 기준
*   **다국어 문서 중심**: `qwen2:7b` 권장 (한/영/중 등 다국어 성능 우수)
*   **영어 위주 + 긴 문서**: `llama3.1:8b` 옵션 (컨텍스트 처리 능력 우수)
*   **GPU 없는 환경**: `qwen2:7b` + Context 2~3개로 제한, 답변 길이 축소

#### 성능 최적화 팁
*   **답변 속도 느릴 때**: Context 개수 2개로 축소, `temperature` 낮춤(0.3~0.5), `max_tokens` 제한(512 이하)
*   **VRAM 부족 에러**: Rerank OFF, Context 2개로 제한
*   **Ollama 실행 확인**: `http://localhost:11434` 접속하여 서비스 상태 확인

### 3. 파라미터 기준선

#### 기본 설정 (MVP - 필수 구현)
*   **ChromaDB HNSW**: `M=16`, `ef_construction=100`, `ef=30`
*   **Chunk Size**: 600~800 (고정)
*   **Chunk Overlap**: 100
*   **Search Pipeline**: BM25 + Vector → 가중 평균 병합 (EnsembleRetriever)
*   **Rerank**: OFF
*   **Query Expansion/Rewriting/Decomposition**: OFF
*   **Context Count**: 3개

#### 고급 설정 (Advanced - 선택적 구현, 16GB+ RAM)
*   **ChromaDB HNSW**: `M=32`, `ef_construction=200`, `ef=50` (메모리 여유 시)
*   **Chunk Size**: 500~1000 (Semantic-Aware Adaptive)
*   **Chunk Overlap**: 150
*   **Search Pipeline**: RRF Fusion + Context Filtering + Rerank
*   **Rerank**: ON (`BAAI/bge-reranker-v2-m3`)
*   **Query Expansion**: Smart Expansion (명사 중심 짧은 쿼리)
*   **Query Rewriting/Decomposition**: LLM 기반 (선택)
*   **Context Count**: 5개

#### 저사양 환경 설정 (8GB RAM)
*   **LLM 모델**: `qwen2:7b`
*   **ChromaDB HNSW**: `M=8`, `ef_construction=50`, `ef=20`
*   **Chunk Size**: 600 (고정)
*   **Context Count**: 2개
*   **Rerank**: OFF (필수)
*   **Query Expansion/Rewriting**: OFF
*   **권장**: 답변 길이 제한, Temperature 낮춤
> [!NOTE]
> 8GB 환경에서는 실행 가능하지만 속도가 느릴 수 있습니다. 메모리 부족 시 브라우저 탭 최소화 권장.

#### 고사양 환경 설정 (32GB+ RAM)
*   **LLM 모델**: `llama3.1:8b` 또는 `qwen2:7b`
*   **ChromaDB HNSW**: `M=32`, `ef_construction=200`, `ef=50`
*   **Context Count**: 5개
*   **Rerank**: ON
*   **Query Expansion**: ON (짧은 질문일 때만)
*   **성능**: 모든 고급 기능 활성화 가능

### 4. 권장 데이터베이스 (Vector Store)
**주요 권장 사항: ChromaDB**
*   **이유**: 오픈 소스이며, 파일 기반 데이터베이스로 로컬에서 실행됩니다 (별도 서버 설정 불필요). LangChain과의 연동성이 매우 뛰어납니다. 데이터를 쉽게 영구 저장할 수 있어 앱을 재시작할 때마다 문서를 다시 인덱싱할 필요가 없습니다.

**대안: FAISS**
*   **이유**: 매우 빠르지만, 일반적으로 메모리 내에서 실행됩니다. 영구 저장을 위해서는 인덱스를 수동으로 저장/로드해야 합니다. 매우 큰 정적 데이터셋에는 좋지만, 이 사용 사례에는 ChromaDB가 개발자 친화적입니다.

## 제안 변경 사항

### [프로젝트 구조]

#### [IMPLEMENTED] [requirements.txt](file:///C:/Users/이은수/withai/rag-v2/requirements.txt)

**필수 패키지 (MVP - 1단계)**
*   `langchain`, `langchain-community` - RAG 오케스트레이션
*   `streamlit` - 웹 UI
*   `chromadb` - 벡터 데이터베이스
*   `sentence-transformers` - 임베딩 모델
*   `pypdf` - PDF 처리
*   `rank_bm25` - BM25 검색
*   `markdown` - Markdown 처리

**선택 패키지 (2단계 - Office 문서)**
*   `openpyxl` - Excel 파일 처리
*   `python-docx` - Word 파일 처리
*   `python-pptx` - PowerPoint 파일 처리

**선택 패키지 (3단계 - OCR)**
*   `pytesseract` - OCR 엔진
*   `pillow` - 이미지 처리

> [!TIP]
> MVP 단계에서는 필수 패키지만 설치하여 초기 복잡도를 낮추고, 2~3단계는 필요 시 점진적으로 추가하세요.

#### [IMPLEMENTED] [app.py](file:///C:/Users/이은수/withai/rag-v2/app.py)
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
        *   **API 인증**: ChromaDB 서버 모드 사용 시 Token 기반 인증 (로컬 임베디드 모드에서는 불필요).
    *   **동시성 관리**:
        *   동시 업로드 방지: 업로드 중에는 업로드 버튼 비활성화 + 상태 메시지 표시.
        *   Chroma collection 동시 접근 시 race condition 방지.
        *   > [!WARNING]
        *   > 이 버전은 단일 사용자(local) 환경 기준으로 설계되었습니다. 다중 사용자 환경에서는 세션 분리 및 DB 동시 접근 제어 로직 추가가 필요합니다.
    *   **UX 강화**:
        *   업로드 후 **샘플 질문 자동 추천** (문서 목차 기반).
        *   답변 내 키워드 **하이라이트** 표시.
        *   최근 대화 기록 자동 저장.
        *   **다국어 UI 지원**: 한국어/영어 인터페이스 전환 기능.

#### [IMPLEMENTED] [rag_engine.py](file:///C:/Users/이은수/withai/rag-v2/rag_engine.py)

**모듈 설계**
> [!IMPORTANT]
> `rag_engine.py`를 단일 파일로 구현하면 파일 길이가 지나치게 길어집니다. 아래와 같이 모듈 분리를 권장합니다.

*   **loaders.py**: 파일 타입별 Loader (PDFLoader, MarkdownLoader, TextLoader, DocxLoader 등)
*   **indexer.py**: Chroma 인덱스 생성/갱신, 해시 비교, 버전 관리
*   **retriever.py**: BM25 + Vector + RRF + Rerank 조합 로직
*   **llm_client.py**: Ollama 호출, timeout / 재시도 / system prompt 관리
*   **rag_engine.py**: 위 모듈들을 조합하는 상위 레이어

**RAG 로직 캡슐화**:
    *   **문서 로딩 (단계별 Multi-format Support)**:
        *   **1단계 (MVP)**: PDF, Markdown (.md), Text (.txt)
        *   **2단계**: Word (.docx), PowerPoint (.pptx), Excel
        *   **3단계**: ZIP 파일 내부 자동 분석, 이미지 OCR
        *   > [!NOTE]
        *   > 포맷별 텍스트 품질 차이가 크므로 단계적 검증 후 확장. 초기에는 PDF/TXT/MD만 지원하여 파서 안정성 확보.
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
    *   **텍스트 분할 (Chunking Strategy)**:
        *   **MVP**: 고정 크기 (600~800 토큰, overlap 100)
        *   **Advanced**: Semantic-Aware Adaptive Chunking (문서 유형별 크기 조정)
            *   일반 문서: 500~700
            *   기술/API 문서: 800~1000
            *   문서 유형 구분: 파일명/경로/키워드 기반 태깅
        *   **Context Enrichment**: 각 Chunk에 문서 제목, 섹션 메타데이터 추가 (검색 품질 향상).
        *   > [!CAUTION]
        *   > Semantic-Aware Chunking은 문서 분류 로직이 필요하여 복잡도가 높음. MVP는 고정 크기로 시작 권장.
    *   Vector DB 생성/로드 (Chroma).
        *   **Incremental Indexing (점진적 인덱싱 - Advanced)**:
            *   **MVP**: 파일 전체 해시 기반 변경 감지, 변경 시 전체 재인덱싱
            *   **Advanced**: 페이지/섹션 단위 변경 감지 및 부분 재인덱싱
            *   > [!WARNING]
            *   > 페이지 단위 Incremental Indexing은 1~2주 수준의 작업량. MVP는 파일 단위로 시작.
        *   **버전 관리**: 파일명/해시 기반 Collection 분리 (이전 데이터 오염 방지).
        *   **최적화**: `HNSW` 파라미터는 **파라미터 기준선** 섹션 참조, `cosine` 거리 사용.
        *   **메타데이터**: 파일명, 페이지 번호, 시작/끝 토큰 인덱스 저장.
    *   **검색 파이프라인 (계층별 구현)**:
        *   **[MVP 기본 파이프라인]**
            1. Vector 검색 (Top K=20)
            2. BM25 검색 (Top K=20)
            3. 단순 가중 평균 병합 (LangChain EnsembleRetriever)
            4. 상위 3~5개 컨텍스트 추출
        *   **[고급 파이프라인 - 단계적 활성화]**
            *   **Query Preprocessing** (비활성화):
                *   Query Rewriting: LLM 기반 질문 명확화
                *   Query Decomposition: 복잡한 질문 분해
                *   > [!TIP]
                *   > Ollama 로컬 LLM 속도 고려 시 체감 지연 발생 가능. 실사용 패턴 확인 후 추가 권장.
            *   **RRF (Reciprocal Rank Fusion)**: BM25/Vector 결과 고급 병합
            *   **Query Expansion**: 명사 위주 짧은 질문에만 적용
            *   **Context Filtering**: 키워드 기반 필터링, 제목/소제목 가중치
            *   **Rerank**: 검색 결과 재순위화 (`BAAI/bge-reranker-v2-m3`)
        *   > [!IMPORTANT]
        *   > MVP는 기본 파이프라인으로 시작. RRF/Rerank/Query Rewriting은 Optional 섹션으로 분리하여 디버깅 복잡도 감소.
    *   Ollama 쿼리 (System Prompt 최적화, Context Max 5).
    *   **Prompt Engineering (연구 기반)**:
        *   **Contextual Instructions**: LLM에게 제공된 Context만 사용하도록 명시적 지시.
        *   **Hallucination Prevention**: 답변 불가 시 "모르겠다"고 답변하도록 설정.
    *   **설정 관리 (Configuration)**:
        *   `config.yaml` 또는 DB를 통해 사용자 정의 설정(Chunk, Prompt, Params) 영구 저장.
        *   **동적 파이프라인**: 설정 변경 시 RAG 파이프라인(Splitter, Retriever) 즉시 반영.
        *   **보안 설정**:
            *   **관리자 비밀번호**: `config.yaml`에 평문 저장 금지. SHA-256 해시 + salt 사용.
            *   실제 비밀번호는 `.env` 파일 또는 OS keyring으로 분리.
            *   예: `password_hash`, `salt` 저장 후 입력값과 비교.
        *   **로그 보안**:
            *   `logs/queries.jsonl`은 로컬에만 저장 (기본값).
            *   클라우드 동기화 경로(OneDrive, Dropbox 등)에 두지 말 것.
            *   질문/답변이 내부 기밀을 포함할 수 있으므로 주의 필요.
        *   **파일 암호화**:
            *   1차 버전에서는 OS 계정 권한으로 보호.
            *   암호화는 향후 옵션으로 고려.
    *   **모델 변경 지원**: `st.session_state` 활용하여 모델 객체 캐싱.
    *   **상세 로깅**:
        *   질문, 답변, 소요 시간.
        *   유사도 점수, BM25 점수, Rerank 점수.
        *   Query Expansion 적용 여부, Routing 규칙.
        *   단계별 Latency (검색, LLM 생성 등).

## 추가 개발 편의 기능

### CLI 테스트 스크립트
*   Streamlit 없이 RAG 엔진만 검증하는 커맨드라인 인터페이스
*   예: `python -m rag_engine.cli "질문" --doc sample.pdf`
*   용도: UI 문제와 분리하여 엔진 로직 디버깅

### 로그 저장 형식
*   **형식**: `logs/queries.jsonl` (JSON Lines)
*   **저장 내용**:
    *   질문, 답변, 타임스탬프
    *   검색 결과 (유사도 점수, BM25 점수, Rerank 점수)
    *   단계별 Latency (검색/LLM 생성)
    *   Query Expansion/Routing 적용 여부
*   **용도**: 검색 품질 튜닝 데이터 분석

### Safe Mode 플래그
*   **목적**: 모든 고급 기능 일시 비활성화하여 베이스라인 성능 측정
*   **비활성화 대상**: Query Expansion, Rerank, Rewriting, Decomposition, RRF
*   **사용 시나리오**: 문제 발생 시 빠른 원인 격리 및 비교
*   **설정 방법**: `config.yaml`의 `safe_mode: true` 또는 환경 변수

## 성능 최적화 및 모니터링
### 제한된 하드웨어 환경 (16GB RAM) 권장 설정
> 자세한 파라미터 설정은 **파라미터 기준선** 섹션 참조

*   **LLM 모델**: `qwen2:7b` (기본), `llama3.1:8b` (선택)
*   **Pipeline Mode**: MVP (기본 파이프라인)
*   **Safe Mode**: 필요 시 활성화하여 고급 기능 비활성화

### 성능 모니터링
*   **메모리 사용량**: Activity Monitor (Mac) / Task Manager (Windows)로 실시간 모니터링
*   **목표**: 총 메모리 사용량 13GB 이하 유지
*   **경고**: 15GB 초과 시 Rerank 비활성화 또는 모델 변경

## 검증 계획

### 자동화 테스트
*   `rag_engine.py`에 대한 단위 테스트를 통해 문서가 올바르게 인덱싱되고 검색되는지 확인합니다.
*   **정확도 평가 세트**:
    *   `eval/qa_pairs.jsonl` 형태로 테스트 질문-답변 쌍 정의:
        ```json
        {"question": "...", "expected_keywords": ["키워드1", "키워드2"]}
        ```
    *   RAG 파이프라인 변경 후 자동 점수 측정:
        *   답변에 `expected_keywords` 포함 여부 체크
        *   향후 Hit Rate, MRR 등 고급 메트릭 추가 고려
    *   Rerank ON/OFF, Query Expansion 등의 효과를 정량 비교 가능

### 수동 검증
1.  Ollama 실행 (`ollama serve`).
2.  `streamlit run app.py` 실행.
3.  샘플 PDF 업로드.
4.  PDF 내용과 관련된 질문 수행.
5.  답변이 정확하고 출처를 인용하는지 확인.

## 운영 및 배포 전략

### 설치 절차
1.  **Python 3.10+ 설치** 확인
2.  **가상환경 생성 및 활성화**:
    ```bash
    python -m venv venv
    venv\Scripts\activate  # Windows
    ```
3.  **패키지 설치**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Ollama 설치 및 모델 다운로드**:
    ```bash
    # Ollama 설치: https://ollama.com/
    ollama pull qwen2:7b
    ollama pull llama3.1:8b
    ```
5.  **앱 실행**:
    ```bash
    streamlit run app.py
    ```

### 백업 및 복구
*   **ChromaDB 저장 경로**: `./chroma_db/` (기본값)
*   **백업 대상**: `chroma_db/` 폴더 + `logs/` 폴더
*   **복구 방법**: 백업한 폴더를 프로젝트 루트에 복사하면 인덱스 자동 복구
*   **주의사항**: 임베딩 모델 변경 시 인덱스 호환성 보장 안 됨 (전체 재인덱싱 필요)

### 업데이트 전략
*   **코드 업데이트**: 인덱스 유지 가능 (RAG 로직만 변경 시)
*   **임베딩 모델 변경**: 인덱스 전체 재생성 필요
    *   예: `BAAI/bge-m3` → `all-MiniLM-L6-v2` 변경 시
    *   기존 `chroma_db/` 폴더 백업 후 삭제, 재인덱싱 수행
*   **Chunk Size/Overlap 변경**: 재인덱싱 권장 (일관성 유지)

### 문제 해결 가이드
*   **Ollama 연결 실패**: `http://localhost:11434` 접속 확인, `ollama serve` 재실행
*   **메모리 부족**: Rerank OFF, Context 2개로 축소
*   **검색 결과 이상**: Safe Mode 활성화 → 베이스라인 성능 확인 → 기능 하나씩 재활성화
*   **인덱싱 실패**: 로그(`logs/`) 확인, 파일 인코딩 체크

## 현재 구현 상태 (2025-12-05)
*   **핵심 기능 (MVP)**: 완료 (PDF/MD/TXT 로드, Vector/BM25 검색, Ollama 연동)
*   **고급 기능 (Advanced)**:
    *   **Office 문서**: 완료 (Word, Excel, PPT)
    *   **OCR**: 완료 (Tesseract + Poppler)
    *   **고급 검색**: 완료 (RRF, Rerank)
*   **테스트**:
    *   단위 테스트 및 통합 테스트 진행 중
    *   평가 스크립트(`evaluate.py`) 동작 확인

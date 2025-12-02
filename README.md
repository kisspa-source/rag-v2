# 로컬 RAG 챗봇 (Ollama)

Windows PC에서 완전히 구동되는 로컬 RAG (검색 증강 생성) 챗봇입니다. Ollama를 LLM으로, LangChain을 오케스트레이션 도구로, ChromaDB를 벡터 데이터베이스로 사용합니다.

## 주요 기능

- 📄 **다양한 문서 지원**: PDF, Markdown, Text 파일
- 🔍 **Hybrid 검색**: BM25 + Vector 검색 결합
- 🤖 **로컬 LLM**: Ollama 기반 (완전 오프라인 실행 가능)
- 💾 **Incremental Indexing**: 변경된 파일만 재인덱싱
- 📊 **출처 추적**: 답변에 문서명 및 페이지 번호 표시
- 🎯 **웹 UI**: Streamlit 기반 사용자 친화적 인터페이스

## 필수 요구사항

- **Python**: 3.10 이상
- **RAM**: 최소 16GB 권장 (8GB 환경에서도 실행 가능하나 느릴 수 있음)
- **Ollama**: 로컬 LLM 실행 환경

## 설치 방법

### 1. Ollama 설치

[Ollama 공식 사이트](https://ollama.com/)에서 다운로드 및 설치

**모델 다운로드**:
```bash
ollama pull qwen2:7b
ollama pull llama3.1:8b
```

### 2. Python 가상환경 생성

```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### 3. 패키지 설치

```bash
pip install -r requirements.txt
```

## 사용 방법

### Streamlit 웹 UI 실행

```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 접속

### CLI 사용 (선택 사항)

**파일 인덱싱**:
```bash
python rag_engine.py --index sample.pdf
```

**질문하기**:
```bash
python rag_engine.py --query "문서의 주요 내용은?"
```

**인덱싱된 파일 목록**:
```bash
python rag_engine.py --list
```

## 프로젝트 구조

```
rag-v2/
├── app.py                 # Streamlit 웹 UI
├── rag_engine.py          # 통합 RAG 엔진
├── loaders.py             # 문서 로더 (PDF, MD, TXT)
├── indexer.py             # 청킹 및 벡터 인덱싱
├── retriever.py           # Hybrid 검색기 (BM25 + Vector)
├── llm_client.py          # Ollama LLM 클라이언트
├── config.yaml            # 설정 파일
├── requirements.txt       # Python 패키지
│
├── chroma_db/             # Vector DB 저장소
├── logs/                  # 쿼리 로그
└── data/                  # 문서 저장소 (선택)
```

## 설정 (config.yaml)

주요 파라미터를 조정할 수 있습니다:

```yaml
rag:
  chunk_size: 800          # 청크 크기
  chunk_overlap: 100       # 청크 오버랩
  context_count: 3         # LLM에 전달할 컨텍스트 개수

llm:
  model_name: "qwen2:7b"   # Ollama 모델
  temperature: 0.3         # Temperature
  max_tokens: 512          # 최대 토큰 수
```

## 성능 최적화

### 환경별 권장 설정

**16GB RAM (기본)**:
- Context Count: 3
- Chunk Size: 800
- Model: `qwen2:7b`

**8GB RAM (저사양)**:
- Context Count: 2
- Chunk Size: 600
- Model: `qwen2:7b`
- Rerank: OFF

**32GB+ RAM (고사양)**:
- Context Count: 5
- Model: `llama3.1:8b`
- 고급 기능 활성화 가능

## 문제 해결

### Ollama 연결 실패
```bash
# Ollama 서비스 상태 확인
curl http://localhost:11434

# Ollama 재시작
ollama serve
```

### 메모리 부족
- `config.yaml`에서 `context_count`를 2로 축소
- Chunk Size를 600으로 축소
- 브라우저 탭 최소화

### 검색 결과 이상
- Safe Mode 활성화 (`config.yaml`에 `safe_mode: true` 추가)
- 로그 확인: `logs/queries.jsonl`

## 로그 및 백업

### 쿼리 로그
- 위치: `logs/queries.jsonl`
- 형식: JSON Lines
- 내용: 질문, 답변, 소요 시간, 점수 등

### 백업
- 백업 대상: `chroma_db/` 폴더
- 복구: 백업한 폴더를 프로젝트 루트에 복사

**주의**: 임베딩 모델 변경 시 전체 재인덱싱 필요

## 개발 정보

- **LangChain**: RAG 파이프라인
- **ChromaDB**: 벡터 데이터베이스 (HNSW 인덱스)
- **Ollama**: 로컬 LLM
- **Streamlit**: 웹 UI
- **BM25**: 키워드 검색
- **BGE-M3**: 다국어 임베딩 모델

## 라이선스

MIT License

## 참고 자료

- [Ollama 공식 문서](https://github.com/ollama/ollama)
- [LangChain 문서](https://python.langchain.com/)
- [ChromaDB 문서](https://docs.trychroma.com/)

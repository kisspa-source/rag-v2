# 🧪 로컬 실행 및 테스트 가이드

이 문서는 **Local RAG Chatbot** 프로젝트를 로컬 macOS 환경에서 실행하고 기능을 검증하기 위한 상세 가이드입니다.

## 1. 사전 요구 사항 (Prerequisites)

시작하기 전에 다음 도구들이 설치되어 있어야 합니다.

*   **Terminal**: macOS 기본 터미널 또는 iTerm2
*   **Homebrew**: macOS 패키지 관리자 (아직 없다면 [설치 필요](https://brew.sh/))
*   **Ollama**: 로컬 LLM 실행 도구 ([다운로드](https://ollama.com/))

## 2. 설치 및 환경 설정 (Installation)

### 2.1 시스템 도구 설치
OCR(광학 문자 인식) 및 PDF 처리를 위해 시스템 라이브러리가 필요합니다.

```bash
brew install tesseract poppler
```

### 2.2 프로젝트 경로 이동
터미널을 열고 프로젝트 폴더로 이동합니다.

```bash
cd /Users/mac/Documents/withAI/Projects/rag-v2
```

### 2.3 가상환경 생성 및 활성화
독립적인 Python 환경을 구성합니다.

```bash
# 가상환경 생성 (최초 1회)
python3 -m venv venv

# 가상환경 활성화 (매번 터미널 열 때마다 필요)
source venv/bin/activate
```
> 활성화되면 터미널 프롬프트 앞에 `(venv)`가 표시됩니다.

### 2.4 파이썬 패키지 설치
필요한 라이브러리들을 설치합니다.

```bash
pip install -r requirements.txt
```

### 2.5 Ollama 모델 다운로드
프로젝트 기본 설정(`config.yaml`)에 맞춰 모델을 다운로드합니다.

```bash
# Ollama 앱이 실행 중인지 확인 후 입력하세요
ollama pull qwen2:7b
```

---

## 3. 앱 실행 방법 (Running the App)

모든 설치가 끝났다면 다음 명령어로 앱을 실행합니다.

```bash
streamlit run app.py
```

브라우저가 자동으로 열리며 `http://localhost:8501` 주소로 접속됩니다.

---

## 4. 기능 테스트 시나리오 (Testing Scenarios)

### ✅ STEP 1: 로그인
*   **비밀번호**: `admin123`
*   설정 파일(`config.yaml`)에 해시로 저장되어 있으며, 초기 비밀번호입니다.

### ✅ STEP 2: 문서 인덱싱 테스트
1.  좌측 사이드바의 **"문서 업로드"** 섹션으로 이동합니다.
2.  테스트할 문서를 업로드합니다 (제공된 `sample.docx` 또는 `README.md` 등).
3.  **"📥 인덱싱 시작"** 버튼을 클릭합니다.
4.  **검증 포인트**:
    *   진행률 바가 표시되는가?
    *   "✅ 인덱싱 완료!" 메시지가 뜨는가?
    *   소요 시간이 합리적인가? (수 초 이내)

### ✅ STEP 3: 채팅 및 검색 답변 테스트
1.  업로드한 문서 내용에 대해 질문합니다.
    *   *예시: "이 프로젝트의 주요 기능은 무엇인가요?"*
2.  **검증 포인트**:
    *   답변이 문서 내용을 바탕으로 생성되는가?
    *   답변 아래에 **"📚 출처:"** 가 올바르게 표시되는가?

### ✅ STEP 4: 고급 기능 테스트 (OCR)
1.  이미지로 된 PDF나 스캔 문서를 업로드해 봅니다 (OCR 테스트용).
2.  텍스트가 정상적으로 추출되어 답변이 가능한지 확인합니다.
3.  *참고: OCR은 시간이 조금 더 소요될 수 있습니다.*

### ✅ STEP 5: 설정 변경 테스트
1.  상단 탭에서 **"⚙️ 설정"**을 클릭합니다.
2.  **성능 프리셋**을 변경해 봅니다 (예: 8GB -> 16GB).
3.  **저장** 버튼을 누르고 앱이 재시작될 때 설정이 유지되는지 확인합니다.

---

## 5. 문제 해결 (Troubleshooting)

**Q. `command not found: ollama` 에러가 나요.**
A. Ollama가 설치되지 않았습니다. [ollama.com](https://ollama.com)에서 다운로드하여 설치하세요.

**Q. `SystemExit: ... poppler is not installed` 에러가 나요.**
A. `brew install poppler` 명령어를 실행했는지 확인하세요.

**Q. 답변이 너무 느려요.**
A. `config.yaml` 또는 설정 탭에서 `Use Rerank` 옵션을 끄거나(False), 모델을 더 작은 것(예: qwen2:1.5b)으로 변경해 보세요.

**Q. 로그인이 안 돼요.**
A. `config.yaml` 파일의 `security` 섹션을 확인하거나, 코드를 수정하여 비밀번호 체크 로직을 임시로 주석 처리할 수 있습니다.

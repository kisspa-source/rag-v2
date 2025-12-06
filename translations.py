"""
Translation dictionary for the application
"""

TRANSLATIONS = {
    # Page Config & Title
    "page_title": {
        "kor": "로컬 RAG 챗봇",
        "eng": "Local RAG Chatbot"
    },
    "app_title": {
        "kor": "🤖 로컬 RAG 챗봇",
        "eng": "🤖 Local RAG Chatbot"
    },
    
    # Login
    "login_header": {
        "kor": "🔐 관리자 로그인",
        "eng": "🔐 Admin Login"
    },
    "password_placeholder": {
        "kor": "비밀번호를 입력하세요",
        "eng": "Enter password"
    },
    "login_button": {
        "kor": "로그인",
        "eng": "Login"
    },
    "login_success": {
        "kor": "로그인 성공!",
        "eng": "Login Successful!"
    },
    "login_failed": {
        "kor": "비밀번호가 올바르지 않습니다.",
        "eng": "Incorrect password."
    },
    
    # Sidebar
    "logout_button": {
        "kor": "로그아웃",
        "eng": "Logout"
    },
    "single_user_warning": {
        "kor": "⚠️ 이 시스템은 단일 사용자 환경(Local)에 최적화되어 있습니다.",
        "eng": "⚠️ This system is optimized for single-user (Local) environment."
    },
    "upload_header": {
        "kor": "📄 문서 업로드",
        "eng": "📄 Document Upload"
    },
    "file_uploader_label": {
        "kor": "PDF, Markdown, Text 파일 업로드",
        "eng": "Upload PDF, Markdown, Text files"
    },
    "file_uploader_help": {
        "kor": "지원 형식: PDF, Markdown (.md), Text (.txt)",
        "eng": "Supported formats: PDF, Markdown (.md), Text (.txt)"
    },
    "start_indexing_button": {
        "kor": "📥 선택한 파일 인덱싱 시작",
        "eng": "📥 Start Indexing Selected Files"
    },
    "connection_status_header": {
        "kor": "🔌 연결 상태",
        "eng": "🔌 Connection Status"
    },
    "test_connection_button": {
        "kor": "연결 테스트",
        "eng": "Test Connection"
    },
    "testing_spinner": {
        "kor": "테스트 중...",
        "eng": "Testing..."
    },
    "connection_success": {
        "kor": "✅ Ollama 연결 성공",
        "eng": "✅ Ollama Connection Successful"
    },
    "connection_fail": {
        "kor": "❌ Ollama 연결 실패",
        "eng": "❌ Ollama Connection Failed"
    },
    "connection_fail_help": {
        "kor": "'ollama serve'를 실행하세요",
        "eng": "Please run 'ollama serve'"
    },
    
    # Tabs
    "tab_chat": {
        "kor": "💬 채팅",
        "eng": "💬 Chat"
    },
    "tab_settings": {
        "kor": "⚙️ 설정",
        "eng": "⚙️ Settings"
    },
    
    # Chat Interface
    "chat_intro": {
        "kor": "문서를 업로드하고 질문하세요!",
        "eng": "Upload documents and ask questions!"
    },
    "processing_status": {
        "kor": "파일 처리 중...",
        "eng": "Processing files..."
    },
    "indexing_processing": {
        "kor": "📄 처리 중",
        "eng": "📄 Processing"
    },
    "indexing_success": {
        "kor": "성공",
        "eng": "Success"
    },
    "indexing_fail": {
        "kor": "실패",
        "eng": "Failed"
    },
    "indexing_error": {
        "kor": "오류",
        "eng": "Error"
    },
    "all_files_indexed": {
        "kor": "✅ 모든 파일({count}개) 인덱싱 완료!",
        "eng": "✅ All files ({count}) indexed successfully!"
    },
    "file_list_updated": {
        "kor": "파일 목록이 갱신되었습니다.",
        "eng": "File list updated."
    },
    "indexing_result_partial": {
        "kor": "⚠️ 완료: 성공 {success}, 실패 {fail}",
        "eng": "⚠️ Done: Success {success}, Fail {fail}"
    },
    "suggested_questions": {
        "kor": "💡 추천 질문",
        "eng": "💡 Suggested Questions"
    },
    "chat_input_placeholder": {
        "kor": "질문을 입력하세요",
        "eng": "Enter your question"
    },
    "warning_upload_first": {
        "kor": "먼저 문서를 업로드하고 인덱싱하세요!",
        "eng": "Please upload and index documents first!"
    },
    "generating_answer": {
        "kor": "답변 생성 중...",
        "eng": "Generating answer..."
    },
    "source_reference": {
        "kor": "📚 출처",
        "eng": "📚 Sources"
    },
    "performance_info": {
        "kor": "⏱️ 성능 정보",
        "eng": "⏱️ Performance Info"
    },
    "total_time": {
        "kor": "총 소요 시간",
        "eng": "Total time"
    },
    "search_time": {
        "kor": "검색",
        "eng": "Search"
    },
    "llm_time": {
        "kor": "LLM",
        "eng": "LLM"
    },
    "clear_history_button": {
        "kor": "🗑️ 대화 기록 지우기",
        "eng": "🗑️ Clear Chat History"
    },

    # Settings Interface
    "settings_header": {
        "kor": "⚙️ 환경 설정",
        "eng": "⚙️ Configuration"
    },
    "load_config_fail": {
        "kor": "설정 파일 로드 실패",
        "eng": "Failed to load config file"
    },
    "performance_preset_header": {
        "kor": "🚀 성능 프리셋",
        "eng": "🚀 Performance Preset"
    },
    "preset_limit_help": {
        "kor": "하드웨어 환경에 맞는 프리셋을 선택하세요",
        "eng": "Select a preset matching your hardware"
    },
    "preset_help_text": {
        "kor": "8GB: 저사양 / 16GB: 기본 / 32GB: 고사양",
        "eng": "8GB: Low / 16GB: Standard / 32GB: High"
    },
    "rag_parameters_header": {
        "kor": "RAG 파라미터",
        "eng": "RAG Parameters"
    },
    "top_k_label": {
        "kor": "Top-K (검색 개수)",
        "eng": "Top-K (Retrieval Count)"
    },
    "context_count_label": {
        "kor": "Context Count (LLM 입력 개수)",
        "eng": "Context Count (LLM Input Count)"
    },
    "llm_settings_header": {
        "kor": "LLM 설정",
        "eng": "LLM Settings"
    },
    "ollama_model_label": {
        "kor": "Ollama 모델",
        "eng": "Ollama Model"
    },
    "timeout_label": {
        "kor": "Timeout (초)",
        "eng": "Timeout (sec)"
    },
    "save_settings_button": {
        "kor": "💾 설정 저장",
        "eng": "💾 Save Settings"
    },
    "settings_saved": {
        "kor": "설정이 저장되었습니다. 적용을 위해 앱을 다시 로드합니다.",
        "eng": "Settings saved. Reloading app to apply."
    },
    
    # History Management
    "history_management_header": {
        "kor": "💾 대화 기록 관리",
        "eng": "💾 Chat History Management"
    },
    "export_json_button": {
        "kor": "📤 대화 기록 내보내기 (JSON)",
        "eng": "📤 Export History (JSON)"
    },
    "export_txt_button": {
        "kor": "📄 대화 기록 내보내기 (TXT)",
        "eng": "📄 Export History (TXT)"
    },
    "import_label": {
        "kor": "대화 기록 불러오기 (JSON)",
        "eng": "Import History (JSON)"
    },
    "import_button": {
        "kor": "📥 불러오기",
        "eng": "📥 Import"
    },
    "import_success": {
        "kor": "대화 기록을 불러왔습니다.",
        "eng": "Chat history imported."
    },
    "import_fail": {
        "kor": "불러오기 실패",
        "eng": "Import failed"
    },
    
    # File Management
    "file_management_header": {
        "kor": "📚 파일 관리",
        "eng": "📚 File Management"
    },
    "refresh_list_button": {
        "kor": "🔄 목록 새로고침",
        "eng": "🔄 Refresh List"
    },
    "indexed_files_list": {
        "kor": "##### 인덱싱된 파일 목록",
        "eng": "##### Indexed Files List"
    },
    "delete_button": {
        "kor": "삭제",
        "eng": "Delete"
    },
    "delete_success": {
        "kor": "삭제됨",
        "eng": "Deleted"
    },
    "delete_fail": {
        "kor": "삭제 실패",
        "eng": "Deletion Failed"
    },
    "delete_all_button": {
        "kor": "🗑️ 전체 파일 삭제",
        "eng": "🗑️ Delete All Files"
    },
    "delete_all_confirm": {
        "kor": "정말 모든 파일을 삭제하시겠습니까?",
        "eng": "Are you sure you want to delete all files?"
    },
    "delete_all_success": {
        "kor": "모든 파일이 삭제되었습니다.",
        "eng": "All files deleted."
    },
    "no_indexed_files": {
        "kor": "인덱싱된 파일이 없습니다.",
        "eng": "No indexed files."
    },
    
    # Auth
    "auth_error": {
        "kor": "인증 오류",
        "eng": "Auth Error"
    },
    "rag_init_fail": {
        "kor": "RAG 엔진 초기화 실패",
        "eng": "RAG Engine Init Failed"
    },
    
    # System Prompt
    "system_prompt_default": {
        "kor": """당신은 제공된 문서를 바탕으로 질문에 답변하는 유능한 어시스턴트입니다.

다음 규칙을 준수하세요:
1. 제공된 컨텍스트 내용만을 사용하여 답변하세요.
2. 컨텍스트에 답이 없으면 "제공된 문서에서 해당 정보를 찾을 수 없습니다"라고 답변하세요.
3. 추측하거나 컨텍스트 외의 정보를 사용하지 마세요.
4. 답변은 명확하고 간결하게 작성하세요.
5. 가능한 경우 출처(문서 이름, 페이지)를 언급하세요.

컨텍스트:
{context}

질문: {question}

답변:""",
        "eng": """You are a capable assistant that answers questions based on the provided documents.

Please follow these rules:
1. Use ONLY the provided context to answer.
2. If the answer is not in the context, say "I cannot find the information in the provided documents."
3. Do not guess or use outside information.
4. Keep answers clear and concise.
5. Mention sources (document name, page) if possible.

Context:
{context}

Question: {question}

Answer:"""
    }
}

def get_text(key, lang='kor', **kwargs):
    """Retrieve translated text"""
    text = TRANSLATIONS.get(key, {}).get(lang, key)
    if kwargs:
        return text.format(**kwargs)
    return text

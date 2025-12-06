"""
Streamlit 기반 RAG 챗봇 UI
"""
import streamlit as st
import time
from pathlib import Path
import tempfile
import os
import json

import hashlib

from rag_engine import RAGEngine
from translations import get_text


# 페이지 설정
st.set_page_config(
    page_title="Local RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)


@st.cache_resource
def initialize_rag_engine():
    """RAG 엔진 초기화 (캐싱)"""
    return RAGEngine()


def check_password(password: str, lang: str) -> bool:
    """비밀번호 검증"""
    try:
        import yaml
        with open('config/config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        stored_hash = config.get('security', {}).get('admin_password_hash')
        salt = config.get('security', {}).get('salt')
        
        if not stored_hash or not salt:
            return True # 보안 설정이 없으면 통과 (또는 False로 막을 수도 있음)
            
        # 입력된 비밀번호 해시 생성
        salted_password = password + salt
        input_hash = hashlib.sha256(salted_password.encode()).hexdigest()
        
        return input_hash == stored_hash
    except Exception as e:
        st.error(f"{get_text('auth_error', lang)}: {e}")
        return False


def main():
    # 언어 설정 초기화
    if 'language' not in st.session_state:
        st.session_state.language = 'kor'

    st.title(get_text('app_title', st.session_state.language))
    
    # Session State 초기화
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    # 로그인 화면
    if not st.session_state.authenticated:
        # 로그인 화면 우측 상단 언어 선택
        lang_col1, lang_col2 = st.columns([8, 2])
        with lang_col2:
            lang_choice = st.radio("Language", ["Korean", "English"], 
                                 index=0 if st.session_state.language == 'kor' else 1,
                                 horizontal=True, label_visibility="collapsed")
            st.session_state.language = 'kor' if lang_choice == "Korean" else 'eng'

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.subheader(get_text('login_header', st.session_state.language))
            password = st.text_input(get_text('password_placeholder', st.session_state.language), type="password")
            if st.button(get_text('login_button', st.session_state.language), type="primary"):
                if check_password(password, st.session_state.language):
                    st.session_state.authenticated = True
                    st.success(get_text('login_success', st.session_state.language))
                    st.rerun()
                else:
                    st.error(get_text('login_failed', st.session_state.language))
        return

    # === 메인 앱 로직 ===
    
    # 사이드바에 로그아웃 버튼 추가
    with st.sidebar:
        # 언어 선택
        lang_choice_sidebar = st.radio("Language", ["Korean", "English"], 
                                     index=0 if st.session_state.language == 'kor' else 1,
                                     horizontal=True, label_visibility="collapsed", key="sidebar_lang")
        st.session_state.language = 'kor' if lang_choice_sidebar == "Korean" else 'eng'
        
        if st.button(get_text('logout_button', st.session_state.language)):
            st.session_state.authenticated = False
            st.rerun()
        st.divider()
        
        # 다중 사용자 경고
        st.info(get_text('single_user_warning', st.session_state.language))

    # RAG 엔진 초기화
    try:
        engine = initialize_rag_engine()
    except Exception as e:
        st.error(f"{get_text('rag_init_fail', st.session_state.language)}: {str(e)}")
        st.stop()
    
    # Session State 초기화 (메시지 등)
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'indexed_files' not in st.session_state:
        st.session_state.indexed_files = engine.get_indexed_files()
        
    if 'sample_questions' not in st.session_state:
        st.session_state.sample_questions = []
    
    # 탭 구성
    # 탭 구성
    tab_chat, tab_settings = st.tabs([get_text('tab_chat', st.session_state.language), get_text('tab_settings', st.session_state.language)])
    
    # === 채팅 탭 ===
    with tab_chat:
        st.markdown(get_text('chat_intro', st.session_state.language))
        
        # 사이드바 (파일 업로드만 유지)
        with st.sidebar:
            st.header(get_text('upload_header', st.session_state.language))
            
            # 파일 업로드 (Multi-file Support)
            uploaded_files = st.file_uploader(
                get_text('file_uploader_label', st.session_state.language),
                type=['pdf', 'md', 'txt'],
                help=get_text('file_uploader_help', st.session_state.language),
                accept_multiple_files=True
            )
            
            if uploaded_files:
                if st.button(get_text('start_indexing_button', st.session_state.language), type="primary"):
                    with st.status(get_text('processing_status', st.session_state.language), expanded=True) as status:
                        success_count = 0
                        fail_count = 0
                        
                        for i, uploaded_file in enumerate(uploaded_files):
                            st.write(f"{get_text('indexing_processing', st.session_state.language)} ({i+1}/{len(uploaded_files)}): {uploaded_file.name}")
                            
                            # 임시 파일로 저장
                            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                tmp_path = tmp_file.name
                            
                            try:
                                # 인덱싱
                                result = engine.load_and_index_file(tmp_path, original_filename=uploaded_file.name)
                                
                                if result['success']:
                                    st.write(f"✅ {uploaded_file.name}: {get_text('indexing_success', st.session_state.language)}")
                                    success_count += 1
                                    
                                    # 샘플 질문 갱신 (마지막 성공 파일 기준)
                                    if result.get('sample_questions'):
                                        st.session_state.sample_questions = result['sample_questions']
                                else:
                                    st.error(f"❌ {uploaded_file.name}: {get_text('indexing_fail', st.session_state.language)} - {result['message']}")
                                    fail_count += 1
                            except Exception as e:
                                st.error(f"❌ {uploaded_file.name}: {get_text('indexing_error', st.session_state.language)} - {str(e)}")
                                fail_count += 1
                            finally:
                                # 임시 파일 삭제
                                if os.path.exists(tmp_path):
                                    os.unlink(tmp_path)
                        
                        # 최종 결과 표시
                        if fail_count == 0:
                            status.update(label=get_text('all_files_indexed', st.session_state.language, count=success_count), state="complete", expanded=False)
                            st.info(get_text('file_list_updated', st.session_state.language))
                        else:
                            status.update(label=get_text('indexing_result_partial', st.session_state.language, success=success_count, fail=fail_count), state="error", expanded=True)
                        
                        # 파일 목록 갱신
                        st.session_state.indexed_files = engine.get_indexed_files()
            
            # Ollama 연결 상태
            st.divider()
            st.subheader(get_text('connection_status_header', st.session_state.language))
            
            if st.button(get_text('test_connection_button', st.session_state.language)):
                with st.spinner(get_text('testing_spinner', st.session_state.language)):
                    if engine.test_connection():
                        st.success(get_text('connection_success', st.session_state.language))
                    else:
                        st.error(get_text('connection_fail', st.session_state.language))
                        st.info(get_text('connection_fail_help', st.session_state.language))

        # 채팅 인터페이스
        
        # 샘플 질문 표시 (메시지가 없거나 샘플 질문이 있을 때)
        if st.session_state.sample_questions and not st.session_state.messages:
            st.info(get_text('suggested_questions', st.session_state.language))
            cols = st.columns(len(st.session_state.sample_questions))
            for i, question in enumerate(st.session_state.sample_questions):
                with cols[i]:
                    if st.button(question, key=f"sample_{i}"):
                        # 질문 입력창에 값을 채우는 것은 불가능하므로 바로 질문 처리
                        # 이를 위해 session_state에 임시 저장 후 rerun하거나
                        # 바로 처리 로직을 호출해야 함.
                        # 여기서는 messages에 추가하고 rerun하는 방식을 사용
                        st.session_state.messages.append({"role": "user", "content": question})
                        # 답변 생성을 위해 플래그 설정
                        st.session_state.trigger_query = question
                        st.rerun()

        # 채팅 히스토리 표시
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # 출처 표시
                if message["role"] == "assistant" and "sources" in message:
                    if message["sources"]:
                        st.caption(f"{get_text('source_reference', st.session_state.language)}: {', '.join(message['sources'])}")
        
        # 질문 처리 로직 (버튼 클릭 또는 입력)
        prompt = st.chat_input(get_text('chat_input_placeholder', st.session_state.language))
        
        # 샘플 질문 버튼으로 트리거된 경우
        if 'trigger_query' in st.session_state:
            prompt = st.session_state.trigger_query
            del st.session_state.trigger_query
        
        if prompt:
            # 인덱싱된 파일 확인
            if not st.session_state.indexed_files:
                st.warning(get_text('warning_upload_first', st.session_state.language))
                st.stop()
            
            # 사용자 메시지 추가 (이미 추가된 경우 중복 방지)
            if not st.session_state.messages or st.session_state.messages[-1]["content"] != prompt:
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
            
            # 답변 생성
            with st.chat_message("assistant"):
                with st.spinner(get_text('generating_answer', st.session_state.language)):
                    result = engine.query(prompt)
                    
                    st.markdown(result['answer'])
                    
                    # 출처 표시
                    if result['sources']:
                        st.caption(f"{get_text('source_reference', st.session_state.language)}: {', '.join(result['sources'])}")
                    
                    # 성능 정보 (선택적으로 표시)
                    with st.expander(get_text('performance_info', st.session_state.language)):
                        st.write(f"{get_text('total_time', st.session_state.language)}: {result['timing']['total']:.2f}s")
                        st.write(f"  - {get_text('search_time', st.session_state.language)}: {result['timing']['search']:.2f}s")
                        st.write(f"  - {get_text('llm_time', st.session_state.language)}: {result['timing']['llm']:.2f}s")
            
            # 어시스턴트 메시지 추가
            st.session_state.messages.append({
                "role": "assistant",
                "content": result['answer'],
                "sources": result['sources']
            })
            
            # 대화 기록 초기화 버튼
            if st.session_state.messages:
                if st.button(get_text('clear_history_button', st.session_state.language)):
                    st.session_state.messages = []
                    st.rerun()

    # === 설정 탭 ===
    with tab_settings:
        st.header(get_text('settings_header', st.session_state.language))
        
        # 설정 파일 로드
        import yaml
        try:
            with open('config/config.yaml', 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            st.error(f"{get_text('load_config_fail', st.session_state.language)}: {e}")
            config = {}

        with st.form("settings_form"):
            st.subheader(get_text('performance_preset_header', st.session_state.language))
            
            current_preset = config.get('presets', {}).get('current', '16GB')
            preset_options = ["8GB", "16GB", "32GB", "Custom"]
            
            # 현재 설정이 프리셋과 일치하는지 확인 (Custom 감지)
            is_custom = True
            if current_preset in ["8GB", "16GB", "32GB"]:
                 # 간단한 체크: chunk_size만 비교해도 충분
                 preset_config = config.get('presets', {}).get(current_preset, {})
                 if preset_config:
                     if config['rag']['chunk_size'] == preset_config['rag']['chunk_size']:
                         is_custom = False
            
            selected_preset_index = preset_options.index(current_preset) if not is_custom and current_preset in preset_options else 3
            
            selected_preset = st.selectbox(
                get_text('preset_limit_help', st.session_state.language), 
                preset_options,
                index=selected_preset_index,
                help=get_text('preset_help_text', st.session_state.language)
            )
            
            # 프리셋 적용 로직 (UI 렌더링용 값 설정)
            if selected_preset != "Custom":
                preset_vals = config.get('presets', {}).get(selected_preset, {})
                rag_vals = preset_vals.get('rag', {})
                llm_vals = preset_vals.get('llm', {})
                
                # 폼 기본값 업데이트
                val_chunk_size = rag_vals.get('chunk_size', 800)
                val_chunk_overlap = rag_vals.get('chunk_overlap', 100)
                val_top_k = rag_vals.get('top_k', 5)
                val_context_count = rag_vals.get('context_count', 3)
                val_model_name = llm_vals.get('model_name', 'qwen2:7b')
                val_max_tokens = llm_vals.get('max_tokens', 512)
            else:
                # 현재 설정값 유지
                val_chunk_size = config.get('rag', {}).get('chunk_size', 800)
                val_chunk_overlap = config.get('rag', {}).get('chunk_overlap', 100)
                val_top_k = config.get('rag', {}).get('top_k', 5)
                val_context_count = config.get('rag', {}).get('context_count', 3)
                val_model_name = config.get('llm', {}).get('model_name', 'qwen2:7b')
                val_max_tokens = config.get('llm', {}).get('max_tokens', 512)

            st.divider()
            st.subheader(get_text('rag_parameters_header', st.session_state.language))
            col1, col2 = st.columns(2)
            with col1:
                chunk_size = st.number_input("Chunk Size", min_value=100, max_value=2000, value=val_chunk_size)
                top_k = st.number_input(get_text('top_k_label', st.session_state.language), min_value=1, max_value=20, value=val_top_k)
            with col2:
                chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=500, value=val_chunk_overlap)
                context_count = st.number_input(get_text('context_count_label', st.session_state.language), min_value=1, max_value=10, value=val_context_count)
            
            st.subheader(get_text('llm_settings_header', st.session_state.language))
            col3, col4 = st.columns(2)
            with col3:
                model_index = 0 if "qwen" in val_model_name else 1
                model_name = st.selectbox(get_text('ollama_model_label', st.session_state.language), ["qwen2:7b", "llama3.1:8b"], index=model_index)
                temperature = st.slider("Temperature", 0.0, 1.0, config.get('llm', {}).get('temperature', 0.3))
            with col4:
                max_tokens = st.number_input("Max Tokens", 100, 4096, value=val_max_tokens)
                timeout = st.number_input(get_text('timeout_label', st.session_state.language), 10, 300, config.get('llm', {}).get('timeout', 120))
            
            # System Prompt 설정
            default_system_prompt = get_text('system_prompt_default', st.session_state.language)
            
            system_prompt = st.text_area(
                "System Prompt", 
                value=config.get('llm', {}).get('system_prompt', default_system_prompt),
                height=300,
                help="{context}와 {question} 변수는 필수입니다."
            )
            
            if st.form_submit_button(get_text('save_settings_button', st.session_state.language)):
                # 설정 업데이트
                config['rag']['chunk_size'] = chunk_size
                config['rag']['chunk_overlap'] = chunk_overlap
                config['rag']['top_k'] = top_k
                config['rag']['context_count'] = context_count
                config['llm']['model_name'] = model_name
                config['llm']['temperature'] = temperature
                config['llm']['max_tokens'] = max_tokens
                config['llm']['timeout'] = timeout
                config['llm']['system_prompt'] = system_prompt
                
                # 프리셋 정보 업데이트
                if selected_preset != "Custom":
                    if 'presets' not in config: config['presets'] = {}
                    config['presets']['current'] = selected_preset
                else:
                    if 'presets' not in config: config['presets'] = {}
                    config['presets']['current'] = "Custom"
                
                # 파일 저장
                with open('config/config.yaml', 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, allow_unicode=True)
                
                st.success(get_text('settings_saved', st.session_state.language))
                st.cache_resource.clear()
                time.sleep(1)
                st.rerun()
        
        st.divider()
        st.subheader(get_text('history_management_header', st.session_state.language))
        col_hist1, col_hist2 = st.columns(2)
        
        with col_hist1:
            # 대화 내보내기
            if st.session_state.messages:
                # JSON 내보내기
                chat_history_json = json.dumps(st.session_state.messages, ensure_ascii=False, indent=2)
                st.download_button(
                    label=get_text('export_json_button', st.session_state.language),
                    data=chat_history_json,
                    file_name="chat_history.json",
                    mime="application/json"
                )
                
                # TXT 내보내기
                chat_history_txt = ""
                for msg in st.session_state.messages:
                    role = "사용자" if msg["role"] == "user" else "AI"
                    chat_history_txt += f"[{role}]: {msg['content']}\n"
                    if msg.get("sources"):
                        chat_history_txt += f"({get_text('source_reference', st.session_state.language)}: {', '.join(msg['sources'])})\n"
                    chat_history_txt += "\n"
                
                st.download_button(
                    label=get_text('export_txt_button', st.session_state.language),
                    data=chat_history_txt,
                    file_name="chat_history.txt",
                    mime="text/plain"
                )
        
        with col_hist2:
            # 대화 불러오기
            uploaded_history = st.file_uploader(get_text('import_label', st.session_state.language), type=['json'])
            if uploaded_history is not None:
                if st.button(get_text('import_button', st.session_state.language)):
                    try:
                        loaded_messages = json.load(uploaded_history)
                        st.session_state.messages = loaded_messages
                        st.success(get_text('import_success', st.session_state.language))
                        st.rerun()
                    except Exception as e:
                        st.error(f"{get_text('import_fail', st.session_state.language)}: {e}")

        st.divider()
        st.divider()
        st.subheader(get_text('file_management_header', st.session_state.language))
        
        col_file1, col_file2 = st.columns([3, 1])
        with col_file2:
            if st.button(get_text('refresh_list_button', st.session_state.language)):
                st.session_state.indexed_files = engine.get_indexed_files()
                st.rerun()
        
        if st.session_state.indexed_files:
            # 파일을 데이터프레임으로 변환하여 표시
            file_list = st.session_state.indexed_files
            
            # 각 파일별 삭제 버튼 생성
            st.markdown(get_text('indexed_files_list', st.session_state.language))
            for file_name in file_list:
                col_name, col_del = st.columns([4, 1])
                with col_name:
                    st.text(f"📄 {file_name}")
                with col_del:
                    if st.button(get_text('delete_button', st.session_state.language), key=f"del_{file_name}", type="secondary", help=f"{file_name}을(를) 삭제합니다"):
                        if engine.delete_file(file_name):
                            st.success(f"{get_text('delete_success', st.session_state.language)}: {file_name}")
                            st.session_state.indexed_files = engine.get_indexed_files()
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error(get_text('delete_fail', st.session_state.language))
            
            if st.button(get_text('delete_all_button', st.session_state.language), type="primary"):
                if st.checkbox(get_text('delete_all_confirm', st.session_state.language)):
                    progress_text = st.empty()
                    for f in file_list:
                        progress_text.text(f"삭제 중: {f}...")
                        engine.delete_file(f)
                    st.success(get_text('delete_all_success', st.session_state.language))
                    st.session_state.indexed_files = []
                    st.rerun()
        else:
            st.info(get_text('no_indexed_files', st.session_state.language))

if __name__ == "__main__":
    main()

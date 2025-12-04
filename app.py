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


# 페이지 설정
st.set_page_config(
    page_title="로컬 RAG 챗봇",
    page_icon="🤖",
    layout="wide"
)


@st.cache_resource
def initialize_rag_engine():
    """RAG 엔진 초기화 (캐싱)"""
    return RAGEngine()


def check_password(password: str) -> bool:
    """비밀번호 검증"""
    try:
        import yaml
        with open('config.yaml', 'r', encoding='utf-8') as f:
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
        st.error(f"인증 오류: {e}")
        return False


def main():
    st.title("🤖 로컬 RAG 챗봇")
    
    # Session State 초기화
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    # 로그인 화면
    if not st.session_state.authenticated:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.subheader("🔐 관리자 로그인")
            password = st.text_input("비밀번호를 입력하세요", type="password")
            if st.button("로그인", type="primary"):
                if check_password(password):
                    st.session_state.authenticated = True
                    st.success("로그인 성공!")
                    st.rerun()
                else:
                    st.error("비밀번호가 올바르지 않습니다.")
        return

    # === 메인 앱 로직 ===
    
    # 사이드바에 로그아웃 버튼 추가
    with st.sidebar:
        if st.button("로그아웃"):
            st.session_state.authenticated = False
            st.rerun()
        st.divider()
        
        # 다중 사용자 경고
        st.info("⚠️ 이 시스템은 단일 사용자 환경(Local)에 최적화되어 있습니다.")

    # RAG 엔진 초기화
    try:
        engine = initialize_rag_engine()
    except Exception as e:
        st.error(f"RAG 엔진 초기화 실패: {str(e)}")
        st.stop()
    
    # Session State 초기화 (메시지 등)
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'indexed_files' not in st.session_state:
        st.session_state.indexed_files = engine.get_indexed_files()
        
    if 'sample_questions' not in st.session_state:
        st.session_state.sample_questions = []
    
    # 탭 구성
    tab_chat, tab_settings = st.tabs(["💬 채팅", "⚙️ 설정"])
    
    # === 채팅 탭 ===
    with tab_chat:
        st.markdown("문서를 업로드하고 질문하세요!")
        
        # 사이드바 (파일 업로드만 유지)
        with st.sidebar:
            st.header("📄 문서 업로드")
            
            # 파일 업로드
            uploaded_file = st.file_uploader(
                "PDF, Markdown, Text 파일 업로드",
                type=['pdf', 'md', 'txt'],
                help="지원 형식: PDF, Markdown (.md), Text (.txt)"
            )
            
            if uploaded_file is not None:
                if st.button("📥 인덱싱 시작", type="primary"):
                    with st.status("파일 처리 중...") as status:
                        # 임시 파일로 저장
                        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        try:
                            # 인덱싱
                            st.write("📖 문서 로드 중...")
                            result = engine.load_and_index_file(tmp_path)
                            
                            if result['success']:
                                status.update(label="✅ 인덱싱 완료!", state="complete")
                                st.success(result['message'])
                                st.info(f"소요 시간: {result['elapsed_time']:.2f}초")
                                
                                # 파일 목록 갱신
                                st.session_state.indexed_files = engine.get_indexed_files()
                                
                                # 샘플 질문 갱신
                                if result.get('sample_questions'):
                                    st.session_state.sample_questions = result['sample_questions']
                            else:
                                status.update(label="❌ 인덱싱 실패", state="error")
                                st.error(result['message'])
                        
                        finally:
                            # 임시 파일 삭제
                            if os.path.exists(tmp_path):
                                os.unlink(tmp_path)
            
            # Ollama 연결 상태
            st.divider()
            st.subheader("🔌 연결 상태")
            
            if st.button("연결 테스트"):
                with st.spinner("테스트 중..."):
                    if engine.test_connection():
                        st.success("✅ Ollama 연결 성공")
                    else:
                        st.error("❌ Ollama 연결 실패")
                        st.info("'ollama serve'를 실행하세요")

        # 채팅 인터페이스
        
        # 샘플 질문 표시 (메시지가 없거나 샘플 질문이 있을 때)
        if st.session_state.sample_questions and not st.session_state.messages:
            st.info("💡 추천 질문")
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
                        st.caption(f"📚 출처: {', '.join(message['sources'])}")
        
        # 질문 처리 로직 (버튼 클릭 또는 입력)
        prompt = st.chat_input("질문을 입력하세요")
        
        # 샘플 질문 버튼으로 트리거된 경우
        if 'trigger_query' in st.session_state:
            prompt = st.session_state.trigger_query
            del st.session_state.trigger_query
        
        if prompt:
            # 인덱싱된 파일 확인
            if not st.session_state.indexed_files:
                st.warning("먼저 문서를 업로드하고 인덱싱하세요!")
                st.stop()
            
            # 사용자 메시지 추가 (이미 추가된 경우 중복 방지)
            if not st.session_state.messages or st.session_state.messages[-1]["content"] != prompt:
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
            
            # 답변 생성
            with st.chat_message("assistant"):
                with st.spinner("답변 생성 중..."):
                    result = engine.query(prompt)
                    
                    st.markdown(result['answer'])
                    
                    # 출처 표시
                    if result['sources']:
                        st.caption(f"📚 출처: {', '.join(result['sources'])}")
                    
                    # 성능 정보 (선택적으로 표시)
                    with st.expander("⏱️ 성능 정보"):
                        st.write(f"총 소요 시간: {result['timing']['total']:.2f}초")
                        st.write(f"  - 검색: {result['timing']['search']:.2f}초")
                        st.write(f"  - LLM: {result['timing']['llm']:.2f}초")
            
            # 어시스턴트 메시지 추가
            st.session_state.messages.append({
                "role": "assistant",
                "content": result['answer'],
                "sources": result['sources']
            })
            
            # 대화 기록 초기화 버튼
            if st.session_state.messages:
                if st.button("🗑️ 대화 기록 지우기"):
                    st.session_state.messages = []
                    st.rerun()

    # === 설정 탭 ===
    with tab_settings:
        st.header("⚙️ 환경 설정")
        
        # 설정 파일 로드
        import yaml
        try:
            with open('config.yaml', 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            st.error(f"설정 파일 로드 실패: {e}")
            config = {}

        with st.form("settings_form"):
            st.subheader("🚀 성능 프리셋")
            
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
                "하드웨어 환경에 맞는 프리셋을 선택하세요", 
                preset_options,
                index=selected_preset_index,
                help="8GB: 저사양 / 16GB: 기본 / 32GB: 고사양"
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
            st.subheader("RAG 파라미터")
            col1, col2 = st.columns(2)
            with col1:
                chunk_size = st.number_input("Chunk Size", min_value=100, max_value=2000, value=val_chunk_size)
                top_k = st.number_input("Top-K (검색 개수)", min_value=1, max_value=20, value=val_top_k)
            with col2:
                chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=500, value=val_chunk_overlap)
                context_count = st.number_input("Context Count (LLM 입력 개수)", min_value=1, max_value=10, value=val_context_count)
            
            st.subheader("LLM 설정")
            col3, col4 = st.columns(2)
            with col3:
                model_index = 0 if "qwen" in val_model_name else 1
                model_name = st.selectbox("Ollama 모델", ["qwen2:7b", "llama3.1:8b"], index=model_index)
                temperature = st.slider("Temperature", 0.0, 1.0, config.get('llm', {}).get('temperature', 0.3))
            with col4:
                max_tokens = st.number_input("Max Tokens", 100, 4096, value=val_max_tokens)
                timeout = st.number_input("Timeout (초)", 10, 300, config.get('llm', {}).get('timeout', 120))
            
            # System Prompt 설정
            default_system_prompt = """당신은 제공된 문서를 바탕으로 질문에 답변하는 유능한 어시스턴트입니다.

다음 규칙을 준수하세요:
1. 제공된 컨텍스트 내용만을 사용하여 답변하세요.
2. 컨텍스트에 답이 없으면 "제공된 문서에서 해당 정보를 찾을 수 없습니다"라고 답변하세요.
3. 추측하거나 컨텍스트 외의 정보를 사용하지 마세요.
4. 답변은 명확하고 간결하게 작성하세요.
5. 가능한 경우 출처(문서 이름, 페이지)를 언급하세요.

컨텍스트:
{context}

질문: {question}

답변:"""
            
            system_prompt = st.text_area(
                "System Prompt", 
                value=config.get('llm', {}).get('system_prompt', default_system_prompt),
                height=300,
                help="{context}와 {question} 변수는 필수입니다."
            )
            
            if st.form_submit_button("💾 설정 저장"):
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
                with open('config.yaml', 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, allow_unicode=True)
                
                st.success("설정이 저장되었습니다. 적용을 위해 앱을 다시 로드합니다.")
                st.cache_resource.clear()
                time.sleep(1)
                st.rerun()
        
        st.divider()
        st.subheader("💾 대화 기록 관리")
        col_hist1, col_hist2 = st.columns(2)
        
        with col_hist1:
            # 대화 내보내기
            if st.session_state.messages:
                # JSON 내보내기
                chat_history_json = json.dumps(st.session_state.messages, ensure_ascii=False, indent=2)
                st.download_button(
                    label="📤 대화 기록 내보내기 (JSON)",
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
                        chat_history_txt += f"(출처: {', '.join(msg['sources'])})\n"
                    chat_history_txt += "\n"
                
                st.download_button(
                    label="📄 대화 기록 내보내기 (TXT)",
                    data=chat_history_txt,
                    file_name="chat_history.txt",
                    mime="text/plain"
                )
        
        with col_hist2:
            # 대화 불러오기
            uploaded_history = st.file_uploader("대화 기록 불러오기 (JSON)", type=['json'])
            if uploaded_history is not None:
                if st.button("📥 불러오기"):
                    try:
                        loaded_messages = json.load(uploaded_history)
                        st.session_state.messages = loaded_messages
                        st.success("대화 기록을 불러왔습니다.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"불러오기 실패: {e}")

        st.divider()
        st.subheader("📚 파일 관리")
        
        if st.session_state.indexed_files:
            # 테이블 형태로 표시
            file_data = [{"File Name": f} for f in st.session_state.indexed_files]
            st.table(file_data)
            
            # 삭제 선택
            file_to_delete = st.selectbox("삭제할 파일 선택", ["선택하세요..."] + st.session_state.indexed_files)
            if file_to_delete != "선택하세요...":
                if st.button(f"🗑️ {file_to_delete} 삭제", type="primary"):
                    if engine.delete_file(file_to_delete):
                        st.success(f"삭제됨: {file_to_delete}")
                        st.session_state.indexed_files = engine.get_indexed_files()
                        st.rerun()
                    else:
                        st.error("삭제 실패")
        else:
            st.info("인덱싱된 파일이 없습니다.")

if __name__ == "__main__":
    main()

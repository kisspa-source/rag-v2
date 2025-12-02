"""
Streamlit 기반 RAG 챗봇 UI
"""
import streamlit as st
import time
from pathlib import Path
import tempfile
import os

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


def main():
    st.title("🤖 로컬 RAG 챗봇")
    st.markdown("문서를 업로드하고 질문하세요!")
    
    # RAG 엔진 초기화
    try:
        engine = initialize_rag_engine()
    except Exception as e:
        st.error(f"RAG 엔진 초기화 실패: {str(e)}")
        st.stop()
    
    # Session State 초기화
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'indexed_files' not in st.session_state:
        st.session_state.indexed_files = engine.get_indexed_files()
    
    # 사이드바
    with st.sidebar:
        st.header("📄 문서 관리")
        
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
                        else:
                            status.update(label="❌ 인덱싱 실패", state="error")
                            st.error(result['message'])
                    
                    finally:
                        # 임시 파일 삭제
                        os.unlink(tmp_path)
        
        # 인덱싱된 파일 목록
        st.divider()
        st.subheader("📚 인덱싱된 파일")
        
        if st.session_state.indexed_files:
            for file_name in st.session_state.indexed_files:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(f"📄 {file_name}")
                with col2:
                    if st.button("🗑️", key=f"del_{file_name}", help="삭제"):
                        if engine.delete_file(file_name):
                            st.success(f"삭제됨: {file_name}")
                            st.session_state.indexed_files = engine.get_indexed_files()
                            st.rerun()
                        else:
                            st.error("삭제 실패")
        else:
            st.info("인덱싱된 파일이 없습니다.")
        
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
    
    # 메인 화면 - 채팅 인터페이스
    st.divider()
    
    # 채팅 히스토리 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # 출처 표시
            if message["role"] == "assistant" and "sources" in message:
                if message["sources"]:
                    st.caption(f"📚 출처: {', '.join(message['sources'])}")
    
    # 질문 입력
    if prompt := st.chat_input("질문을 입력하세요"):
        # 인덱싱된 파일 확인
        if not st.session_state.indexed_files:
            st.warning("먼저 문서를 업로드하고 인덱싱하세요!")
            st.stop()
        
        # 사용자 메시지 추가
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


if __name__ == "__main__":
    main()

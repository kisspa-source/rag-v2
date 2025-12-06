"""
통합 RAG 엔진 - 모든 모듈을 조합하는 상위 레이어
"""
import logging
import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

import yaml
from langchain_core.documents import Document

from loaders import DocumentLoader, load_documents
from indexer import DocumentIndexer
from retriever import HybridRetriever
from llm_client import LLMClient

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


import gc
import threading

# ... imports ...

class RAGEngine:
    """통합 RAG 엔진 - 문서 로드, 인덱싱, 검색, 답변 생성"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        RAG 엔진 초기화
        
        Args:
            config_path: 설정 파일 경로
        """
        # 설정 로드
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 동시성 제어를 위한 Lock
        self.lock = threading.Lock()
        
        # 로그 경로 설정
        self.query_log_path = Path("logs/queries.jsonl")
        self.query_log_path.parent.mkdir(exist_ok=True)
        
        # 모듈 초기화
        logger.info("RAG 엔진 초기화 중...")
        
        # 문서 로더
        self.loader = DocumentLoader()
        
        # 인덱서 (임베딩 모델 포함)
        self.indexer = DocumentIndexer(self.config)
        
        # 리트리버 (BM25 + Vector)
        self.retriever = HybridRetriever(
            vector_indexer=self.indexer,
            config=self.config
        )
        
        # LLM 클라이언트
        self.llm_client = LLMClient(self.config)
        
        logger.info("RAG 엔진 초기화 완료")

    def load_and_index_file(self, file_path: str, original_filename: str = None) -> Dict[str, Any]:
        """
        파일 로드 및 인덱싱
        """
        start_time = time.time()
        
        try:
            # 문서 로드
            display_name = original_filename if original_filename else file_path
            logger.info(f"파일 로드 중: {display_name}")
            documents = self.loader.load_file(file_path, original_filename=original_filename)
            
            if not documents:
                return {
                    'success': False,
                    'message': '문서를 로드할 수 없습니다.',
                    'elapsed_time': time.time() - start_time
                }
            
            # 인덱싱
            logger.info(f"인덱싱 중: {display_name}")
            
            with self.lock:
                success = self.indexer.index_documents(documents, file_path, original_filename=original_filename)
                
                if not success:
                    return {
                        'success': False,
                        'message': '인덱싱 실패',
                        'elapsed_time': time.time() - start_time
                    }
                
                # BM25 인덱스 재구축
                logger.info("BM25 인덱스 재구축 중...")
                self._rebuild_bm25_index()
            
            # 샘플 질문 생성
            logger.info("샘플 질문 생성 중...")
            sample_questions = []
            if documents:
                # 첫 몇 페이지의 텍스트를 사용하여 질문 생성
                context = "\n".join([doc.page_content for doc in documents[:3]])
                sample_questions = self.llm_client.generate_questions(context)
            
            # 메모리 정리
            gc.collect()
            
            elapsed_time = time.time() - start_time
            
            return {
                'success': True,
                'message': f'성공적으로 인덱싱되었습니다. ({len(documents)}개 페이지)',
                'num_documents': len(documents),
                'elapsed_time': elapsed_time,
                'sample_questions': sample_questions
            }
            
        except Exception as e:
            logger.error(f"파일 처리 실패: {str(e)}")
            return {
                'success': False,
                'message': f'오류 발생: {str(e)}',
                'elapsed_time': time.time() - start_time
            }
    
    def _rebuild_bm25_index(self):
        """모든 인덱싱된 문서로 BM25 인덱스 재구축"""
        # ChromaDB에서 모든 문서 가져오기
        all_data = self.indexer.collection.get()
        
        if not all_data['documents']:
            logger.warning("인덱싱된 문서가 없습니다.")
            return
        
        # Document 객체로 변환
        documents = []
        for i in range(len(all_data['ids'])):
            doc = Document(
                page_content=all_data['documents'][i],
                metadata=all_data['metadatas'][i] if all_data['metadatas'] else {}
            )
            documents.append(doc)
        
        # BM25 인덱스 구축
        self.retriever.build_bm25_index(documents)
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        질문에 대한 답변 생성 (E2E RAG 파이프라인)
        
        Args:
            question: 사용자 질문
            
        Returns:
            답변 및 메타데이터
        """
        start_time = time.time()
        
        try:
            # 1. 컨텍스트 검색
            logger.info(f"질문: {question}")
            
            context_count = self.config['rag'].get('context_count', 3)
            search_start = time.time()
            
            # 검색은 Read-only이므로 Lock 불필요 (ChromaDB가 읽기 동시성 지원 가정)
            # 하지만 BM25가 재구축 중일 수 있으므로 안전하게 Lock 사용 고려
            # 성능을 위해 읽기 Lock은 생략하거나 RLock 사용 가능. 여기서는 간단히 Lock 사용.
            # with self.lock:
            context, search_results = self.retriever.get_context_for_llm(
                question, 
                top_k=context_count
            )
            search_time = time.time() - search_start
            
            logger.info(f"검색 완료 ({search_time:.2f}초)")
            
            # 2. LLM으로 답변 생성
            llm_start = time.time()
            llm_result = self.llm_client.generate_answer(question, context)
            llm_time = time.time() - llm_start
            
            # 3. 출처 정보 추출
            sources = self._extract_sources(search_results)
            
            # 4. 전체 소요 시간
            total_time = time.time() - start_time
            
            result = {
                'question': question,
                'answer': llm_result['answer'],
                'sources': sources,
                'search_results': search_results,
                'timing': {
                    'total': total_time,
                    'search': search_time,
                    'llm': llm_time
                }
            }
            
            # 로그 저장
            self._log_query(result)
            
            logger.info(f"답변 생성 완료 (총 {total_time:.2f}초)")
            
            # 메모리 정리
            gc.collect()
            
            return result
            
        except Exception as e:
            logger.error(f"쿼리 처리 실패: {str(e)}")
            return {
                'question': question,
                'answer': f"오류가 발생했습니다: {str(e)}",
                'sources': [],
                'search_results': [],
                'timing': {
                    'total': time.time() - start_time,
                    'search': 0,
                    'llm': 0
                }
            }
    
    def _extract_sources(self, search_results: List[Dict[str, Any]]) -> List[str]:
        """검색 결과에서 출처 정보 추출"""
        sources = []
        seen = set()
        
        for result in search_results:
            metadata = result.get('metadata', {})
            source_file = metadata.get('source_file', 'unknown')
            page = metadata.get('page', '?')
            
            source_str = f"{source_file} p.{page}"
            
            if source_str not in seen:
                sources.append(source_str)
                seen.add(source_str)
        
        return sources
    
    def _log_query(self, result: Dict[str, Any]):
        """쿼리 로그 저장"""
        try:
            # 민감 정보 마스킹
            question = self._mask_sensitive_data(result['question'])
            answer = self._mask_sensitive_data(result['answer'])
            
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'question': question,
                'answer': answer,
                'sources': result['sources'],
                'timing': result['timing'],
                'model': self.config['llm']['model_name']
            }
            
            # 점수 정보 추가
            if result['search_results']:
                log_entry['search_scores'] = [
                    {
                        'score': r.get('score', 0),
                        'vector_score': r.get('vector_score', 0),
                        'bm25_score': r.get('bm25_score', 0)
                    }
                    for r in result['search_results']
                ]
            
            # JSON Lines 형식으로 저장
            with open(self.query_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
                
        except Exception as e:
            logger.error(f"로그 저장 실패: {str(e)}")
            
    def _mask_sensitive_data(self, text: str) -> str:
        """민감 정보 마스킹 (이메일, 전화번호, 주민등록번호 등)"""
        import re
        
        if not text:
            return ""
            
        # 이메일 마스킹
        text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '***@***.***', text)
        
        # 전화번호 마스킹 (010-1234-5678)
        text = re.sub(r'(\d{2,3})-(\d{3,4})-(\d{4})', r'\1-****-\3', text)
        
        # 주민등록번호 마스킹 (123456-1234567)
        text = re.sub(r'(\d{6})-(\d{7})', r'\1-*******', text)
        
        return text
    
    def get_indexed_files(self) -> List[str]:
        """인덱싱된 파일 목록 반환"""
        return self.indexer.get_indexed_files()
    
    def delete_file(self, file_name: str) -> bool:
        """파일 인덱스 삭제"""
        with self.lock:
            success = self.indexer.delete_file(file_name)
            
            if success:
                # BM25 인덱스 재구축
                self._rebuild_bm25_index()
            
            return success
    
    def test_connection(self) -> bool:
        """Ollama 연결 테스트"""
        return self.llm_client.test_connection()


if __name__ == "__main__":
    # CLI 테스트
    import sys
    
    # RAG 엔진 초기화
    engine = RAGEngine()
    
    # Ollama 연결 테스트
    if not engine.test_connection():
        print("❌ Ollama 연결 실패. 'ollama serve'로 서비스를 시작하세요.")
        sys.exit(1)
    
    print("✅ Ollama 연결 성공\n")
    
    # CLI 모드
    if len(sys.argv) > 1:
        # 파일 인자가 있으면 인덱싱
        if sys.argv[1] == '--index':
            if len(sys.argv) < 3:
                print("사용법: python rag_engine.py --index <file_path>")
                sys.exit(1)
            
            file_path = sys.argv[2]
            result = engine.load_and_index_file(file_path)
            print(f"결과: {result['message']}")
            print(f"소요 시간: {result['elapsed_time']:.2f}초")
        
        # 질문 모드
        elif sys.argv[1] == '--query':
            if len(sys.argv) < 3:
                print("사용법: python rag_engine.py --query \"질문\"")
                sys.exit(1)
            
            question = sys.argv[2]
            result = engine.query(question)
            
            print(f"\n질문: {result['question']}")
            print(f"\n답변: {result['answer']}")
            print(f"\n출처: {', '.join(result['sources'])}")
            print(f"\n소요 시간: {result['timing']['total']:.2f}초")
            print(f"  - 검색: {result['timing']['search']:.2f}초")
            print(f"  - LLM: {result['timing']['llm']:.2f}초")
        
        # 파일 목록
        elif sys.argv[1] == '--list':
            files = engine.get_indexed_files()
            print(f"인덱싱된 파일 ({len(files)}개):")
            for f in files:
                print(f"  - {f}")
        
        else:
            print("알 수 없는 명령어")
            print("사용법:")
            print("  python rag_engine.py --index <file_path>")
            print("  python rag_engine.py --query \"질문\"")
            print("  python rag_engine.py --list")
    else:
        print("사용법:")
        print("  python rag_engine.py --index <file_path>")
        print("  python rag_engine.py --query \"질문\"")
        print("  python rag_engine.py --list")

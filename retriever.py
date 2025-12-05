"""
검색기 모듈 - BM25, Vector, Hybrid Search 구현
"""
import logging
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
import torch

logger = logging.getLogger(__name__)


class HybridRetriever:
    """BM25 + Vector Search를 결합한 Hybrid 검색기"""
    
    def __init__(self, vector_indexer, config: Dict[str, Any]):
        """
        Args:
            vector_indexer: DocumentIndexer 인스턴스
            config: 설정 딕셔너리
        """
        self.vector_indexer = vector_indexer
        self.config = config
        self.bm25_index = None
        self.corpus = []  # BM25용 문서 코퍼스
        self.corpus_metadata = []  # 문서 메타데이터
        
        # Reranker 모델 초기화 (Lazy Loading)
        self.reranker = None
        
    def _load_reranker(self):
        """Reranker 모델 로드"""
        if self.reranker is None:
            model_name = self.config['rag'].get('reranker_model', 'BAAI/bge-reranker-v2-m3')
            logger.info(f"Reranker 모델 로드 중: {model_name}")
            try:
                self.reranker = CrossEncoder(model_name)
                logger.info("Reranker 모델 로드 완료")
            except Exception as e:
                logger.error(f"Reranker 모델 로드 실패: {str(e)}")
        
    def build_bm25_index(self, documents: List[Document]):
        """
        BM25 인덱스 구축
        
        Args:
            documents: Document 리스트
        """
        logger.info("BM25 인덱스 구축 중...")
        
        self.corpus = []
        self.corpus_metadata = []
        
        for doc in documents:
            # 토큰화 (공백 기준)
            tokens = doc.page_content.split()
            self.corpus.append(tokens)
            self.corpus_metadata.append({
                'content': doc.page_content,
                'metadata': doc.metadata
            })
        
        if self.corpus:
            self.bm25_index = BM25Okapi(self.corpus)
            logger.info(f"BM25 인덱스 구축 완료: {len(self.corpus)}개 문서")
        else:
            logger.warning("BM25 인덱스 구축 실패: 문서가 비어있음")
    
    def bm25_search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        BM25 검색
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 개수
            
        Returns:
            검색 결과 리스트
        """
        if not self.bm25_index:
            logger.warning("BM25 인덱스가 초기화되지 않음")
            return []
        
        # 쿼리 토큰화
        query_tokens = query.split()
        
        # BM25 점수 계산
        scores = self.bm25_index.get_scores(query_tokens)
        
        # 점수 순으로 정렬
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # 점수가 0보다 큰 것만 반환
                results.append({
                    'id': f'bm25_{idx}',
                    'content': self.corpus_metadata[idx]['content'],
                    'metadata': self.corpus_metadata[idx]['metadata'],
                    'score': float(scores[idx]),
                    'source': 'bm25'
                })
        
        logger.info(f"BM25 검색 결과: {len(results)}개")
        return results
    
    def vector_search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Vector 검색
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 개수
            
        Returns:
            검색 결과 리스트
        """
        results = self.vector_indexer.search(query, top_k=top_k)
        
        # 소스 추가
        for result in results:
            result['source'] = 'vector'
        
        logger.info(f"Vector 검색 결과: {len(results)}개")
        return results
    
    def hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.5) -> List[Dict[str, Any]]:
        """
        Hybrid Search (BM25 + Vector 결합)
        
        Args:
            query: 검색 쿼리
            top_k: 최종 반환할 결과 개수
            alpha: Vector 검색 가중치 (0~1, BM25 가중치는 1-alpha)
            
        Returns:
            검색 결과 리스트
        """
        # BM25와 Vector 검색을 각각 수행 (더 많은 결과 요청)
        bm25_results = self.bm25_search(query, top_k=20)
        vector_results = self.vector_search(query, top_k=20)
        
        # 결과 병합 (가중 평균)
        merged_results = {}
        
        # Vector 점수 정규화 및 추가
        if vector_results:
            max_vector_score = max([r['score'] for r in vector_results])
            for result in vector_results:
                content = result['content']
                normalized_score = result['score'] / max_vector_score if max_vector_score > 0 else 0
                
                if content not in merged_results:
                    merged_results[content] = {
                        'content': content,
                        'metadata': result['metadata'],
                        'vector_score': normalized_score * alpha,
                        'bm25_score': 0
                    }
                else:
                    merged_results[content]['vector_score'] = normalized_score * alpha
        
        # BM25 점수 정규화 및 추가
        if bm25_results:
            max_bm25_score = max([r['score'] for r in bm25_results])
            for result in bm25_results:
                content = result['content']
                normalized_score = result['score'] / max_bm25_score if max_bm25_score > 0 else 0
                
                if content not in merged_results:
                    merged_results[content] = {
                        'content': content,
                        'metadata': result['metadata'],
                        'vector_score': 0,
                        'bm25_score': normalized_score * (1 - alpha)
                    }
                else:
                    merged_results[content]['bm25_score'] = normalized_score * (1 - alpha)
        
        # 최종 점수 계산 및 정렬
        final_results = []
        for content, data in merged_results.items():
            final_score = data['vector_score'] + data['bm25_score']
            final_results.append({
                'content': content,
                'metadata': data['metadata'],
                'score': final_score,
                'vector_score': data['vector_score'],
                'bm25_score': data['bm25_score']
            })
        
        # 점수 순으로 정렬하고 상위 k개 반환
        final_results.sort(key=lambda x: x['score'], reverse=True)
        top_results = final_results[:top_k]
        
        logger.info(f"Hybrid 검색 결과: {len(top_results)}개 (alpha={alpha})")
        
        return top_results

    def rrf_merge(self, bm25_results: List[Dict[str, Any]], vector_results: List[Dict[str, Any]], k: int = 60) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion (RRF) 병합
        Score = 1 / (k + rank)
        """
        merged_scores = {}
        content_map = {}
        
        # BM25 순위 점수 계산
        for rank, result in enumerate(bm25_results):
            content = result['content']
            content_map[content] = result
            if content not in merged_scores:
                merged_scores[content] = 0.0
            merged_scores[content] += 1 / (k + rank + 1)
            
        # Vector 순위 점수 계산
        for rank, result in enumerate(vector_results):
            content = result['content']
            content_map[content] = result  # 덮어쓰기 (메타데이터는 동일 가정)
            if content not in merged_scores:
                merged_scores[content] = 0.0
            merged_scores[content] += 1 / (k + rank + 1)
            
        # 결과 정렬
        final_results = []
        for content, score in merged_scores.items():
            result = content_map[content].copy()
            result['score'] = score
            result['source'] = 'rrf'
            final_results.append(result)
            
        final_results.sort(key=lambda x: x['score'], reverse=True)
        return final_results

    def rerank(self, query: str, results: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Cross-Encoder를 사용한 재순위화 (Rerank)
        """
        if not results:
            return []
            
        self._load_reranker()
        if not self.reranker:
            logger.warning("Reranker 모델이 없어 재순위화를 건너뜁니다.")
            return results[:top_k]
            
        # (Query, Document) 쌍 생성
        pairs = [[query, r['content']] for r in results]
        
        # 점수 계산
        scores = self.reranker.predict(pairs)
        
        # 결과에 점수 반영 및 정렬
        reranked_results = []
        for i, result in enumerate(results):
            new_result = result.copy()
            new_result['rerank_score'] = float(scores[i])
            reranked_results.append(new_result)
            
        reranked_results.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        logger.info(f"Rerank 완료 (Top {top_k})")
        return reranked_results[:top_k]
    
    def get_context_for_llm(self, query: str, top_k: int = 3) -> tuple[str, List[Dict[str, Any]]]:
        """
        LLM에 전달할 컨텍스트 생성
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 컨텍스트 개수
            
        Returns:
            (컨텍스트 문자열, 검색 결과 리스트)
        """
        # 설정 확인
        search_type = self.config['rag'].get('search_type', 'hybrid_weighted')
        use_rerank = self.config['rag'].get('use_rerank', False)
        
        # 1. 1차 검색 (Retrieval)
        if search_type == 'hybrid_rrf':
            # RRF 사용 시 더 많은 후보군 가져오기
            bm25_res = self.bm25_search(query, top_k=50)
            vector_res = self.vector_search(query, top_k=50)
            results = self.rrf_merge(bm25_res, vector_res)
            # Rerank 안 할거면 여기서 자르기
            if not use_rerank:
                results = results[:top_k]
        else:
            # 기본 Weighted Hybrid
            results = self.hybrid_search(query, top_k=50 if use_rerank else top_k)
            
        # 2. 2차 검색 (Reranking)
        if use_rerank:
            results = self.rerank(query, results, top_k=top_k)
        
        if not results:
            return "관련 문서를 찾을 수 없습니다.", []
        
        # 컨텍스트 문자열 생성
        context_parts = []
        for i, result in enumerate(results):
            source_file = result['metadata'].get('source_file', 'unknown')
            page = result['metadata'].get('page', '?')
            
            context_parts.append(
                f"[문서 {i+1}] {source_file} (p.{page})\n{result['content']}"
            )
        
        context = "\n\n".join(context_parts)
        
        return context, results


if __name__ == "__main__":
    # 테스트 코드
    print("HybridRetriever 모듈 로드 완료")

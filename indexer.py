"""
인덱서 모듈 - 문서 청킹 및 벡터 DB 인덱싱
"""
import hashlib
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class DocumentIndexer:
    """문서 청킹 및 ChromaDB 인덱싱을 담당하는 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 설정 딕셔너리
                - rag.chunk_size: 청크 크기
                - rag.chunk_overlap: 청크 오버랩
                - vector_db.path: ChromaDB 저장 경로
                - vector_db.collection_name: 컬렉션 이름
                - vector_db.hnsw: HNSW 파라미터
        """
        self.config = config
        
        # 텍스트 분할기 초기화
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config['rag']['chunk_size'],
            chunk_overlap=config['rag']['chunk_overlap'],
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        
        # 임베딩 모델 초기화
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # ChromaDB 클라이언트 초기화
        db_path = config['vector_db']['path']
        Path(db_path).mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(
                anonymized_telemetry=False
            )
        )
        
        # 컬렉션 초기화
        collection_name = config['vector_db']['collection_name']
        hnsw_params = config['vector_db']['hnsw']
        
        try:
            self.collection = self.client.get_collection(
                name=collection_name
            )
            logger.info(f"기존 컬렉션 로드: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "hnsw:M": hnsw_params['M'],
                    "hnsw:construction_ef": hnsw_params['ef_construction'],
                    "hnsw:search_ef": hnsw_params['ef']
                }
            )
            logger.info(f"새 컬렉션 생성: {collection_name}")
        
        # 파일 해시 캐시 초기화
        self.hash_cache_path = Path(db_path) / "file_hashes.json"
        self.file_hashes = self._load_hash_cache()
    
    def create_chunks(self, documents: List[Document]) -> List[Document]:
        """
        문서를 청크로 분할하고 메타데이터 보강
        
        Args:
            documents: 원본 Document 리스트
            
        Returns:
            청크화된 Document 리스트
        """
        all_chunks = []
        
        for doc in documents:
            # 텍스트 분할
            chunks = self.text_splitter.split_documents([doc])
            
            # 메타데이터 보강: Context Enrichment
            for i, chunk in enumerate(chunks):
                # 문서 제목 추가 (파일명 사용)
                chunk.metadata['source_file'] = doc.metadata.get('file_name', 'unknown')
                chunk.metadata['chunk_index'] = i
                chunk.metadata['total_chunks'] = len(chunks)
                
                # 청크 텍스트 길이 저장
                chunk.metadata['chunk_length'] = len(chunk.page_content)
                
                all_chunks.append(chunk)
        
        logger.info(f"총 {len(all_chunks)}개의 청크 생성")
        
        # 청크 크기 통계
        chunk_sizes = [len(chunk.page_content) for chunk in all_chunks]
        if chunk_sizes:
            avg_size = sum(chunk_sizes) / len(chunk_sizes)
            logger.info(f"평균 청크 크기: {avg_size:.0f} 문자")
            logger.info(f"최소/최대: {min(chunk_sizes)}/{max(chunk_sizes)} 문자")
        
        return all_chunks
    
    def index_documents(self, documents: List[Document], file_path: str) -> bool:
        """
        문서를 인덱싱 (Incremental Indexing 지원)
        
        Args:
            documents: 인덱싱할 Document 리스트
            file_path: 원본 파일 경로
            
        Returns:
            성공 여부
        """
        try:
            # 파일 해시 계산
            file_hash = self._calculate_file_hash(file_path)
            file_name = Path(file_path).name
            
            # 기존 해시와 비교 (Incremental Indexing)
            if file_name in self.file_hashes:
                if self.file_hashes[file_name] == file_hash:
                    logger.info(f"파일이 변경되지 않음, 인덱싱 스킵: {file_name}")
                    return True
                else:
                    logger.info(f"파일 변경 감지, 재인덱싱: {file_name}")
                    self._delete_file_from_index(file_name)
            else:
                logger.info(f"새 파일 인덱싱: {file_name}")
            
            # 청크 생성
            chunks = self.create_chunks(documents)
            
            if not chunks:
                logger.warning(f"청크가 생성되지 않음: {file_name}")
                return False
            
            # ChromaDB에 추가
            chunk_ids = []
            chunk_texts = []
            chunk_metadatas = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{file_name}_{i}"
                chunk_ids.append(chunk_id)
                chunk_texts.append(chunk.page_content)
                chunk_metadatas.append(chunk.metadata)
            
            # 임베딩 생성 및 저장
            logger.info(f"임베딩 생성 중... ({len(chunk_texts)}개 청크)")
            embeddings = self.embeddings.embed_documents(chunk_texts)
            
            self.collection.add(
                ids=chunk_ids,
                embeddings=embeddings,
                documents=chunk_texts,
                metadatas=chunk_metadatas
            )
            
            # 해시 캐시 업데이트
            self.file_hashes[file_name] = file_hash
            self._save_hash_cache()
            
            logger.info(f"인덱싱 완료: {file_name} ({len(chunks)}개 청크)")
            return True
            
        except Exception as e:
            logger.error(f"인덱싱 실패 ({file_path}): {str(e)}")
            return False
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        벡터 검색
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 개수
            
        Returns:
            검색 결과 리스트
        """
        try:
            # 쿼리 임베딩 생성
            query_embedding = self.embeddings.embed_query(query)
            
            # ChromaDB 검색
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # 결과 포맷팅
            formatted_results = []
            if results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    formatted_results.append({
                        'id': results['ids'][0][i],
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i],
                        'score': 1 - results['distances'][0][i]  # 거리를 유사도로 변환
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"검색 실패: {str(e)}")
            return []
    
    def get_indexed_files(self) -> List[str]:
        """인덱싱된 파일 목록 반환"""
        return list(self.file_hashes.keys())
    
    def delete_file(self, file_name: str) -> bool:
        """파일 인덱스 삭제"""
        try:
            self._delete_file_from_index(file_name)
            
            # 해시 캐시에서 제거
            if file_name in self.file_hashes:
                del self.file_hashes[file_name]
                self._save_hash_cache()
            
            logger.info(f"파일 삭제 완료: {file_name}")
            return True
            
        except Exception as e:
            logger.error(f"파일 삭제 실패 ({file_name}): {str(e)}")
            return False
    
    def _delete_file_from_index(self, file_name: str):
        """ChromaDB에서 파일의 모든 청크 삭제"""
        # 해당 파일의 모든 청크 ID 찾기
        all_ids = self.collection.get()['ids']
        file_chunk_ids = [id for id in all_ids if id.startswith(f"{file_name}_")]
        
        if file_chunk_ids:
            self.collection.delete(ids=file_chunk_ids)
            logger.info(f"삭제된 청크 수: {len(file_chunk_ids)}")
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """파일의 SHA-256 해시 계산"""
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            # 큰 파일을 위해 청크 단위로 읽기
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest()
    
    def _load_hash_cache(self) -> Dict[str, str]:
        """해시 캐시 로드"""
        if self.hash_cache_path.exists():
            try:
                with open(self.hash_cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"해시 캐시 로드 실패: {str(e)}")
                return {}
        return {}
    
    def _save_hash_cache(self):
        """해시 캐시 저장"""
        try:
            with open(self.hash_cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.file_hashes, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"해시 캐시 저장 실패: {str(e)}")


if __name__ == "__main__":
    # 테스트 코드
    import yaml
    from loaders import load_documents
    
    # 설정 로드
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 인덱서 초기화
    indexer = DocumentIndexer(config)
    
    print("인덱싱된 파일 목록:", indexer.get_indexed_files())

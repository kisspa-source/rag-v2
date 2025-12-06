"""
LLM 클라이언트 모듈 - Ollama 연동 및 RAG 체인 구성
"""
import logging
import time
from typing import Dict, Any, Optional

from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)


class LLMClient:
    """Ollama LLM을 사용하는 RAG 클라이언트"""
    
    DEFAULT_SYSTEM_PROMPT = """당신은 제공된 문서를 바탕으로 질문에 답변하는 유능한 어시스턴트입니다.

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
    
    def __init__(self, config: Dict[str, Any], system_prompt: Optional[str] = None):
        """
        Args:
            config: 설정 딕셔너리
                - llm.model_name: Ollama 모델 이름
                - llm.temperature: Temperature 설정
                - llm.max_tokens: 최대 토큰 수
                - llm.timeout: 타임아웃 (초)
            system_prompt: 커스텀 시스템 프롬프트 (선택)
        """
        self.config = config
        # Config에서 system_prompt가 있으면 우선 사용, 없으면 인자값 또는 기본값 사용
        config_system_prompt = config.get('llm', {}).get('system_prompt')
        self.system_prompt = config_system_prompt or system_prompt or self.DEFAULT_SYSTEM_PROMPT
        
        # Ollama LLM 초기화
        llm_config = config['llm']
        self.llm = Ollama(
            model=llm_config['model_name'],
            temperature=llm_config.get('temperature', 0.3),
            num_predict=llm_config.get('max_tokens', 512),
            timeout=llm_config.get('timeout', 120)
        )
        
        logger.info(f"LLM 초기화 완료: {llm_config['model_name']}")
        
        # 프롬프트 템플릿 생성
        self.prompt_template = PromptTemplate(
            template=self.system_prompt,
            input_variables=["context", "question"]
        )
        
        # RAG 체인 구성
        self.rag_chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )
    
    def generate_answer(self, question: str, context: str) -> Dict[str, Any]:
        """
        질문과 컨텍스트를 바탕으로 답변 생성
        
        Args:
            question: 사용자 질문
            context: 검색된 컨텍스트
            
        Returns:
            {
                'answer': 답변 텍스트,
                'elapsed_time': 소요 시간 (초)
            }
        """
        try:
            start_time = time.time()
            
            logger.info(f"답변 생성 중... (모델: {self.config['llm']['model_name']})")
            
            # RAG 체인 실행
            answer = self.rag_chain.invoke({
                "context": context,
                "question": question
            })
            
            elapsed_time = time.time() - start_time
            
            logger.info(f"답변 생성 완료 ({elapsed_time:.2f}초)")
            
            return {
                'answer': answer.strip(),
                'elapsed_time': elapsed_time
            }
            
        except Exception as e:
            logger.error(f"답변 생성 실패: {str(e)}")
            return {
                'answer': f"답변 생성 중 오류가 발생했습니다: {str(e)}",
                'elapsed_time': 0
            }
    
    def test_connection(self) -> bool:
        """
        Ollama 연결 테스트
        
        Returns:
            연결 성공 여부
        """
        try:
            logger.info("Ollama 연결 테스트 중...")
            response = self.llm.invoke("안녕하세요")
            logger.info(f"연결 성공: {response[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Ollama 연결 실패: {str(e)}")
            logger.error("Ollama 서비스가 실행 중인지 확인하세요: http://localhost:11434")
            return False
    
    def update_system_prompt(self, new_prompt: str):
        """시스템 프롬프트 업데이트"""
        self.system_prompt = new_prompt
        self.prompt_template = PromptTemplate(
            template=self.system_prompt,
            input_variables=["context", "question"]
        )
        
        # RAG 체인 재구성
        self.rag_chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )
        
        logger.info("시스템 프롬프트 업데이트 완료")
    
    def generate_questions(self, context: str, num_questions: int = 3) -> list[str]:
        """
        컨텍스트를 바탕으로 추천 질문 생성
        """
        try:
            prompt = f"""다음 텍스트를 읽고, 사용자가 물어볼 만한 핵심 질문 {num_questions}개를 한국어로 작성해주세요.
질문만 나열하고, 번호나 다른 텍스트는 포함하지 마세요. 각 질문은 줄바꿈으로 구분하세요.

텍스트:
{context[:2000]}...

질문:"""
            
            response = self.llm.invoke(prompt)
            questions = [q.strip() for q in response.split('\n') if q.strip() and '?' in q]
            return questions[:num_questions]
            
        except Exception as e:
            logger.error(f"질문 생성 실패: {e}")
            return []


if __name__ == "__main__":
    # 테스트 코드
    import yaml
    
    # 설정 로드
    with open('config/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # LLM 클라이언트 초기화
    client = LLMClient(config)
    
    # 연결 테스트
    if client.test_connection():
        print("✅ Ollama 연결 성공")
        
        # 간단한 질문 테스트
        test_context = "파이썬은 고수준 프로그래밍 언어입니다. 읽기 쉽고 배우기 쉬운 것이 특징입니다."
        test_question = "파이썬의 특징은 무엇인가요?"
        
        result = client.generate_answer(test_question, test_context)
        print(f"\n질문: {test_question}")
        print(f"답변: {result['answer']}")
        print(f"소요 시간: {result['elapsed_time']:.2f}초")
    else:
        print("❌ Ollama 연결 실패")

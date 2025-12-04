import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from rag_engine import RAGEngine

def load_qa_pairs(file_path: str) -> List[Dict[str, Any]]:
    """QA 데이터셋 로드"""
    qa_pairs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                qa_pairs.append(json.loads(line))
    return qa_pairs

def evaluate_rag(engine: RAGEngine, qa_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """RAG 성능 평가"""
    results = []
    total_questions = len(qa_pairs)
    correct_count = 0
    
    print(f"총 {total_questions}개의 질문에 대한 평가를 시작합니다...")
    
    for i, item in enumerate(qa_pairs):
        question = item['question']
        expected_keywords = item['expected_keywords']
        
        print(f"\n[{i+1}/{total_questions}] 질문: {question}")
        
        # RAG 쿼리 실행
        start_time = time.time()
        response = engine.query(question)
        elapsed_time = time.time() - start_time
        
        answer = response['answer']
        print(f"답변: {answer[:100]}...")
        
        # 키워드 매칭 평가
        matched_keywords = [k for k in expected_keywords if k.lower() in answer.lower()]
        hit_rate = len(matched_keywords) / len(expected_keywords) if expected_keywords else 0
        is_correct = hit_rate >= 0.5  # 50% 이상 키워드 포함 시 정답 처리
        
        if is_correct:
            correct_count += 1
            print("✅ 정답")
        else:
            print(f"❌ 오답 (포함된 키워드: {matched_keywords})")
            
        results.append({
            'question': question,
            'answer': answer,
            'expected_keywords': expected_keywords,
            'matched_keywords': matched_keywords,
            'hit_rate': hit_rate,
            'is_correct': is_correct,
            'elapsed_time': elapsed_time,
            'sources': response['sources']
        })
        
    accuracy = correct_count / total_questions
    avg_time = sum(r['elapsed_time'] for r in results) / total_questions
    
    return {
        'accuracy': accuracy,
        'avg_time': avg_time,
        'details': results
    }

def main():
    parser = argparse.ArgumentParser(description='RAG Evaluation Script')
    parser.add_argument('--data', type=str, default='eval/qa_pairs.jsonl', help='Path to QA dataset')
    parser.add_argument('--output', type=str, default='eval/results.json', help='Path to save results')
    parser.add_argument('--index', type=str, help='File to index before evaluation')
    args = parser.parse_args()
    
    # RAG 엔진 초기화
    try:
        engine = RAGEngine()
    except Exception as e:
        print(f"RAG 엔진 초기화 실패: {e}")
        return

    # 선택적 인덱싱
    if args.index:
        print(f"파일 인덱싱 중: {args.index}")
        result = engine.load_and_index_file(args.index)
        if result['success']:
            print("인덱싱 성공")
        else:
            print(f"인덱싱 실패: {result['message']}")
            return

    # 데이터셋 로드
    try:
        qa_pairs = load_qa_pairs(args.data)
    except Exception as e:
        print(f"데이터셋 로드 실패: {e}")
        return
        
    # 평가 실행
    eval_result = evaluate_rag(engine, qa_pairs)
    
    # 결과 출력
    print("\n" + "="*50)
    print(f"평가 결과")
    print("="*50)
    print(f"정확도 (Keyword Hit Rate >= 0.5): {eval_result['accuracy']*100:.1f}%")
    print(f"평균 응답 시간: {eval_result['avg_time']:.2f}초")
    print("="*50)
    
    # 결과 저장
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(eval_result, f, ensure_ascii=False, indent=2)
    print(f"상세 결과가 {args.output}에 저장되었습니다.")

if __name__ == "__main__":
    main()

"""
Lambda 핸들러 모듈
API Gateway를 통한 질의응답 처리
"""

import json
from src.rag.pipeline import RAGPipeline


def lambda_handler(event, context):
    """
    Lambda 함수 핸들러
    
    Args:
        event: API Gateway 이벤트
        context: Lambda 컨텍스트
        
    Returns:
        API Gateway 응답
    """
    try:
        # 요청 본문 파싱
        body = json.loads(event.get('body', '{}'))
        question = body.get('question', '')
        top_k = body.get('top_k', 5)
        
        if not question:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'question 파라미터가 필요합니다.'
                })
            }
        
        # RAG 파이프라인 초기화 (실제로는 환경 변수에서 설정 로드)
        rag_pipeline = RAGPipeline()
        
        # 질의 처리
        result = rag_pipeline.process_query(question, top_k)
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(result)
        }
    
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }


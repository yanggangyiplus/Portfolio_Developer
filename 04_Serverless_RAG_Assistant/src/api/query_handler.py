"""
Lambda 핸들러 모듈
API Gateway를 통한 질의응답 처리
"""

import json
import os
from src.rag.pipeline import RAGPipeline


def get_cors_headers():
    """
    CORS 헤더 반환
    프로덕션에서는 특정 도메인만 허용하도록 설정
    """
    allowed_origins = os.getenv('ALLOWED_ORIGINS', '*').split(',')
    origin = os.getenv('HTTP_ORIGIN', '*')
    
    # 프로덕션에서는 특정 도메인만 허용
    if origin in allowed_origins or '*' in allowed_origins:
        return {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': origin,
            'Access-Control-Allow-Methods': 'POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type'
        }
    else:
        return {
            'Content-Type': 'application/json'
        }


def lambda_handler(event, context):
    """
    Lambda 함수 핸들러
    
    Args:
        event: API Gateway 이벤트
        context: Lambda 컨텍스트
        
    Returns:
        API Gateway 응답
    """
    cors_headers = get_cors_headers()
    
    # OPTIONS 요청 처리 (CORS preflight)
    if event.get('httpMethod') == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': cors_headers,
            'body': ''
        }
    
    try:
        # 요청 본문 파싱
        body = json.loads(event.get('body', '{}'))
        question = body.get('question', '')
        top_k = body.get('top_k', 5)
        
        if not question:
            return {
                'statusCode': 400,
                'headers': cors_headers,
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
            'headers': cors_headers,
            'body': json.dumps(result)
        }
    
    except json.JSONDecodeError as e:
        return {
            'statusCode': 400,
            'headers': cors_headers,
            'body': json.dumps({
                'error': f'잘못된 JSON 형식: {str(e)}'
            })
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': cors_headers,
            'body': json.dumps({
                'error': str(e)
            })
        }

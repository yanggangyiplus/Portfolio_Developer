"""
AWS CDK 스택 정의
Serverless RAG 서비스 인프라 구성
"""

from aws_cdk import (
    Stack,
    Duration,
    RemovalPolicy,
    aws_lambda as lambda_,
    aws_apigateway as apigateway,
    aws_s3 as s3,
    aws_dynamodb as dynamodb,
)
from constructs import Construct


class RagServerlessStack(Stack):
    """Serverless RAG 서비스 인프라 스택"""
    
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)
        
        # S3 버킷 (문서 저장)
        documents_bucket = s3.Bucket(
            self, "RagDocumentsBucket",
            bucket_name=f"rag-documents-{self.account}-{self.region}",
            versioned=True,
            auto_delete_objects=True,
            removal_policy=RemovalPolicy.DESTROY
        )
        
        # DynamoDB 테이블 (벡터 스토어)
        vector_store_table = dynamodb.Table(
            self, "RagVectorStoreTable",
            table_name="rag-documents",
            partition_key=dynamodb.Attribute(
                name="document_id",
                type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.DESTROY
        )
        
        # Lambda 함수 (질의응답 핸들러)
        query_handler = lambda_.Function(
            self, "RagQueryHandler",
            function_name="rag-query-handler",
            runtime=lambda_.Runtime.PYTHON_3_11,
            handler="query_handler.lambda_handler",
            code=lambda_.Code.from_asset("src/api"),
            environment={
                "VECTOR_STORE_TABLE": vector_store_table.table_name,
                "DOCUMENTS_BUCKET": documents_bucket.bucket_name,
            },
            timeout=Duration.seconds(30)
        )
        
        # API Gateway
        api = apigateway.RestApi(
            self, "RagApi",
            rest_api_name="RAG Assistant API",
            description="Serverless RAG Assistant REST API"
        )
        
        # /rag/query 엔드포인트
        query_resource = api.root.add_resource("rag").add_resource("query")
        query_resource.add_method(
            "POST",
            apigateway.LambdaIntegration(query_handler)
        )
        
        # DynamoDB 읽기 권한 부여
        vector_store_table.grant_read_data(query_handler)
        documents_bucket.grant_read(query_handler)


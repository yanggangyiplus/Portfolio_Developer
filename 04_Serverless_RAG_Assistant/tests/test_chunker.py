"""
문서 청킹 모듈 테스트
"""

import pytest
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.chunker import DocumentChunker


class TestDocumentChunker:
    """문서 청킹 모듈 테스트 클래스"""
    
    def test_chunk_text(self):
        """텍스트 청킹 테스트"""
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
        
        text = " ".join(["word"] * 200)  # 200단어 텍스트
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) > 0
        assert all(len(chunk.split()) <= 100 for chunk in chunks)
    
    def test_chunk_documents(self):
        """여러 문서 청킹 테스트"""
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
        
        documents = [
            " ".join(["word"] * 100),
            " ".join(["word"] * 100)
        ]
        
        chunks = chunker.chunk_documents(documents)
        assert len(chunks) > 0
    
    def test_empty_text(self):
        """빈 텍스트 처리 테스트"""
        chunker = DocumentChunker()
        chunks = chunker.chunk_text("")
        assert chunks == []


if __name__ == "__main__":
    pytest.main([__file__])


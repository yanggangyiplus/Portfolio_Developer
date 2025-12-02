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
    
    def test_invalid_chunk_size(self):
        """잘못된 chunk_size 파라미터 테스트"""
        with pytest.raises(ValueError):
            DocumentChunker(chunk_size=0)
        
        with pytest.raises(ValueError):
            DocumentChunker(chunk_size=-1)
    
    def test_invalid_chunk_overlap(self):
        """잘못된 chunk_overlap 파라미터 테스트"""
        with pytest.raises(ValueError):
            DocumentChunker(chunk_size=100, chunk_overlap=-1)
        
        with pytest.raises(ValueError):
            DocumentChunker(chunk_size=100, chunk_overlap=100)
        
        with pytest.raises(ValueError):
            DocumentChunker(chunk_size=100, chunk_overlap=150)
    
    def test_infinite_loop_prevention(self):
        """무한 루프 방지 테스트 (chunk_size == chunk_overlap)"""
        # chunk_size와 chunk_overlap이 같으면 step_size가 1이 되어야 함
        chunker = DocumentChunker(chunk_size=10, chunk_overlap=9)
        text = " ".join(["word"] * 100)
        chunks = chunker.chunk_text(text)
        
        # 무한 루프 없이 정상 종료되어야 함
        assert len(chunks) > 0
        assert len(chunks) <= len(text.split())
    
    def test_single_word_text(self):
        """단일 단어 텍스트 처리 테스트"""
        chunker = DocumentChunker(chunk_size=10, chunk_overlap=2)
        chunks = chunker.chunk_text("word")
        assert len(chunks) == 1
        assert chunks[0] == "word"
    
    def test_chunk_size_larger_than_text(self):
        """청크 크기가 텍스트보다 큰 경우 테스트"""
        chunker = DocumentChunker(chunk_size=1000, chunk_overlap=100)
        text = " ".join(["word"] * 10)
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) == 1
        assert chunks[0] == text


if __name__ == "__main__":
    pytest.main([__file__])

"""
문서 청킹 모듈
문서를 적절한 크기로 나누어 검색 정확도를 향상시킴
"""

from typing import List


class DocumentChunker:
    """문서를 청크로 나누는 클래스"""
    
    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 50):
        """
        Args:
            chunk_size: 청크 크기 (토큰 수)
            chunk_overlap: 청크 간 겹치는 부분 (토큰 수)
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size는 0보다 커야 합니다.")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap은 0 이상이어야 합니다.")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap은 chunk_size보다 작아야 합니다.")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.step_size = max(1, chunk_size - chunk_overlap)  # 무한 루프 방지
    
    def chunk_text(self, text: str) -> List[str]:
        """
        텍스트를 청크로 나눔
        
        Args:
            text: 청킹할 텍스트
            
        Returns:
            청크 리스트
        """
        if not text:
            return []
        
        # 간단한 토큰 기반 청킹 (실제로는 더 정교한 토크나이저 사용)
        words = text.split()
        
        if len(words) == 0:
            return []
        
        chunks = []
        i = 0
        
        while i < len(words):
            chunk_words = words[i:i + self.chunk_size]
            chunk = ' '.join(chunk_words)
            chunks.append(chunk)
            
            # step_size만큼 이동 (무한 루프 방지)
            i += self.step_size
            
            # 종료 조건: 더 이상 새로운 청크를 만들 수 없을 때
            if i >= len(words):
                break
        
        return chunks
    
    def chunk_documents(self, documents: List[str]) -> List[str]:
        """
        여러 문서를 청크로 나눔
        
        Args:
            documents: 문서 리스트
            
        Returns:
            청크 리스트
        """
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_text(doc)
            all_chunks.extend(chunks)
        return all_chunks

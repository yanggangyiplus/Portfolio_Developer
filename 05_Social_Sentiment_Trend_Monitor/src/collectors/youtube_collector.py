"""
YouTube API 수집기 모듈
댓글 및 영상 메타데이터 수집
"""

from typing import List, Dict, Optional
import os
import warnings


class YouTubeCollector:
    """YouTube API를 통한 데이터 수집 클래스"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: YouTube Data API 키
            
        Raises:
            ValueError: API 키가 제공되지 않았을 때 (프로덕션 환경)
        """
        self.api_key = api_key or os.getenv('YOUTUBE_API_KEY', '')
        
        # 개발 환경에서는 경고만, 프로덕션에서는 에러 발생
        if not self.api_key:
            warnings.warn(
                "YouTube API 키가 설정되지 않았습니다. Mock 데이터를 사용합니다.",
                UserWarning
            )
        # 실제로는 googleapiclient.discovery.build 사용
        # if not self.api_key:
        #     raise ValueError("YouTube API 키가 필요합니다. 환경 변수 YOUTUBE_API_KEY를 설정하거나 api_key 파라미터를 제공하세요.")
    
    def search_videos(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        비디오 검색
        
        Args:
            query: 검색 쿼리
            max_results: 최대 결과 수
            
        Returns:
            비디오 정보 리스트
        """
        if not query:
            return []
        
        if not self.api_key:
            # Mock 응답 (개발 환경)
            return [
                {
                    'video_id': 'mock_video_1',
                    'title': f'{query} 관련 영상 1',
                    'description': '영상 설명',
                    'published_at': '2024-01-01T00:00:00Z'
                }
            ]
        
        # 실제 API 호출 구현
        # youtube = googleapiclient.discovery.build('youtube', 'v3', developerKey=self.api_key)
        # request = youtube.search().list(q=query, part='snippet', maxResults=max_results)
        # response = request.execute()
        # ...
        
        return []
    
    def get_comments(self, video_id: str, max_results: int = 100) -> List[Dict]:
        """
        비디오 댓글 수집
        
        Args:
            video_id: 비디오 ID
            max_results: 최대 댓글 수
            
        Returns:
            댓글 리스트
        """
        if not video_id:
            return []
        
        if not self.api_key:
            # Mock 응답 (개발 환경)
            return [
                {
                    'comment_id': 'comment_1',
                    'text': '좋은 영상이네요!',
                    'author': 'user1',
                    'published_at': '2024-01-01T00:00:00Z',
                    'like_count': 10
                }
            ]
        
        # 실제 API 호출 구현
        # youtube = googleapiclient.discovery.build('youtube', 'v3', developerKey=self.api_key)
        # request = youtube.commentThreads().list(part='snippet', videoId=video_id, maxResults=max_results)
        # response = request.execute()
        # ...
        
        return []
    
    def collect_data(self, query: str, max_videos: int = 10, max_comments_per_video: int = 100) -> List[Dict]:
        """
        검색어에 대한 전체 데이터 수집
        
        Args:
            query: 검색 쿼리
            max_videos: 최대 비디오 수
            max_comments_per_video: 비디오당 최대 댓글 수
            
        Returns:
            수집된 데이터 리스트
        """
        if not query:
            return []
        
        videos = self.search_videos(query, max_videos)
        all_data = []
        
        for video in videos:
            comments = self.get_comments(video['video_id'], max_comments_per_video)
            for comment in comments:
                all_data.append({
                    'video_id': video['video_id'],
                    'video_title': video['title'],
                    'comment_text': comment['text'],
                    'author': comment['author'],
                    'published_at': comment['published_at']
                })
        
        return all_data

from typing import List, Dict, Any
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
import httpx
import isodate
import logging
import traceback
import json

logger = logging.getLogger(__name__)

class YouTubeAPI:
    def __init__(self, api_key: str):
        logger.info("YouTubeAPI :: def __init__")
        self.api_key = api_key
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        
    async def search_videos(self, query: str, max_results: int = 20, video_duration: str = None) -> List[Dict[str, Any]]:
        logger.info("YouTubeAPI :: def search_videos")
        """
        Busca vídeos no YouTube Kids.
        
        Args:
            query: Termo de busca
            max_results: Número máximo de resultados
            video_duration: Duração dos vídeos ('short', 'medium', 'long' ou None para todos)
        
        Returns:
            List[Dict[str, Any]]: Lista de vídeos encontrados
        """
        try:
            # Busca por vídeos sem adicionar "for kids" para aumentar resultados
            safe_query = query
            logger.info(f"Searching for query: {safe_query}")
            
            # Fazendo a busca com parâmetros básicos
            search_params = {
                'q': safe_query,
                'part': 'snippet',
                'maxResults': max_results,
                'type': 'video',
                'relevanceLanguage': 'pt',
                'safeSearch': 'strict'
            }
            
            logger.info("Using strict safeSearch mode for child-appropriate content")
            
            # Adiciona o parâmetro de duração se especificado
            if video_duration and video_duration in ['short', 'medium', 'long']:
                search_params['videoDuration'] = video_duration
                logger.info(f"Filtering by duration: {video_duration}")
            
            logger.info(f"Search parameters: {json.dumps(search_params)}")
            
            search_response = self.youtube.search().list(**search_params).execute()
            
            logger.info(f"Search response received")
            
            if not search_response.get('items'):
                logger.warning(f"No videos found in search response for query: {query}")
                return []

            # Coleta IDs dos vídeos para buscar mais informações
            video_ids = [item['id']['videoId'] for item in search_response['items']]
            logger.info(f"Found {len(video_ids)} video IDs")
            
            if not video_ids:
                logger.warning("No video IDs extracted from search results")
                return []
                
            # Busca detalhes dos vídeos
            logger.info(f"Fetching details for {len(video_ids)} videos")
            videos_response = self.youtube.videos().list(
                part='snippet,contentDetails,statistics',
                id=','.join(video_ids)
            ).execute()
            
            videos_count = len(videos_response.get('items', []))
            logger.info(f"Videos details response received with {videos_count} items")
            
            if videos_count == 0:
                logger.warning("No video details found")
                return []
            
            videos = []
            for item in videos_response.get('items', []):
                try:
                    # Converte duração ISO 8601 para segundos
                    duration_str = item['contentDetails']['duration']
                    duration = isodate.parse_duration(duration_str).total_seconds()
                    
                    # Cria objeto com dados do vídeo
                    video_data = {
                        'id': item['id'],
                        'title': item['snippet']['title'],
                        'description': item['snippet']['description'],
                        'thumbnail': item['snippet']['thumbnails']['high']['url'] if 'high' in item['snippet']['thumbnails'] else item['snippet']['thumbnails']['default']['url'],
                        'channel_title': item['snippet']['channelTitle'],
                        'published_at': item['snippet']['publishedAt'],
                        'duration': duration_str,
                        'duration_seconds': duration,
                        'view_count': int(item['statistics'].get('viewCount', 0)),
                        'like_count': int(item['statistics'].get('likeCount', 0)),
                        'comment_count': int(item['statistics'].get('commentCount', 0))
                    }
                    videos.append(video_data)
                    logger.info(f"Processed video: '{video_data['title']}' - Duration: {duration} seconds")
                except Exception as e:
                    logger.error(f"Error processing video {item.get('id', 'unknown')}: {str(e)}")
                    logger.error(traceback.format_exc())
                    continue
            
            logger.info(f"Successfully processed {len(videos)} videos")
            
            # Retorna todos os vídeos mesmo que não tenham sido processados os filtros
            return videos
            
        except Exception as e:
            logger.error(f"Error in YouTube API search: {str(e)}")
            logger.error(traceback.format_exc())
            return []
            
    async def get_video_data(self, video_id: str) -> Dict[str, Any]:
        logger.info(f"YouTubeAPI :: def get_video_data for {video_id}")
        """
        Coleta dados detalhados de um vídeo.
        """
        try:
            # Obtém detalhes do vídeo
            video_response = self.youtube.videos().list(
                part='snippet,contentDetails,statistics',
                id=video_id
            ).execute()
            
            if not video_response.get('items'):
                logger.warning(f"No data found for video ID: {video_id}")
                return {}
                
            video_info = video_response['items'][0]
            
            # Tenta obter a transcrição
            transcript_text = ''
            try:
                transcript = YouTubeTranscriptApi.get_transcript(
                    video_id,
                    languages=['pt', 'en']
                )
                transcript_text = ' '.join(item['text'] for item in transcript)
                logger.debug(f"Successfully retrieved transcript for video {video_id}")
            except Exception as e:
                logger.warning(f"Could not get transcript for video {video_id}: {str(e)}")
            
            # Retorna dados consolidados
            video_data = {
                'id': video_id,
                'title': video_info['snippet']['title'],
                'description': video_info['snippet']['description'],
                'duration': video_info['contentDetails']['duration'],
                'duration_seconds': isodate.parse_duration(video_info['contentDetails']['duration']).total_seconds(),
                'view_count': int(video_info['statistics'].get('viewCount', 0)),
                'like_count': int(video_info['statistics'].get('likeCount', 0)),
                'comment_count': int(video_info['statistics'].get('commentCount', 0)),
                'transcript': transcript_text,
                'tags': video_info['snippet'].get('tags', []),
                'category_id': video_info['snippet'].get('categoryId', ''),
                'thumbnail': video_info['snippet']['thumbnails']['high']['url'] if 'high' in video_info['snippet']['thumbnails'] else video_info['snippet']['thumbnails']['default']['url'],
                'channel_title': video_info['snippet']['channelTitle']
            }
            
            logger.debug(f"Successfully processed video data for {video_id}")
            return video_data
            
        except Exception as e:
            logger.error(f"Error getting video data for {video_id}: {str(e)}")
            logger.error(traceback.format_exc())
            return {} 
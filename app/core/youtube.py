from typing import List, Dict, Any
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import *
import httpx
import isodate
import logging
import traceback
import json
import re
from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class YouTubeAPI:
    def __init__(self, api_key: str):
        logger.info("YouTubeAPI :: def __init__")
        self.api_key = api_key
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        
    async def search_videos(self, query: str, max_results: int = None, video_duration: str = None, nlp_filters_enabled: List[str] = None) -> List[Dict[str, Any]]:
        logger.info("YouTubeAPI :: def search_videos")
        """
        Busca vídeos no YouTube Kids.
        
        Args:
            query: Termo de busca
            max_results: Número máximo de resultados (padrão: configuração MAX_SEARCH_RESULTS)
            video_duration: Duração dos vídeos ('short', 'medium', 'long' ou None para todos)
            nlp_filters_enabled: Lista de filtros de PLN habilitados
        
        Returns:
            List[Dict[str, Any]]: Lista de vídeos encontrados
        """
        # Usa a configuração padrão se max_results não for especificado
        if max_results is None:
            max_results = settings.MAX_SEARCH_RESULTS
            
        # Lista de filtros de PLN que requerem transcrição
        nlp_filter_names = ["Análise de Sentimentos", "Detecção de Toxicidade", "Classificação Educacional", "Detecção de Linguagem Imprópria"]
        
        # Verifica se algum filtro de PLN está habilitado
        needs_transcription = False
        if nlp_filters_enabled:
            needs_transcription = any(filter_name in nlp_filters_enabled for filter_name in nlp_filter_names)
            logger.info(f"Filtros de PLN habilitados: {nlp_filters_enabled}")
            logger.info(f"Transcrição necessária: {needs_transcription}")
            
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
                    
                    # Obtém frases do vídeo se filtros de PLN estão habilitados
                    if needs_transcription:
                        logger.info(f"Obtendo frases para vídeo: {video_data['title']}")
                        sentences = await self.get_video_sentences(item['id'])
                        video_data['sentences'] = sentences
                        logger.info(f"Frases obtidas: início='{sentences['start'][:50]}...', meio='{sentences['middle'][:50]}...', fim='{sentences['end'][:50]}...'")
                    else:
                        video_data['sentences'] = {"start": "", "middle": "", "end": ""}
                    
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
    
    async def get_video_sentences(self, video_id: str) -> Dict[str, str]:
        """
        Extrai três frases de um vídeo (início, meio e fim) a partir da transcrição.
        
        Args:
            video_id: ID do vídeo do YouTube
            
        Returns:
            Dict[str, str]: Dicionário com as frases do início, meio e fim
        """
        logger.info(f"YouTubeAPI :: get_video_sentences for {video_id}")
        
        # Verifica se a transcrição está habilitada
        if not settings.ENABLE_VIDEO_TRANSCRIPTION:
            logger.info("Video transcription is disabled in settings")
            return {"start": "", "middle": "", "end": ""}
        
        try:
            # Tenta obter transcrições em português primeiro
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Procura por transcrições em português
            portuguese_transcripts = []
            for transcript in transcript_list:
                is_portuguese = (
                    transcript.language_code.startswith('pt') or 
                    'portuguese' in transcript.language.lower() or
                    'português' in transcript.language.lower()
                )
                
                if is_portuguese:
                    portuguese_transcripts.append(transcript)
            
            if not portuguese_transcripts:
                logger.warning(f"No Portuguese transcripts found for video {video_id}")
                return {"start": "", "middle": "", "end": ""}
            
            # Escolhe a melhor transcrição (manual tem prioridade sobre auto-gerada)
            best_transcript = None
            for transcript in portuguese_transcripts:
                if not transcript.is_generated:
                    best_transcript = transcript
                    logger.info(f"Using manual transcript: {transcript.language}")
                    break
            
            if not best_transcript:
                best_transcript = portuguese_transcripts[0]
                logger.info(f"Using auto-generated transcript: {best_transcript.language}")
            
            # Baixa a transcrição
            transcript_data = best_transcript.fetch()
            
            if not transcript_data:
                logger.warning(f"Empty transcript data for video {video_id}")
                return {"start": "", "middle": "", "end": ""}
            
            # Extrai frases do início, meio e fim
            total_segments = len(transcript_data)
            
            # Início: primeiros 10% dos segmentos
            start_end = max(1, total_segments // 10)
            start_text = ' '.join(segment['text'] for segment in transcript_data[:start_end])
            
            # Meio: segmentos do meio (40% a 60%)
            middle_start = int(total_segments * 0.4)
            middle_end = int(total_segments * 0.6)
            middle_text = ' '.join(segment['text'] for segment in transcript_data[middle_start:middle_end])
            
            # Fim: últimos 10% dos segmentos
            end_start = max(0, total_segments - (total_segments // 10))
            end_text = ' '.join(segment['text'] for segment in transcript_data[end_start:])
            
            # Limpa e extrai primeira frase de cada parte
            def extract_first_sentence(text: str) -> str:
                if not text:
                    return ""
                
                # Remove quebras de linha e espaços extras
                clean_text = re.sub(r'\s+', ' ', text.strip())
                
                # Procura por fim de frase (ponto, exclamação, interrogação)
                sentence_end = re.search(r'[.!?]', clean_text)
                if sentence_end:
                    return clean_text[:sentence_end.end()].strip()
                
                # Se não encontrar fim de frase, pega até 100 caracteres
                return clean_text[:100].strip() + ("..." if len(clean_text) > 100 else "")
            
            result = {
                "start": extract_first_sentence(start_text),
                "middle": extract_first_sentence(middle_text),
                "end": extract_first_sentence(end_text)
            }
            
            logger.info(f"Successfully extracted sentences for video {video_id}")
            return result
            
        except NoTranscriptFound:
            logger.warning(f"No transcript found for video {video_id}")
            return {"start": "", "middle": "", "end": ""}
        except VideoUnavailable:
            logger.warning(f"Video {video_id} is unavailable")
            return {"start": "", "middle": "", "end": ""}
        except Exception as e:
            logger.error(f"Error extracting sentences for video {video_id}: {str(e)}")
            logger.error(traceback.format_exc())
            return {"start": "", "middle": "", "end": ""}
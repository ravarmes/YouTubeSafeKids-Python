from typing import List, Dict, Any
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import *
import isodate
import logging
import traceback
import json
import re
import asyncio
from .config import get_settings
from app.cache import TranscriptionCacheManager

logger = logging.getLogger(__name__)
settings = get_settings()

class YouTubeAPI:
    def __init__(self, api_key: str):
        logger.info("YouTubeAPI :: def __init__")
        self.api_key = api_key
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        self._cache_manager = TranscriptionCacheManager()
    
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
        
        # Verifica se é uma consulta de teste com cache de transcrições
        if self._cache_manager.is_cached_query(query):
            cached_results = await self._cache_manager.build_cached_results(
                query, max_results, self.youtube
            )
            if cached_results is not None:
                return cached_results
            
        # Lista de filtros de PLN que requerem transcrição (nomes devem corresponder aos registrados no main.py)
        nlp_filter_names = settings.NLP_FILTER_NAMES
        
        # Verifica se algum filtro de PLN está habilitado
        needs_transcription = False
        if nlp_filters_enabled:
            needs_transcription = any(filter_name in nlp_filters_enabled for filter_name in nlp_filter_names)
            logger.info(f"Filtros de PLN habilitados: {nlp_filters_enabled}")
            logger.info(f"Transcrição necessária: {needs_transcription}")
            
        try:
            safe_query = query
            logger.info(f"Searching for query: {safe_query}")
            
            # Busca mais candidatos quando filtros NLP estão ativos, para
            # compensar vídeos sem transcrição. Isto NÃO gasta mais cota:
            # - search().list() = 100 unidades (independente de maxResults)
            # - videos().list() = 1 unidade (independente de quantos IDs)
            # - youtube-transcript-api = 0 unidades (scraping gratuito)
            fetch_count = max_results * 4 if needs_transcription else max_results
            fetch_count = min(fetch_count, 15)  # Busca ampla mas limitada para evitar rate limit
            
            search_params = {
                'q': safe_query,
                'part': 'snippet',
                'maxResults': fetch_count,
                'type': 'video',
                'relevanceLanguage': 'pt',
                'safeSearch': 'strict'
            }
            
            logger.info("Using strict safeSearch mode for child-appropriate content")
            
            if video_duration and video_duration in ['short', 'medium', 'long']:
                search_params['videoDuration'] = video_duration
                logger.info(f"Filtering by duration: {video_duration}")
            
            logger.info(f"Search parameters: {json.dumps(search_params)}")
            
            search_response = self.youtube.search().list(**search_params).execute()
            
            logger.info(f"Search response received")
            
            if not search_response.get('items'):
                logger.warning(f"No videos found in search response for query: {query}")
                return []

            video_ids = [item['id']['videoId'] for item in search_response['items']]
            logger.info(f"Found {len(video_ids)} video IDs")
            
            if not video_ids:
                logger.warning("No video IDs extracted from search results")
                return []
                
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
            skipped = 0
            rate_limited = False
            for item in videos_response.get('items', []):
                # Já temos vídeos suficientes
                if len(videos) >= max_results:
                    break
                    
                try:
                    duration_str = item['contentDetails']['duration']
                    duration = isodate.parse_duration(duration_str).total_seconds()
                    
                    video_title = item['snippet']['title']
                    video_data = {
                        'id': item['id'],
                        'title': video_title,
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
                    
                    if needs_transcription:
                        # 1. Verifica cache local primeiro (sem requisição)
                        cached = self._cache_manager.find_cached_sentences(video_title)
                        if cached:
                            video_data['sentences'] = cached
                            videos.append(video_data)
                            logger.info(f"CACHE HIT ({len(videos)}/{max_results}): '{video_title}'")
                            logger.info(f"  início='{cached['start'][:50]}...', meio='{cached['middle'][:50]}...', fim='{cached['end'][:50]}...'")
                            continue
                        
                        # 2. Se rate limited, pula sem tentar a API
                        if rate_limited:
                            skipped += 1
                            logger.warning(f"Rate limited, pulando: '{video_title}'")
                            continue
                        
                        # 3. Tenta API de transcrição (com delay)
                        if skipped > 0 or len(videos) > 0:
                            await asyncio.sleep(1.5)
                        
                        logger.info(f"Verificando transcrição ({len(videos)}/{max_results}): {video_title}")
                        try:
                            sentences = await self.get_video_sentences(item['id'])
                        except Exception as transcript_error:
                            error_msg = str(transcript_error)
                            if '429' in error_msg or 'Too Many Requests' in error_msg:
                                rate_limited = True
                                logger.error(f"Rate limited pelo YouTube! Parando verificações de transcrição.")
                                skipped += 1
                                continue
                            raise
                        
                        has_transcription = any(
                            sentences.get(k, '').strip()
                            for k in ['start', 'middle', 'end']
                        )
                        
                        if not has_transcription:
                            skipped += 1
                            logger.warning(f"Sem transcrição, pulando: '{video_title}' ({skipped} pulados)")
                            continue
                        
                        video_data['sentences'] = sentences
                        logger.info(f"Transcrição OK: início='{sentences['start'][:50]}...', meio='{sentences['middle'][:50]}...', fim='{sentences['end'][:50]}...'")
                    else:
                        video_data['sentences'] = {"start": "", "middle": "", "end": ""}
                    
                    videos.append(video_data)
                    logger.info(f"Vídeo aceito ({len(videos)}/{max_results}): '{video_title}'")
                except Exception as e:
                    logger.error(f"Error processing video {item.get('id', 'unknown')}: {str(e)}")
                    logger.error(traceback.format_exc())
                    continue
            
            logger.info(f"Resultado: {len(videos)} vídeos com transcrição ({skipped} sem transcrição ignorados, rate_limited={rate_limited})")
            
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
            error_msg = str(e)
            # Re-raise rate limiting errors para o caller detectar
            if '429' in error_msg or 'Too Many Requests' in error_msg:
                logger.error(f"Rate limited ao buscar transcrição para {video_id}")
                raise
            logger.error(f"Error extracting sentences for video {video_id}: {error_msg}")
            logger.error(traceback.format_exc())
            return {"start": "", "middle": "", "end": ""}
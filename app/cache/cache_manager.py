"""
Cache Manager para Transcrições de Vídeos
------------------------------------------
Gerencia o cache de transcrições para consultas de teste específicas.
Toda a lógica de cache fica isolada neste módulo.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import logging
import isodate

logger = logging.getLogger(__name__)


class TranscriptionCacheManager:
    """Gerencia cache de transcrições para consultas de teste."""

    # Mapeamento de consultas de teste → arquivo de cache
    QUERY_CACHE_MAP = {
        'desenho pernalonga português': 'cache_pernalonga.json',
        'desenho south park português': 'cache_southpark.json',
    }

    def __init__(self):
        self._cache_dir = Path(__file__).parent
        self._transcription_cache = self._load_all_caches()

    # ------------------------------------------------------------------
    # Interface pública
    # ------------------------------------------------------------------

    def is_cached_query(self, query: str) -> bool:
        """Retorna True se a query corresponde a uma pesquisa com cache."""
        query_lower = query.lower().strip()
        return any(
            test_q in query_lower or query_lower in test_q
            for test_q in self.QUERY_CACHE_MAP
        )

    def find_cached_sentences(self, video_title: str) -> Optional[Dict[str, str]]:
        """Busca transcrição em cache pelo título do vídeo (correspondência parcial)."""
        # Correspondência exata
        if video_title in self._transcription_cache:
            return self._transcription_cache[video_title]

        # Correspondência parcial
        title_lower = video_title.lower()
        for cached_title, sentences in self._transcription_cache.items():
            cached_lower = cached_title.lower()
            if cached_lower in title_lower or title_lower in cached_lower:
                return sentences

        return None

    async def build_cached_results(
        self,
        query: str,
        max_results: int,
        youtube_client,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Para consultas de teste: busca cada vídeo do cache pelo seu título
        na YouTube Data API para obter metadados reais, e usa transcrições do cache.

        Args:
            query: Termo de busca
            max_results: Número máximo de resultados
            youtube_client: Instância autenticada de googleapiclient (youtube v3)

        Returns:
            Lista de vídeos com metadados reais + transcrições do cache,
            ou None se a query não corresponder a nenhum cache.
        """
        cache_data = self._resolve_cache_for_query(query)
        if cache_data is None:
            return None

        # Para cada título no cache, busca o vídeo pelo título no YouTube
        video_ids_map = {}  # video_id -> (cached_title, cached_sentences)

        for cached_title, sentences in cache_data.items():
            try:
                search_response = youtube_client.search().list(
                    q=cached_title,
                    part='snippet',
                    maxResults=3,
                    type='video',
                    relevanceLanguage='pt'
                ).execute()

                if search_response.get('items'):
                    for search_item in search_response['items']:
                        vid = search_item['id']['videoId']
                        yt_title = search_item['snippet']['title']
                        if self._titles_match(cached_title, yt_title):
                            video_ids_map[vid] = (cached_title, sentences)
                            logger.info(f"Encontrado no YouTube: '{yt_title}' (ID: {vid})")
                            break
                    else:
                        vid = search_response['items'][0]['id']['videoId']
                        video_ids_map[vid] = (cached_title, sentences)
                        logger.info(f"Usando primeiro resultado para '{cached_title}' (ID: {vid})")
                else:
                    logger.warning(f"Nenhum resultado no YouTube para: '{cached_title}'")
            except Exception as e:
                logger.error(f"Erro ao buscar '{cached_title}': {e}")

        if not video_ids_map:
            logger.warning("Nenhum vídeo do cache encontrado no YouTube")
            return []

        # Busca detalhes de todos os vídeos em batch (1 chamada)
        videos_response = youtube_client.videos().list(
            part='snippet,contentDetails,statistics',
            id=','.join(video_ids_map.keys())
        ).execute()

        videos: List[Dict[str, Any]] = []
        for item in videos_response.get('items', []):
            vid = item['id']
            if vid not in video_ids_map:
                continue

            cached_title, cached_sentences = video_ids_map[vid]

            try:
                duration_str = item['contentDetails']['duration']
                duration = isodate.parse_duration(duration_str).total_seconds()

                video_data = {
                    'id': vid,
                    'title': item['snippet']['title'],
                    'description': item['snippet']['description'],
                    'thumbnail': (
                        item['snippet']['thumbnails']['high']['url']
                        if 'high' in item['snippet']['thumbnails']
                        else item['snippet']['thumbnails']['default']['url']
                    ),
                    'channel_title': item['snippet']['channelTitle'],
                    'published_at': item['snippet']['publishedAt'],
                    'duration': duration_str,
                    'duration_seconds': duration,
                    'view_count': int(item['statistics'].get('viewCount', 0)),
                    'like_count': int(item['statistics'].get('likeCount', 0)),
                    'comment_count': int(item['statistics'].get('commentCount', 0)),
                    'sentences': cached_sentences,
                }
                videos.append(video_data)
                logger.info(
                    f"CACHE+API ({len(videos)}/{len(cache_data)}): "
                    f"'{item['snippet']['title']}'"
                )
            except Exception as e:
                logger.error(f"Erro ao processar vídeo {vid}: {e}")

        logger.info(
            f"Resultado cache: {len(videos)} vídeos com metadados reais "
            f"+ transcrições do cache"
        )
        return videos[:max_results]

    # ------------------------------------------------------------------
    # Métodos internos
    # ------------------------------------------------------------------

    def _load_all_caches(self) -> Dict[str, Dict[str, str]]:
        """Carrega cache de transcrições de todos os JSON em app/cache/."""
        cache: Dict[str, Dict[str, str]] = {}
        if not self._cache_dir.exists():
            logger.info("Diretório de cache não encontrado")
            return cache

        for json_file in self._cache_dir.glob('*.json'):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    cache.update(data)
                    logger.info(f"Cache carregado: {json_file.name} ({len(data)} vídeos)")
            except Exception as e:
                logger.error(f"Erro ao carregar cache {json_file}: {e}")

        logger.info(f"Total de transcrições em cache: {len(cache)}")
        return cache

    def _resolve_cache_for_query(self, query: str) -> Optional[Dict]:
        """Retorna os dados do cache correspondente à query, ou None."""
        query_lower = query.lower().strip()

        for test_query, filename in self.QUERY_CACHE_MAP.items():
            if test_query in query_lower or query_lower in test_query:
                cache_path = self._cache_dir / filename
                if not cache_path.exists():
                    logger.warning(f"Arquivo de cache não encontrado: {cache_path}")
                    return None
                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    logger.info(f"Cache carregado para teste: {filename} ({len(data)} vídeos)")
                    return data
                except Exception as e:
                    logger.error(f"Erro ao ler cache {cache_path}: {e}")
                    return None

        return None

    @staticmethod
    def _titles_match(cached_title: str, youtube_title: str) -> bool:
        """Verifica se dois títulos correspondem (parcial, case-insensitive)."""
        a = cached_title.lower().strip()
        b = youtube_title.lower().strip()
        return a in b or b in a

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Dict, Any
from ...filters import get_all_filters, filter_manager
from ...core.youtube import YouTubeAPI
from ...core.config import get_settings
import json
import logging
import sys
import traceback
import time

router = APIRouter()
settings = get_settings()
youtube_api = YouTubeAPI(settings.YOUTUBE_API_KEY)
logger = logging.getLogger(__name__)

# Configuração específica do logger para este módulo
logger.setLevel(logging.DEBUG)
# Remove handlers existentes para evitar duplicação
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
# Adiciona novo handler
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

@router.get("/search/")
async def search_videos(
    query: str,
    filter_weights: str = Query(...),
) -> Dict[str, Any]:
    """
    Search for videos and apply filters.
    
    Args:
        query: Search query
        filter_weights: JSON string containing filter weights
        
    Returns:
        Dict[str, Any]: Search results with processed videos
    """
    start_time = time.time()
    logger.info("="*80)
    logger.info(f"NOVA BUSCA DE VÍDEOS")
    logger.info("="*80)
    logger.info(f"Termo de busca: '{query}'")
    
    try:
        # Parse filter weights from JSON string
        filter_weights_dict = {}
        try:
            filter_weights_dict = json.loads(filter_weights)
            logger.info(f"Filtros recebidos: {json.dumps(filter_weights_dict, indent=2)}")
        except json.JSONDecodeError as e:
            logger.error(f"Erro ao decodificar filter_weights: {str(e)}")
            logger.error(f"filter_weights recebido: {filter_weights}")
            filter_weights_dict = {}  # Continua sem filtros
        
        # Verifica se realmente há filtros habilitados
        if not filter_weights_dict:
            logger.warning("Nenhum filtro habilitado recebido do frontend!")
        else:
            filter_names = list(filter_weights_dict.keys())
            logger.info(f"Filtros habilitados ({len(filter_names)}): {', '.join(filter_names)}")
        
        # Verifica se o filtro de duração está habilitado
        video_duration = None
        if 'Duração' in filter_weights_dict:
            duration_info = filter_weights_dict['Duração']
            # Se for um dicionário com tipo, extraímos o tipo de duração
            if isinstance(duration_info, dict) and 'type' in duration_info:
                video_duration = duration_info['type']
                logger.info(f"Usando filtro de duração: {video_duration}")
        
        # Identifica filtros de PLN habilitados
        nlp_filter_names = ["Análise de Sentimentos", "Detecção de Toxicidade", "Classificação Educacional", "Detecção de Linguagem Imprópria"]
        nlp_filters_enabled = [filter_name for filter_name in nlp_filter_names if filter_name in filter_weights_dict]
        
        if nlp_filters_enabled:
            logger.info(f"Filtros de PLN detectados: {nlp_filters_enabled}")
        
        # Busca vídeos no YouTube
        logger.info(f"Buscando vídeos no YouTube para o termo '{query}'...")
        videos = await youtube_api.search_videos(query, video_duration=video_duration, nlp_filters_enabled=nlp_filters_enabled)
        logger.info(f"API do YouTube retornou {len(videos)} vídeos")
        
        if not videos:
            elapsed = time.time() - start_time
            logger.warning(f"Nenhum vídeo encontrado na API do YouTube (tempo: {elapsed:.2f}s)")
            return {"videos": [], "total": 0}
        
        # Log de alguns vídeos encontrados
        logger.info(f"Primeiros {min(3, len(videos))} vídeos encontrados:")
        for i, video in enumerate(videos[:3], 1):
            duration = video.get('duration_seconds', 0) / 60
            logger.info(f"  {i}. '{video.get('title', 'Unknown')}' ({duration:.1f} min)")
            
        # Process each video through filters
        processed_videos = []
        logger.info(f"Processando {len(videos)} vídeos com filtros...")
        
        for i, video in enumerate(videos, 1):
            try:
                # Process video through enabled filters only
                logger.info(f"\nProcessando vídeo {i}/{len(videos)}: {video.get('title', 'Unknown')}")
                
                # Process video through enabled filters only
                processed_video = filter_manager.process_video(video, filter_weights_dict)
                processed_videos.append(processed_video)
                
                # Log do score final
                score = processed_video.get('final_score', 0)
                logger.info(f"Vídeo {i} processado. Score final: {score:.4f}")
                
            except Exception as e:
                logger.error(f"Erro ao processar vídeo {i}: {str(e)}")
                logger.error(traceback.format_exc())
                # Adiciona o vídeo mesmo com erro, mas sem score
                video['final_score'] = 0
                video['filter_scores'] = {}
                processed_videos.append(video)
        
        # Sort videos by final score
        logger.info(f"Ordenando {len(processed_videos)} vídeos por score final...")
        processed_videos.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        
        # Log dos melhores resultados
        if processed_videos:
            logger.info(f"\nMelhores resultados:")
            for i, video in enumerate(processed_videos[:5], 1):
                score = video.get('final_score', 0)
                duration = video.get('duration_seconds', 0) / 60
                logger.info(f"  {i}. Score: {score:.4f} - '{video.get('title', 'Desconhecido')}' ({duration:.1f} min)")
                
        elapsed = time.time() - start_time
        logger.info(f"\nBusca concluída em {elapsed:.2f} segundos. Retornando {len(processed_videos)} vídeos.")
        
        return {
            "videos": processed_videos,
            "total": len(processed_videos)
        }
        
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Erro durante a busca: {str(e)}")
        logger.error(traceback.format_exc())
        logger.error(f"Busca falhou após {elapsed:.2f} segundos")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/filters/")
async def get_filters() -> List[Dict[str, Any]]:
    """Get information about all available filters."""
    logger.info("Obtendo informações dos filtros")
    return filter_manager.get_filter_info()
from typing import Dict, Any
import re
import logging
from .base import BaseFilter

logger = logging.getLogger(__name__)

class DurationFilter(BaseFilter):
    """Filter that classifies videos by duration."""
    
    def __init__(self):
        logger.info("DurationFilter :: def __init__")
        super().__init__(
            name="Duração",
            description="Filtra por duração",
            default_enabled=True
        )
        
        self.duration_types = {
            "short": {"min": 0, "max": 4},
            "medium": {"min": 4, "max": 20},
            "long": {"min": 20, "max": float('inf')}
        }
    
    def _parse_duration(self, duration: str) -> float:
        logger.info("DurationFilter :: def _parse_duration")
        """Parse ISO 8601 duration to minutes."""
        try:
            # Remove PT prefix
            duration = duration.replace('PT', '')
            
            # Extract hours, minutes, seconds
            hours = re.search(r'(\d+)H', duration)
            minutes = re.search(r'(\d+)M', duration)
            seconds = re.search(r'(\d+)S', duration)
            
            total_minutes = 0
            if hours:
                total_minutes += int(hours.group(1)) * 60
            if minutes:
                total_minutes += int(minutes.group(1))
            if seconds:
                total_minutes += int(seconds.group(1)) / 60
                
            logger.debug(f"Parsed duration: {duration} -> {total_minutes} minutes")
            return total_minutes
            
        except Exception as e:
            logger.error(f"Error parsing duration {duration}: {str(e)}")
            return 0
    
    def process(self, video: Dict[str, Any]) -> float:
        logger.info("DurationFilter :: def process")
        """Process video duration and return a score."""
        try:
            if not self.validate_video(video):
                logger.warning(f"Invalid video data for DurationFilter")
                return 0
                
            self.logger.info(f"Processing duration for video: {video['title']}")
            
            # Get duration in minutes
            if 'duration_seconds' in video:
                duration = video['duration_seconds'] / 60  # Convert seconds to minutes
                self.logger.info(f"Using duration_seconds: {duration} minutes")
            else:
                duration = self._parse_duration(video['duration'])
            
            self.logger.info(f"Video duration in minutes: {duration}")
            
            # Get selected duration type from video context or use default
            # Special check: both duração_type and duration_type could be used
            if 'duração_type' in video:
                duration_type = video['duração_type']
                self.logger.info(f"Using duração_type from video: {duration_type}")
            elif 'duration_type' in video:
                duration_type = video['duration_type']
                self.logger.info(f"Using duration_type from video: {duration_type}")
            else:
                # Default to 'long' if no type specified
                duration_type = 'long'
                self.logger.info(f"No duration type specified, using default: {duration_type}")
            
            # Validar o tipo de duração
            if duration_type not in self.duration_types:
                self.logger.warning(f"Invalid duration type: {duration_type}, defaulting to 'long'")
                duration_type = 'long'
                
            # Get duration range
            duration_range = self.duration_types[duration_type]
            self.logger.info(f"Duration range: min={duration_range['min']}, max={duration_range['max']}")
            
            # Calculate score - CORRIGIDO
            # Verifica se a duração está dentro do intervalo especificado
            if duration_range['min'] <= duration <= duration_range['max']:
                score = 1.0
                self.logger.info(f"Video duration matches criteria ({duration_range['min']}-{duration_range['max']} min). Score: {score}")
            else:
                score = 0.0
                self.logger.info(f"Video duration does not match criteria ({duration_range['min']}-{duration_range['max']} min). Duration: {duration} min. Score: {score}")
                
            return score
            
        except Exception as e:
            logger.error(f"Error in DurationFilter.process: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return 0
    
    def get_filter_info(self) -> Dict[str, Any]:
        logger.info("DurationFilter :: def get_filter_info")
        """Get filter information."""
        return {
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
            "weight": self.weight,
            "type": "duration",
            "options": [
                {"value": "short", "label": "Menos de 4 minutos"},
                {"value": "medium", "label": "4 a 20 minutos"},
                {"value": "long", "label": "Mais de 20 minutos"}
            ]
        } 
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.requests import Request
from .core.config import get_settings
from .core.logging import setup_logging
from .api.endpoints import videos
from .filters import filter_manager
from .filters.duration import DurationFilter
from .filters.age_rating import AgeRatingFilter
from .filters.educational import EducationalFilter
from .filters.toxicity import ToxicityFilter
from .filters.language import LanguageFilter
from .filters.diversity import DiversityFilter
from .filters.interactivity import InteractivityFilter
from .filters.engagement import EngagementFilter
from .filters.sentiment import SentimentFilter
from .filters.sensitive import SensitiveFilter
import logging
import sys

# Configuração de logs para garantir que tudo seja exibido
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("app.log", mode="a", encoding="utf-8")
    ]
)

# Setup logging
logger = setup_logging()

# Get settings
settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION
)

logger.info("="*80)
logger.info("=== Inicializando Servidor ===")
logger.info("="*80)
logger.info(f"Versão: {settings.VERSION}")
logger.info(f"API Base URL: {settings.API_V1_STR}")

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Templates
templates = Jinja2Templates(directory="app/templates")

# Register API routes
app.include_router(
    videos.router,
    prefix=settings.API_V1_STR + "/videos",
    tags=["videos"]
)

# Log registered routes
logger.info("Rotas registradas:")
for route in app.routes:
    logger.info(f"- {route.path}")

# Register filters
logger.info("")
logger.info("=== Registrando filtros ===")
filter_manager.register_filter("Duração", DurationFilter())
filter_manager.register_filter("Faixa Etária", AgeRatingFilter())
filter_manager.register_filter("Educacional", EducationalFilter())
filter_manager.register_filter("Toxicidade", ToxicityFilter())
filter_manager.register_filter("Linguagem Imprópria", LanguageFilter())
filter_manager.register_filter("Diversidade", DiversityFilter())
filter_manager.register_filter("Interatividade", InteractivityFilter())
filter_manager.register_filter("Engajamento", EngagementFilter())
filter_manager.register_filter("Sentimento", SentimentFilter())
filter_manager.register_filter("Conteúdo Sensível", SensitiveFilter())

logger.info("")
logger.info("="*80)
logger.info("=== Servidor Inicializado ===")
logger.info("="*80)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the initial HTML page."""
    logger.info("Acessando página inicial")
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "filters": filter_manager.get_filter_info()
        }
    ) 
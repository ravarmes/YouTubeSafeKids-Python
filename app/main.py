from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.requests import Request
from .core.config import get_settings
from .core.logging import setup_logging
from .api.endpoints import videos
from .filters import (
    filter_manager,
    DurationFilter, EngagementFilter, SentimentFilter, LanguageFilter,
    ToxicityFilter, AgeRatingFilter, DiversityFilter, InteractivityFilter,
    EducationalFilter, SensitiveFilter,
)
import logging

# Setup logging (configuração centralizada em core/logging.py)
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

_FILTERS = {
    "Duração": DurationFilter(),
    "Engajamento": EngagementFilter(),
    "Sentimento": SentimentFilter(),
    "Linguagem Imprópria": LanguageFilter(),
    "Toxicidade": ToxicityFilter(),
    "Faixa Etária": AgeRatingFilter(),
    "Diversidade": DiversityFilter(),
    "Interatividade": InteractivityFilter(),
    "Tópicos Educacionais": EducationalFilter(),
    "Conteúdo Sensível": SensitiveFilter(),
}

for name, instance in _FILTERS.items():
    filter_manager.register_filter(name, instance)

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
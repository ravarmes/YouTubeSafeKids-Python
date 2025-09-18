import logging
import sys
import os
from pathlib import Path

def setup_logging():
    """
    Configure o sistema de logging para a aplicação.
    
    - Configura um logger raiz com nível DEBUG
    - Cria um handler para o console (stdout)
    - Configura um formato detalhado para as mensagens
    - Garante que a pasta de logs existe
    - Adiciona um handler para arquivo
    
    Returns:
        logging.Logger: O logger configurado
    """
    # Cria uma pasta para os logs
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Reseta quaisquer configurações anteriores
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Configuração básica do logger raiz
    root_logger.setLevel(logging.DEBUG)
    
    # Handler para console com formato detalhado
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)
    
    # Handler para arquivo
    log_file = logs_dir / "app.log"
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    root_logger.addHandler(file_handler)
    
    # Logger específico da aplicação
    logger = logging.getLogger("app")
    logger.info("Logging system initialized")
    
    return logger 
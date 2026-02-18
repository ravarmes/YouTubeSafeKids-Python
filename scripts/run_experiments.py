#!/usr/bin/env python3
"""
Script principal para executar experimentos automatizados com modelos BERTimbau.

Este script orquestra todo o pipeline de experimentação, desde o pré-processamento
dos dados até a avaliação e comparação dos modelos.

Uso:
    python run_experiments.py --config config/experiment_config.json
    python run_experiments.py --config config/experiment_config.json --models sentiment toxicity
    python run_experiments.py --quick-test
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess
import shutil
from datetime import datetime

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description='Executar experimentos automatizados')
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/experiment_config.json',
        help='Caminho para o arquivo de configuração'
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['sentiment', 'toxicity', 'language', 'educational'],
        help='Modelos específicos para treinar (padrão: todos)'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        help='Diretório com dados de treinamento'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Diretório de saída para resultados'
    )
    
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Executar teste rápido com configurações reduzidas'
    )
    
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Pular treinamento e usar modelos existentes'
    )
    
    parser.add_argument(
        '--skip-evaluation',
        action='store_true',
        help='Pular avaliação'
    )
    
    parser.add_argument(
        '--skip-comparison',
        action='store_true',
        help='Pular comparação de modelos'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Mostrar comandos sem executar'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Logging verboso'
    )
    
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Carrega configuração de experimento.
    
    Args:
        config_path: Caminho para o arquivo de configuração
        
    Returns:
        Dict: Configuração carregada
    """
    logger.info(f"Carregando configuração: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return config

def setup_experiment_directory(base_dir: str, experiment_name: str) -> str:
    """
    Configura diretório do experimento.
    
    Args:
        base_dir: Diretório base
        experiment_name: Nome do experimento
        
    Returns:
        str: Caminho do diretório do experimento
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    
    # Criar estrutura de diretórios
    directories = [
        'models',
        'logs',
        'results',
        'data',
        'plots',
        'reports'
    ]
    
    for directory in directories:
        (exp_dir / directory).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Diretório do experimento criado: {exp_dir}")
    
    return str(exp_dir)

def run_command(command: List[str], cwd: Optional[str] = None, dry_run: bool = False) -> bool:
    """
    Executa comando do sistema.
    
    Args:
        command: Comando a ser executado
        cwd: Diretório de trabalho
        dry_run: Se True, apenas mostra o comando
        
    Returns:
        bool: True se sucesso, False caso contrário
    """
    cmd_str = ' '.join(command)
    logger.info(f"Executando: {cmd_str}")
    
    if dry_run:
        logger.info("[DRY RUN] Comando não executado")
        return True
    
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True
        )
        
        if result.stdout:
            logger.info(f"STDOUT: {result.stdout}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Erro ao executar comando: {e}")
        if e.stdout:
            logger.error(f"STDOUT: {e.stdout}")
        if e.stderr:
            logger.error(f"STDERR: {e.stderr}")
        return False

def preprocess_data(config: Dict[str, Any], data_dir: str, exp_dir: str, 
                   model_types: List[str], dry_run: bool = False) -> Dict[str, str]:
    """
    Executa pré-processamento de dados.
    
    Args:
        config: Configuração do experimento
        data_dir: Diretório com dados brutos
        exp_dir: Diretório do experimento
        model_types: Tipos de modelos
        dry_run: Se True, apenas mostra comandos
        
    Returns:
        Dict: Mapeamento de tipo de modelo para arquivo de dados processados
    """
    logger.info("=== Iniciando Pré-processamento de Dados ===")
    
    preprocessing_config = config.get('data', {}).get('preprocessing', {})
    processed_files = {}
    
    for model_type in model_types:
        logger.info(f"Pré-processando dados para {model_type}")
        
        # Definir arquivos de entrada e saída
        input_file = Path(data_dir) / f"{model_type}_raw.csv"
        output_file = Path(exp_dir) / "data" / f"{model_type}_processed.csv"
        
        if not input_file.exists():
            logger.warning(f"Arquivo de dados não encontrado: {input_file}")
            continue
        
        # Construir comando de pré-processamento
        command = [
            'python', 'scripts/data_preprocessing.py',
            '--input', str(input_file),
            '--output', str(output_file),
            '--model', model_type
        ]
        
        # Adicionar parâmetros de configuração
        if preprocessing_config.get('clean_text', False):
            command.append('--clean-text')
        
        if preprocessing_config.get('remove_duplicates', False):
            command.append('--remove-duplicates')
        
        if preprocessing_config.get('balance_classes', False):
            command.append('--balance')
        
        command.extend([
            '--min-length', str(preprocessing_config.get('min_length', 10)),
            '--max-length', str(preprocessing_config.get('max_length', 512))
        ])
        
        # Dividir em train/test se configurado
        split_config = config.get('data', {}).get('split', {})
        if split_config.get('test_size', 0) > 0:
            command.extend([
                '--split',
                '--test-size', str(split_config.get('test_size', 0.2))
            ])
        
        # Executar pré-processamento
        if run_command(command, dry_run=dry_run):
            processed_files[model_type] = str(output_file)
        else:
            logger.error(f"Falha no pré-processamento para {model_type}")
    
    return processed_files

def train_models(config: Dict[str, Any], processed_files: Dict[str, str], 
                exp_dir: str, model_types: List[str], dry_run: bool = False) -> Dict[str, str]:
    """
    Treina modelos.
    
    Args:
        config: Configuração do experimento
        processed_files: Arquivos de dados processados
        exp_dir: Diretório do experimento
        model_types: Tipos de modelos
        dry_run: Se True, apenas mostra comandos
        
    Returns:
        Dict: Mapeamento de tipo de modelo para arquivo do modelo treinado
    """
    logger.info("=== Iniciando Treinamento de Modelos ===")
    
    trained_models = {}
    
    for model_type in model_types:
        if model_type not in processed_files:
            logger.warning(f"Dados processados não encontrados para {model_type}")
            continue
        
        logger.info(f"Treinando modelo {model_type}")
        
        # Configuração específica do modelo
        model_config = config.get('models', {}).get(model_type, {})
        training_config = config.get('training', {})
        
        # Definir arquivos
        train_file = processed_files[model_type].replace('.csv', '_train.csv')
        if not Path(train_file).exists():
            train_file = processed_files[model_type]  # Usar arquivo completo se não dividido
        
        output_dir = Path(exp_dir) / "models" / model_type
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Construir comando de treinamento
        command = [
            'python', 'scripts/train_model.py',
            '--model', model_type,
            '--train-data', train_file,
            '--output-dir', str(output_dir)
        ]
        
        # Adicionar parâmetros de configuração
        command.extend([
            '--epochs', str(model_config.get('epochs', 3)),
            '--batch-size', str(model_config.get('batch_size', 16)),
            '--learning-rate', str(model_config.get('learning_rate', 2e-5)),
            '--max-length', str(model_config.get('max_length', 512))
        ])
        
        # Parâmetros de treinamento
        if training_config.get('mixed_precision', False):
            command.append('--mixed-precision')
        
        if training_config.get('early_stopping', {}).get('enabled', False):
            command.append('--early-stopping')
        
        # Executar treinamento
        if run_command(command, dry_run=dry_run):
            model_path = output_dir / "best_model.pt"
            trained_models[model_type] = str(model_path)
        else:
            logger.error(f"Falha no treinamento para {model_type}")
    
    return trained_models

def evaluate_models(config: Dict[str, Any], trained_models: Dict[str, str], 
                   processed_files: Dict[str, str], exp_dir: str, 
                   model_types: List[str], dry_run: bool = False) -> Dict[str, str]:
    """
    Avalia modelos treinados.
    
    Args:
        config: Configuração do experimento
        trained_models: Modelos treinados
        processed_files: Arquivos de dados processados
        exp_dir: Diretório do experimento
        model_types: Tipos de modelos
        dry_run: Se True, apenas mostra comandos
        
    Returns:
        Dict: Mapeamento de tipo de modelo para arquivo de resultados
    """
    logger.info("=== Iniciando Avaliação de Modelos ===")
    
    evaluation_results = {}
    eval_config = config.get('evaluation', {})
    
    for model_type in model_types:
        if model_type not in trained_models:
            logger.warning(f"Modelo treinado não encontrado para {model_type}")
            continue
        
        logger.info(f"Avaliando modelo {model_type}")
        
        # Definir arquivos
        test_file = processed_files[model_type].replace('.csv', '_test.csv')
        if not Path(test_file).exists():
            # Usar uma porção dos dados processados para teste
            test_file = processed_files[model_type]
        
        model_path = trained_models[model_type]
        output_dir = Path(exp_dir) / "results" / model_type
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Construir comando de avaliação
        command = [
            'python', 'scripts/evaluate_model.py',
            '--model', model_type,
            '--test-data', test_file,
            '--model-path', model_path,
            '--output-dir', str(output_dir)
        ]
        
        # Adicionar parâmetros de configuração
        command.extend([
            '--batch-size', str(eval_config.get('batch_size', 32))
        ])
        
        if eval_config.get('save_predictions', False):
            command.append('--save-predictions')
        
        # Executar avaliação
        if run_command(command, dry_run=dry_run):
            results_file = output_dir / "evaluation_results.json"
            evaluation_results[model_type] = str(results_file)
        else:
            logger.error(f"Falha na avaliação para {model_type}")
    
    return evaluation_results

def compare_models(config: Dict[str, Any], trained_models: Dict[str, str], 
                  processed_files: Dict[str, str], exp_dir: str, 
                  model_types: List[str], dry_run: bool = False):
    """
    Compara modelos treinados.
    
    Args:
        config: Configuração do experimento
        trained_models: Modelos treinados
        processed_files: Arquivos de dados processados
        exp_dir: Diretório do experimento
        model_types: Tipos de modelos
        dry_run: Se True, apenas mostra comandos
    """
    logger.info("=== Iniciando Comparação de Modelos ===")
    
    # Agrupar modelos por tipo para comparação
    model_groups = {}
    for model_type in model_types:
        if model_type in trained_models:
            model_groups[model_type] = [trained_models[model_type]]
    
    for model_type, models in model_groups.items():
        if len(models) < 2:
            logger.info(f"Apenas um modelo para {model_type}, pulando comparação")
            continue
        
        logger.info(f"Comparando modelos {model_type}")
        
        # Definir arquivos
        test_file = processed_files[model_type].replace('.csv', '_test.csv')
        if not Path(test_file).exists():
            test_file = processed_files[model_type]
        
        output_prefix = Path(exp_dir) / "reports" / f"{model_type}_comparison"
        
        # Construir comando de comparação
        command = [
            'python', 'scripts/model_comparison.py',
            '--models'] + models + [
            '--test-data', test_file,
            '--model-type', model_type,
            '--output', str(output_prefix),
            '--save-predictions'
        ]
        
        # Executar comparação
        run_command(command, dry_run=dry_run)

def generate_final_report(config: Dict[str, Any], exp_dir: str, 
                         model_types: List[str], start_time: float):
    """
    Gera relatório final do experimento.
    
    Args:
        config: Configuração do experimento
        exp_dir: Diretório do experimento
        model_types: Tipos de modelos
        start_time: Tempo de início do experimento
    """
    logger.info("=== Gerando Relatório Final ===")
    
    end_time = time.time()
    duration = end_time - start_time
    
    report = {
        'experiment_info': {
            'name': config.get('experiment_name', 'Unknown'),
            'version': config.get('version', '1.0.0'),
            'start_time': datetime.fromtimestamp(start_time).isoformat(),
            'end_time': datetime.fromtimestamp(end_time).isoformat(),
            'duration_seconds': duration,
            'duration_formatted': f"{duration//3600:.0f}h {(duration%3600)//60:.0f}m {duration%60:.0f}s"
        },
        'models_trained': model_types,
        'configuration': config,
        'directory_structure': {
            'experiment_dir': exp_dir,
            'models': str(Path(exp_dir) / "models"),
            'results': str(Path(exp_dir) / "results"),
            'reports': str(Path(exp_dir) / "reports"),
            'logs': str(Path(exp_dir) / "logs")
        }
    }
    
    # Salvar relatório
    report_path = Path(exp_dir) / "final_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Criar resumo em texto
    summary_path = Path(exp_dir) / "experiment_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=== RESUMO DO EXPERIMENTO ===\n\n")
        f.write(f"Nome: {report['experiment_info']['name']}\n")
        f.write(f"Versão: {report['experiment_info']['version']}\n")
        f.write(f"Duração: {report['experiment_info']['duration_formatted']}\n")
        f.write(f"Modelos Treinados: {', '.join(model_types)}\n")
        f.write(f"Diretório: {exp_dir}\n\n")
        
        f.write("=== ARQUIVOS GERADOS ===\n")
        f.write(f"- Modelos: {report['directory_structure']['models']}\n")
        f.write(f"- Resultados: {report['directory_structure']['results']}\n")
        f.write(f"- Relatórios: {report['directory_structure']['reports']}\n")
        f.write(f"- Logs: {report['directory_structure']['logs']}\n")
    
    logger.info(f"Relatório final salvo em: {report_path}")
    logger.info(f"Resumo salvo em: {summary_path}")

def main():
    """Função principal do script de experimentos."""
    args = parse_arguments()
    start_time = time.time()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("=== Iniciando Experimentos Automatizados ===")
    
    try:
        # Carregar configuração
        config = load_config(args.config)
        
        # Aplicar configurações de teste rápido
        if args.quick_test:
            logger.info("Modo de teste rápido ativado")
            for model_type in config.get('models', {}):
                config['models'][model_type]['epochs'] = 1
                config['models'][model_type]['batch_size'] = 8
            config['data']['preprocessing']['balance_classes'] = False
        
        # Sobrescrever configurações com argumentos
        if args.data_dir:
            config['paths']['data_dir'] = args.data_dir
        if args.output_dir:
            config['paths']['results_dir'] = args.output_dir
        
        # Determinar modelos a treinar
        model_types = args.models or list(config.get('models', {}).keys())
        logger.info(f"Modelos a treinar: {model_types}")
        
        # Configurar diretório do experimento
        exp_dir = setup_experiment_directory(
            config['paths'].get('results_dir', 'results'),
            config.get('experiment_name', 'experiment')
        )
        
        # Salvar configuração usada
        config_path = Path(exp_dir) / "experiment_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # Pipeline de experimentação
        processed_files = {}
        trained_models = {}
        evaluation_results = {}
        
        # 1. Pré-processamento de dados
        data_dir = config['paths'].get('data_dir', 'data')
        if Path(data_dir).exists():
            processed_files = preprocess_data(
                config, data_dir, exp_dir, model_types, args.dry_run
            )
        else:
            logger.warning(f"Diretório de dados não encontrado: {data_dir}")
        
        # 2. Treinamento de modelos
        if not args.skip_training and processed_files:
            trained_models = train_models(
                config, processed_files, exp_dir, model_types, args.dry_run
            )
        
        # 3. Avaliação de modelos
        if not args.skip_evaluation and trained_models:
            evaluation_results = evaluate_models(
                config, trained_models, processed_files, exp_dir, model_types, args.dry_run
            )
        
        # 4. Comparação de modelos
        if not args.skip_comparison and trained_models:
            compare_models(
                config, trained_models, processed_files, exp_dir, model_types, args.dry_run
            )
        
        # 5. Relatório final
        if not args.dry_run:
            generate_final_report(config, exp_dir, model_types, start_time)
        
        logger.info("=== Experimentos Concluídos com Sucesso ===")
        logger.info(f"Resultados salvos em: {exp_dir}")
        
    except Exception as e:
        logger.error(f"Erro durante os experimentos: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
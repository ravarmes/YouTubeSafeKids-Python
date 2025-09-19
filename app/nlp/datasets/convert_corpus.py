#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para converter valores textuais para numéricos no corpus.csv
conforme o mapeamento definido em corpus.md
"""

import pandas as pd
import os
from pathlib import Path

def get_conversion_mappings():
    """
    Retorna os mapeamentos de conversão conforme definido em corpus.md
    """
    mappings = {
        'AS': {  # Análise de Sentimentos
            'Negativo': 0,
            'Neutro': 1,
            'Positivo': 2
        },
        'TOX': {  # Toxicidade
            'Não tóxico': 0,
            'Leve': 1,
            'Moderado': 2,
            'Severo': 3
        },
        'LI': {  # Linguagem Imprópria
            'Nenhuma': 0,
            'Leve': 1,
            'Severa': 2
        },
        'TE': {  # Tópicos Educacionais
            'Não educacional': 0,
            'Parcialmente educacional': 1,
            'Educacional': 2
        }
    }
    return mappings

def convert_corpus_values(csv_path):
    """
    Converte os valores textuais para numéricos no arquivo corpus.csv
    
    Args:
        csv_path (str): Caminho para o arquivo corpus.csv
    """
    print(f"Carregando arquivo: {csv_path}")
    
    # Carrega o CSV com separador de ponto e vírgula
    try:
        df = pd.read_csv(csv_path, sep=';', encoding='utf-8')
        print(f"Arquivo carregado com sucesso. Total de linhas: {len(df)}")
    except Exception as e:
        print(f"Erro ao carregar o arquivo: {e}")
        return False
    
    # Obtém os mapeamentos
    mappings = get_conversion_mappings()
    
    # Cria backup do arquivo original
    backup_path = csv_path.replace('.csv', '_backup.csv')
    df.to_csv(backup_path, sep=';', index=False, encoding='utf-8')
    print(f"Backup criado em: {backup_path}")
    
    # Aplica as conversões
    conversions_made = {}
    
    for column, mapping in mappings.items():
        if column in df.columns:
            print(f"\nConvertendo coluna {column}:")
            
            # Mostra valores únicos antes da conversão
            unique_values_before = df[column].unique()
            print(f"  Valores únicos encontrados: {unique_values_before}")
            
            # Aplica o mapeamento
            df[column] = df[column].map(mapping)
            
            # Verifica se há valores não mapeados (NaN)
            nan_count = df[column].isna().sum()
            if nan_count > 0:
                print(f"  ATENÇÃO: {nan_count} valores não foram mapeados na coluna {column}")
                unmapped_values = df[df[column].isna()][column].unique()
                print(f"  Valores não mapeados: {unmapped_values}")
            else:
                print(f"  ✓ Conversão concluída com sucesso")
            
            conversions_made[column] = {
                'original_values': unique_values_before.tolist(),
                'unmapped_count': nan_count
            }
        else:
            print(f"AVISO: Coluna {column} não encontrada no arquivo")
    
    # Salva o arquivo convertido
    try:
        df.to_csv(csv_path, sep=';', index=False, encoding='utf-8')
        print(f"\n✓ Arquivo convertido salvo em: {csv_path}")
        
        # Mostra estatísticas finais
        print("\n=== RESUMO DAS CONVERSÕES ===")
        for column, info in conversions_made.items():
            print(f"{column}: {len(info['original_values'])} valores únicos convertidos")
            if info['unmapped_count'] > 0:
                print(f"  ⚠️  {info['unmapped_count']} valores não mapeados")
        
        return True
        
    except Exception as e:
        print(f"Erro ao salvar o arquivo convertido: {e}")
        return False

def validate_conversions(csv_path):
    """
    Valida se as conversões foram aplicadas corretamente
    
    Args:
        csv_path (str): Caminho para o arquivo corpus.csv
    """
    print("\n=== VALIDAÇÃO DAS CONVERSÕES ===")
    
    try:
        df = pd.read_csv(csv_path, sep=';', encoding='utf-8')
        mappings = get_conversion_mappings()
        
        for column in ['AS', 'TOX', 'LI', 'TE']:
            if column in df.columns:
                unique_values = df[column].unique()
                expected_values = list(mappings[column].values())
                
                print(f"\n{column}:")
                print(f"  Valores encontrados: {sorted(unique_values)}")
                print(f"  Valores esperados: {sorted(expected_values)}")
                
                # Verifica se todos os valores estão no range esperado
                unexpected_values = [v for v in unique_values if v not in expected_values and not pd.isna(v)]
                if unexpected_values:
                    print(f"  ⚠️  Valores inesperados: {unexpected_values}")
                else:
                    print(f"  ✓ Todos os valores estão corretos")
        
        return True
        
    except Exception as e:
        print(f"Erro na validação: {e}")
        return False

def main():
    """
    Função principal
    """
    # Define o caminho do arquivo
    script_dir = Path(__file__).parent
    csv_path = script_dir / 'corpus.csv'
    
    if not csv_path.exists():
        print(f"Erro: Arquivo não encontrado em {csv_path}")
        return
    
    print("=== CONVERSÃO DE VALORES DO CORPUS ===")
    print(f"Arquivo: {csv_path}")
    
    # Executa a conversão
    success = convert_corpus_values(str(csv_path))
    
    if success:
        # Valida as conversões
        validate_conversions(str(csv_path))
        print("\n✓ Processo concluído com sucesso!")
    else:
        print("\n❌ Erro durante o processo de conversão")

if __name__ == "__main__":
    main()
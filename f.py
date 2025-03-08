#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FlexSim2 - Aplicativo de simulação com integração OpenAI
Arquivo: f.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import openai
import os
import json
import requests
from dotenv import load_dotenv
from PIL import Image
import nltk
from nltk.tokenize import word_tokenize
from datetime import datetime, timedelta

# Carregar variáveis de ambiente
load_dotenv()

# Configurar a API key da OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Configuração da página Streamlit
st.set_page_config(
    page_title="FlexSim2 - Simulador Inteligente",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Definição de estilos CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f7ff;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .result-area {
        background-color: #f5f5f5;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Funções auxiliares
def get_openai_response(prompt, model="gpt-4o", temperature=0.7, max_tokens=1000):
    """
    Obtém resposta da API da OpenAI.
    
    Args:
        prompt (str): Texto de entrada para o modelo.
        model (str): Modelo a ser usado (default: gpt-4o).
        temperature (float): Controle de aleatoriedade (0.0 a 1.0).
        max_tokens (int): Número máximo de tokens na resposta.
        
    Returns:
        str: Resposta gerada pelo modelo.
    """
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Você é um assistente de simulação especializado em flexsim, um software de simulação industrial."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erro na API: {str(e)}"

def process_simulation_data(data, parameters):
    """
    Processa dados de simulação com base nos parâmetros fornecidos.
    
    Args:
        data (pd.DataFrame): Dados de entrada para processamento.
        parameters (dict): Parâmetros de configuração.
        
    Returns:
        pd.DataFrame: Dados processados.
    """
    # Simulação de processamento de dados
    processed_data = data.copy()
    
    # Aplicar fator de escala baseado nos parâmetros
    if 'scale_factor' in parameters:
        processed_data['value'] = processed_data['value'] * parameters['scale_factor']
    
    # Aplicar filtros se necessário
    if 'min_threshold' in parameters:
        processed_data = processed_data[processed_data['value'] >= parameters['min_threshold']]
    
    return processed_data

def generate_plot(data, plot_type="line"):
    """
    Gera visualização dos dados processados.
    
    Args:
        data (pd.DataFrame): Dados para visualização.
        plot_type (str): Tipo de gráfico (line, bar, scatter).
        
    Returns:
        fig: Objeto de figura plotly.
    """
    if plot_type == "line":
        fig = px.line(data, x='date', y='value', title="Simulação FlexSim - Linha do Tempo")
    elif plot_type == "bar":
        fig = px.bar(data, x='date', y='value', title="Simulação FlexSim - Gráfico de Barras")
    elif plot_type == "scatter":
        fig = px.scatter(data, x='date', y='value', title="Simulação FlexSim - Gráfico de Dispersão")
    else:
        fig = px.line(data, x='date', y='value', title="Simulação FlexSim - Visualização")
    
    fig.update_layout(
        xaxis_title="Data",
        yaxis_title="Valor",
        template="plotly_white"
    )
    
    return fig

# Definição dos tipos de simulação disponíveis
simulation_types = {
    "Manufatura": {
        "description": "Simulação de linhas de produção e processos de manufatura",
        "icon": "🏭",
        "parameters": ["taxa_produção", "tempo_setup", "disponibilidade"]
    },
    "Logística": {
        "description": "Simulação de operações logísticas e cadeia de suprimentos",
        "icon": "🚚",
        "parameters": ["tempo_transporte", "capacidade_armazenamento", "demanda"]
    },
    "Saúde": {
        "description": "Simulação de fluxos de pacientes e operações hospitalares",
        "icon": "🏥",
        "parameters": ["tempo_atendimento", "número_médicos", "leitos_disponíveis"]
    },
    "Serviços": {
        "description": "Simulação de operações de serviços e atendimento ao cliente",
        "icon": "👨‍💼",
        "parameters": ["tempo_atendimento", "número_atendentes", "taxa_chegada"]
    },
    "Trein": {
        "description": "Simulação de operações ferroviárias e logística de trens",
        "icon": "🚂",
        "parameters": ["intervalo_trens", "capacidade_vagões", "velocidade_média"]
    }
}

# Função para gerar dados de exemplo para simulação
def generate_sample_data(simulation_type, days=30):
    """
    Gera dados de amostra para demonstração.
    
    Args:
        simulation_type (str): Tipo de simulação.
        days (int): Número de dias para gerar dados.
        
    Returns:
        pd.DataFrame: DataFrame com dados de amostra.
    """
    np.random.seed(42)  # Para reprodutibilidade
    
    start_date = datetime.now() - timedelta(days=days)
    dates = [start_date + timedelta(days=i) for i in range(days)]
    
    if simulation_type == "Manufatura":
        base_value = 100
        noise = np.random.normal(0, 10, days)
        trend = np.linspace(0, 20, days)
        values = base_value + noise + trend
    elif simulation_type == "Logística":
        base_value = 80
        noise = np.random.normal(0, 15, days)
        seasonal = 20 * np.sin(np.linspace(0, 2*np.pi, days))
        values = base_value + noise + seasonal
    elif simulation_type == "Saúde":
        base_value = 50
        noise = np.random.normal(0, 5, days)
        weekend_effect = np.array([5 if i % 7 >= 5 else 0 for i in range(days)])
        values = base_value + noise - weekend_effect
    elif simulation_type == "Serviços":
        base_value = 70
        noise = np.random.normal(0, 8, days)
        trend = np.linspace(0, -15, days)  # Tendência decrescente
        values = base_value + noise + trend
    elif simulation_type == "Trein":
        base_value = 120
        noise = np.random.normal(0, 12, days)
        pattern = 25 * np.sin(np.linspace(0, 4*np.pi, days))
        values = base_value + noise + pattern
    else:
        # Caso padrão
        base_value = 90
        noise = np.random.normal(0, 10, days)
        values = base_value + noise
    
    # Garantir que não há valores negativos
    values = np.maximum(values, 0)

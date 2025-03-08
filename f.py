import streamlit as st
import openai
import os
from dotenv import load_dotenv

# Carregar variáveis de ambiente (para a API key)
load_dotenv()

# Configuração da página
st.set_page_config(
    page_title="ChatBot IA",
    page_icon="🤖",
    layout="wide"
)

# Configurar API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Título principal
st.title("ChatBot IA")
st.markdown("---")

# Inicializar histórico de chat se não existir
if "messages" not in st.session_state:
    st.session_state.messages = []

# Função para obter resposta da OpenAI
def get_ai_response(prompt):
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Você é um assistente útil e amigável."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erro: {str(e)}"

# Exibir mensagens anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Campo de entrada do usuário
prompt = st.chat_input("Digite sua mensagem aqui...")

# Processar input do usuário
if prompt:
    # Adicionar mensagem do usuário ao histórico
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Exibir mensagem do usuário
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Obter resposta do assistente
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            response = get_ai_response(prompt)
            st.markdown(response)
    
    # Adicionar resposta do assistente ao histórico
    st.session_state.messages.append({"role": "assistant", "content": response})

# Botão para limpar o histórico
if st.button("Limpar Conversa"):
    st.session_state.messages = []
    st.experimental_rerun()

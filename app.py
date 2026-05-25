from flask import Flask, request, jsonify
import requests
import os
import json
import re
from google import genai
from google.genai import types

app = Flask(__name__)

# Chaves configuradas diretamente no código
CHAVE_GEMINI = "AIzaSyClWzUlQlPCSGD9xepoydGZMSTV83R9Teg"
SUPABASE_URL = "https://rmlyubbislrtgwdshmpd.supabase.co"
API_KEY_SUPABASE = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJtbHl1YmJpc2xydGd3ZHNobXBkIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3ODk5MTM1NSwiZXhwIjoyMDk0NTY3MzU1fQ.A4x1u2ZnxfPwiRinWf7mfUGcIOVWy2Q9U_KTRVtbZyI"

ai_client = genai.Client(api_key=CHAVE_GEMINI)

HEADERS_SUPABASE = {
    "apikey": API_KEY_SUPABASE,
    "Authorization": f"Bearer {API_KEY_SUPABASE}",
    "Content-Type": "application/json"
}

def recuperar_contexto_supabase(query_usuario):
    """Busca fatos ou checagens no banco de dados local primeiro."""
    try:
        endpoint = f"{SUPABASE_URL}/rest/v1/fact_checks?query=ilike.*{query_usuario}*&limit=3"
        resposta = requests.get(endpoint, headers=HEADERS_SUPABASE)
        
        if resposta.status_code == 200:
            registros = resposta.json()
            contexto = ""
            for reg in registros:
                contexto += f"Fato cadastrado localmente: '{reg.get('query')}' tem o veredito confirmado como {reg.get('verdict')}.\n"
            return contexto if contexto else "Nenhum histórico interno encontrado para este termo."
    except Exception as e:
        print(f"Erro ao recuperar contexto: {e}")
    return "Erro ao recuperar base de conhecimento."


def salvar_no_supabase(query, verdict, confidence):
    """Salva a predição gerada pela IA na tabela do Supabase respeitando as restrições de coluna."""
    try:
        payload = {
            "query": query,
            "claim": query,
            "verdict": verdict,
            "confidence": str(confidence), 
            # Alterado de 'rag_gemini_web' para 'ml' para não quebrar a restrição (constraint) do banco
            "source": "rag_gemini_web", 
            "status": "predicted"
        }
        resposta = requests.post(
            f"{SUPABASE_URL}/rest/v1/fact_checks",
            headers=HEADERS_SUPABASE,
            json=payload
        )
        
        if resposta.status_code != 201 and resposta.status_code != 200:
            print(f"❌ Erro retornado pelo Supabase: {resposta.text}")
            
        resposta.raise_for_status() 
        print(f"-> Salvo no Supabase com sucesso! Status: {resposta.status_code}")
    except Exception as e:
        print(f"Erro ao salvar resultado no Supabase: {e}")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'erro': 'JSON inválido. Use a chave "query".'}), 400
            
        noticia_usuario = data.get('query')

        # 1. RETRIEVAL LOCAL: Busca primeiro no seu banco do Supabase
        contexto_recuperado = recuperar_contexto_supabase(noticia_usuario)

        # 2. PROMPT DO SISTEMA: Instrução explícita para gerar a estrutura JSON manualmente
        prompt_sistema = f"""
        Você é um assistente especialista em Fact-Checking (checagem de fatos).
        Sua missão é analisar se a frase enviada pelo usuário é verdadeira (true), falsa (false) ou incerta (uncertain).

        Regras de Busca:
        1. Verifique o 'Contexto Local' fornecido abaixo. Se ele contiver a resposta, use-o.
        2. Se o 'Contexto Local' disser que não encontrou histórico, utilize a ferramenta Google Search integrada para pesquisar o fato na internet em fontes de notícias confiáveis e agências de checagem.

        Contexto Local Atual:
        {contexto_recuperado}

        ATENÇÃO: Você DEVE responder OBRIGATORIAMENTE no formato JSON puro, sem blocos de código markdown (como ```json ... 
```), contendo exatamente esta estrutura:
        {{"verdict": "true", "confidence": 1.0}}
        Onde 'verdict' pode ser 'true', 'false' ou 'uncertain', e 'confidence' é um float entre 0.0 e 1.0.
        """

        # 3. GENERATION COM GOOGLE SEARCH
        resposta_ia = ai_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=noticia_usuario,
            config=types.GenerateContentConfig(
                system_instruction=prompt_sistema,
                temperature=0.1,
                tools=[{"google_search": {}}] 
            ),
        )

        texto_resposta = resposta_ia.text.strip()
        
        # Limpeza de segurança caso a IA insira blocos de código markdown
        if texto_resposta.startswith("```"):
            texto_resposta = re.sub(r'^```[a-zA-Z]*\n|```$', '', texto_resposta, flags=re.MULTILINE).strip()

        # Converte o texto JSON puro para dicionário Python
        dados_resposta = json.loads(texto_resposta)
        
        # 4. SALVAMENTO AUTOMÁTICO: Envia os dados para o Supabase
        salvar_no_supabase(
            query=noticia_usuario, 
            verdict=dados_resposta.get("verdict"), 
            confidence=dados_resposta.get("confidence", 1.0)
        )

        # Adiciona a tag identificadora para o retorno da API
        dados_resposta["source"] = "rag_gemini_web"

        return jsonify(dados_resposta)

    except json.JSONDecodeError:
        return jsonify({'erro': 'A IA não retornou um JSON válido.', 'resposta_bruta': resposta_ia.text}), 500
    except Exception as e:
        return jsonify({'erro': f'Erro interno no RAG: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
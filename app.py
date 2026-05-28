import os
import requests
import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import joblib
from sentence_transformers import SentenceTransformer

# Carrega as credenciais do ambiente (.env)
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
API_KEY_SUPABASE = os.getenv("API_KEY_SUPABASE")

app = Flask(__name__)
CORS(app) # Permite conexões de outras portas (como o seu Front-end)

# ==========================================
# 🧠 CARREGAMENTO DOS COMPONENTES LOCAIS (OFFLINE)
# ==========================================
print("[INFO] Carregando extrator de embeddings MiniLM local...")
encoder_local = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

CAMINHO_MODELO = "modelo_fact_checker.pkl"

if os.path.exists(CAMINHO_MODELO):
    print(f"[INFO] Carregando cérebro estatístico '{CAMINHO_MODELO}'...")
    modelo_logistica = joblib.load(CAMINHO_MODELO)
    print("[INFO] Microserviço de ML pronto para receber requisições!")
else:
    print(f"❌ ERRO CRÍTICO: O arquivo '{CAMINHO_MODELO}' não foi encontrado!")
    print("Execute o script 'retreinar.py' primeiro para gerar o modelo.")
    exit(1)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        dados_requisicao = request.get_json()
        if not dados_requisicao:
            return jsonify({"error": "Payload JSON ausente ou inválido"}), 400

        # Flexibilidade para ler qualquer mapeamento de texto enviado pelo cliente
        texto_noticia = dados_requisicao.get("query") or dados_requisicao.get("claim") or dados_requisicao.get("text")

        if not texto_noticia:
            return jsonify({"error": "Texto da notícia não fornecido no campo 'query' ou 'claim'"}), 400

        # 1. Extração de Features Semânticas (384 dimensões) - 100% OFFLINE
        print(f"\n[ML LOCAL] Vetorizando entrada: '{texto_noticia}'")
        embedding_numerico = encoder_local.encode(texto_noticia).tolist()

        # 2. Predição via Regressão Logística Local
        # Enviamos como uma matriz de uma única linha: [embedding_numerico]
        probabilidades = modelo_logistica.predict_proba([embedding_numerico])[0]
        confianca_calculada = float(np.max(probabilidades))
        classe_predita = int(np.argmax(probabilidades))

        # Determina o veredito baseado na maior probabilidade matemática
        veredito_final = "true" if classe_predita == 1 else "false"

        # 🚀 FILTRO DE CONFIABILIDADE (THRESHOLD): 
        # Se a certeza estatística for menor que 75%, o sistema recua e crava 'uncertain'.
        # Isso blinda o front contra erros lógicos (como as negações políticas).
        if confianca_calculada < 0.75:
            print(f"[ML LOCAL] Confiança de {confianca_calculada*100:.2f}% abaixo do limite. Classificado como UNCERTAIN.")
            veredito_final = "uncertain"
        else:
            print(f"[ML LOCAL] Decisão cravada como '{veredito_final}' com {confianca_calculada*100:.2f}% de certeza.")

        # 3. SALVAMENTO DA INFERÊNCIA NO HISTÓRICO (SUPABASE)
        # Formata o array numérico no formato string vetorial exigido pelo pgvector: '[x, y, z...]'
        embedding_string = "[" + ",".join(map(str, embedding_numerico)) + "]"
        
        # 🚀 CORREÇÃO: Preenchemos query E claim para satisfazer a restrição Not-Null do banco
        payload_supabase = {
            "query": texto_noticia,  
            "claim": texto_noticia,
            "verdict": veredito_final,
            "confidence": round(confianca_calculada, 2),
            "embedding": embedding_string,
            "source": "ml",
            "status": "predicted",
        }

        print("[API] Enviando dados para gravação no Supabase...")
        resposta_supabase = requests.post(
            f"{SUPABASE_URL}/rest/v1/fact_checks",
            headers={
                "apikey": API_KEY_SUPABASE,
                "Authorization": f"Bearer {API_KEY_SUPABASE}",
                "Content-Type": "application/json",
                "Prefer": "return=representation"
            },
            json=payload_supabase
        )

        # Monitoramento ativo do status da requisição ao banco
        if resposta_supabase.status_code not in [200, 201, 204]:
            print(f"⚠️ Alerta: Inferência calculada, mas rejeitada pelo Supabase.")
            print(f"   -> Código HTTP: {resposta_supabase.status_code}")
            print(f"   -> Motivo técnico: {resposta_supabase.text}")
        else:
            print("✅ Histórico de inferência persistido com sucesso no banco!")

        # 4. Resposta entregue para o cliente (Ex: teste_api.py ou Front)
        return jsonify({
            "verdict": veredito_final,
            "confidence": round(confianca_calculada, 2),
            "source": "offline_ml_local"
        }), 200

    except Exception as e:
        print(f"❌ Erro interno na execução do Flask: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Roda o microserviço local na porta padrão 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
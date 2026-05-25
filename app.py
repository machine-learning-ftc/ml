from flask import Flask, request, jsonify
import joblib
import requests
import threading

app = Flask(__name__)

print("Carregando modelo SVM e vetorizador...")
modelo = joblib.load('modelo_eleicoes.pkl')
vetorizador = joblib.load('vetorizador.pkl')
print("IA Pronta para uso!")

LIMIAR_CONFIANCA = 0.60
RETREINAR_A_CADA = 5  # retreina a cada 5 inferências
contador_inferencias = 0

SUPABASE_URL = "https://rmlyubbislrtgwdshmpd.supabase.co"
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJtbHl1YmJpc2xydGd3ZHNobXBkIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3ODk5MTM1NSwiZXhwIjoyMDk0NTY3MzU1fQ.A4x1u2ZnxfPwiRinWf7mfUGcIOVWy2Q9U_KTRVtbZyI"
HEADERS_SUPABASE = {
    "apikey": API_KEY,
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def salvar_no_supabase(query, verdict, confidence):
    try:
        payload = {
            "query": query,
            "claim": query,
            "verdict": verdict,
            "confidence": confidence,
            "source": "ml",
            "status": "predicted"
        }
        resposta = requests.post(
            f"{SUPABASE_URL}/rest/v1/fact_checks",
            headers=HEADERS_SUPABASE,
            json=payload
        )
        print(f"Supabase status: {resposta.status_code}")
        print(f"Supabase resposta: {resposta.text}")
    except Exception as e:
        print(f"Erro ao salvar inferência: {e}")

def retreinar_em_background():
    """Executa o retreino sem travar a API"""
    import subprocess
    print("Retreinando modelo em background...")
    subprocess.run(["python", "retreinar.py"])
    
    # Recarrega o modelo após retreino
    global modelo, vetorizador
    modelo = joblib.load('modelo_eleicoes.pkl')
    vetorizador = joblib.load('vetorizador.pkl')
    print("Modelo atualizado!")

@app.route('/predict', methods=['POST'])
def predict():
    global contador_inferencias

    try:
        data = request.get_json()
        noticia = data.get('query')

        if not noticia:
            return jsonify({'erro': 'JSON inválido. Use a chave "query".'}), 400

        texto_vetorizado = vetorizador.transform([noticia])
        probabilidades = modelo.predict_proba(texto_vetorizado)[0]
        predicao = modelo.predict(texto_vetorizado)[0]
        confianca = max(probabilidades)

        if confianca < LIMIAR_CONFIANCA:
            veredito_final = "uncertain"
            confianca_exibida = 0.0
        elif predicao == 1:
            veredito_final = "true"
            confianca_exibida = round(float(confianca), 4)
        elif predicao == 0:
            veredito_final = "false"
            confianca_exibida = round(float(confianca), 4)
        else:
            veredito_final = "uncertain"
            confianca_exibida = round(float(confianca), 4)

        # Salva a inferência no Supabase
        salvar_no_supabase(noticia, veredito_final, confianca_exibida)

        # Incrementa contador e retreina se necessário
        contador_inferencias += 1
        if contador_inferencias >= RETREINAR_A_CADA:
            contador_inferencias = 0
            thread = threading.Thread(target=retreinar_em_background)
            thread.daemon = True
            thread.start()

        return jsonify({
            "verdict": veredito_final,
            "confidence": confianca_exibida,
            "source": "machine_learning"
        })

    except Exception as e:
        return jsonify({'erro': f'Erro interno no servidor: {str(e)}'}), 500
    
@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        data = request.get_json()
        query = data.get('query')
        verdict_correto = data.get('verdict')

        if not query or not verdict_correto:
            return jsonify({'erro': 'JSON inválido. Use as chaves "query" e "verdict".'}), 400

        if verdict_correto not in ['true', 'false', 'uncertain']:
            return jsonify({'erro': 'Verdict inválido. Use "true", "false" ou "uncertain".'}), 400

        busca = requests.get(
            f"{SUPABASE_URL}/rest/v1/fact_checks?query=eq.{query}&order=id.desc&limit=1",
            headers=HEADERS_SUPABASE
        )
        registros = busca.json()

        if not registros:
            resposta = requests.post(
                f"{SUPABASE_URL}/rest/v1/fact_checks",
                headers=HEADERS_SUPABASE,
                json={
                    "query": query,
                    "claim": query,
                    "verdict": verdict_correto,
                    "source": "ml",
                    "status": "predicted",
                    "confidence": 1.0,
                }
            )
            print(f"Feedback insert status: {resposta.status_code}")
            print(f"Feedback insert resposta: {resposta.text}")
            acao = "inserido"
        else:
            registro_id = registros[0]['id']
            resposta = requests.patch(
                f"{SUPABASE_URL}/rest/v1/fact_checks?id=eq.{registro_id}",
                headers=HEADERS_SUPABASE,
                json={"verdict": verdict_correto}
            )
            print(f"Feedback update status: {resposta.status_code}")
            print(f"Feedback update resposta: {resposta.text}")
            acao = "corrigido"

        return jsonify({
            "status": "ok",
            "acao": acao,
            "query": query,
            "verdict_corrigido": verdict_correto
        })

    except Exception as e:
        return jsonify({'erro': f'Erro interno no servidor: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
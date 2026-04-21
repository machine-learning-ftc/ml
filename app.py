from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# 1. Carregar o .pkl do ML
print("Carregando modelo SVM e vetorizador...")
modelo = joblib.load('modelo_eleicoes.pkl')
vetorizador = joblib.load('vetorizador.pkl')
print("IA Pronta para uso!")

# 2. A Rota de previsão
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Pega o JSON do Back
        data = request.get_json()
        
        #  1: Usar a chave "query" em vez de "texto"
        noticia = data.get('query')
        
        if not noticia:
            return jsonify({'erro': 'JSON invalido. Use a chave "query".'}), 400

        # Passa pelo processamento NLP
        texto_vetorizado = vetorizador.transform([noticia])
        
        # Faz a previsão matemática
        resultado = modelo.predict(texto_vetorizado)[0]
        
        # Probabilidade decimal 
        probabilidades = modelo.predict_proba(texto_vetorizado)[0]
        confianca_decimal = max(probabilidades)

        # 2: Formatação 
        if resultado == 1:
            veredito_final = "true"
        else:
            veredito_final = "false"

        # 3: Montando o JSON 
        resposta = {
            "verdict": veredito_final,
            "confidence": round(confianca_decimal, 4), # Arredonda para 4 casas decimais
            "source": "machine_learning"
        }

        return jsonify(resposta)

    except Exception as e:
        return jsonify({'erro': f'Erro interno no servidor: {str(e)}'}), 500

# 3. Ligar o servidor na porta 5001
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
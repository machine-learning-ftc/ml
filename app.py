from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# 1. Carregar os arquivos .pkl da IA
print("Carregando modelo SVM e vetorizador...")
modelo = joblib.load('modelo_eleicoes.pkl')
vetorizador = joblib.load('vetorizador.pkl')
print("IA Pronta para uso!")

# DEFINIÇÃO DO LIMIAR DE CONFIANÇA (THRESHOLD) PARA AS UNCERTAIN
# 0.5 é um valor inicial. 
# Quanto maior este número, mais conservadora e "exigente" a IA fica antes de dar um veredito.
LIMIAR_CONFIANCA = 0.5

# 2. Rota de previsão
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Pega o JSON enviado pelo Back-end
        data = request.get_json()
        
        noticia = data.get('query')
        
        if not noticia:
            return jsonify({'erro': 'JSON inválido. Use a chave "query".'}), 400

        # Passa o texto recebido pelo processamento NLP (Vetorizador)
        texto_vetorizado = vetorizador.transform([noticia])
        
        # Calcula a distância matemática até a fronteira de decisão (Certeza da IA)
        # abs() garante que o valor seja sempre positivo
        distancia_decisao = modelo.decision_function(texto_vetorizado)[0]
        confianca_absoluta = abs(distancia_decisao)
        
        # Faz a previsão bruta da máquina (1 para verdadeiro, 0 para falso)
        predicao_bruta = modelo.predict(texto_vetorizado)[0]

        # 3. A TRAVA DE SEGURANÇA (Verificação do Limiar)
        if confianca_absoluta < LIMIAR_CONFIANCA:
            # Se a IA ficou muito em cima do muro, ela assume que não sabe
            veredito_final = "uncertain"
            confianca_exibida = 0.0
        else:
            # Se ela passou do limiar de certeza, valida o resultado bruto
            if predicao_bruta == 1:
                veredito_final = "true"
            else:
                veredito_final = "false"
            confianca_exibida = round(float(confianca_absoluta), 4)

        # 4. Montando o JSON 
        resposta = {
            "verdict": veredito_final,
            "confidence": confianca_exibida, 
            "source": "machine_learning"
        }

        return jsonify(resposta)

    except Exception as e:
        return jsonify({'erro': f'Erro interno no servidor: {str(e)}'}), 500

# 3. Ligar o servidor na porta 5001
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
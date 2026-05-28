import requests

url = 'http://127.0.0.1:5000/predict'

#EXEMPLO
dados = {
    "query": "Flávio bolsonaro será candidato a presidente em 2026"
}

print(f" Enviando dados para o microserviço: '{dados['query']}'\n")

resposta = requests.post(url, json=dados)

print(" Resposta recebida do servidor:")
print(resposta.json())
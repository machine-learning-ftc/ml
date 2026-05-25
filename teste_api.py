import requests

url = 'http://127.0.0.1:5001/predict'

#EXEMPLO
dados = {
    "query": "Jair bolsonaro foi eleito em 2018"
}

print(f" Enviando dados para o microserviço: '{dados['query']}'\n")

resposta = requests.post(url, json=dados)

print(" Resposta recebida do servidor:")
print(resposta.json())
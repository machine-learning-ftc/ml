import requests
import json


SUPABASE_URL = "https://rmlyubbislrtgwdshmpd.supabase.co"
API_KEY = "COLE_A_CHAVE_AQUI"


endpoint = f"{SUPABASE_URL}/rest/v1/{NOME_TABELA}?select=*"
headers = {
    "apikey": API_KEY,
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

print(f" Conectando ao banco de dados em: {SUPABASE_URL}...")

try:
    
    resposta = requests.get(endpoint, headers=headers)
    
    if resposta.status_code == 200:
        dados = resposta.json()
        print(f" Sucesso! Foram encontradas {len(dados)} validações no banco.")
        
       
        if len(dados) > 0:
            print("\nEstrutura da primeira linha recebida:")
            print(json.dumps(dados[0], indent=2, ensure_ascii=False))
        else:
            print(" A tabela existe e conectou, mas ainda está vazia (0 registros).")
    else:
        print(f" Erro ao puxar dados: {resposta.status_code}")
        print(resposta.text)

except Exception as e:
    print(f" Erro de conexão: {str(e)}")
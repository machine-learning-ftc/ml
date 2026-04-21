/////////Tecnologias e Bibliotecas Utilizadas/////////
O microserviço foi construído com foco em leveza e alta performance matemática, utilizando a seguinte stack:

Python 3.x: Linguagem base do projeto.

Flask: Framework web minimalista utilizado para encapsular o modelo e expor a rota da API (porta 5001).

Scikit-Learn: Biblioteca principal de Machine Learning.

Modelo Algorítmico: LinearSVC (Support Vector Machine) configurado com pesos balanceados (class_weight='balanced') para traçar fronteiras de decisão otimizadas.

Vetorização (NLP): TfidfVectorizer utilizando análise de N-grams (blocos de palavras) e preservação de caracteres numéricos para não perder o contexto de datas e mandatos.

Pandas: Manipulação e limpeza do dataset (estruturação do CSV).

Feedparser: Extração automatizada de manchetes reais via RSS (ex: G1) para a etapa de treino do modelo.

Joblib: Serialização (exportação) dos ficheiros binários .pkl contendo o cérebro treinado e o dicionário de palavras.

////// Como Usar e Executar Localmente //////
1. Preparação do Ambiente
Recomenda-se a utilização de um ambiente virtual (venv). Na raiz do diretório do microserviço, instala todas as dependências necessárias através do ficheiro de configuração:

Bash
pip install -r requirements.txt
2. Iniciar o Servidor Flask
Para carregar o modelo treinado (.pkl) para a memória e iniciar a escuta de requisições, executa:

Bash
python app.py
A API ficará ativa e a escutar requisições em http://127.0.0.1:5001.

3. Como Consumir a API (Exemplo de Requisição)
O microserviço espera uma requisição POST na rota /predict. O corpo (body) deve ser um objeto JSON contendo a chave query.

Exemplo utilizando curl:

Bash
curl -X POST http://127.0.0.1:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"query": "As eleições ocorrem no primeiro domingo de outubro"}'
Resposta de Sucesso Esperada:
O servidor devolverá um JSON formatado especificamente para ser consumido de forma assíncrona pelo Back-end principal:

JSON
{
  "verdict": "true",
  "confidence": 0.9412,
  "source": "machine_learning"
}
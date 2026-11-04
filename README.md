# Aplicativo de análise de sentimentos

Este projeto gera publicações sobre ocorrências de roubo de veículos, analisa o sentimento dos textos, atualiza uma tabela Q de aprendizado por reforço e cria um mapa das ocorrências e um gráfico de sentimentos.

## Requisitos

- Python instalado.
- Dependências: `pandas`, `openpyxl`, `folium`, `matplotlib`, `nltk` e `openai`.
- Uma chave `OPENROUTER_API_KEY` ou `OPENAI_API_KEY` para gerar mensagens usando o modelo configurado no OpenRouter. Sem uma chave, o script usa mensagens locais.

Instale as dependências com:

```bash
python -m pip install pandas openpyxl folium matplotlib nltk openai
```

O script baixa o léxico VADER do NLTK na primeira execução, se necessário.

## Dados de entrada

Coloque a planilha `VeiculosSubtraidos_2024.xlsx` em `data/raw/`. O script lê a aba `AMOSTRAGEM` e usa campos como `BAIRRO`, `DATA_OCORRENCIA_BO`, `HORA_OCORRENCIA`, `DESCR_TIPO_VEICULO`, `LATITUDE` e `LONGITUDE`.

## Execução

Na raiz do projeto, execute:

```bash
python rl_sentimental_app.py
```

As mensagens são geradas em inglês. O sentimento é calculado com VADER. A tabela Q escolhe entre três prompts para a mensagem sobre policiamento e é atualizada com o score de sentimento dessa mensagem.

## Arquivos gerados

- `data/processed/mensagens_geradas.txt`: publicações e resultados das análises de sentimento.
- `output/grafico_sentimentos.html`: gráfico com os scores de sentimento.
- `output/mapa_crimes.html`: mapa interativo das ocorrências.
- `output/q_table.json`: tabela Q persistida entre execuções.

Se `output/q_table.json` ainda não existir, o script tenta carregar a tabela legada `q_table.json` da raiz do projeto. As atualizações seguintes são gravadas em `output/q_table.json`.

Quando a API está configurada, o script aguarda três segundos entre as ocorrências para espaçar as chamadas.

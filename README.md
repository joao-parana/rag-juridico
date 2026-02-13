# CONECTA+ | Agentes de IA para DEVs

## Projeto de RAG Jurídico
Este projeto tem como objetivo desenvolver um sistema de RAG (Retrieval-Augmented Generation) para o domínio jurídico (CDC - Código de Defesa do Consumidor e LGPD - Lei Geral de Proteção de Dados), utilizando a biblioteca LangChain, a API da OpenAI e o banco vetorial Chroma. O sistema é capaz de recuperar informações relevantes de uma base de dados jurídica e gerar respostas precisas para perguntas dos usuários.

## Quadro de tarefas
O direcionamento das tarefas a serem desenvolvidas estão neste [quadro do Trello](https://trello.com/b/hLEcKGPE/agentes-para-devs-sprint-1).

## Bibliotecas necessárias
```bash
pip install langchain langchain-openai langchain-community langchain-chroma pypdf python-dotenv
```

## Estrutura do projeto
- `bd.py`: arquivo responsável pela criação e gerenciamento da base vetorial utilizando a biblioteca Chroma, incluindo a função de indexação dos documentos jurídicos.
- `rag.py`: arquivo principal contendo a implementação do sistema de RAG, incluindo a configuração do modelo de linguagem, a base vetorial e as funções de recuperação e geração de respostas.
- `app.py`: arquivo contendo a implementação de uma interface simples para interação com o sistema de RAG, permitindo que os usuários façam perguntas e recebam respostas baseadas nos documentos jurídicos indexados.
- `dados/`: pasta contendo os arquivos PDF do CDC e da LGPD, que serão processados e indexados na base vetorial.
- `.env.exemplo`: arquivo de exemplo para configuração das variáveis de ambiente, como a chave da API da OpenAI e o caminho dos dados.
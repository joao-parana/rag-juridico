from typing import Any
import dotenv
import bd

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate


config = dotenv.dotenv_values()

banco_vetorial = bd.carrega_banco_vetorial()
retriever = banco_vetorial.as_retriever(search_type='similarity', search_kwargs={'k': 5})

llm = ChatOpenAI(model=config['LLM_MODEL'], openai_api_key=config['OPENAI_KEY'])
llm_com_rag = RetrievalQA.from_chain_type(
    llm=llm, 
    retriever=retriever, 
    return_source_documents=True,
    chain_type='stuff'
)


def extrai_fonte(documento: Document) -> str:
    fonte = documento.metadata.get('fonte', 'desconhecida')
    pagina = documento.metadata.get('page', 'desconhecida')

    return f'Fonte: {fonte} - Página: {pagina}'


def executa_prompt(prompt: str) -> dict[str, Any]:
    prompt_final = f'''
Você é um assistente jurídico especializado em responder perguntas sobre o Código de Defesa do Consumidor (CDC) e a Lei Geral de Proteção de Dados (LGPD).

Utilize o contexto fornecido para responder à pergunta.
Se a pergunta não estiver relacionada ao CDC ou à LGPD, responda exatamente esta frase: "Desculpe, só posso responder perguntas sobre o CDC e a LGPD."

Pergunta: {prompt}
'''

    completion = llm_com_rag.invoke({"query": prompt_final})
    resposta = {
        'resultado': completion['result'],
        'fontes': []
    }
    
    if resposta['resultado'].strip() != 'Desculpe, só posso responder perguntas sobre o CDC e a LGPD.':
        resposta['fontes'] = [extrai_fonte(doc) for doc in completion['source_documents']]

    return resposta

####
# RANKING MANUAL (Reranking)
#
# 1. Recuperação inicial: a base vetorial é consultada para recuperar um conjunto de documentos similares ao prompt original do usuário
#    (essa etapa é rápida e tem o objetivo de reduzir o número de documentos a serem avaliados).
# 2. Reranking: a IA é invocada para avaliar a relevância de cada documento recuperado em relação à pergunta do usuário.
# 3. Por fim, os documentos são vinculados ao seu ranking (tupla: documento, score) e ordenados do mais relevante para o menos relevante.
####
def rankeia_documentos(prompt_original):
    
    # O mesmo prompt para todos os documentos.
    prompt_de_rankeamento = PromptTemplate(
        input_variables=["pergunta", "trecho"],
        template='''
Você é um especialista no Código de Defesa do Consumidor (CDC) e na Lei Geral de Proteção de Dados (LGPD).

Avalie a relevância de um trecho de documento para responder à pergunta. 
Atribua uma nota entre 0 e 10, onde 0 significa "não é relevante" e 10 significa "extremamente relevante".

Responda apenas com um número de 0 a 10.

Pergunta: {pergunta}
Trecho: {trecho}
'''
    )

    # Consulta a base vetorial
    documentos_iniciais = banco_vetorial.similarity_search(prompt_original, k=15)
    
    rankeados = []
    for doc in documentos_iniciais:

        ### Perceba que estamos invocando a IA generativa SEM o retriever da base vetorial (variável llm_com_rag). 
        ## Ou seja, a IA generativa não tem acesso a outros documentos além do trecho que estamos avaliando.
        resposta = llm.invoke(prompt_de_rankeamento.format(pergunta=prompt_original, trecho=doc.page_content)).content.strip()

        try:
            score = float(resposta)
        except ValueError:
            score = 0

        rankeados.append((doc, score))

    # Rankeia e mapeia para um array somente com os documentos.
    rankeados.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in rankeados]


####
# RERANKING MANUAL
# 
# Neste cenário, somente o Top 4 documentos mais relevantes (após o processo de rankeamento) são utilizados para compor o contexto que será fornecido 
# à IA generativa para responder à pergunta do usuário.
####
def executa_prompt_reranking(prompt: str) -> dict[str, Any]:
    documentos_rankeados = rankeia_documentos(prompt)
    top_4_documentos = documentos_rankeados[:4]

    contexto_consolidado = "\n\n".join([doc.page_content for doc in top_4_documentos])
    prompt_final = f'''
Você é um assistente jurídico especializado em responder perguntas sobre o Código de Defesa do Consumidor (CDC) e a Lei Geral de Proteção de Dados (LGPD).

Utilize o contexto fornecido para responder à pergunta.
Se a pergunta não estiver relacionada ao CDC ou à LGPD, responda exatamente esta frase: "Desculpe, só posso responder perguntas sobre o CDC e a LGPD."

Pergunta: {prompt}
Contexto: {contexto_consolidado}
'''
    
    # Perceba que com o Reranking manual, a IA generativa é invocada diretamente (variável llm) e não mais através do chain de RAG (variável llm_com_rag).
    completion = llm.invoke(prompt_final)
    
    # A estrutura da resposta é diferente do método com RAG, pois não temos mais acesso aos "source_documents" (documentos de origem) que o chain de RAG fornecia.
    resposta = {
        'resultado': completion.content,
        'fontes': []
    }

    if resposta['resultado'].strip() != 'Desculpe, só posso responder perguntas sobre o CDC e a LGPD.':
        resposta['fontes'] = [extrai_fonte(doc) for doc in top_4_documentos]

    return resposta
from typing import Any
from itertools import chain
import dotenv

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda, RunnableParallel

import bd
config = dotenv.dotenv_values()


####
# FUNÇÕES AUXILIARES
####

def extrai_fonte(documento: Document) -> str:
    fonte = documento.metadata.get('fonte', 'desconhecida')
    pagina = documento.metadata.get('page', 'desconhecida')

    return f'Fonte: {fonte} - Página: {pagina}'


def monta_resposta_com_fontes(completion: str, documentos: list[Document]) -> dict[str, Any]:
    resposta = {
        'resultado': completion.strip(),
        'fontes': []
    }

    if resposta['resultado'] != 'Desculpe, só posso responder perguntas sobre o CDC e a LGPD.':
        resposta['fontes'] = [extrai_fonte(doc) for doc in documentos]

    return resposta


def gera_contexto_de_documentos(documentos: list[Document]) -> str:
    return "\n\n".join([doc.page_content for doc in documentos])


####
# Setup de RAG e LLM
####

banco_vetorial = bd.carrega_banco_vetorial()
retriever = banco_vetorial.as_retriever(search_type='similarity', search_kwargs={'k': 5})

llm = ChatOpenAI(model=config['LLM_MODEL'], openai_api_key=config['OPENAI_API_KEY'])
rag_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Você é um assistente jurídico especializado em responder perguntas sobre o "
        "Código de Defesa do Consumidor (CDC) e a Lei Geral de Proteção de Dados (LGPD). "
        "Use o contexto para responder. Se a pergunta não estiver relacionada ao CDC ou à LGPD, responda exatamente: "
        "Desculpe, só posso responder perguntas sobre o CDC e a LGPD.",
    ),
    (
        "human", 
        "Pergunta:\n{pergunta}\nContexto:\n{contexto}"
    )
])

monta_contexto = RunnablePassthrough.assign(contexto=lambda payload: gera_contexto_de_documentos(payload['documentos']))
invoca_llm = RunnablePassthrough.assign(resposta=RunnablePassthrough() | rag_prompt | llm | StrOutputParser())
monta_resultado = RunnableLambda(lambda payload: monta_resposta_com_fontes(payload['resposta'], payload['documentos']))


def retriever_padrao_strategy() -> RunnableParallel:
    return RunnableParallel(pergunta=RunnablePassthrough(), documentos=retriever)

####
# IMPLEMENTA TÉCNICA DE CONSULTA REWRITE-RETRIEVE-READ
####

def rewrite_retrieve_read_strategy() -> Runnable:
    prompt_rewrite = PromptTemplate(
        input_variables=["pergunta"],
        template='''
Você é um especialista no Código de Defesa do Consumidor (CDC) e na Lei Geral de Proteção de Dados (LGPD).
Sua tarefa é reescrever a pergunta do usuário para torná-la mais clara e específica, facilitando a recuperação de informações relevantes sobre o CDC ou a LGPD.
Se a pergunta já for clara e específica, mantenha seu conteúdo, mas reescreva-a de forma mais formal e objetiva.
Se a pergunta não estiver relacionada ao CDC ou à LGPD, reescreva-a de forma a indicar que o assistente só pode responder perguntas sobre esses temas.

**Retorne somente a pergunta reescrita, sem explicações adicionais.**
Reescreva a seguinte pergunta:
{pergunta}
'''
    )

    return (
        (lambda pergunta: {"pergunta": pergunta})
        | prompt_rewrite
        | llm
        | StrOutputParser()
        | RunnableParallel(
            pergunta=RunnablePassthrough(), 
            documentos=retriever
        )
    )
    


# ####
# # IMPLEMENTA TÉCNICA MULTI QUERY RETRIEVER
# ####
def elimina_duplicatas(documentos: list[Document]) -> list[Document]:
    vistos = set()
    resultado = []
    for doc in documentos:
        if doc.page_content not in vistos:
            resultado.append(doc)
            vistos.add(doc.page_content)

    return resultado

def multi_query_retriever_strategy() -> Runnable:
    prompt_multi_query = PromptTemplate(
        input_variables=["pergunta"],
        template='''
Você é um especialista no Código de Defesa do Consumidor (CDC) e na Lei Geral de Proteção de Dados (LGPD).
Sua tarefa é gerar 3 variações da pergunta do usuário, mantendo o mesmo significado, mas utilizando palavras e estruturas diferentes. 
O objetivo é aumentar a diversidade de documentos recuperados, melhorando as chances de encontrar informações relevantes sobre o CDC ou a LGPD.
Se a pergunta não estiver relacionada ao CDC ou à LGPD, reescreva-a de forma a indicar que o assistente só pode responder perguntas sobre esses temas.

**Retorne somente as 3 variações da pergunta, separadas por linha, sem explicações adicionais.**
Gere variações para a seguinte pergunta:
{pergunta}
'''
    )

    return (
        (lambda pergunta: {"pergunta": pergunta})
        | RunnablePassthrough.assign(novas_perguntas=prompt_multi_query | llm | StrOutputParser() | (lambda resposta: resposta.split('\n')))
        | RunnableParallel(
            pergunta=lambda payload: payload['pergunta'], 
            # documentos=lambda payload: [doc for doc in retriever.invoke(pergunta) for pergunta in payload['novas_perguntas']]
            documentos=lambda payload: elimina_duplicatas(chain.from_iterable([retriever.invoke(pergunta) for pergunta in payload['novas_perguntas']]))
        )
    )


def executa_prompt(prompt: str, query_strategy: callable = retriever_padrao_strategy) -> dict[str, Any]:
    pesquisa_documentos = query_strategy()

    llm_com_rag = (
        pesquisa_documentos # {pergunta: str, documentos: list[Document]}
        | monta_contexto # {pergunta: str, documentos: list[Document], contexto: str}
        | invoca_llm # {pergunta: str, documentos: list[Document], contexto: str, resposta: str}
        | monta_resultado # {resultado: str, fontes: list[str]}
    )
    
    return llm_com_rag.invoke(prompt)


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

    documentos_iniciais = banco_vetorial.similarity_search(prompt_original, k=15)
    msgs_rankeamento = [prompt_de_rankeamento.format(pergunta=prompt_original, trecho=doc.page_content) for doc in documentos_iniciais]
    
    respostas = llm.batch(msgs_rankeamento)
    rankeados = [(doc, float(resposta.content.strip())) for doc, resposta in zip(documentos_iniciais, respostas)]

    # Ordena do mais relevante para o menos em uma lista de documentos
    rankeados.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in rankeados]


####
# RERANKING MANUAL
# 
# Neste cenário, somente o Top 4 documentos mais relevantes (após o processo de rankeamento) são utilizados para compor o contexto que será fornecido 
# à IA generativa para responder à pergunta do usuário.
####
def executa_prompt_reranking(prompt: str) -> dict[str, Any]:
    prompt_final = PromptTemplate(
        input_variables=["prompt", "contexto"],
        template='''
Você é um assistente jurídico especializado em responder perguntas sobre o Código de Defesa do Consumidor (CDC) e a Lei Geral de Proteção de Dados (LGPD).

Utilize o contexto fornecido para responder à pergunta.
Se a pergunta não estiver relacionada ao CDC ou à LGPD, responda exatamente esta frase: "Desculpe, só posso responder perguntas sobre o CDC e a LGPD."

Pergunta: {prompt}
Contexto: {contexto}
'''
    )

    cadeia = (
        RunnableParallel(prompt=RunnablePassthrough(), documentos=rankeia_documentos) 
        | (lambda payload: {'prompt': payload["prompt"], 'documentos': payload["documentos"][:4]})
        | RunnablePassthrough.assign(contexto=lambda payload: gera_contexto_de_documentos(payload["documentos"]))
        | RunnablePassthrough.assign(resposta=RunnablePassthrough() | prompt_final | llm | StrOutputParser())
        | monta_resposta
    )

    return cadeia.invoke(prompt)
    
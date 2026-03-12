from typing import Any
import dotenv
import bd

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel


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


def extrai_fonte(documento: Document) -> str:
    fonte = documento.metadata.get('fonte', 'desconhecida')
    pagina = documento.metadata.get('page', 'desconhecida')

    return f'Fonte: {fonte} - Página: {pagina}'


config = dotenv.dotenv_values()

banco_vetorial = bd.carrega_banco_vetorial()
retriever = banco_vetorial.as_retriever(search_type='similarity', search_kwargs={'k': 5})

if bd.use_ollama:
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model='llama3.2')
else:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model=config['LLM_MODEL'], openai_api_key=config['OPENAI_KEY'])
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

pesquisa_documentos = RunnableParallel(pergunta=RunnablePassthrough(), documentos=retriever)
monta_contexto = RunnablePassthrough.assign(contexto=lambda payload: gera_contexto_de_documentos(payload['documentos']))
executa_prompt = RunnablePassthrough.assign(resposta=RunnablePassthrough() | rag_prompt | llm | StrOutputParser())
monta_resposta = RunnableLambda(lambda payload: monta_resposta_com_fontes(payload['resposta'], payload['documentos']))

llm_com_rag = (
    pesquisa_documentos # {pergunta: str, documentos: list[Document]}
    | monta_contexto # {pergunta: str, documentos: list[Document], contexto: str}
    | executa_prompt # {pergunta: str, documentos: list[Document], contexto: str, resposta: str}
    | monta_resposta # {resultado: str, fontes: list[str]}
)


def executa_prompt(prompt: str) -> dict[str, Any]:
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
    

    

    
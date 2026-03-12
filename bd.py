import dotenv

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter


config = dotenv.dotenv_values()
db_dir = Path(config['CHROMA_DB_PATH'])

use_ollama = config.get('USE_OLLAMA', 'False').strip().lower() == 'true'

if use_ollama:
    from langchain_ollama import OllamaEmbeddings
    embedding = OllamaEmbeddings(model='nomic-embed-text')
    collection_name = 'documentos_juridicos_ollama'
else:
    from langchain_openai import OpenAIEmbeddings
    embedding = OpenAIEmbeddings(model=config['EMBEDDINGS_MODEL'], openai_api_key=config['OPENAI_API_KEY'])
    collection_name = 'documentos_juridicos'

def le_pdf(caminho: str) -> list[Document]:
    loader = PyPDFLoader(caminho)
    return loader.load()


def configura_metadado(documento: Document, fonte: str) -> Document:
    documento.metadata['fonte'] = fonte
    return documento


def quebra_por_tamanho(documentos: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documentos)


def quebra_por_paragrafos(documentos: list[Document]) -> list[Document]:
    splitter = CharacterTextSplitter(separator='\n\n', chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documentos)


def carrega_documentos() -> dict[str, list[Document]]:
    documentos_cdc = [configura_metadado(doc, 'cdc') for doc in le_pdf(config['CDC_PATH'])]
    documentos_lgpd = [configura_metadado(doc, 'lgpd') for doc in le_pdf(config['LGPD_PATH'])]

    total_cdc = len(documentos_cdc)
    total_lgpd = len(documentos_lgpd)
    total_de_documentos = total_cdc + total_lgpd
    print(f'Documentos CDC: {total_cdc}. Documentos LGPD: {total_lgpd}. Total de documentos: {total_de_documentos}.')

    return {
        'cdc': documentos_cdc,
        'lgpd': documentos_lgpd
    }


def cria_chunks(documentos: dict[str, list[Document]]) -> dict[str, list[Document]]:
    chunks_cdc = quebra_por_tamanho(documentos['cdc'])
    chunks_lgpd = quebra_por_tamanho(documentos['lgpd'])

    total_chunks_cdc = len(chunks_cdc)
    total_chunks_lgpd = len(chunks_lgpd)
    total_de_chunks = total_chunks_cdc + total_chunks_lgpd

    print(f'Chunks CDC: {total_chunks_cdc}. Chunks LGPD: {total_chunks_lgpd}. Total de chunks: {total_de_chunks}.')
    
    return {
        'cdc': chunks_cdc,
        'lgpd': chunks_lgpd
    }


def carrega_banco_vetorial() -> Chroma:
    if db_dir.exists():
        print('Banco de dados vetorial já existe. Carregando...\n')
        return Chroma(embedding_function=embedding, persist_directory=str(db_dir), collection_name=collection_name)

    documentos = carrega_documentos()

    chunks = cria_chunks(documentos)
    todos_os_chunks = chunks['cdc'] + chunks['lgpd']

    return Chroma.from_documents(todos_os_chunks, embedding, persist_directory=str(db_dir), collection_name=collection_name)

if __name__ == "__main__":
    banco = carrega_banco_vetorial()

    documentos_recuperados = banco.similarity_search('Em que casos o consentimento é obrigatório?', k=5)
    for i, doc in enumerate(documentos_recuperados, start=1):
        print(f'Documento {i}:')
        print(f'Conteúdo: {doc.page_content}')
        print(f'Metadados: {doc.metadata["fonte"]}')
        print('---')
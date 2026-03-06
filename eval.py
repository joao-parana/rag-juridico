import os
import json
from dotenv import load_dotenv

from langsmith import Client
from langsmith.evaluation import evaluate
from langchain_core.prompts import ChatPromptTemplate

from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT

import rag


load_dotenv()
langsmith_client = Client()

def garantir_dataset():
    """
    Cria o dataset se não existir e adiciona exemplos.
    """
    dataset_name = os.environ['RAG_JURIDICO_DATASET']

    datasets = list(langsmith_client.list_datasets(dataset_name=dataset_name))
    if datasets:
        print(f"Dataset '{dataset_name}' já existe.")
        return

    dataset = langsmith_client.create_dataset(
        dataset_name=dataset_name,
        description="Perguntas e respostas esperadas para avaliar o assistente jurídico sobre CDC e LGPD."
    )
    print(f"Dataset criado: {dataset.name}")

    with open('dados/base-eval.json', 'r') as f:
        exemplos = json.load(f)
    

    langsmith_client.create_examples(dataset_id=dataset.id, examples=exemplos)
    print("Exemplos adicionados ao dataset.")


def target(inputs: dict) -> dict:
    """
    Função alvo avaliada pelo LangSmith.
    Recebe os inputs de uma linha do dataset e devolve o output da aplicação.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Responda à seguinte pergunta precisamente."),
        ("user", "{pergunta}"),
    ])

    resposta = rag.executa_prompt(inputs["question"])

    return { 'answer': resposta['resultado'] }


def correctness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    evaluator = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        model="openai:gpt-5-mini",
        feedback_key="correctness",
    )

    return evaluator(
        inputs=inputs,
        outputs=outputs,
        reference_outputs=reference_outputs
    )


if __name__ == "__main__":
    garantir_dataset()
    # print(CORRECTNESS_PROMPT)
    
    resultados_do_experimento = langsmith_client.evaluate(
        target,
        data=os.environ['RAG_JURIDICO_DATASET'],
        evaluators=[
            correctness_evaluator,
        ],
        experiment_prefix="eval1-rag-juridico",
        max_concurrency=2,
    )

    print(resultados_do_experimento)
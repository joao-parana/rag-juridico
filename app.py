from dotenv import load_dotenv
load_dotenv()

import rag

def efetua_pergunta():
    print('---\n')
    return input('# PERGUNTA (digite "sair" para encerrar) \n')


def imprime_fontes(resposta: dict[str, list[str]]):
    print('## FONTES:')

    for fonte in resposta['fontes']:
        print(f'  - {fonte}')

def inicia_chat():
    print('### BEM-VINDO AO ASSISTENTE JURÍDICO! FAÇA SUAS PERGUNTAS SOBRE O CDC E A LGPD. ###')
    
    prompt = efetua_pergunta()
    
    while prompt.strip().lower() != 'sair':
        # resposta = rag.executa_prompt(prompt)
        resposta = rag.executa_prompt_reranking(prompt)

        print(f'\n# RESPOSTA\n{resposta["resultado"]}\n')
        if resposta['fontes']:
            imprime_fontes(resposta)

        prompt = efetua_pergunta()

    print('### OBRIGADO POR USAR O ASSISTENTE JURÍDICO! ATÉ LOGO! ###')


if __name__ == "__main__":
    inicia_chat()
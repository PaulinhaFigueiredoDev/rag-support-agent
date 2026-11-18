# Este arquivo inicia o Loja Assist: valida a configuração do RAG e conduz
# o atendimento interativo no terminal, enviando perguntas ao agente.

from loja_exemplo_assistant.agent import run_turn
from loja_exemplo_assistant.config import get_rag_top_k
from loja_exemplo_assistant.session import Session


def main() -> None:
    # Valida a configuração antes de iniciar a conversa.
    try:
        get_rag_top_k()
    except ValueError as exc:
        print(f"Configuração inválida: {exc}")
        return

    # Mantém o contexto da conversa entre as perguntas desta execução.
    session = Session()
    print("Loja Exemplo Assist (Ollama). Digite 'sair' para encerrar.")

    # Repete o atendimento até receber um comando de saída ou uma interrupção.
    while True:
        try:
            question = input("Cliente: ").strip()
        except (EOFError, KeyboardInterrupt):
            # Trata fim de entrada (EOF) e Ctrl+C como encerramento normal.
            print("\nAtendimento encerrado.")
            break

        # Aceita comandos de saída em português ou inglês, sem diferenciar maiúsculas.
        if question.lower() in {"sair", "exit"}:
            print("Atendimento encerrado.")
            break

        # Ignora entradas vazias; perguntas preenchidas seguem para o agente.
        if question:
            answer = run_turn(session, question)
            print("Agente:", answer)


if __name__ == "__main__":
    main()

from loja_exemplo_assistant.agent import run_turn
from loja_exemplo_assistant.session import Session


def main() -> None:
    session = Session()
    print("Loja Exemplo Assist (Ollama). Digite 'sair' para encerrar.")
    while True:
        try:
            question = input("Cliente: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAtendimento encerrado.")
            break
        if question.lower() in {"sair", "exit"}:
            print("Atendimento encerrado.")
            break
        if question:
            answer = run_turn(session, question)
            print("Agente:", answer)


if __name__ == "__main__":
    main()

from emporio_assistant.agent import run_turn
from emporio_assistant.session import Session


def main() -> None:
    session = Session()
    print("Empório Assist (Ollama). Digite 'sair' para encerrar.")
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

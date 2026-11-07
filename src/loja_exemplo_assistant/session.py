"""Estado de uma conversa, independente do provedor de modelo."""

from dataclasses import dataclass, field

from loja_exemplo_assistant.prompts import SYSTEM_PROMPT


@dataclass
class Session:
    messages: list[dict] = field(default_factory=lambda: [{"role": "system", "content": SYSTEM_PROMPT}])
    pending_return_question: str | None = None

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

    @classmethod
    def from_messages(cls, messages: list[dict]) -> "Session":
        """Adaptador para chamadas antigas que passam somente o histórico."""
        from loja_exemplo_assistant.policy import PURCHASE_QUESTION, is_return_intent

        pending = None
        if messages and messages[-1].get("role") == "assistant" and messages[-1].get("content") == PURCHASE_QUESTION:
            pending = next(
                (item["content"] for item in reversed(messages[:-1])
                 if item.get("role") == "user" and is_return_intent(item.get("content", ""))),
                None,
            )
        return cls(messages=messages, pending_return_question=pending)

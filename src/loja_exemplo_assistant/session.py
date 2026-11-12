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

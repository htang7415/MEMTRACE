"""Actor model abstraction."""


class ActorModel:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        del max_tokens
        raise NotImplementedError


class StubActorModel(ActorModel):
    def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        del max_tokens
        return f"[stubbed model output from {self.model_name}] {prompt[:80]}"

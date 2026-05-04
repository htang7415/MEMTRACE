"""MLX actor runner."""

from memtrace.config import TEMPERATURE, TOP_P
from memtrace.models.actor import ActorModel, StubActorModel
from memtrace.models.profile_runner import ProfileActorModel


class MLXActorModel(ActorModel):
    def __init__(self, model_name: str) -> None:
        super().__init__(model_name)
        try:
            from mlx_lm import generate, load
            from mlx_lm.sample_utils import make_sampler
        except ImportError as exc:
            raise RuntimeError("MLX actor inference requires the `mlx-lm` package") from exc

        self._generate = generate
        self._make_sampler = make_sampler
        self._model, self._tokenizer = load(model_name)

    def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        sampler = self._make_sampler(temp=TEMPERATURE, top_p=TOP_P)
        output = self._generate(
            self._model,
            self._tokenizer,
            prompt=_format_prompt(self._tokenizer, prompt),
            sampler=sampler,
            max_tokens=max_tokens,
            verbose=False,
        )
        return _strip_prompt_continuation(output).strip()


def load_actor(model_name: str, backend: str = "mlx") -> ActorModel:
    if backend == "stub":
        return StubActorModel(model_name=model_name)
    if backend == "profile":
        return ProfileActorModel(model_name=model_name)
    if backend == "mlx":
        return MLXActorModel(model_name=model_name)
    raise ValueError(f"Unknown actor backend: {backend}")


def _format_prompt(tokenizer, prompt: str):
    if not getattr(tokenizer, "has_chat_template", False):
        return prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _strip_prompt_continuation(output: str) -> str:
    text = output
    for marker in ("<|endoftext|>", "<|im_end|>", "\nHuman:", "\nUser:"):
        if marker in text:
            text = text.split(marker, 1)[0]
    return text

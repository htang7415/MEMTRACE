"""MLX runner placeholder."""

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
            prompt=prompt,
            sampler=sampler,
            max_tokens=max_tokens,
            verbose=False,
        )
        return output.strip()


def load_actor(model_name: str, backend: str = "mlx") -> ActorModel:
    if backend == "stub":
        return StubActorModel(model_name=model_name)
    if backend == "profile":
        return ProfileActorModel(model_name=model_name)
    if backend == "mlx":
        return MLXActorModel(model_name=model_name)
    raise ValueError(f"Unknown actor backend: {backend}")

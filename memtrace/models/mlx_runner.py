"""MLX runner placeholder."""

from memtrace.models.actor import ActorModel


def load_actor(model_name: str) -> ActorModel:
    return ActorModel(model_name=model_name)


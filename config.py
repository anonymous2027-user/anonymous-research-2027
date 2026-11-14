"""Portable experiment settings. All paths are supplied by the caller."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model_id: str = 'Qwen/Qwen2-Audio-7B-Instruct'
    revision: str = '0a095220c30b7b31434169c3086508ef3ea5bf0a'
    seed: int = 42
    epochs: int = 3
    segments: int = 5
    batch: int = 21
    learning_rate: float = 3e-6
    warmup_epochs: float = 0.1
    weight_decay: float = 0.001
    max_grad_norm: float = 0.5
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    proximal_mu: float = 0.01
    windows_per_person: int = 200
    audio_limit_seconds: int = 30

    @classmethod
    def read(cls, path=None):
        result = cls(**json.loads(Path(path).read_text())) if path else cls()
        if result.epochs != 3 or result.segments != 5 or result.windows_per_person != 200:
            raise ValueError('This release fixes three epochs, five segments and 200 windows/person')
        if result.batch <= 0 or result.learning_rate <= 0 or result.proximal_mu < 0:
            raise ValueError('Invalid optimization settings')
        return result


def write_settings(path):
    Path(path).write_text(json.dumps(asdict(Settings()), indent=2) + '\n')

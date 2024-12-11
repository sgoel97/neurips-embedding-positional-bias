import pandas as pd
from pathlib import Path
from itertools import product
from utils.string_utils import truncate
from pydantic import BaseModel, model_validator
from typing import Literal


class NeedleConfig(BaseModel):
    corpus_path: Path | str
    mode: Literal["insert", "remove"]
    size: float
    posn: float
    corpus: str = "lorem ipsum"
    name: str = "needle"

    @model_validator(mode="after")
    def validate(self):
        assert self.mode in ["insert", "remove"], "Mode must be 'insert' or 'remove'."
        assert 0 <= self.posn <= 1, "Percent location must be between 0 and 1 (inclusive)."

        if self.mode == "insert":
            assert 0 <= self.size, "Needle size must be greater than or equal to 0."
        else:
            assert 0 <= self.size <= 1, "Needle size must be between 0 and 1 (inclusive)."

        self.corpus_path = Path(self.corpus_path)
        assert self.corpus_path.exists(), "Needle corpus file does not exist."

        self.corpus = self.corpus_path.read_text("utf-8")
        self.name = f"needle_{self.mode}_{self.posn}pos_{self.size}sz_{self.corpus_path.stem}"

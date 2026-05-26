from __future__ import annotations

from pathlib import Path
from typing import Union

import yaml

from .schema import Task


def load_task(path: Union[str, Path]) -> Task:
    """Parse a YAML task file into a :class:`Task`.

    Validation is delegated to the Pydantic model. Raises ``FileNotFoundError``
    if the path is missing and ``pydantic.ValidationError`` for schema issues.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Task file not found: {p}")
    with p.open("r") as f:
        data = yaml.safe_load(f)
    task = Task.model_validate(data)
    task._source_path = p
    return task

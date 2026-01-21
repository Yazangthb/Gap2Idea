from dataclasses import dataclass
from pathlib import Path

@dataclass
class Paths:
    root: Path
    data: Path
    artifacts: Path
    pdfs: Path
    texts: Path

def get_paths(root: str | Path = ".") -> Paths:
    root = Path(root).resolve()
    data = root / "data"
    artifacts = root / "artifacts"
    pdfs = data / "pdfs"
    texts = data / "texts"
    return Paths(root=root, data=data, artifacts=artifacts, pdfs=pdfs, texts=texts)
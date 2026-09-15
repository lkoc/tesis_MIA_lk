"""Resolve archived artifact locations after copying the complete workspace."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def artifact_path(value):
    text=str(value).replace('\\','/')
    relative=text.split('/Benchmarks/',1)[1] if '/Benchmarks/' in text else (text[len('Benchmarks/'):] if text.startswith('Benchmarks/') else None)
    if relative is not None:
        current=(ROOT/'Benchmarks'/relative).resolve()
        if not current.is_relative_to(ROOT/'Benchmarks'):raise ValueError('Artifact escapes the benchmark directory')
        if current.exists():return current
    original=Path(value)
    if original.exists():return original
    raise FileNotFoundError(f'Artifact unavailable in the current checkout or recorded location: {value}')

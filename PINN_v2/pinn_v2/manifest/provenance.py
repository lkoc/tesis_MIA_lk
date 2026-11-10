"""Source hashing and archiving; numpy-free so the WSL runner can import it."""
from pathlib import Path
import hashlib

PACKAGE=Path(__file__).resolve().parents[1]


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def package_sources():
    return sorted(p for p in PACKAGE.rglob('*.py') if '__pycache__' not in p.parts)


def archive_sources(output):
    """Copy every package module to output/source under its digest; return relative path to digest."""
    archive=Path(output)/'source';archive.mkdir(parents=True,exist_ok=True);hashes={}
    for path in package_sources():
        data=path.read_bytes();digest=sha256_bytes(data)
        hashes[path.relative_to(PACKAGE.parent).as_posix()]=digest
        target=archive/(digest+'_'+path.name)
        if not target.exists():target.write_bytes(data)
    return hashes
from pathlib import Path

from auditoria_config import AuditConfig, prepare_document


def test_prepare_document_uses_generic_word_limits(tmp_path: Path):
    source = tmp_path / "document.txt"
    source.write_text("alpha beta gamma delta epsilon", encoding="utf-8")
    config = AuditConfig(source, tmp_path / "output", chunk_mode="words", max_words=2)
    document, chunks = prepare_document(config)
    assert document.text.split() == "alpha beta gamma delta epsilon".split()
    assert [len(chunk.text.split()) for chunk in chunks] == [2, 2, 1]
    assert " ".join(chunk.text for chunk in chunks).split() == document.text.split()
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


def test_dry_run_executes_without_network(tmp_path, monkeypatch, capsys):
    import auditar_plan
    from auditoria_config import run_audit

    source = tmp_path / "human.txt"
    source.write_text("A human document with sufficient words and context. " * 12, encoding="utf-8")
    def forbidden(*args, **kwargs):
        raise AssertionError("dry_run must not access network or Docker")
    monkeypatch.setattr(auditar_plan, "post_json", forbidden)
    monkeypatch.setattr(auditar_plan, "diagnose_service", forbidden)
    report = run_audit(AuditConfig(source, tmp_path / "out", max_words=40, dry_run=True))
    assert report.is_file()
    assert "100.0%" in capsys.readouterr().out
    assert "dry_run" in report.read_text(encoding="utf-8")


def test_scores_progress_and_prepared_document(tmp_path, monkeypatch, capsys):
    import auditar_plan
    from auditoria_config import run_audit
    from documentos import Document, TextBlock, TextChunk

    source = tmp_path / "human.txt"
    text = "This passage was prepared externally and must remain unchanged. " * 6
    source.write_text("different original text", encoding="utf-8")
    document = Document(text, [TextBlock(text, "para_0")], {"format": "txt"})
    chunks = [TextChunk("chunk_001", text, ["para_0"], ["Heading"])]
    responses = iter([{"overall_score": 0, "engines": {"one": {"score": 0.9}}}, {"overall_score": 82}])
    sent = []
    def post(api, endpoint, payload, timeout):
        sent.append((endpoint, payload["text"]))
        return next(responses)
    monkeypatch.setattr(auditar_plan, "post_json", post)
    config = AuditConfig(source, tmp_path / "out")
    report = run_audit(config, document, chunks, {"health": {"engines": {"loaded": 12, "total": 23}}})
    output = capsys.readouterr().out
    assert "50.0%" in output and "100.0%" in output
    assert "0.00/100" in output and "82.00/100" in output
    assert "ATENCION" in output
    assert sent == [("/api/analyze", text), ("/api/analyze", text)]
    assert "82.0" in report.read_text(encoding="utf-8")


def test_timeout_stops_new_submissions(tmp_path, monkeypatch, capsys):
    import auditar_plan
    import requests
    from auditoria_config import run_audit

    source = tmp_path / "human.txt"
    source.write_text("A sufficient test passage for analysis, containing several words. " * 20, encoding="utf-8")
    calls = []
    def post(*args, **kwargs):
        calls.append(args)
        raise requests.Timeout("test timeout")
    monkeypatch.setattr(auditar_plan, "post_json", post)
    report = run_audit(AuditConfig(source, tmp_path / "out", max_words=40), diagnostics={"health": {"status": "healthy"}})
    assert len(calls) == 1
    assert "timeout" in report.read_text(encoding="utf-8")
    assert "100.0%" in capsys.readouterr().out


def test_global_score_never_uses_engine_score():
    from auditar_plan import find_score

    assert find_score({"engines": {"one": {"score": 0.99}}}) is None
    assert find_score({"overall_score": 0}) == 0
    assert find_score({"overall_score": float("nan")}) is None
    assert find_score({"overall_score": True}) is None
    assert find_score({"result": {"overall_score": 23.5}}) == 23.5
"""The licence summary is an audit trail, not an inventory of this disk.

VOICE_LICENCES.md records what each voice is licensed under. It used to be
regenerated purely from the `*.onnx` files present in voices/, which meant
installing one voice into a fresh checkout rewrote the file with only that
voice in it -- silently deleting the public-domain records for the three
shipped voices.

Found by cloning the repository fresh and following the documented bake-off
instructions:

    python -m scripts.get_voices --voice en_US-bryce-medium
    git diff voices/VOICE_LICENCES.md    # three rows deleted

That is a licensing record disappearing because of an unrelated local action,
and a developer could commit the deletion without noticing. The invariant
these tests pin: provisioning or removing a local voice must never delete an
existing row for another voice.
"""

from __future__ import annotations

from pathlib import Path

from scripts.get_voices import existing_rows, write_summary

CARD = """# Model card for {name}

* Language: en_US (English, United States)

## Dataset

* License: public domain

## Training

Single speaker.
"""

SHIPPED = ("en_US-john-medium", "en_US-ljspeech-high", "en_US-norman-medium")


def rows_in(voices: Path) -> list[str]:
    return [
        line
        for line in (voices / "VOICE_LICENCES.md").read_text(encoding="utf-8").splitlines()
        if line.startswith("| `")
    ]


def names_in(voices: Path) -> set[str]:
    return set(existing_rows(voices / "VOICE_LICENCES.md"))


def install(voices: Path, name: str) -> None:
    """A voice that is present on this machine: model plus its card."""
    voices.mkdir(parents=True, exist_ok=True)
    (voices / f"{name}.onnx").write_bytes(b"not a real model")
    (voices / f"{name}.MODEL_CARD").write_text(CARD.format(name=name), encoding="utf-8")


def seed_summary(voices: Path, names: tuple[str, ...]) -> None:
    """The file as it arrives in a fresh checkout: rows for voices whose
    models are NOT installed."""
    voices.mkdir(parents=True, exist_ok=True)
    body = ["# Voice licences", "", "| Voice | Verdict | Licence | Notes |", "|---|---|---|---|"]
    body += [f"| `{n}` | public domain | public domain | - |" for n in names]
    (voices / "VOICE_LICENCES.md").write_text("\n".join(body) + "\n", encoding="utf-8")


# -- the regression ---------------------------------------------------------


def test_installing_one_voice_keeps_the_records_of_the_others(tmp_path):
    """The defect, exactly as reproduced from a fresh clone."""
    voices = tmp_path / "voices"
    seed_summary(voices, SHIPPED)
    install(voices, "en_US-bryce-medium")

    write_summary(voices)

    names = names_in(voices)
    for shipped in SHIPPED:
        assert shipped in names, f"{shipped} lost its licence record"
    assert "en_US-bryce-medium" in names, "the installed voice must be recorded"


def test_removing_a_local_voice_does_not_delete_its_record(tmp_path):
    """The other half of the invariant. A model deleted off this disk says
    nothing about what that voice is licensed under."""
    voices = tmp_path / "voices"
    install(voices, "en_US-bryce-medium")
    write_summary(voices)
    assert "en_US-bryce-medium" in names_in(voices)

    (voices / "en_US-bryce-medium.onnx").unlink()
    write_summary(voices)

    assert "en_US-bryce-medium" in names_in(voices)


def test_rewriting_is_idempotent(tmp_path):
    """Provisioning twice must not duplicate rows or churn the file."""
    voices = tmp_path / "voices"
    seed_summary(voices, SHIPPED)
    install(voices, "en_US-bryce-medium")

    write_summary(voices)
    once = (voices / "VOICE_LICENCES.md").read_text(encoding="utf-8")
    write_summary(voices)
    write_summary(voices)
    thrice = (voices / "VOICE_LICENCES.md").read_text(encoding="utf-8")

    assert once == thrice, "repeated runs must not change the file"
    rows = rows_in(voices)
    assert len(rows) == len(set(rows)), f"duplicate rows: {rows}"


def test_rows_stay_in_a_stable_order(tmp_path):
    """Otherwise every provisioning run produces a noisy diff."""
    voices = tmp_path / "voices"
    seed_summary(voices, SHIPPED)
    install(voices, "en_US-bryce-medium")
    write_summary(voices)
    assert rows_in(voices) == sorted(rows_in(voices))


def test_an_installed_voice_refreshes_its_own_row(tmp_path):
    """Preserving rows must not mean going stale: a voice that IS installed
    here is re-read from its MODEL_CARD."""
    voices = tmp_path / "voices"
    seed_summary(voices, ("en_US-bryce-medium",))
    # Same name, but the card on disk now declares something restrictive.
    voices.mkdir(parents=True, exist_ok=True)
    (voices / "en_US-bryce-medium.onnx").write_bytes(b"x")
    (voices / "en_US-bryce-medium.MODEL_CARD").write_text(
        "# Model card\n\n## Dataset\n\n* License: CC BY-NC-SA 4.0\n",
        encoding="utf-8",
    )

    write_summary(voices)

    row = existing_rows(voices / "VOICE_LICENCES.md")["en_US-bryce-medium"]
    assert "restricted" in row, f"stale row kept instead of refreshed: {row}"


def test_a_missing_summary_is_created(tmp_path):
    voices = tmp_path / "voices"
    install(voices, "en_US-john-medium")
    write_summary(voices)
    assert (voices / "VOICE_LICENCES.md").exists()
    assert "en_US-john-medium" in names_in(voices)


def test_existing_rows_ignores_everything_that_is_not_a_row(tmp_path):
    summary = tmp_path / "VOICE_LICENCES.md"
    summary.write_text(
        "# Voice licences\n\nprose\n\n| Voice | Verdict |\n|---|---|\n"
        "| `en_US-john-medium` | public domain | public domain | - |\n\nmore prose\n",
        encoding="utf-8",
    )
    assert set(existing_rows(summary)) == {"en_US-john-medium"}


def test_existing_rows_on_a_missing_file_is_empty(tmp_path):
    assert existing_rows(tmp_path / "absent.md") == {}

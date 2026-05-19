"""CSV and JSON readers/writers for JX-3P patch banks.

JSON is the canonical interchange format (see schemas/). CSV matches the
columns the C decoder emits and the C encoder accepts, byte-for-byte.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema

from jx3p.patch import JX3PPatch
from jx3p.sequence import (
    JX3PSequence,
    JX3PStep,
    JX3PVoice,
    PAGES_PER_SEQUENCE,
    STEPS_PER_PAGE,
    VOICES_PER_STEP,
)


FORMAT_VERSION = "1.0"

# CSV header line as emitted by c/decoder/src/decoder_patch.c print_csv_header.
# Reproduced verbatim so write_csv produces byte-identical output.
_CSV_HEADER = (
    "patch, "
    "A01 (DCO-1 Range), A02 (DCO-1 Waveform), A03 (DCO-1 LFO Mod), A04 (DCO-1 ENV Mod), "
    "A05 (DCO-2 Range), A06 (DCO-2 Waveform), A07 (DCO-2 Cross Modulation), A08 (DCO-2 Tune), "
    "A09 (DCO-2 Fine Tune), A10 (DCO-2 LFO Mod), A11 (DCO-2 ENV Mod), A12 (DCO LFO Mod), "
    "A13 (DCO ENV Mod), A14 (DCO ENV polarity), A15 (VCF Mix), A16 (VCF High Pass), "
    "B01 (VCF Cutoff Frequency), B02 (VCF LFO Mod), B03 (VCF Pitch Follow), B04 (VCF Resonance), "
    "B05 (VCF ENV Mod), B06 (VCF ENV polarity), B07 (VCA Mode), B08 (VCA Level), "
    "B09 (Chorus off/on), B10 (LFO Waveform), B11 (LFO Delay), B12 (LFO Rate), "
    "B13 (ENV Attack), B14 (ENV Decay), B15 (ENV Sustain), B16 (ENV Release), "
    "mystery"
)

# (attr_on_JX3PPatch, kind) for each data column of the CSV, in order.
# kinds:
#   "str"       — value is already a string (multi-bit enum or polarity/mode label)
#   "int"       — base-10 integer 0..255 (or 0..15 for mystery)
#   "bool_01"   — bool rendered "0"/"1" in CSV
#   "bool_onoff"— bool rendered "off"/"on" in CSV (chorus only)
_CSV_COLUMNS: tuple[tuple[str, str], ...] = (
    ("dco1_range",        "str"),
    ("dco1_waveform",     "str"),
    ("dco1_fmod_lfo",     "bool_01"),
    ("dco1_fmod_env",     "bool_01"),
    ("dco2_range",        "str"),
    ("dco2_waveform",     "str"),
    ("dco2_crossmod",     "str"),
    ("dco2_tune",         "int"),
    ("dco2_fine_tune",    "int"),
    ("dco2_fmod_lfo",     "bool_01"),
    ("dco2_fmod_env",     "bool_01"),
    ("dco_lfo_amount",    "int"),
    ("dco_env_amount",    "int"),
    ("dco_env_polarity",  "str"),
    ("vcf_mix",           "int"),
    ("vcf_hpf",           "int"),
    ("vcf_cutoff",        "int"),
    ("vcf_lfo_mod",       "int"),
    ("vcf_pitch_follow",  "int"),
    ("vcf_resonance",     "int"),
    ("vcf_env_mod",       "int"),
    ("vcf_env_polarity",  "str"),
    ("vca_mode",          "str"),
    ("vca_level",         "int"),
    ("chorus",            "bool_onoff"),
    ("lfo_waveform",      "str"),
    ("lfo_delay",         "int"),
    ("lfo_rate",          "int"),
    ("env_attack",        "int"),
    ("env_decay",         "int"),
    ("env_sustain",       "int"),
    ("env_release",       "int"),
    ("mystery",           "int"),
)


# --- JSON ------------------------------------------------------------------

def read_json(path: str | Path) -> list[list[JX3PPatch]]:
    """Read a 2x16 patch collection from a JSON file conforming to bank.schema.json."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    validate_bank_json(data)
    return [
        [JX3PPatch.from_dict(p) for p in bank]
        for bank in data["banks"]
    ]


def write_json(path: str | Path, banks: list[list[JX3PPatch]], *, indent: int = 2) -> None:
    """Write a 2x16 patch collection to a JSON file conforming to bank.schema.json."""
    _check_banks_shape(banks)
    data = {
        "format_version": FORMAT_VERSION,
        "banks": [[p.to_dict() for p in bank] for bank in banks],
    }
    validate_bank_json(data)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent)
        fh.write("\n")


def validate_bank_json(data: dict[str, Any]) -> None:
    """Validate a parsed JSON dict against bank.schema.json. Raises ValidationError."""
    _bank_validator().validate(data)


# --- sequence JSON ---------------------------------------------------------

def read_seq_json(path: str | Path) -> JX3PSequence:
    """Read a JX-3P sequence dump from a JSON file."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if data.get("kind") != "sequence":
        raise ValueError("not a sequence JSON file (missing 'kind': 'sequence')")
    raw_pages = data.get("pages") or []
    if len(raw_pages) != PAGES_PER_SEQUENCE:
        raise ValueError(
            f"expected {PAGES_PER_SEQUENCE} pages, got {len(raw_pages)}"
        )
    seq = JX3PSequence()
    for i, raw_page in enumerate(raw_pages):
        seq.pages[i] = _page_from_dict(raw_page, i)
    return seq


def write_seq_json(path: str | Path, seq: JX3PSequence, *, indent: int = 2) -> None:
    """Write a JX-3P sequence dump to a JSON file."""
    if len(seq.pages) != PAGES_PER_SEQUENCE:
        raise ValueError(f"sequence must have {PAGES_PER_SEQUENCE} pages")
    data = {
        "format_version": FORMAT_VERSION,
        "kind": "sequence",
        "pages": [_page_to_dict(p) for p in seq.pages],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent)
        fh.write("\n")


def _page_to_dict(page: list[JX3PStep] | None) -> list[dict] | None:
    if page is None:
        return None
    return [_step_to_dict(s) for s in page]


def _page_from_dict(raw_page: list[dict] | None, page_idx: int) -> list[JX3PStep] | None:
    if raw_page is None:
        return None
    if len(raw_page) != STEPS_PER_PAGE:
        raise ValueError(
            f"page {page_idx}: expected {STEPS_PER_PAGE} steps, got {len(raw_page)}"
        )
    return [_step_from_dict(s) for s in raw_page]


def _step_to_dict(step: JX3PStep) -> dict[str, Any]:
    return {
        "voices": [None if v is None else {"note": v.note, "tied": v.tied}
                   for v in step.voices],
        "byte7": step.byte7,
    }


def _step_from_dict(raw: dict[str, Any]) -> JX3PStep:
    voices_raw = raw.get("voices") or []
    if len(voices_raw) != VOICES_PER_STEP:
        raise ValueError(
            f"step has {len(voices_raw)} voices, expected {VOICES_PER_STEP}"
        )
    voices = [
        None if v is None else JX3PVoice(note=int(v["note"]), tied=bool(v["tied"]))
        for v in voices_raw
    ]
    byte7 = int(raw.get("byte7", 0x01))
    return JX3PStep(voices=voices, byte7=byte7)


# --- CSV -------------------------------------------------------------------

def read_csv(path: str | Path) -> list[list[JX3PPatch]]:
    """Read a 2x16 patch collection from a CSV file matching the C decoder's output."""
    with open(path, encoding="utf-8") as fh:
        lines = [line.rstrip("\r\n") for line in fh if line.strip()]
    if len(lines) < 33:  # header + 32 patches
        raise ValueError(f"CSV must have header + 32 patch rows, got {len(lines)} lines")

    banks: list[list[JX3PPatch]] = [[JX3PPatch() for _ in range(16)] for _ in range(2)]
    for row_idx, line in enumerate(lines[1:33]):
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) != len(_CSV_COLUMNS) + 1:
            raise ValueError(
                f"row {row_idx} ({cells[0] if cells else '?'}): expected "
                f"{len(_CSV_COLUMNS) + 1} columns, got {len(cells)}"
            )
        bank_idx = row_idx // 16
        slot = row_idx % 16
        banks[bank_idx][slot] = _patch_from_csv_cells(cells)
    return banks


def write_csv(path: str | Path, banks: list[list[JX3PPatch]]) -> None:
    """Write a 2x16 patch collection to a CSV file (byte-identical to the C decoder)."""
    _check_banks_shape(banks)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(_CSV_HEADER + "\n")
        for bank_idx in range(2):
            for slot in range(16):
                row = patch_to_csv_row(bank_idx, slot, banks[bank_idx][slot])
                fh.write(", ".join(row) + "\n")


# --- helpers (also used by tests) -----------------------------------------

def patch_label(bank_idx: int, slot: int) -> str:
    """Return the C-style row label, e.g. ``C01`` or ``D16``."""
    return f"{'CD'[bank_idx]}{slot + 1:02d}"


def patch_to_csv_row(bank_idx: int, slot: int, patch: JX3PPatch) -> list[str]:
    """Render one patch as the C decoder's CSV cell list (label + 33 fields)."""
    cells: list[str] = [patch_label(bank_idx, slot)]
    for attr, kind in _CSV_COLUMNS:
        cells.append(_render_cell(getattr(patch, attr), kind))
    return cells


# --- internals -------------------------------------------------------------

def _render_cell(value: Any, kind: str) -> str:
    if kind == "str":
        return str(value)
    if kind == "int":
        return str(int(value))
    if kind == "bool_01":
        return "1" if value else "0"
    if kind == "bool_onoff":
        return "on" if value else "off"
    raise AssertionError(f"unknown CSV column kind: {kind!r}")


def _parse_cell(cell: str, kind: str) -> Any:
    if kind == "str":
        return cell
    if kind == "int":
        return int(cell)
    if kind == "bool_01":
        if cell not in ("0", "1"):
            raise ValueError(f"expected 0 or 1, got {cell!r}")
        return cell == "1"
    if kind == "bool_onoff":
        if cell not in ("off", "on"):
            raise ValueError(f"expected off or on, got {cell!r}")
        return cell == "on"
    raise AssertionError(f"unknown CSV column kind: {kind!r}")


def _patch_from_csv_cells(cells: list[str]) -> JX3PPatch:
    """Inverse of patch_to_csv_row. The label cell (0) is ignored."""
    kwargs: dict[str, Any] = {}
    for (attr, kind), cell in zip(_CSV_COLUMNS, cells[1:]):
        kwargs[attr] = _parse_cell(cell, kind)
    return JX3PPatch(**kwargs)


def _check_banks_shape(banks: list[list[JX3PPatch]]) -> None:
    if len(banks) != 2 or any(len(b) != 16 for b in banks):
        raise ValueError("banks must be a 2x16 list of JX3PPatch")


_VALIDATOR: jsonschema.Draft202012Validator | None = None


def _bank_validator() -> jsonschema.Draft202012Validator:
    """Build and cache a validator for bank.schema.json that resolves the patch $ref.

    Schema files live in jx3p/schemas/ in an installed wheel (per
    pyproject.toml's force-include) and at the repo-root schemas/ in a
    source checkout. We try both locations.
    """
    global _VALIDATOR
    if _VALIDATOR is not None:
        return _VALIDATOR

    from referencing import Registry, Resource

    schemas_dir = _schemas_dir()
    with open(schemas_dir / "bank.schema.json", encoding="utf-8") as fh:
        bank_schema = json.load(fh)
    with open(schemas_dir / "patch.schema.json", encoding="utf-8") as fh:
        patch_schema = json.load(fh)

    # bank.schema.json refers to patch.schema.json via a relative $ref; register
    # the patch resource under both the bare filename and its $id so resolution
    # succeeds regardless of how the $ref is written.
    patch_resource = Resource.from_contents(patch_schema)
    registry_entries = [("patch.schema.json", patch_resource)]
    if "$id" in patch_schema:
        registry_entries.append((patch_schema["$id"], patch_resource))
    registry = Registry().with_resources(registry_entries)

    _VALIDATOR = jsonschema.Draft202012Validator(bank_schema, registry=registry)
    return _VALIDATOR


def _schemas_dir() -> Path:
    pkg_dir = Path(__file__).parent
    if (pkg_dir / "schemas").is_dir():
        return pkg_dir / "schemas"
    return pkg_dir.parent / "schemas"

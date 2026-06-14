"""Real-hardware regression: the auto-boost decodes a genuinely quiet capture.

`fixtures/quiet-unit-seq.wav` is an ACTUAL JX-3P sequence tape dump Daniel
recorded 2026-06-14 (downstairs unit, KT cable, ~12% peak — exactly the kind of
quiet capture that motivated raising AUTO_BOOST_TARGET to 0.92). It decoded
correctly in the app; these pin that the shipped boost keeps rescuing it, so a
future change to the boost or the detector can't silently regress real
low-level captures.

History: an exploratory decode-time gain/limiter "sweep" briefly lived here
(2026-06-12). It was dropped 2026-06-14 — the plain peak-boost handles every
real case, and the limiter rungs over-clip a clean FSK tone (whose tone IS its
own peak), decoding nothing. See docs/auto-calibration-handoff.md in the app
repo for the full reasoning.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from jx3p import codec

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "quiet-unit-seq.wav"


def test_quiet_real_capture_is_boosted_to_target() -> None:
    """The ~12% capture is lifted to AUTO_BOOST_TARGET so the ±0.15 detector
    band sees it — this is the mechanism the v0.8.1 bump fixed."""
    samples = codec._load_wav_mono_float(FIXTURE)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    assert peak >= codec.AUTO_BOOST_TARGET - 1e-6


def test_quiet_real_capture_decodes_all_page_records() -> None:
    """Every page record is recovered from the real quiet dump (the failure
    mode this guards against is zero records)."""
    samples = codec._load_wav_mono_float(FIXTURE)
    records = codec._decode_sequence_records(
        codec._demodulate_bits(codec._detect_crossings(samples)))
    assert len(records) == codec.PAGES_PER_SEQUENCE


def test_quiet_real_capture_decodes_via_public_api() -> None:
    """End-to-end through read_seq_wav: the dump's content survives."""
    seq = codec.read_seq_wav(FIXTURE)
    populated = [i for i, page in enumerate(seq.pages) if page is not None]
    assert populated, "real quiet capture decoded to an empty sequence (regression)"

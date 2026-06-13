"""WAV ↔ patch struct codec.

Three-stage pipeline matching the C decoder (c/decoder/src/{detector,demodulator,decoder_patch}.c):

    audio samples → zero-crossing intervals → bits → 26-byte patch records

The JX-3P encodes patches as FSK audio with a sampled pulse per bit (a 0 is a
single ~11-sample high-frequency cycle, a 1 is a single ~50-sample
low-frequency cycle). Each byte is wrapped in an 11-bit serial frame
(start = 0, 8 data bits LSB first, two stop = 1). A bank holds 16 patches
and is preceded by a long pilot tone of 1-bits; each patch is transmitted
twice with a 48-bit separator tone of 1-bits between transmissions.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from jx3p.patch import JX3PPatch
from jx3p.sequence import (
    JX3PSequence,
    PAGES_PER_SEQUENCE,
    SEQUENCE_DATATYPE,
    build_seq_record,
    decode_page_payload,
    encode_page_payload,
    parse_seq_record,
)


# --- format constants (mirror c/decoder & c/encoder) -----------------------

QUIESCENCE_THRESHOLD = 0.15  # detector Schmitt-trigger band, normalised -1..1
SHORT_THRESHOLD = 0.625      # crossing < long_avg * SHORT_THRESHOLD => short cycle
LONG_BIT_PHASES = 2          # consecutive long crossings to emit one 1 bit
SHORT_BIT_PHASES = 2         # consecutive short crossings to emit one 0 bit

# Quiet-recording rescue. Tape dumps recorded with too little Mac/interface
# input gain never reach QUIESCENCE_THRESHOLD, so the detector emits zero
# crossings and the decoder finds no records (the user sees an empty JSON).
# After loading a WAV we look at its peak amplitude: if it is below
# AUTO_BOOST_TARGET we scale the samples up so the peak equals the target.
# Loud-enough inputs (peak >= target) pass through untouched.
#
# Target raised 0.7 -> 0.92 (2026-06-12). The app's Record-from-JX
# calibration normalizes the capture PEAK to ~0.78, so the old 0.7 target
# sat BELOW the calibration aim and never fired for a calibrated user.
# That left the weak passages of a dump dipping toward the ±0.15 detector
# band on quieter JX units, dropping crossings (sequences fail first).
# 0.92 lifts every loaded WAV to a hot, crossing-reliable level at decode
# time without the user having to over-drive (and clip) the capture.
AUTO_BOOST_TARGET = 0.92

PATCH_DATA_LENGTH = 286      # bits per patch record (26 bytes * 11-bit frame)
SEQ_DATA_LENGTH = 1463       # bits per sequence record (133 bytes * 11-bit frame)
INTERTONE_RUN = 11           # >this many consecutive 1-bits resets to searching

# encoder: pre-recorded one-cycle waveforms, sampled from a real JX-3P at 44.1kHz
WAVE_SAMPLE_RATE = 44100

# Mirrors c/encoder/audio.c
_AUDIO_BIT_ZERO = np.array([
    12244, 19836, 19449, 16342, 14058, 2774,
    -15558, -19175, -18706, -15082, -13165,
], dtype=np.int16)

_AUDIO_BIT_ONE = np.array([
    9644, 19672, 19315, 17117, 13591, 11004, 8259, 6310, 4539, 3152, 2081,
    1008, 374, -343, -646, -1094, -1375, -1550, -1882, -1716, -2225, -1727,
    -2518, -1356, -5901, -22039, -28390, -25740, -22163, -17463, -14003,
    -10592, -8088, -5874, -4142, -2784, -1503, -635, 122, 706, 1060, 1571,
    1629, 2142, 1972, 2407, 2262, 2410, 2418, 2643,
], dtype=np.int16)

_PILOT_BITS = 4096
_SEPARATOR_BITS = 48


# --- public API ------------------------------------------------------------

def read_wav(path: str | Path) -> list[list[JX3PPatch]]:
    """Decode a JX-3P tape-dump WAV into a 2x16 list of patches.

    Returned shape: ``[bank][slot]``. Bank 0 = panel C, bank 1 = panel D.
    Slots are 0..15. Patches that failed to decode (bad checksum on both
    transmissions) are returned as freshly-defaulted ``JX3PPatch`` instances.
    """
    samples = _load_wav_mono_float(Path(path))
    crossings = _detect_crossings(samples)
    bits = _demodulate_bits(crossings)
    raw_patches = _decode_patches(bits)

    banks: list[list[JX3PPatch]] = [[JX3PPatch() for _ in range(16)] for _ in range(2)]
    for bank_idx, slot, raw in raw_patches:
        banks[bank_idx][slot] = JX3PPatch.from_bytes(raw)
    return banks


def write_wav(path: str | Path, banks: list[list[JX3PPatch]]) -> None:
    """Encode a 2x16 list of patches into a JX-3P tape-dump WAV file."""
    if len(banks) != 2 or any(len(b) != 16 for b in banks):
        raise ValueError("banks must be a 2x16 list of patches")

    chunks: list[np.ndarray] = []
    for bank_idx, bank in enumerate(banks):
        chunks.append(_render_run_of_ones(_PILOT_BITS))
        for slot, patch in enumerate(bank):
            payload = patch.to_bytes(bank_idx, slot)
            chunks.append(_render_patch(payload))
            chunks.append(_render_run_of_ones(_SEPARATOR_BITS))
            chunks.append(_render_patch(payload))
            chunks.append(_render_run_of_ones(_SEPARATOR_BITS))

    audio = np.concatenate(chunks).astype(np.int16, copy=False)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(WAVE_SAMPLE_RATE)
        fh.writeframes(audio.tobytes())


# --- WAV I/O ---------------------------------------------------------------

def _load_wav_mono_float(path: Path) -> np.ndarray:
    """Read a WAV file and return mono float64 samples in -1..1.

    For multi-channel inputs only the first channel is used (matching the C
    decoder, which iterates ``si += sfinfo.channels`` and ignores the rest).
    """
    with wave.open(str(path), "rb") as fh:
        n_channels = fh.getnchannels()
        sampwidth = fh.getsampwidth()
        n_frames = fh.getnframes()
        raw = fh.readframes(n_frames)

    if sampwidth == 1:
        # WAV 8-bit is unsigned (0..255 with bias 128)
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.int16) - 128
        scale = 1.0 / 128.0
    elif sampwidth == 2:
        data = np.frombuffer(raw, dtype=np.int16)
        scale = 1.0 / 32768.0
    elif sampwidth == 3:
        # 24-bit packed little-endian; expand to int32
        u8 = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        data = (
            (u8[:, 0].astype(np.int32))
            | (u8[:, 1].astype(np.int32) << 8)
            | (u8[:, 2].astype(np.int8).astype(np.int32) << 16)
        )
        scale = 1.0 / (1 << 23)
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype=np.int32)
        scale = 1.0 / (1 << 31)
    else:
        raise ValueError(f"unsupported sample width: {sampwidth} bytes")

    if n_channels > 1:
        data = data.reshape(-1, n_channels)[:, 0]
    samples = data.astype(np.float64) * scale

    # Quiet-recording rescue (see AUTO_BOOST_TARGET above). Only ever amplifies;
    # never attenuates a healthy recording. Digital silence (peak == 0) is left
    # alone to avoid division by zero.
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if 0.0 < peak < AUTO_BOOST_TARGET:
        samples = samples * (AUTO_BOOST_TARGET / peak)
    return samples


# --- detector --------------------------------------------------------------

def _detect_crossings(samples: np.ndarray) -> np.ndarray:
    """Return an array of inter-crossing intervals (in samples).

    Implements the same Schmitt-trigger zero-crossing detector as
    c/decoder/src/detector.c. The "phase" is +1 above +threshold, -1 below
    -threshold, and 0 between (sticky). A crossing is recorded when phase
    flips sign.
    """
    thresh = QUIESCENCE_THRESHOLD

    # Per-sample marker: +1, -1, or 0 (in band).
    markers = np.zeros(samples.shape, dtype=np.int8)
    markers[samples > thresh] = 1
    markers[samples < -thresh] = -1

    # Forward-fill the non-zero markers so each in-band sample inherits the
    # most recent above/below-threshold sign — matching the C "sticky phase"
    # behaviour. Samples before the first non-zero marker stay at 0 (NEUTRAL).
    nonzero = markers != 0
    if not nonzero.any():
        return np.zeros(0, dtype=np.int64)
    idx = np.where(nonzero, np.arange(samples.size), 0)
    filled = np.maximum.accumulate(idx)
    phase = markers[filled]
    # Zero-out the leading neutral region.
    first_nonzero = int(np.argmax(nonzero))
    phase[:first_nonzero] = 0

    # A crossing is the first sample where the new sign differs from prev.
    # We mirror the C loop: the crossing is registered when the *current*
    # sample is opposite-sign to the *previous* phase. Because the phase
    # stays sticky until the next non-zero sample, the crossing fires at the
    # sample that crosses the opposite threshold.
    prev_phase = np.empty_like(phase)
    prev_phase[0] = 0
    prev_phase[1:] = phase[:-1]
    crossings_mask = ((prev_phase == 1) & (samples < -thresh)) | (
        (prev_phase == -1) & (samples > thresh)
    )
    crossing_indices = np.flatnonzero(crossings_mask)

    if crossing_indices.size == 0:
        return np.zeros(0, dtype=np.int64)

    # Inter-crossing distance = current_sample - previous_crossing_sample.
    # The first interval is measured from sample 0 (matches C's
    # last_crossing_sample = 0 init).
    intervals = np.empty(crossing_indices.size + 1, dtype=np.int64)
    intervals[0] = crossing_indices[0]
    intervals[1:-1] = np.diff(crossing_indices)
    # EOF hack from the C detector: append a copy of the last interval so the
    # demodulator gets one final crossing event to flush its bit-phase counter.
    intervals[-1] = intervals[-2] if intervals.size >= 2 else intervals[0]
    return intervals


# --- demodulator -----------------------------------------------------------

def _demodulate_bits(crossings: np.ndarray) -> list[int]:
    """Convert inter-crossing intervals into a bitstream.

    First, sync-lock to the long pilot tone by averaging incoming crossing
    lengths until a short crossing appears (= start of data). After lock,
    classify each crossing as short (0) or long (1) using the running
    long-cycle width as a reference. Two consecutive same-type crossings
    emit one bit (LONG_BIT_PHASES / SHORT_BIT_PHASES, both = 2).
    """
    bits: list[int] = []
    sync_locked = False
    tone_sum = 0      # accumulator while pilot tone is being measured
    tone_count = 0
    long_width = 0.0
    bit_phase_count = 0
    bit_value = 0

    skipped_first = False

    for length in crossings:
        if not sync_locked:
            if not skipped_first:
                # The C code throws away the first crossing (likely the
                # ramp-up edge of the pilot tone).
                skipped_first = True
                continue
            if tone_count == 0:
                tone_sum = int(length)
                tone_count = 1
                long_width = float(tone_sum) / tone_count
                continue

            if length < long_width * SHORT_THRESHOLD:
                # First short crossing == start of data; SOD_PHASES = 0 so
                # we lock immediately and fall through to the bit-decode.
                sync_locked = True
            else:
                tone_sum += int(length)
                tone_count += 1
                long_width = float(tone_sum) / tone_count
                continue

        # sync_locked branch
        bitval = 0 if length < long_width * SHORT_THRESHOLD else 1

        if bit_phase_count == 0:
            bit_value = bitval
        bit_phase_count += 1

        end_phases = LONG_BIT_PHASES if bit_value == 1 else SHORT_BIT_PHASES

        # Bit must hold its type for end_phases consecutive crossings;
        # otherwise discard and resync.
        if bit_phase_count <= end_phases and bitval != bit_value:
            bit_phase_count = 0
            bit_value = 1  # C resets to LONG_VALUE on bad bit
            continue

        if bit_phase_count == end_phases:
            bits.append(bit_value)
            bit_phase_count = 0
            bit_value = 1

    return bits


# --- patch decoder ---------------------------------------------------------

def _decode_patches(bits: list[int]) -> list[tuple[int, int, bytes]]:
    """Find patch records in the bitstream and return ``(bank, slot, raw26)``.

    Searches for the inter-record run of 1-bits (>11 of them); the next 0
    bit starts a 286-bit payload (26 bytes × 11-bit serial frame). Each
    patch is transmitted twice for redundancy; we keep the first one with a
    valid checksum at each bank/slot position.
    """
    DS_SEARCHING = 0
    DS_COLLECTING = 1

    state = DS_COLLECTING  # match C init_decoder_state()
    bucket: list[int] = []
    one_count = 0
    found: dict[tuple[int, int], bytes] = {}

    for bit in bits:
        if bit:
            one_count += 1
        else:
            one_count = 0
            if state == DS_SEARCHING:
                state = DS_COLLECTING
                bucket.clear()

        if one_count > INTERTONE_RUN and state != DS_SEARCHING:
            # Hit intertone mid-record: abandon and resync.
            bucket.clear()
            state = DS_SEARCHING
            continue

        if state == DS_COLLECTING:
            bucket.append(bit)
            if len(bucket) >= PATCH_DATA_LENGTH:
                raw = _convert_bucket_to_bytes(bucket)
                if raw is not None:
                    bank_cd = (raw[2] & 0x3) - 2
                    patch_num = raw[3] & 0xF
                    if 0 <= bank_cd < 2 and 0 <= patch_num < 16:
                        key = (bank_cd, patch_num)
                        if key not in found:
                            found[key] = raw
                bucket.clear()
                state = DS_SEARCHING

    return [(bank, slot, raw) for (bank, slot), raw in found.items()]


def _convert_bucket_to_bytes(bucket: list[int]) -> bytes | None:
    """Pack 286 bits into 26 bytes (11-bit serial frame, LSB first).

    Returns ``None`` if the trailing checksum byte does not match the sum of
    the preceding 25.
    """
    out = bytearray(26)
    for byte_idx in range(26):
        b = 0
        for bit_idx in range(8):
            # bucket[byte_idx*11 + 0] is the start bit (always 0)
            # bucket[byte_idx*11 + 1] is data bit 0 (LSB)
            # bucket[byte_idx*11 + 9..10] are stop bits (always 1)
            b |= (bucket[byte_idx * 11 + 1 + bit_idx] & 1) << bit_idx
        out[byte_idx] = b
    expected = sum(out[:25]) & 0xFF
    if expected != out[25]:
        return None
    return bytes(out)


# --- encoder helpers -------------------------------------------------------

def _render_patch(payload: bytes) -> np.ndarray:
    """Render a 26-byte patch as an int16 sample stream (26 frames of 11 bits)."""
    if len(payload) != 26:
        raise ValueError(f"patch payload must be 26 bytes, got {len(payload)}")
    chunks: list[np.ndarray] = []
    for byte in payload:
        chunks.append(_AUDIO_BIT_ZERO)  # start bit
        for bit_idx in range(8):
            chunks.append(_AUDIO_BIT_ONE if (byte >> bit_idx) & 1 else _AUDIO_BIT_ZERO)
        chunks.append(_AUDIO_BIT_ONE)  # stop bit
        chunks.append(_AUDIO_BIT_ONE)  # stop bit
    return np.concatenate(chunks)


def _render_run_of_ones(n: int) -> np.ndarray:
    """Render ``n`` consecutive 1-bits as a single sample stream (pilot/separator)."""
    return np.tile(_AUDIO_BIT_ONE, n)


# --- sequence WAV codec ----------------------------------------------------
#
# The JX-3P sequencer tape format reuses the patch tape's FSK audio carrier and
# 11-bit serial framing, so the audio stages (_load_wav_mono_float,
# _detect_crossings, _demodulate_bits) decode any tape dump regardless of
# datatype. Sequence-specific differences live in the byte-layer decoding
# below: 133-byte records (vs 26 for patches), datatype = 1 (not 2), and
# 4-byte headers / per-step voice arrays defined in jx3p.sequence.

def read_seq_wav(path: str | Path) -> JX3PSequence:
    """Decode a JX-3P sequencer tape-dump WAV into a :class:`JX3PSequence`."""
    samples = _load_wav_mono_float(Path(path))
    crossings = _detect_crossings(samples)
    bits = _demodulate_bits(crossings)
    raw_pages = _decode_sequence_records(bits)

    seq = JX3PSequence()
    for page_idx, raw in raw_pages.items():
        _idx, payload = parse_seq_record(raw)
        seq.pages[page_idx] = decode_page_payload(payload)
    return seq


def write_seq_wav(path: str | Path, seq: JX3PSequence) -> None:
    """Encode a :class:`JX3PSequence` as a JX-3P sequencer tape-dump WAV.

    Mirrors :func:`write_wav` (patches): pilot tone, then for each page two
    transmissions of the record separated by inter-record 1-bits.
    """
    if len(seq.pages) != PAGES_PER_SEQUENCE:
        raise ValueError(
            f"sequence must have {PAGES_PER_SEQUENCE} pages, got {len(seq.pages)}"
        )

    chunks: list[np.ndarray] = [_render_run_of_ones(_PILOT_BITS)]
    for page_idx, page in enumerate(seq.pages):
        payload = encode_page_payload(page)
        record = build_seq_record(page_idx, payload)
        chunks.append(_render_seq_record(record))
        chunks.append(_render_run_of_ones(_SEPARATOR_BITS))
        chunks.append(_render_seq_record(record))
        chunks.append(_render_run_of_ones(_SEPARATOR_BITS))

    audio = np.concatenate(chunks).astype(np.int16, copy=False)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(WAVE_SAMPLE_RATE)
        fh.writeframes(audio.tobytes())


def _decode_sequence_records(bits: list[int]) -> dict[int, bytes]:
    """Find sequence records in the bitstream and return ``{page_idx: raw133}``.

    Parallels :func:`_decode_patches` but for 133-byte records. Each page is
    transmitted twice; the first valid copy wins.
    """
    DS_SEARCHING = 0
    DS_COLLECTING = 1

    state = DS_COLLECTING
    bucket: list[int] = []
    one_count = 0
    found: dict[int, bytes] = {}

    for bit in bits:
        if bit:
            one_count += 1
        else:
            one_count = 0
            if state == DS_SEARCHING:
                state = DS_COLLECTING
                bucket.clear()

        if one_count > INTERTONE_RUN and state != DS_SEARCHING:
            bucket.clear()
            state = DS_SEARCHING
            continue

        if state == DS_COLLECTING:
            bucket.append(bit)
            if len(bucket) >= SEQ_DATA_LENGTH:
                raw = _convert_seq_bucket_to_bytes(bucket)
                if raw is not None and (raw[0] >> 6) & 0x3 == SEQUENCE_DATATYPE:
                    page_idx = raw[2]
                    if 0 <= page_idx < PAGES_PER_SEQUENCE and page_idx not in found:
                        found[page_idx] = raw
                bucket.clear()
                state = DS_SEARCHING

    return found


def _convert_seq_bucket_to_bytes(bucket: list[int]) -> bytes | None:
    """Pack 1463 bits (133 bytes × 11-bit serial frame, LSB first) into 133 bytes.

    Returns ``None`` if the trailing checksum byte does not match the sum of
    the preceding 132.
    """
    out = bytearray(133)
    for byte_idx in range(133):
        b = 0
        for bit_idx in range(8):
            b |= (bucket[byte_idx * 11 + 1 + bit_idx] & 1) << bit_idx
        out[byte_idx] = b
    expected = sum(out[:132]) & 0xFF
    if expected != out[132]:
        return None
    return bytes(out)


def _render_seq_record(record: bytes) -> np.ndarray:
    """Render a 133-byte sequence record as an int16 sample stream."""
    if len(record) != 133:
        raise ValueError(f"sequence record must be 133 bytes, got {len(record)}")
    chunks: list[np.ndarray] = []
    for byte in record:
        chunks.append(_AUDIO_BIT_ZERO)  # start bit
        for bit_idx in range(8):
            chunks.append(_AUDIO_BIT_ONE if (byte >> bit_idx) & 1 else _AUDIO_BIT_ZERO)
        chunks.append(_AUDIO_BIT_ONE)   # stop bit
        chunks.append(_AUDIO_BIT_ONE)   # stop bit
    return np.concatenate(chunks)

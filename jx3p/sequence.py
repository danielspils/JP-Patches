"""JX3PSequence dataclass and conversion helpers.

The JX-3P sequencer stores one continuous ~128-step memory that is split into
**8 pages × 16 steps** for tape transfer. Each step holds up to 7 voice slots
(the synth is 6-voice polyphonic; 7 slots are exposed in the wire format).
The whole sequencer dump is 8 records × 133 bytes (4-byte header + 128-byte
payload + 1-byte checksum), and each record is transmitted twice for
redundancy — identical wrapping to the patch tape format.

Wire encoding of a voice byte:
    0x7F           = empty slot (no note in this voice)
    1nnnnnn0       = note-on; note value = (byte >> 1) & 0x3F, MIDI = value + 36
    0nnnnnn0       = tied; same note continues from the prior step

The JX-3P keyboard spans MIDI 36..83 (49 keys, C2-C6 in Roland nomenclature).
A "rest" entered via the JX REST button is encoded as a tied continuation of
the previously played note, not as a separate silence marker.

Byte 7 of each step is metadata (likely the initial polyphony count at record
time). The codec preserves it byte-for-byte for round-trip fidelity but does
not interpret it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


JX_LOWEST_MIDI = 36         # the synth's lowest key
JX_KEYBOARD_KEYS = 49       # C2 to C6
PAGES_PER_SEQUENCE = 8
STEPS_PER_PAGE = 16
VOICES_PER_STEP = 7
EMPTY_VOICE_BYTE = 0x7F     # sentinel: no note in this voice slot
EMPTY_BYTE7 = 0x7F          # sentinel on byte 7 of an empty step
DEFAULT_BYTE7 = 0x01        # observed default for steps with a single voice
SEQUENCE_DATATYPE = 1       # top-two-bits of byte 0; the JX emits 0b01 for sequence
SEQUENCE_HEADER_BYTE0 = (SEQUENCE_DATATYPE << 6)  # = 0x40


@dataclass
class JX3PVoice:
    """A note in one of the 7 voice slots of a sequencer step."""

    note: int   # MIDI note number, JX-3P keyboard range 36..83
    tied: bool  # True = continues from the prior step (bit 7 of source byte was clear)


@dataclass
class JX3PStep:
    """One step in a sequencer page.

    ``voices`` is always length-VOICES_PER_STEP. A slot may be ``None`` to
    indicate an empty voice. ``byte7`` is opaque metadata preserved verbatim
    on round-trip.
    """

    voices: list[Optional[JX3PVoice]] = field(
        default_factory=lambda: [None] * VOICES_PER_STEP
    )
    byte7: int = DEFAULT_BYTE7

    def is_empty(self) -> bool:
        return all(v is None for v in self.voices)


@dataclass
class JX3PSequence:
    """A full JX-3P sequencer tape dump.

    ``pages`` is always length-PAGES_PER_SEQUENCE. A page may be ``None`` to
    indicate the user never reached/programmed that page; otherwise it is a
    length-STEPS_PER_PAGE list of :class:`JX3PStep`.
    """

    pages: list[Optional[list[JX3PStep]]] = field(
        default_factory=lambda: [None] * PAGES_PER_SEQUENCE
    )


# --- byte ↔ structure conversion -------------------------------------------

def decode_voice_byte(b: int) -> Optional[JX3PVoice]:
    if b == EMPTY_VOICE_BYTE:
        return None
    note_value = (b >> 1) & 0x3F
    midi = note_value + JX_LOWEST_MIDI
    tied = (b & 0x80) == 0
    return JX3PVoice(note=midi, tied=tied)


def encode_voice(voice: Optional[JX3PVoice]) -> int:
    if voice is None:
        return EMPTY_VOICE_BYTE
    note_value = voice.note - JX_LOWEST_MIDI
    if not (0 <= note_value < 64):
        raise ValueError(
            f"note {voice.note} outside JX-3P keyboard range "
            f"({JX_LOWEST_MIDI}..{JX_LOWEST_MIDI + 63})"
        )
    byte = (note_value & 0x3F) << 1
    if not voice.tied:
        byte |= 0x80
    return byte & 0xFF


def decode_page_payload(payload: bytes) -> Optional[list[JX3PStep]]:
    """Decode a 128-byte page payload into 16 steps, or ``None`` if empty."""
    if len(payload) != STEPS_PER_PAGE * 8:
        raise ValueError(f"page payload must be {STEPS_PER_PAGE * 8} bytes")
    # An entirely-empty page has 0x7f in every voice slot.
    if all(payload[s * 8 + v] == EMPTY_VOICE_BYTE
           for s in range(STEPS_PER_PAGE)
           for v in range(VOICES_PER_STEP)):
        return None
    steps: list[JX3PStep] = []
    for s in range(STEPS_PER_PAGE):
        chunk = payload[s * 8:(s + 1) * 8]
        voices = [decode_voice_byte(chunk[v]) for v in range(VOICES_PER_STEP)]
        steps.append(JX3PStep(voices=voices, byte7=chunk[7]))
    return steps


def encode_page_payload(page: Optional[list[JX3PStep]]) -> bytes:
    """Encode 16 steps (or ``None``) as a 128-byte payload."""
    if page is None:
        # Entirely empty page: 0x7f in every byte position.
        return bytes([EMPTY_VOICE_BYTE] * (STEPS_PER_PAGE * 8))
    if len(page) != STEPS_PER_PAGE:
        raise ValueError(f"page must have {STEPS_PER_PAGE} steps, got {len(page)}")
    out = bytearray(STEPS_PER_PAGE * 8)
    for s, step in enumerate(page):
        if len(step.voices) != VOICES_PER_STEP:
            raise ValueError(
                f"step {s} has {len(step.voices)} voices, expected {VOICES_PER_STEP}"
            )
        for v in range(VOICES_PER_STEP):
            out[s * 8 + v] = encode_voice(step.voices[v])
        out[s * 8 + 7] = step.byte7 & 0xFF
    return bytes(out)


def build_seq_record(page_idx: int, payload_128: bytes) -> bytes:
    """Assemble a 133-byte sequence record (header + payload + checksum)."""
    if not (0 <= page_idx < PAGES_PER_SEQUENCE):
        raise ValueError(f"page_idx out of range: {page_idx}")
    if len(payload_128) != STEPS_PER_PAGE * 8:
        raise ValueError(f"payload must be {STEPS_PER_PAGE * 8} bytes")
    out = bytearray(133)
    out[0] = SEQUENCE_HEADER_BYTE0
    out[1] = 0x00
    out[2] = page_idx & 0xFF
    out[3] = page_idx & 0xFF
    out[4:132] = payload_128
    out[132] = sum(out[:132]) & 0xFF
    return bytes(out)


def parse_seq_record(raw: bytes) -> tuple[int, bytes]:
    """Validate a 133-byte sequence record and return ``(page_idx, payload)``.

    Raises ``ValueError`` if the record is malformed or the checksum is wrong.
    """
    if len(raw) != 133:
        raise ValueError(f"sequence record must be 133 bytes, got {len(raw)}")
    if (raw[0] >> 6) & 0x3 != SEQUENCE_DATATYPE:
        raise ValueError(f"datatype mismatch: byte 0 = 0x{raw[0]:02x}")
    expected = sum(raw[:132]) & 0xFF
    if expected != raw[132]:
        raise ValueError(f"checksum: expected 0x{expected:02x}, got 0x{raw[132]:02x}")
    return raw[2], bytes(raw[4:132])

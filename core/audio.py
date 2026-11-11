"""Síntesis aditiva de piano + exportación a WAV y MIDI (sin dependencias externas)."""
import io
import numpy as np
import scipy.io.wavfile as wavfile
from .notas import nombre_a_freq, nombre_a_midi, midi_a_freq

SR = 44100
ARMONICOS = [(1, 1.0), (2, 0.5), (3, 0.25), (4, 0.12), (5, 0.06)]


def _a_wav(señal: np.ndarray, sr: int = SR) -> bytes:
    buf = io.BytesIO()
    wavfile.write(buf, sr, (np.clip(señal, -1, 1) * 32767).astype(np.int16))
    return buf.getvalue()


def _nota(freq: float, dur: float, sr: int = SR) -> np.ndarray:
    n = int(sr * dur)
    t = np.arange(n) / sr
    onda = sum(a * np.sin(2 * np.pi * freq * k * t) for k, a in ARMONICOS if freq * k < sr / 2)
    env = np.exp(-3.0 * t)                               # decaimiento tipo piano
    na = max(1, int(sr * 0.008))                          # ataque de 8 ms (evita "clicks")
    env[:na] *= np.linspace(0, 1, na)
    nr = min(int(sr * 0.04), n)                           # release de 40 ms
    env[-nr:] *= np.linspace(1, 0, nr)
    return onda * env


def tono(nota: str, dur: float = 1.5) -> bytes:
    """WAV de una sola nota (para las teclas del piano)."""
    w = _nota(nombre_a_freq(nota), dur)
    return _a_wav(w / np.max(np.abs(w)) * 0.9)


def sintetizar(notas, bpm: int = 100, solape: float = 1.6) -> bytes:
    """Melodía completa. Las notas se solapan un poco (pedal) y se normaliza globalmente."""
    if not notas:
        return b""
    paso = 60.0 / bpm
    largo = int(SR * paso * (len(notas) - 1 + solape)) + 1
    buf = np.zeros(largo)
    for i, n in enumerate(notas):
        w = _nota(nombre_a_freq(n), paso * solape)
        ini = int(i * paso * SR)
        fin = min(ini + len(w), largo)
        buf[ini:fin] += w[: fin - ini]
    pico = np.max(np.abs(buf))
    return _a_wav(buf / pico * 0.9 if pico > 0 else buf)


def sintetizar_acordes(acordes, bpm: int = 90, pulsos: int = 2) -> bytes:
    """acordes: lista de listas de números MIDI. Cada acorde dura `pulsos` negras."""
    if not acordes:
        return b""
    paso = 60.0 / bpm * pulsos
    dur = paso * 1.2
    largo = int(SR * (paso * (len(acordes) - 1) + dur)) + 1
    buf = np.zeros(largo)
    for i, ch in enumerate(acordes):
        w = sum(_nota(midi_a_freq(m), dur) for m in ch)
        ini = int(i * paso * SR)
        fin = min(ini + len(w), largo)
        buf[ini:fin] += w[: fin - ini]
    pico = np.max(np.abs(buf))
    return _a_wav(buf / pico * 0.9 if pico > 0 else buf)


def _vlq(v: int) -> bytes:
    out = [v & 0x7F]
    v >>= 7
    while v:
        out.append((v & 0x7F) | 0x80)
        v >>= 7
    return bytes(reversed(out))


def a_midi(notas, bpm: int = 100) -> bytes:
    """Archivo MIDI (formato 0) con una nota negra por elemento."""
    tpq = 480
    pista = bytearray()
    pista += b"\x00\xFF\x51\x03" + int(60_000_000 / bpm).to_bytes(3, "big")
    for n in notas:
        m = nombre_a_midi(n)
        pista += b"\x00" + bytes([0x90, m, 90])
        pista += _vlq(tpq) + bytes([0x80, m, 0])
    pista += b"\x00\xFF\x2F\x00"
    cab = b"MThd" + (6).to_bytes(4, "big") + (0).to_bytes(2, "big") + (1).to_bytes(2, "big") + tpq.to_bytes(2, "big")
    return cab + b"MTrk" + len(pista).to_bytes(4, "big") + bytes(pista)

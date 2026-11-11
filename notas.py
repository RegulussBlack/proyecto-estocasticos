"""Utilidades de notas musicales (nomenclatura latina: Do, Re, Mi...)."""
import re

NOMBRES = ["Do", "Do#", "Re", "Re#", "Mi", "Fa", "Fa#", "Sol", "Sol#", "La", "La#", "Si"]
OCTAVAS = (3, 4, 5)
MIDI_MIN = 12 * (OCTAVAS[0] + 1)        # Do3 = 48
MIDI_MAX = 12 * (OCTAVAS[-1] + 1) + 11  # Si5 = 83

NOTAS_ORDENADAS = [f"{n}{o}" for o in OCTAVAS for n in NOMBRES]


def nombre_a_midi(nota: str) -> int:
    base, octava = nota[:-1], int(nota[-1])
    return 12 * (octava + 1) + NOMBRES.index(base)


def midi_a_nombre(m: int) -> str:
    return f"{NOMBRES[m % 12]}{m // 12 - 1}"


def midi_a_freq(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


def nombre_a_freq(nota: str) -> float:
    return midi_a_freq(nombre_a_midi(nota))


def es_valida(nota: str) -> bool:
    try:
        return midi_en_rango(nombre_a_midi(nota))
    except (ValueError, IndexError):
        return False


def midi_en_rango(m: int) -> bool:
    return MIDI_MIN <= m <= MIDI_MAX


def clave_orden(token: str) -> int:
    """Orden natural para notas ('Do4') o intervalos ('+2')."""
    try:
        return nombre_a_midi(token)
    except (ValueError, IndexError):
        return int(token)


def parsear_secuencia(texto: str):
    """Convierte texto libre ('do4, re4 mi4 -> fa4') en (notas_validas, invalidas)."""
    partes = [p for p in re.split(r"[\s,;>→\-]+", texto.strip()) if p]
    validas, invalidas = [], []
    for p in partes:
        p = p.capitalize()
        (validas if es_valida(p) else invalidas).append(p)
    return validas, invalidas

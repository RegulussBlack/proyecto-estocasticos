"""Teoría musical: escalas, intervalos, tonalidad, acordes diatónicos y progresiones."""
import numpy as np
from .notas import NOMBRES, nombre_a_midi, midi_a_nombre

ESCALA_MAYOR = [0, 2, 4, 5, 7, 9, 11]
ESCALA_MENOR = [0, 2, 3, 5, 7, 8, 10]

ESCALAS = {
    "Mayor (jónica)": ESCALA_MAYOR,
    "Menor natural (eólica)": ESCALA_MENOR,
    "Menor armónica": [0, 2, 3, 5, 7, 8, 11],
    "Dórica": [0, 2, 3, 5, 7, 9, 10],
    "Mixolidia": [0, 2, 4, 5, 7, 9, 10],
    "Pentatónica mayor": [0, 2, 4, 7, 9],
    "Pentatónica menor": [0, 3, 5, 7, 10],
    "Blues": [0, 3, 5, 6, 7, 10],
    "Cromática": list(range(12)),
}

# (semitonos, nombre, razón de frecuencias en entonación justa, carácter)
INTERVALOS = [
    (0, "Unísono", "1/1", "Consonancia perfecta"),
    (1, "2ª menor", "16/15", "Disonante"),
    (2, "2ª mayor", "9/8", "Disonante suave"),
    (3, "3ª menor", "6/5", "Consonancia imperfecta"),
    (4, "3ª mayor", "5/4", "Consonancia imperfecta"),
    (5, "4ª justa", "4/3", "Consonancia perfecta"),
    (6, "Tritono", "45/32", "Disonante (tensión máxima)"),
    (7, "5ª justa", "3/2", "Consonancia perfecta"),
    (8, "6ª menor", "8/5", "Consonancia imperfecta"),
    (9, "6ª mayor", "5/3", "Consonancia imperfecta"),
    (10, "7ª menor", "9/5", "Disonante suave"),
    (11, "7ª mayor", "15/8", "Disonante"),
    (12, "Octava", "2/1", "Consonancia perfecta"),
]

# (tonalidad mayor, relativa menor, armadura)
CIRCULO_QUINTAS = [
    ("Do", "La m", "sin alteraciones"), ("Sol", "Mi m", "1 ♯"), ("Re", "Si m", "2 ♯"),
    ("La", "Fa♯ m", "3 ♯"), ("Mi", "Do♯ m", "4 ♯"), ("Si", "Sol♯ m", "5 ♯"),
    ("Fa♯ / Sol♭", "Re♯ m / Mi♭ m", "6 ♯ / 6 ♭"), ("Re♭", "Si♭ m", "5 ♭"),
    ("La♭", "Fa m", "4 ♭"), ("Mi♭", "Do m", "3 ♭"), ("Si♭", "Sol m", "2 ♭"), ("Fa", "Re m", "1 ♭"),
]

GRADOS = [  # (nombre del grado, función tonal)
    ("Tónica", "Tónica"), ("Supertónica", "Subdominante"), ("Mediante", "Tónica"),
    ("Subdominante", "Subdominante"), ("Dominante", "Dominante"),
    ("Superdominante", "Tónica"), ("Sensible / Subtónica", "Dominante"),
]
ROMANOS = ["I", "II", "III", "IV", "V", "VI", "VII"]

# Perfiles de Krumhansl-Kessler (1982)
PERFIL_MAYOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
PERFIL_MENOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def detectar_tonalidad(seq):
    """Krumhansl-Schmuckler. Devuelve (tónica_pc, modo, correlación, pcs_de_la_escala)."""
    pc = np.bincount([nombre_a_midi(n) % 12 for n in seq], minlength=12).astype(float)
    if pc.std() == 0:
        return 0, "mayor", 0.0, set(ESCALA_MAYOR)
    mejor = (-2.0, 0, "mayor")
    for tonica in range(12):
        for modo, perfil in (("mayor", PERFIL_MAYOR), ("menor", PERFIL_MENOR)):
            r = np.corrcoef(pc, np.roll(perfil, tonica))[0, 1]
            if r > mejor[0]:
                mejor = (r, tonica, modo)
    r, tonica, modo = mejor
    offs = ESCALA_MAYOR if modo == "mayor" else ESCALA_MENOR
    return tonica, modo, float(r), {(tonica + o) % 12 for o in offs}


def nombre_tonalidad(tonica, modo):
    return f"{NOMBRES[tonica]} {modo}"


def acordes_diatonicos(tonica_pc: int, modo: str) -> dict:
    """Tríadas de la tonalidad, indexadas por número romano (I, ii, iii, IV, V, vi, vii°...)."""
    offs = ESCALA_MAYOR if modo == "mayor" else ESCALA_MENOR
    pcs = [(tonica_pc + o) % 12 for o in offs]
    out = {}
    for i in range(7):
        raiz = 48 + pcs[i]
        t = raiz + (pcs[(i + 2) % 7] - pcs[i]) % 12
        q = t + (pcs[(i + 4) % 7] - pcs[(i + 2) % 7]) % 12
        terc, quin = t - raiz, q - raiz
        if (terc, quin) == (4, 7):
            cal = "mayor"
        elif (terc, quin) == (3, 7):
            cal = "menor"
        elif (terc, quin) == (3, 6):
            cal = "disminuido"
        else:
            cal = "aumentado"
        rom = ROMANOS[i] if cal in ("mayor", "aumentado") else ROMANOS[i].lower()
        rom += "°" if cal == "disminuido" else ""
        nombre = NOMBRES[pcs[i]] + {"mayor": "", "menor": "m", "disminuido": "°", "aumentado": "+"}[cal]
        out[rom] = dict(romano=rom, grado=i + 1, nombre=nombre, calidad=cal, midi=[raiz, t, q],
                        notas=[midi_a_nombre(x) for x in (raiz, t, q)],
                        grado_nombre=GRADOS[i][0], funcion=GRADOS[i][1])
    return out


def _rep(texto, n):
    return (texto.split() * n)


PROGRESIONES = {
    "mayor": {
        "Pop: I–V–vi–IV": _rep("I V vi IV", 4),
        "Años 50: I–vi–IV–V": _rep("I vi IV V", 4),
        "Canon de Pachelbel": _rep("I V vi iii IV I IV V", 2),
        "Blues de 12 compases": _rep("I I I I IV IV I I V IV I V", 2),
        "Jazz: ii–V–I": "ii V I vi ii V I I ii V I vi ii V I I".split(),
        "Cadencial clásica": "I IV V I I vi ii V I IV I V I".split(),
        "Folk: I–IV–V": _rep("I IV I V I IV V I", 2),
    },
    "menor": {
        "Épica: i–VI–III–VII": _rep("i VI III VII", 4),
        "Andaluza: i–VII–VI–v": _rep("i VII VI v", 4),
        "Menor clásica": "i iv v i i VI iv v i i iv ii° v i".split(),
        "Pop menor: i–iv–VII–III": "i iv VII III VI ii° v i i iv VII III VI ii° v i".split(),
    },
}

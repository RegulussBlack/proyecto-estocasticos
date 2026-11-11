"""Modelo de Markov de orden k con suavizado jerárquico, muestreo con temperatura,
representación por notas o por intervalos, y herramientas de evaluación."""
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from .notas import (nombre_a_midi, midi_a_nombre, clave_orden, MIDI_MIN, MIDI_MAX)


# ---------------------------------------------------------------- representaciones
def a_intervalos(seq):
    """['Do4','Mi4','Re4'] -> ['+4','-2']  (invariante a la transposición)."""
    m = [nombre_a_midi(n) for n in seq]
    return [f"{b - a:+d}" for a, b in zip(m, m[1:])]


def desde_intervalos(nota0, intervalos, lo=MIDI_MIN, hi=MIDI_MAX):
    m, out = nombre_a_midi(nota0), []
    for t in intervalos:
        m += int(t)
        while m > hi:
            m -= 12
        while m < lo:
            m += 12
        out.append(midi_a_nombre(m))
    return out


def tokenizar(seq, representacion):
    return a_intervalos(seq) if representacion == "intervalos" else list(seq)


# ---------------------------------------------------------------- modelo
class ModeloMarkov:
    """P(w | contexto) con contexto de longitud <= orden.

    Suavizado jerárquico (Dirichlet / Laplace): el nivel k usa como "prior" la
    distribución del nivel k-1. El nivel 0 usa suavizado de Laplace sobre el vocabulario:
        P_k(w|c) = (cuenta(c,w) + β·P_{k-1}(w|c')) / (cuenta(c) + β),   β = α·|V|
    Si el contexto nunca se vio, se hace *backoff* al nivel inferior.
    """

    def __init__(self, orden: int = 1, alpha: float = 0.5):
        self.orden, self.alpha = orden, alpha

    def entrenar(self, secuencias, vocab=None, clave=None):
        self.vocab = sorted(vocab or {t for s in secuencias for t in s}, key=clave or clave_orden)
        self.idx = {t: i for i, t in enumerate(self.vocab)}
        self.conteos = [defaultdict(Counter) for _ in range(self.orden + 1)]
        for s in secuencias:
            for i, w in enumerate(s):
                for k in range(0, min(self.orden, i) + 1):
                    self.conteos[k][tuple(s[i - k:i])][w] += 1
        return self

    def _dist(self, ctx):
        V = len(self.vocab)
        if not ctx:
            c = self.conteos[0].get((), Counter())
            cnt = np.array([c.get(w, 0) for w in self.vocab], dtype=float)
            return (cnt + self.alpha) / (cnt.sum() + self.alpha * V)
        prior = self._dist(ctx[1:])
        c = self.conteos[len(ctx)].get(ctx)
        if not c:
            return prior
        cnt = np.array([c.get(w, 0) for w in self.vocab], dtype=float)
        beta = self.alpha * V
        return (cnt + beta * prior) / (cnt.sum() + beta)

    def dist(self, historial):
        ctx = tuple(historial[-self.orden:]) if self.orden > 0 else ()
        return self._dist(ctx)

    # ---- evaluación
    def evaluar_segmento(self, seq, ini=1):
        """Log-verosimilitud, nº de tokens, aciertos top-1 y top-3 en seq[ini:]."""
        ll, n, a1, a3 = 0.0, 0, 0, 0
        for i in range(ini, len(seq)):
            p = self.dist(seq[:i])
            j = self.idx[seq[i]]
            ll += np.log(p[j])
            n += 1
            orden_desc = np.argsort(-p)
            a1 += int(orden_desc[0] == j)
            a3 += int(j in orden_desc[:3])
        return ll, n, a1, a3

    # ---- generación
    def generar(self, semilla, n, temperatura=1.0, top_k=0, permitidos=None, rng=None):
        rng = rng or np.random.default_rng()
        hist, out = list(semilla), []
        for _ in range(n):
            p = self.dist(hist)
            p = _ajustar(p, temperatura, top_k)
            if permitidos is not None:
                mask = np.array([t in permitidos for t in self.vocab], dtype=float)
                if (p * mask).sum() > 0:
                    p = p * mask
                    p /= p.sum()
            w = self.vocab[rng.choice(len(self.vocab), p=p)]
            hist.append(w)
            out.append(w)
        return out


def _ajustar(p, temperatura, top_k):
    p = np.power(p, 1.0 / max(temperatura, 1e-3))
    if top_k and top_k < len(p):
        corte = np.sort(p)[-top_k]
        p = np.where(p >= corte, p, 0.0)
    return p / p.sum()


def generar_melodia(secuencias_notas, orden, alpha, representacion, n, temperatura=1.0,
                    top_k=0, semilla_notas=None, permitidos_pc=None, rng=None):
    """Entrena con las piezas dadas y devuelve (melodía de notas, modelo, tokens_semilla)."""
    secs = [tokenizar(s, representacion) for s in secuencias_notas if len(s) > 1]
    modelo = ModeloMarkov(orden, alpha).entrenar(secs)
    semilla_notas = list(semilla_notas or [secuencias_notas[0][0]])

    if representacion == "intervalos":
        tokens = modelo.generar(a_intervalos(semilla_notas), n, temperatura, top_k, None, rng)
        return desde_intervalos(semilla_notas[-1], tokens), modelo
    permitidos = None
    if permitidos_pc is not None:
        permitidos = {t for t in modelo.vocab if nombre_a_midi(t) % 12 in permitidos_pc}
    return modelo.generar(semilla_notas, n, temperatura, top_k, permitidos, rng), modelo


def novedad(generada, secuencias_notas, n=4):
    """% de n-gramas de la melodía generada que NO aparecen literalmente en el entrenamiento."""
    vistos = set()
    for s in secuencias_notas:
        vistos |= {tuple(s[i:i + n]) for i in range(len(s) - n + 1)}
    gen = [tuple(generada[i:i + n]) for i in range(len(generada) - n + 1)]
    return float(np.mean([g not in vistos for g in gen])) if gen else float("nan")


# ---------------------------------------------------------------- análisis de la cadena
def matriz_transicion(secuencias, vocab, alpha, clave=None) -> pd.DataFrame:
    m = ModeloMarkov(1, alpha).entrenar(secuencias, vocab, clave)
    P = np.vstack([m.dist([s]) for s in m.vocab])
    return pd.DataFrame(P, index=m.vocab, columns=m.vocab)


def distribucion_estacionaria(P: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eig(P.T)
    v = np.real(vecs[:, np.argmin(np.abs(vals - 1))])
    return v / v.sum()


def tasa_entropia(P: np.ndarray, pi: np.ndarray) -> float:
    """H = -Σ_i π_i Σ_j P_ij log2 P_ij  (bits por nota)."""
    return float(-(pi[:, None] * P * np.log2(np.where(P > 0, P, 1))).sum())


def segundo_autovalor(P: np.ndarray) -> float:
    return float(np.sort(np.abs(np.linalg.eigvals(P)))[::-1][1]) if len(P) > 1 else 0.0


def convergencia(P: np.ndarray, pi: np.ndarray, estado: int, pasos: int = 25) -> pd.DataFrame:
    """Distancia de variación total entre δ_estado·P^t y π."""
    d, filas = np.zeros(len(P)), []
    d[estado] = 1.0
    for t in range(pasos + 1):
        filas.append((t, 0.5 * np.abs(d - pi).sum()))
        d = d @ P
    return pd.DataFrame(filas, columns=["paso", "distancia"])


# ---------------------------------------------------------------- evaluación del modelo
def evaluar(piezas: dict, orden: int, alpha: float, representacion: str,
            modo: str = "temporal", frac: float = 0.75) -> dict:
    """modo='temporal': entrena con el primer 75% de cada pieza y prueba con el 25% final.
       modo='pieza'   : leave-one-piece-out (entrena con las demás, prueba con la pieza completa)."""
    seqs = {k: tokenizar(v, representacion) for k, v in piezas.items() if len(v) > 4}
    vocab = {t for s in seqs.values() for t in s}
    ll = n = a1 = a3 = 0
    if modo == "temporal":
        cortes = {k: max(2, int(len(s) * frac)) for k, s in seqs.items()}
        m = ModeloMarkov(orden, alpha).entrenar([s[:cortes[k]] for k, s in seqs.items()], vocab)
        for k, s in seqs.items():
            r = m.evaluar_segmento(s, cortes[k])
            ll, n, a1, a3 = ll + r[0], n + r[1], a1 + r[2], a3 + r[3]
    else:
        for k, s in seqs.items():
            otros = [t for kk, t in seqs.items() if kk != k]
            if not otros:
                continue
            m = ModeloMarkov(orden, alpha).entrenar(otros, vocab)
            r = m.evaluar_segmento(s, 1)
            ll, n, a1, a3 = ll + r[0], n + r[1], a1 + r[2], a3 + r[3]
    if n == 0:
        return dict(perplejidad=np.nan, top1=np.nan, top3=np.nan, n=0, V=len(vocab))
    return dict(perplejidad=float(np.exp(-ll / n)), top1=a1 / n, top3=a3 / n, n=n, V=len(vocab))


def rejilla(piezas, ordenes, alphas, representacion, modo):
    filas = []
    for o in ordenes:
        for a in alphas:
            r = evaluar(piezas, o, a, representacion, modo)
            filas.append(dict(orden=o, alpha=a, **r))
    return pd.DataFrame(filas)

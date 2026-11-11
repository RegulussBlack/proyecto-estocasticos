"""Minería de datos musical: extracción de características, tonalidad, PCA, clustering."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from .notas import NOMBRES, nombre_a_midi

# Perfiles de Krumhansl-Kessler (1982)
PERFIL_MAYOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
PERFIL_MENOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
ESCALA_MAYOR = [0, 2, 4, 5, 7, 9, 11]
ESCALA_MENOR = [0, 2, 3, 5, 7, 8, 10]
INTERVALOS = list(range(-7, 8))


def _pc_hist(seq):
    return np.bincount([nombre_a_midi(n) % 12 for n in seq], minlength=12).astype(float)


def detectar_tonalidad(seq):
    """Algoritmo de Krumhansl-Schmuckler. Devuelve (tónica_pc, modo, correlación, pcs_escala)."""
    pc = _pc_hist(seq)
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


def caracteristicas(seq) -> dict:
    """Vector de rasgos, invariante a la transposición (histograma relativo a la tónica)."""
    midis = np.array([nombre_a_midi(n) for n in seq])
    iv = np.diff(midis)
    tonica, *_ = detectar_tonalidad(seq)
    pc = np.roll(_pc_hist(seq), -tonica)
    pc = pc / pc.sum()
    f = {f"grado_{i}": pc[i] for i in range(12)}
    hist_iv = np.array([np.sum(np.clip(iv, -7, 7) == k) for k in INTERVALOS], dtype=float)
    hist_iv = hist_iv / max(hist_iv.sum(), 1)
    f.update({f"int_{k:+d}": v for k, v in zip(INTERVALOS, hist_iv)})
    p = pc[pc > 0]
    f["entropia_pc"] = float(-(p * np.log2(p)).sum())
    f["salto_medio"] = float(np.mean(np.abs(iv))) if len(iv) else 0.0
    f["prop_repetidas"] = float(np.mean(iv == 0)) if len(iv) else 0.0
    f["prop_grados_conj"] = float(np.mean(np.abs(iv) <= 2)) if len(iv) else 0.0
    f["prop_saltos"] = float(np.mean(np.abs(iv) >= 5)) if len(iv) else 0.0
    f["ambito"] = float(midis.max() - midis.min())
    f["notas_distintas"] = float(len(set(midis)))
    return f


def tabla_caracteristicas(piezas: dict) -> pd.DataFrame:
    return pd.DataFrame({k: caracteristicas(v) for k, v in piezas.items() if len(v) > 2}).T


def analisis_estilo(X: pd.DataFrame, k: int):
    """Estandariza, aplica PCA(2) y KMeans(k). Devuelve (df_pca, varianza_explicada, silueta, similitud)."""
    Z = StandardScaler().fit_transform(X.values)
    pca = PCA(n_components=2, random_state=0).fit(Z)
    coords = pca.transform(Z)
    etiquetas = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Z)
    sil = float(silhouette_score(Z, etiquetas)) if 2 <= k < len(X) else float("nan")
    df = pd.DataFrame({"pieza": X.index, "PC1": coords[:, 0], "PC2": coords[:, 1],
                       "cluster": etiquetas.astype(str)})
    sim = pd.DataFrame(cosine_similarity(Z), index=X.index, columns=X.index)
    return df, pca.explained_variance_ratio_, sil, sim

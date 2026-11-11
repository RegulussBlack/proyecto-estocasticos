# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 22:18:21 2026

@author: Fabricio
"""

import streamlit as st
from pianoroll_streamlit import pianoroll
import numpy as np

st.set_page_config(page_title="Compositor Markov", layout="wide")

st.title("🎹 Compositor Estocástico con Cadenas de Markov")
st.caption("Proyecto de Procesos Estocásticos y Series de Tiempo — Música Clásica")

tab_teoria, tab_piano, tab_generacion, tab_metricas = st.tabs([
    "📚 Teoría: Cadenas de Markov",
    "🎹 Piano interactivo",
    "🎼 Generación (próximamente)",
    "📊 Métricas (próximamente)"
])

# ============================================================
# PESTAÑA 1: TEORÍA
# ============================================================
with tab_teoria:
    st.header("¿Qué es una Cadena de Markov?")

    st.markdown(r"""
    Una **Cadena de Markov** es un proceso estocástico (una secuencia de variables
    aleatorias) que cumple la **propiedad de Markov**:

    > El futuro depende del presente **solo** a través del estado actual,
    > no de la historia pasada.

    Formalmente, si $X_0, X_1, X_2, \dots$ es una secuencia de estados, se cumple:

    $$P(X_{t+1} = j \mid X_t = i, X_{t-1} = i_{t-1}, \dots, X_0 = i_0) = P(X_{t+1} = j \mid X_t = i)$$

    Es decir, **"el futuro es independiente del pasado dado el presente"**.
    """)

    st.subheader("Matriz de transición")
    st.markdown(r"""
    Las probabilidades $P(X_{t+1} = j \mid X_t = i)$ se organizan en una
    **matriz de transición** $P$, donde la entrada $P_{ij}$ es la probabilidad
    de pasar del estado $i$ al estado $j$.

    Cada fila **suma 1** (es una distribución de probabilidad):

    $$\sum_{j} P_{ij} = 1 \quad \forall i$$
    """)

    st.subheader("Estimación desde datos")
    st.markdown(r"""
    En la práctica, no conocemos $P$. La **estimamos** contando las transiciones
    observadas en una secuencia:

    $$\hat{P}_{ij} = \frac{\text{número de veces que } i \to j}{\text{número de veces que salimos de } i}$$

    Para evitar probabilidades cero en transiciones no vistas, se aplica el
    **suavizado de Laplace**:

    $$\hat{P}_{ij} = \frac{\text{count}(i \to j) + \alpha}{\sum_k \text{count}(i \to k) + \alpha \cdot |S|}$$

    donde $\alpha > 0$ es un parámetro pequeño y $|S|$ es el número de estados posibles.
    """)

    st.subheader("Orden de la cadena")
    st.markdown("""
    - **Orden 1**: el siguiente estado depende solo de la nota actual.
    - **Orden 2**: depende de las **dos** últimas notas (más contexto musical).
    - **Orden $k$**: depende de las $k$ últimas notas.

    A mayor orden, más "memoria" musical, pero también más datos necesarios
    para estimar la matriz (más estados posibles).
    """)

    st.subheader("Cadena NO homogénea en el tiempo")
    st.markdown("""
    En este proyecto, la matriz **cambia mientras el usuario toca**:
    cada nota nueva actualiza los conteos y, por tanto, la ley de transición.

    Esto convierte el proceso en una **cadena de Markov no homogénea**,
    porque $P$ depende del tiempo (del momento en que se estima). Es un punto
    clave del proyecto y lo discutiremos en la defensa.
    """)

    st.subheader("¿Cómo se aplica a la música?")
    st.markdown("""
    - **Estado** = una nota (o acorde) que tocas en el piano.
    - **Transición** = pasar de una nota a la siguiente.
    - **Generación** = muestrear la siguiente nota desde $P(\cdot \mid \text{contexto})$.

    Así, el modelo **aprende tu estilo** a partir de las notas que tocas y
    propone una **continuación estocásticamente coherente** con lo que tocaste.
    """)

    st.info("👉 Cuando termines de leer, pasa a la pestaña **🎹 Piano interactivo**.")

# ============================================================
# PESTAÑA 2: PIANO INTERACTIVO
# ============================================================
with tab_piano:
    st.header("Toca algunas notas")
    st.markdown("""
    Usa el piano de abajo (clic con el mouse). Las notas que toques serán la
    **entrada** para estimar la matriz de transición de Markov.

    En el siguiente paso del proyecto, estas notas alimentarán el modelo
    y generarán una continuación automática.
    """)

    notas = pianoroll()

    if notas:
        st.success(f"Notas capturadas: {notas}")
        st.markdown("**Secuencia interpretada como estados de la cadena:**")
        st.code(" → ".join(str(n) for n in notas))
    else:
        st.info("Aún no has tocado ninguna nota.")

# ============================================================
# PESTAÑA 3: GENERACIÓN (placeholder)
# ============================================================
with tab_generacion:
    st.header("Generación con Markov (próximamente)")
    st.markdown("""
    Aquí aparecerá la **continuación generada** por la cadena de Markov
    estimada a partir de tus notas.

    El botón estará conectado a:
    1. Estimación de la matriz de transición (con suavizado de Laplace).
    2. Muestreo de la siguiente nota dado el contexto.
    3. Reproducción de la melodía generada.
    """)
    st.button("🎵 Generar continuación (aún no implementado)", disabled=True)

# ============================================================
# PESTAÑA 4: MÉTRICAS (placeholder)
# ============================================================
with tab_metricas:
    st.header("Métricas del modelo (próximamente)")
    st.markdown("""
    Aquí mostraremos:
    - **Entropía** de la matriz de transición (medida de incertidumbre).
    - **Número de estados únicos** vistos.
    - **Número de transiciones** observadas.
    - Evolución de la matriz mientras tocas (gráfico en vivo).
    """)
    st.button("📊 Ver métricas (aún no implementado)", disabled=True)
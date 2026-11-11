import streamlit as st
import numpy as np
import io
import scipy.io.wavfile as wavfile
from collections import defaultdict, Counter

st.set_page_config(page_title="Compositor Markov", layout="wide")

# ============================================================
# 1. SÍNTESIS DE AUDIO (Piano)
# ============================================================
def generate_piano_tone(frequency, duration=1.5, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = (1.0 * np.sin(2 * np.pi * frequency * t) +
            0.5 * np.sin(2 * np.pi * frequency * 2 * t) +
            0.2 * np.sin(2 * np.pi * frequency * 3 * t))
    envelope = np.exp(-4 * t)
    wave = wave * envelope
    wave = wave / np.max(np.abs(wave))
    wave_int16 = (wave * 32767).astype(np.int16)
    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, wave_int16)
    buffer.seek(0)
    return buffer

NOTAS_FRECUENCIAS = {
    "Do4": 261.63, "Do#4": 277.18, "Re4": 293.66, "Re#4": 311.13,
    "Mi4": 329.63, "Fa4": 349.23, "Fa#4": 369.99, "Sol4": 392.00,
    "Sol#4": 415.30, "La4": 440.00, "La#4": 466.16, "Si4": 493.88,
    "Do5": 523.25
}

# ============================================================
# 2. LÓGICA DE CADENAS DE MARKOV
# ============================================================
def construir_matriz_markov(secuencia, alpha=1.0):
    """Cuenta las transiciones y aplica suavizado de Laplace."""
    transiciones = defaultdict(Counter)
    for i in range(len(secuencia) - 1):
        actual = secuencia[i]
        siguiente = secuencia[i+1]
        transiciones[actual][siguiente] += 1
    
    # Aplicar suavizado de Laplace y calcular probabilidades
    matriz = {}
    vocabulario = list(set(secuencia))
    V = len(vocabulario)
    
    for nota, conteos in transiciones.items():
        total = sum(conteos.values())
        matriz[nota] = {}
        for sig_nota in vocabulario:
            # Suavizado: (conteo + alpha) / (total + alpha * V)
            matriz[nota][sig_nota] = (conteos.get(sig_nota, 0) + alpha) / (total + alpha * V)
    
    return matriz, vocabulario

def generar_continuacion(matriz, vocabulario, ultima_nota, longitud=10):
    """Genera notas nuevas muestreando de la matriz de Markov."""
    generado = []
    actual = ultima_nota
    
    for _ in range(longitud):
        if actual in matriz:
            probabilidades = [matriz[actual].get(n, 1e-5) for n in vocabulario]
            # Normalizar por si acaso
            probabilidades = np.array(probabilidades) / np.sum(probabilidades)
            siguiente = np.random.choice(vocabulario, p=probabilidades)
        else:
            siguiente = np.random.choice(vocabulario) # Fallback
        generado.append(siguiente)
        actual = siguiente
        
    return generado

# ============================================================
# 3. ESTADO DE LA SESIÓN
# ============================================================
if "notas_tocadas" not in st.session_state:
    st.session_state.notas_tocadas = []

# ============================================================
# 4. INTERFAZ (PESTAÑAS)
# ============================================================
st.title("🎹 Compositor Estocástico con Cadenas de Markov")

tab_teoria, tab_piano, tab_generacion, tab_metricas = st.tabs([
    "📚 Teoría", "🎹 Piano interactivo", "🎼 Generación (Markov)", "📊 Métricas"
])

# --- PESTAÑA 1: TEORÍA ---
with tab_teoria:
    st.header("¿Qué es una Cadena de Markov?")
    st.markdown(r"""
    Una **Cadena de Markov** es un proceso estocástico que cumple la **propiedad de Markov**:
    > El futuro depende del presente **solo** a través del estado actual, no de la historia pasada.

    Formalmente: $P(X_{t+1} = j \mid X_t = i, X_{t-1} = i_{t-1}, \dots) = P(X_{t+1} = j \mid X_t = i)$
    """)
    st.markdown(r"""
    **Matriz de transición:** Las probabilidades $P_{ij}$ de pasar del estado $i$ al $j$. 
    Se estiman contando transiciones y aplicando **suavizado de Laplace**:
    $\hat{P}_{ij} = \frac{\text{count}(i \to j) + \alpha}{\sum_k \text{count}(i \to k) + \alpha \cdot |S|}$
    """)

# --- PESTAÑA 2: PIANO ---
with tab_piano:
    st.header("Toca algunas notas")
    
    if st.session_state.notas_tocadas:
        st.success(f"Secuencia actual: {' → '.join(st.session_state.notas_tocadas)}")
        if st.button("🗑️ Borrar secuencia"):
            st.session_state.notas_tocadas = []
            st.rerun()
    else:
        st.info("Aún no has tocado ninguna nota.")

    cols = st.columns(7)
    teclas_blancas = ["Do4", "Re4", "Mi4", "Fa4", "Sol4", "La4", "Si4"]
    for i, nota in enumerate(teclas_blancas):
        with cols[i]:
            if st.button(f"🎹 {nota}", use_container_width=True):
                st.session_state.notas_tocadas.append(nota)
                st.audio(generate_piano_tone(NOTAS_FRECUENCIAS[nota]), format="audio/wav")

    cols_negras = st.columns(7)
    teclas_negras = ["", "Do#4", "Re#4", "", "Fa#4", "Sol#4", "La#4"]
    for i, nota in enumerate(teclas_negras):
        with cols_negras[i]:
            if nota:
                if st.button(f"🎵 {nota}", use_container_width=True):
                    st.session_state.notas_tocadas.append(nota)
                    st.audio(generate_piano_tone(NOTAS_FRECUENCIAS[nota]), format="audio/wav")

# --- PESTAÑA 3: GENERACIÓN ---
with tab_generacion:
    st.header("Generación automática con Markov")
    
    if len(st.session_state.notas_tocadas) < 2:
        st.warning("⚠️ Necesitas tocar al menos 2 notas en el piano para que Markov pueda aprender las transiciones.")
    else:
        st.write(f"El modelo ha aprendido de tus **{len(st.session_state.notas_tocadas)} notas**.")
        
        longitud = st.slider("¿Cuántas notas quieres que genere?", 3, 20, 8)
        
        if st.button("✨ Generar continuación"):
            # 1. Construir matriz
            matriz, vocabulario = construir_matriz_markov(st.session_state.notas_tocadas)
            
            # 2. Generar secuencia
            ultima_nota = st.session_state.notas_tocadas[-1]
            generado = generar_continuacion(matriz, vocabulario, ultima_nota, longitud)
            
            st.success(f"Melodía generada: **{' → '.join(generado)}**")
            
            # 3. Reproducir cada nota generada
            st.write("Escucha la melodía generada:")
            for nota in generado:
                if nota in NOTAS_FRECUENCIAS:
                    st.audio(generate_piano_tone(NOTAS_FRECUENCIAS[nota]), format="audio/wav")

# --- PESTAÑA 4: MÉTRICAS ---
with tab_metricas:
    st.header("Análisis del modelo")
    if len(st.session_state.notas_tocadas) < 2:
        st.info("Toca algunas notas en el piano para ver las métricas.")
    else:
        matriz, vocabulario = construir_matriz_markov(st.session_state.notas_tocadas)
        
        st.subheader("Estados detectados")
        st.write(f"Has utilizado **{len(vocabulario)} notas distintas** en tu secuencia.")
        st.write(", ".join(vocabulario))
        
        st.subheader("Matriz de Transición (Probabilidades)")
        st.write("Muestra la probabilidad de pasar de una nota a otra según lo que tocaste. (Suavizado con Laplace α=1.0).")
        
        # Mostrar matriz como texto formateado
        for nota, probs in matriz.items():
            st.write(f"**Si tocas {nota}, ¿qué sigue?**")
            # Ordenar por probabilidad descendente
            probs_ordenadas = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            for sig_nota, prob in probs_ordenadas:
                if prob > 0.05: # Solo mostrar las más probables
                    st.progress(float(prob), text=f"{sig_nota}: {prob*100:.1f}%")
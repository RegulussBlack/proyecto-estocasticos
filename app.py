import streamlit as st
import numpy as np
import io
import scipy.io.wavfile as wavfile

st.set_page_config(page_title="Compositor Markov", layout="wide")

# ============================================================
# FUNCIÓN PARA SINTETIZAR EL SONIDO DEL PIANO
# ============================================================
def generate_piano_tone(frequency, duration=1.5, sample_rate=44100):
    """
    Genera una onda de audio que simula una nota de piano.
    Usa una onda fundamental + armónicos + decaimiento exponencial.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Onda compuesta (fundamental + armónicos para darle timbre)
    wave = (1.0 * np.sin(2 * np.pi * frequency * t) +
            0.5 * np.sin(2 * np.pi * frequency * 2 * t) +
            0.2 * np.sin(2 * np.pi * frequency * 3 * t))
    
    # Envolvente exponencial (simula el decaimiento del piano)
    envelope = np.exp(-4 * t)
    wave = wave * envelope
    
    # Normalizar y convertir a 16-bit PCM
    wave = wave / np.max(np.abs(wave))
    wave_int16 = (wave * 32767).astype(np.int16)
    
    # Guardar en un buffer de memoria
    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, wave_int16)
    buffer.seek(0)
    return buffer

# Diccionario de frecuencias (Notas de Do4 a Do5)
NOTAS_FRECUENCIAS = {
    "Do4": 261.63, "Do#4": 277.18, "Re4": 293.66, "Re#4": 311.13,
    "Mi4": 329.63, "Fa4": 349.23, "Fa#4": 369.99, "Sol4": 392.00,
    "Sol#4": 415.30, "La4": 440.00, "La#4": 466.16, "Si4": 493.88,
    "Do5": 523.25
}

# ============================================================
# GESTIÓN DE ESTADO (Para guardar las notas que tocas)
# ============================================================
if "notas_tocadas" not in st.session_state:
    st.session_state.notas_tocadas = []

# ============================================================
# INTERFAZ PRINCIPAL
# ============================================================
st.title("🎹 Compositor Estocástico con Cadenas de Markov")
st.caption("Proyecto de Procesos Estocásticos y Series de Tiempo")

tab_teoria, tab_piano = st.tabs([
    "📚 Teoría: Cadenas de Markov",
    "🎹 Piano interactivo"
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
    st.info("👉 Ve a la pestaña **🎹 Piano interactivo** para tocar notas y ver el modelo en acción.")

# --- PESTAÑA 2: PIANO ---
with tab_piano:
    st.header("Toca algunas notas")
    st.write("Haz clic en las teclas para escuchar el sonido sintetizado. Las notas se guardarán para Markov.")

    # Mostrar las notas que se han tocado
    if st.session_state.notas_tocadas:
        st.success(f"Secuencia actual: {' → '.join(st.session_state.notas_tocadas)}")
        if st.button("🗑️ Borrar secuencia"):
            st.session_state.notas_tocadas = []
            st.rerun()
    else:
        st.info("Aún no has tocado ninguna nota.")

    # Renderizar el teclado (Dividido en columnas)
    st.subheader("Teclado Musical (Do4 a Do5)")
    
    cols = st.columns(7) # 7 columnas para las teclas blancas
    
    # Teclas blancas
    teclas_blancas = ["Do4", "Re4", "Mi4", "Fa4", "Sol4", "La4", "Si4"]
    for i, nota in enumerate(teclas_blancas):
        with cols[i]:
            if st.button(f"🎹 {nota}", use_container_width=True):
                st.session_state.notas_tocadas.append(nota)
                freq = NOTAS_FRECUENCIAS[nota]
                audio_buffer = generate_piano_tone(freq)
                st.audio(audio_buffer, format="audio/wav")

    # Teclas negras (Simuladas en una fila extra, un poco más pequeñas)
    cols_negras = st.columns(7)
    teclas_negras = ["", "Do#4", "Re#4", "", "Fa#4", "Sol#4", "La#4"]
    for i, nota in enumerate(teclas_negras):
        with cols_negras[i]:
            if nota:
                if st.button(f"🎵 {nota}", use_container_width=True):
                    st.session_state.notas_tocadas.append(nota)
                    freq = NOTAS_FRECUENCIAS[nota]
                    audio_buffer = generate_piano_tone(freq)
                    st.audio(audio_buffer, format="audio/wav")

    st.divider()
    st.write("*(En el siguiente paso conectaremos esta secuencia con la matriz de Markov para generar la melodía).*")
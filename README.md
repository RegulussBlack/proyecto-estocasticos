# 🎹 Compositor Estocástico con Cadenas de Markov

Proyecto de Ciencia de Datos e IA: aprende el estilo de melodías con cadenas de Markov de orden *k*,
genera melodías y progresiones de acordes, y evalúa el modelo con métricas de ML.

## Ejecutar localmente
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Desplegar en Streamlit Community Cloud
1. Sube esta carpeta a un repositorio de GitHub (`app.py` en la raíz).
2. En share.streamlit.io elige el repo y `app.py`.
3. En *Advanced settings* selecciona **Python 3.11**.

## Estructura
```
app.py              Interfaz (8 pestañas)
core/notas.py       Nomenclatura, MIDI, frecuencias
core/audio.py       Síntesis aditiva (notas y acordes), WAV y exportación MIDI
core/markov.py      Modelo orden k, suavizado jerárquico, muestreo, evaluación
core/teoria.py      Escalas, intervalos, tonalidad (Krumhansl-Schmuckler), acordes, progresiones
core/corpus.py      Melodías de entrenamiento
```

## Ideas para seguir ampliándolo
- Duraciones como parte del estado (altura × duración).
- Subir archivos MIDI propios (`mido`/`pretty_midi`) como corpus.
- Modelo oculto de Markov (acordes como estados latentes) con `hmmlearn`.
- Comparar con un LSTM pequeño (PyTorch) usando la misma perplejidad.

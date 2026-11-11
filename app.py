"""🎹 Compositor Estocástico con Cadenas de Markov — proyecto de Ciencia de Datos / IA."""
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from core import audio, markov as mk, teoria as te
from core.corpus import CORPUS
from core.notas import (NOMBRES, NOTAS_ORDENADAS, nombre_a_midi, midi_a_nombre, midi_a_freq,
                        parsear_secuencia, clave_orden)

st.set_page_config(page_title="Compositor Markov", page_icon="🎹", layout="wide")

st.markdown("""
<style>
div.stButton > button {width: 100%;}
div.stDownloadButton > button {width: 100%;}
.block-container {padding-top: 2rem;}
</style>
""", unsafe_allow_html=True)

# ============================================================ ESTADO
st.session_state.setdefault("notas", [])
st.session_state.setdefault("ultimo_audio", None)
st.session_state.setdefault("reproducir", False)
st.session_state.setdefault("gen", None)
st.session_state.setdefault("msg", None)


def tocar(nota):
    st.session_state.notas.append(nota)
    st.session_state.ultimo_audio = audio.tono(nota)
    st.session_state.reproducir = True


def deshacer():
    if st.session_state.notas:
        st.session_state.notas.pop()


def borrar():
    st.session_state.notas = []
    st.session_state.ultimo_audio = None


def cargar_texto():
    validas, invalidas = parsear_secuencia(st.session_state.get("texto_seq", ""))
    st.session_state.notas = validas
    st.session_state.msg = (f"Se ignoraron tokens no válidos: {', '.join(invalidas)}" if invalidas else None)


def cargar_pieza():
    st.session_state.notas = list(CORPUS[st.session_state.pieza_sel])


# ============================================================ GRÁFICOS
def heatmap(df, x_title, y_title, scheme="viridis", fmt=".2f", texto=True, altura=None, invertir=False):
    largo = df.stack().rename("valor").reset_index()
    largo.columns = ["y", "x", "valor"]
    base = alt.Chart(largo).encode(
        x=alt.X("x:O", sort=list(df.columns), title=x_title),
        y=alt.Y("y:O", sort=list(df.index), title=y_title))
    capa = base.mark_rect().encode(
        color=alt.Color("valor:Q", scale=alt.Scale(scheme=scheme, reverse=invertir), title=None),
        tooltip=["y", "x", alt.Tooltip("valor:Q", format=fmt)])
    if texto:
        umbral = (largo["valor"].max() + largo["valor"].min()) / 2
        capa += base.mark_text(fontSize=10).encode(
            text=alt.Text("valor:Q", format=fmt),
            color=alt.condition(alt.datum.valor > umbral, alt.value("white"), alt.value("black")))
    props = dict(width="container")
    if altura:
        props["height"] = altura
    return capa.properties(**props)


def piano_roll(notas):
    lo = min(nombre_a_midi(n) for n in notas)
    hi = max(nombre_a_midi(n) for n in notas)
    dominio = [n for n in NOTAS_ORDENADAS if lo <= nombre_a_midi(n) <= hi][::-1]
    df = pd.DataFrame({"inicio": range(len(notas)), "fin": [i + 0.92 for i in range(len(notas))],
                       "nota": notas})
    return alt.Chart(df).mark_rect(cornerRadius=3).encode(
        x=alt.X("inicio:Q", title="Tiempo (negras)"), x2="fin:Q",
        y=alt.Y("nota:O", sort=dominio, scale=alt.Scale(domain=dominio), title=None),
        color=alt.Color("nota:N", legend=None, scale=alt.Scale(scheme="tableau20")),
        tooltip=["nota"]).properties(width="container", height=max(120, 18 * len(dominio)))


def reproductor(wav, midi, nombre):
    st.audio(wav, format="audio/wav")
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Descargar WAV", wav, f"{nombre}.wav", "audio/wav", key=f"wav_{nombre}")
    c2.download_button("⬇️ Descargar MIDI", midi, f"{nombre}.mid", "audio/midi", key=f"mid_{nombre}")


# ============================================================ SIDEBAR
with st.sidebar:
    st.title("⚙️ Configuración")
    st.subheader("Modelo")
    representacion = st.radio(
        "Representación de la melodía", ["notas", "intervalos"],
        help="**Notas**: los estados son las alturas (Do4, Re4…). **Intervalos**: los estados son los "
             "saltos entre notas (+2, -1…), lo que hace al modelo invariante a la transposición.")
    orden = st.slider("Orden de la cadena (k)", 1, 4, 2,
                      help="Cuántas notas previas recuerda el modelo. k=1 es la cadena de Markov clásica.")
    alpha = st.select_slider("Suavizado α", options=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0], value=0.25,
                             help="α alto → más aleatoriedad / menos fidelidad al corpus.")
    st.subheader("Datos de entrenamiento")
    sel = st.multiselect("Piezas del corpus", list(CORPUS),
                         default=["Estrellita (Twinkle Twinkle)", "Oda a la Alegría (Beethoven)"])
    incluir = st.checkbox("Incluir mis notas del piano", value=True)
    peso = st.slider("Peso de mis notas", 1, 10, 1, disabled=not incluir,
                     help="Cuántas veces cuentan tus notas frente al corpus.")
    st.caption("💡 Todo el sitio reacciona a estos controles.")


def piezas_entrenamiento(con_peso=True):
    d = {k: CORPUS[k] for k in sel}
    if incluir and len(st.session_state.notas) >= 2:
        for i in range(peso if con_peso else 1):
            d[f"★ Mis notas #{i + 1}"] = list(st.session_state.notas)
    return d


PIEZAS = piezas_entrenamiento()
SECS_NOTAS = list(PIEZAS.values())
SECS_TOK = [mk.tokenizar(s, representacion) for s in SECS_NOTAS]
VOCAB = sorted({t for s in SECS_TOK for t in s}, key=clave_orden)

# ============================================================ CABECERA
st.title("🎹 Compositor Estocástico con Cadenas de Markov")
st.caption("Un proyecto de Ciencia de Datos e IA: modelado probabilístico de secuencias, síntesis de audio, "
           "armonía, teoría musical y evaluación con métricas de ML.")

tabs = st.tabs(["🏠 Proyecto", "📚 Teoría Markov", "🎵 Teoría musical", "🎹 Piano",
                "🎼 Melodías", "🎶 Armonía", "📊 Análisis", "🧪 Evaluación"])
t_proy, t_teo, t_music, t_piano, t_gen, t_arm, t_ana, t_eval = tabs

# ============================================================ 1. PROYECTO
with t_proy:
    st.header("Sobre el proyecto")
    st.markdown("""
**Problema.** ¿Se puede *aprender* el estilo de una melodía a partir de ejemplos y generar música nueva
que suene coherente, sin reglas escritas a mano?

**Enfoque.** Se modela la música como un **proceso estocástico**: cada nota se trata como una variable
aleatoria que depende de las notas anteriores. Se estima la dependencia contando transiciones en un corpus
(*aprendizaje estadístico*) y se genera música **muestreando** de las distribuciones aprendidas.

**Qué hace distinto a este proyecto de una demo simple**
- Cadenas de Markov de **orden k** con **suavizado jerárquico** y *backoff* (no solo orden 1).
- Dos representaciones: **notas** e **intervalos** (invariante a la transposición).
- Control creativo: **temperatura**, **top-k** y restricción a una **tonalidad** detectada automáticamente.
- **Análisis formal** de la cadena: distribución estacionaria, tasa de entropía, tiempo de mezcla.
- **Teoría musical aplicada**: escalas, intervalos, acordes y tonalidad detectada automáticamente.
- **Armonía**: una segunda cadena de Markov sobre **progresiones de acordes** (I–V–vi–IV, blues, jazz…).
- **Evaluación rigurosa**: perplejidad y accuracy top-k con partición temporal y *leave-one-piece-out*.
- Exportación a **WAV** y **MIDI**.
""")
    st.subheader("Pipeline")
    st.graphviz_chart("""
digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef2ff", fontname="Helvetica"];
  A [label="Corpus +\\nnotas del usuario"]; B [label="Tokenización\\n(notas / intervalos)"];
  C [label="Estimación\\nP(w | contexto)\\n+ suavizado"]; D [label="Muestreo\\n(temperatura, top-k,\\ntonalidad)"];
  E [label="Síntesis\\nWAV / MIDI"]; F [label="Evaluación\\nperplejidad, top-k"];
  G [label="Teoría musical:\\ntonalidad, escalas,\\nacordes"];
  A -> B -> C -> D -> E; C -> F; A -> G; }
""")
    st.subheader("Cómo usarlo")
    st.markdown("""
1. En la barra lateral elige las **piezas del corpus** y el **orden** del modelo.
2. En **🎵 Teoría musical** repasa escalas, intervalos y acordes (con ejemplos sonoros).
3. En **🎹 Piano** toca tu propia melodía; en **🎼 Melodías** genera una nueva y descárgala.
4. En **🎶 Armonía** genera progresiones de acordes; en **📊 Análisis** inspecciona qué aprendió la cadena.
5. En **🧪 Evaluación** compara órdenes y valores de α con métricas objetivas.
""")
    c1, c2, c3 = st.columns(3)
    c1.metric("Piezas en el corpus", len(CORPUS))
    c2.metric("Piezas seleccionadas", len(PIEZAS))
    c3.metric("Estados (|V|)", len(VOCAB))
    with st.expander("⚠️ Limitaciones y trabajo futuro"):
        st.markdown("""
- El modelo **no representa ritmo ni armonía**: todas las notas duran una negra y la melodía es monofónica.
- El corpus es pequeño; con pocos datos los órdenes altos **memorizan** (ver Evaluación).
- Una cadena de Markov no captura estructura de largo plazo (frases, forma ABA).
- **Trabajo futuro:** modelar duraciones (estados = altura × duración), modelos ocultos de Markov con
  acordes como estados latentes, redes recurrentes/Transformers, cargar archivos MIDI propios.
""")
    with st.expander("📖 Referencias"):
        st.markdown("""
- Markov, A. A. (1906). *Extension of the law of large numbers to dependent quantities.*
- Shannon, C. E. (1948). *A Mathematical Theory of Communication.* Bell System Technical Journal.
- Hiller, L. & Isaacson, L. (1957). *Illiac Suite* — primera obra compuesta por computadora con métodos estocásticos.
- Chen, S. & Goodman, J. (1999). *An empirical study of smoothing techniques for language modeling.*
- Krumhansl, C. (1990). *Cognitive Foundations of Musical Pitch.* Oxford University Press.
- Jurafsky, D. & Martin, J. *Speech and Language Processing* (capítulo de modelos de lenguaje n-grama).
""")

# ============================================================ 2. TEORÍA
with t_teo:
    st.header("Fundamentos teóricos")
    with st.expander("1️⃣ Procesos estocásticos y cadenas de Markov", expanded=True):
        st.markdown(r"""
Un **proceso estocástico** es una familia de variables aleatorias $\{X_t\}_{t\ge 0}$. Aquí, $X_t$ es la
nota número $t$ de la melodía y el conjunto de valores posibles $S$ (el **espacio de estados**) son las notas.

Una **cadena de Markov** cumple la **propiedad de Markov**: el futuro depende del pasado solo a través
del presente.""")
        st.latex(r"P(X_{t+1}=j \mid X_t=i, X_{t-1}, \dots, X_0) = P(X_{t+1}=j \mid X_t=i) = P_{ij}")
        st.markdown(r"""
La **matriz de transición** $P=(P_{ij})$ es *estocástica por filas*: $P_{ij}\ge 0$ y $\sum_j P_{ij}=1$.
La distribución tras $n$ pasos es $\mu_n=\mu_0 P^n$.""")
    with st.expander("2️⃣ Estimación: máxima verosimilitud y suavizado de Laplace"):
        st.markdown(r"""
Por **máxima verosimilitud**, $\hat P_{ij}=\dfrac{c(i\to j)}{c(i)}$. Problema: si una transición nunca se
vio, su probabilidad es **0** y la log-verosimilitud se vuelve $-\infty$. El **suavizado de Laplace (aditivo)**
añade un pseudo-conteo $\alpha$ (equivale a un prior de Dirichlet simétrico):""")
        st.latex(r"\hat P_{ij}=\frac{c(i\to j)+\alpha}{c(i)+\alpha\,|S|}")
        st.markdown("Con $\\alpha\\to 0$ el modelo es fiel al corpus; con $\\alpha\\to\\infty$ tiende a la distribución uniforme (puro azar).")
    with st.expander("3️⃣ Orden k y backoff jerárquico"):
        st.markdown(r"""
Una cadena de **orden k** condiciona en las últimas $k$ notas: $P(w_t \mid w_{t-1},\dots,w_{t-k})$. Es
equivalente a una cadena de orden 1 sobre el espacio de estados ampliado $S^k$ (**n-gramas**).
El compromiso es el clásico **sesgo–varianza**: más orden → más contexto, pero **contextos más raros**
(maldición de la dimensionalidad) y riesgo de memorizar el corpus.

Este proyecto usa un suavizado **jerárquico**, donde la distribución de orden $k-1$ actúa como prior del orden $k$:""")
        st.latex(r"P_k(w\mid c)=\frac{c(c,w)+\beta\,P_{k-1}(w\mid c')}{c(c)+\beta},\qquad \beta=\alpha|S|")
        st.markdown("donde $c'$ es el contexto sin su nota más antigua. Si el contexto nunca se vio, se hace *backoff* al orden inferior.")
    with st.expander("4️⃣ Propiedades de la cadena: estacionariedad, entropía, mezcla"):
        st.markdown(r"""
- **Distribución estacionaria** $\pi$: cumple $\pi P=\pi$. Es el vector propio izquierdo de $P$ con autovalor 1.
  Si la cadena es irreducible y aperiódica, converge a $\pi$ desde cualquier estado (teorema ergódico). Gracias al
  suavizado, **todas las entradas son positivas**, así que esto siempre se cumple. Interpretación musical: la
  "nota más frecuente a largo plazo".
- **Tasa de entropía** (Shannon): la incertidumbre media por nota, en bits.""")
        st.latex(r"H=-\sum_i \pi_i\sum_j P_{ij}\log_2 P_{ij}")
        st.markdown(r"""
  Baja entropía → melodía predecible; alta → casi aleatoria. El máximo es $\log_2|S|$.
- **Velocidad de mezcla**: gobernada por el segundo mayor autovalor en módulo $|\lambda_2|$. Cuanto más pequeño,
  más rápido "olvida" la cadena su estado inicial: $\|\mu_n-\pi\|_{TV}\sim|\lambda_2|^n$.""")
    with st.expander("5️⃣ Generación: muestreo, temperatura y top-k"):
        st.markdown(r"""
Se genera nota a nota muestreando de $P(\cdot\mid\text{contexto})$. La **temperatura** $T$ reescala la distribución
(como en los modelos de lenguaje):""")
        st.latex(r"p_i(T)=\frac{p_i^{1/T}}{\sum_j p_j^{1/T}}")
        st.markdown("$T<1$ → conservador (casi siempre la nota más probable); $T>1$ → más sorpresa. **Top-k** descarta todo salvo las $k$ notas más probables.")
    with st.expander("6️⃣ Notas vs. intervalos (invariancia a la transposición)"):
        st.markdown(r"""
Una melodía en Do mayor y la misma en Re mayor comparten **intervalos** ($+2,+2,-1,\dots$) pero no notas.
Modelar intervalos agrupa estadísticamente piezas de distintas tonalidades y **aprovecha mejor datos escasos**.
Se reconstruye la melodía sumando intervalos a una nota inicial y plegando por octavas al rango disponible (Do3–Si5).""")
    with st.expander("7️⃣ Evaluación: perplejidad"):
        st.markdown(r"""
La **perplejidad** mide qué tan "sorprendido" está el modelo ante datos no vistos (menor = mejor):""")
        st.latex(r"PP=\exp\!\Big(-\tfrac1N\sum_{t=1}^N\ln P(w_t\mid w_{t-k:t-1})\Big)")
        st.markdown(r"""
Se interpreta como el **número efectivo de opciones** entre las que duda el modelo. Un modelo aleatorio uniforme
tiene $PP=|S|$. También se reporta la **accuracy top-1 / top-3**: ¿la nota real estaba entre las 1 o 3 más probables?""")
    with st.expander("8️⃣ Markov aplicado a la armonía"):
        st.markdown(r"""
Los acordes también forman secuencias con dependencia local: en la **armonía funcional** cada acorde tiende a
resolver en otro (el dominante $V$ "quiere" ir a la tónica $I$). Si usamos como estados los **grados** de la
tonalidad (I, ii, iii, IV, V, vi, vii°), la matriz de transición **aprende esas tendencias** a partir de ejemplos
(pop, blues, jazz, clásica) y la cadena genera progresiones nuevas que suenan "correctas". Como los grados son
relativos a la tónica, la misma progresión se puede transportar a cualquier tonalidad.""")

# ============================================================ 3. TEORÍA MUSICAL
with t_music:
    st.header("Teoría musical esencial")
    st.write("Lo necesario para entender **qué está modelando** la cadena de Markov. "
             "Las secciones con 🔊 se pueden escuchar.")

    with st.expander("1️⃣ Sonido, altura y frecuencia", expanded=True):
        st.markdown(r"""
Un sonido musical es una vibración periódica; su **altura** (grave ↔ agudo) depende de la **frecuencia** $f$ en
hercios (Hz). La referencia universal es **La4 = 440 Hz**. Subir una **octava** duplica la frecuencia
(La5 = 880 Hz) y las notas con el mismo nombre en distintas octavas se perciben como "la misma nota".

En la música occidental la octava se divide en **12 semitonos iguales** (*temperamento igual*), de modo que cada
semitono multiplica la frecuencia por $2^{1/12}\approx 1.0595$. Numerando las notas con el estándar **MIDI**
(Do4 = 60, La4 = 69):""")
        st.latex(r"f(m)=440\cdot 2^{(m-69)/12}")
        st.markdown("""
Las 12 clases de altura son: **Do, Do♯, Re, Re♯, Mi, Fa, Fa♯, Sol, Sol♯, La, La♯, Si**. Las teclas blancas son las
7 notas naturales; las negras son las alteraciones (♯ sube un semitono, ♭ lo baja; Do♯ y Re♭ suenan igual: **enarmonía**).""")
        tabla = pd.DataFrame({"Nota": [f"{n}4" for n in NOMBRES],
                              "MIDI": [60 + i for i in range(12)],
                              "Frecuencia (Hz)": [round(midi_a_freq(60 + i), 2) for i in range(12)]}).set_index("Nota")
        st.dataframe(tabla)

    with st.expander("2️⃣ Intervalos: la distancia entre notas"):
        st.markdown(r"""
Un **intervalo** mide la distancia entre dos notas, en **semitonos**. Cada intervalo tiene un nombre y un carácter
(consonante o disonante). En **entonación justa** los intervalos más consonantes corresponden a razones de números
enteros pequeños (la quinta justa es 3:2); el temperamento igual las aproxima con $2^{s/12}$.

Esto es clave para el proyecto: el modelo por **intervalos** aprende justamente estas distancias, y por eso una
melodía transportada a otra tonalidad "suena igual" para el modelo.""")
        tabla = pd.DataFrame([{"Semitonos": s_, "Intervalo": n, "Razón justa": r, "Razón temperada": round(2 ** (s_ / 12), 4),
                               "Carácter": c} for s_, n, r, c in te.INTERVALOS]).set_index("Semitonos")
        st.dataframe(tabla)
        c1, c2 = st.columns([1, 2])
        sem = c1.slider("🔊 Escuchar un intervalo (semitonos)", 0, 12, 7, key="iv_sem")
        nombre_iv = te.INTERVALOS[sem][1]
        base_m = 60
        par = [midi_a_nombre(base_m), midi_a_nombre(base_m + sem)]
        c2.write(f"**{nombre_iv}**: {par[0]} → {par[1]}  ·  {te.INTERVALOS[sem][3]}")
        if c2.button("▶️ Melódico (una nota tras otra)", key="iv_mel"):
            st.audio(audio.sintetizar(par, 90), format="audio/wav", autoplay=True)
        if c2.button("▶️ Armónico (a la vez)", key="iv_arm"):
            st.audio(audio.sintetizar_acordes([[base_m, base_m + sem]], 60, 3), format="audio/wav", autoplay=True)

    with st.expander("3️⃣ Escalas y modos"):
        st.markdown(r"""
Una **escala** es un subconjunto ordenado de notas que define el "vocabulario" de una pieza. Se describe por su
patrón de **tonos (T = 2 semitonos)** y **semitonos (S = 1)**:

| Escala | Patrón | Carácter |
|---|---|---|
| **Mayor** | T T S T T T S | brillante, estable |
| **Menor natural** | T S T T S T T | oscuro, melancólico |
| **Menor armónica** | T S T T S 1½ S | tensión "exótica" (sensible elevada) |
| **Pentatónica** | 5 notas, sin semitonos | muy consonante; rock, folk, blues |
| **Blues** | pentatónica menor + ♭5 | la "blue note" |

Los **modos** (dórico, mixolidio…) son rotaciones de la escala mayor. La **tónica** es la nota "de reposo" a la que
la melodía tiende a volver. Esa **gravedad tonal** es justo lo que la cadena de Markov captura estadísticamente:
tras la sensible (nota 7) es muy probable la tónica.""")

    with st.expander("4️⃣ Tonalidad, armadura y círculo de quintas"):
        st.markdown("""
Una **tonalidad** es la combinación de una tónica y una escala (p. ej. *Do mayor*, *La menor*). Cada tonalidad mayor
comparte las mismas notas con su **relativa menor** (3 semitonos por debajo). La **armadura** indica cuántos ♯ o ♭
lleva la tonalidad. Al ordenar las tonalidades de **quinta en quinta** se forma el **círculo de quintas**: las
tonalidades vecinas comparten casi todas las notas, por eso modulan con facilidad.""")
        st.dataframe(pd.DataFrame(te.CIRCULO_QUINTAS, columns=["Mayor", "Relativa menor", "Armadura"]).set_index("Mayor"))
        st.markdown("""
**Detección automática (Krumhansl–Schmuckler).** En el proyecto, la tonalidad de una melodía se estima
correlacionando su histograma de notas con *perfiles psicológicos* de las 24 tonalidades (los perfiles
Krumhansl–Kessler, obtenidos con experimentos de oyentes). Se usa para restringir la generación a notas de la escala.
Puede confundir una tonalidad con su relativa porque comparten las mismas notas.""")

    with st.expander("5️⃣ Grados, funciones tonales y acordes"):
        st.markdown(r"""
Cada nota de la escala es un **grado** (I a VII). Sobre cada grado se construye una **tríada** (acorde de 3 notas)
apilando terceras **dentro de la escala**. En una tonalidad mayor resultan:
mayor (I, IV, V), menor (ii, iii, vi) y disminuido (vii°). Convención: **mayúscula = acorde mayor**, minúscula = menor,
° = disminuido.

| Calidad | Estructura (semitonos) | Sonido |
|---|---|---|
| Mayor | 0 – 4 – 7 | alegre, estable |
| Menor | 0 – 3 – 7 | triste, suave |
| Disminuido | 0 – 3 – 6 | tenso, inestable |
| Aumentado | 0 – 4 – 8 | misterioso, suspendido |

**Funciones tonales** (armonía funcional): **Tónica (T)** da reposo (I, iii, vi); **Subdominante (S)** se aleja del
reposo (ii, IV); **Dominante (D)** crea tensión que pide resolver (V, vii°). La secuencia típica es **T → S → D → T**,
y es la regularidad que aprenderá la cadena de Markov de la pestaña 🎶 Armonía.""")

    with st.expander("6️⃣ Ritmo y limitaciones de este proyecto"):
        st.markdown("""
El **pulso** es la unidad básica de tiempo y el **tempo** se mide en BPM (pulsos por minuto). Agrupar pulsos en
**compases** crea acentos (4/4, 3/4...). En este proyecto todas las notas duran **una negra** y no hay compases:
modelamos *qué* nota viene después, no *cuándo* ni *cuánto dura*. Añadir duraciones como parte del estado
(altura × duración) es una extensión natural.""")

    st.divider()
    st.subheader("🔬 Laboratorio de escalas y acordes")
    c1, c2, c3 = st.columns(3)
    ton_lab = c1.selectbox("Tónica", NOMBRES, key="lab_ton")
    esc_lab = c2.selectbox("Escala", list(te.ESCALAS), key="lab_esc")
    bpm_lab = c3.slider("Tempo (BPM)", 60, 200, 120, key="lab_bpm")
    tpc = NOMBRES.index(ton_lab)
    offs = te.ESCALAS[esc_lab]
    midis = [60 + tpc + o for o in offs] + [72 + tpc]
    notas_esc = [midi_a_nombre(m) for m in midis]
    st.success("  →  ".join(notas_esc))
    pasos = [b - a for a, b in zip(offs, offs[1:] + [12])]
    st.caption("Distancia en semitonos entre notas consecutivas: " + " · ".join(map(str, pasos))
               + "  (2 = tono, 1 = semitono)")
    if st.button("▶️ Escuchar la escala", key="lab_play"):
        st.audio(audio.sintetizar(notas_esc, bpm_lab), format="audio/wav", autoplay=True)

    modo_lab = st.radio("Acordes diatónicos de la tonalidad", ["mayor", "menor"], horizontal=True, key="lab_modo")
    ac_lab = te.acordes_diatonicos(tpc, modo_lab)
    st.dataframe(pd.DataFrame([{"Grado": a["grado"], "Acorde": a["romano"], "Nombre": a["nombre"],
                                "Notas": " - ".join(a["notas"]), "Calidad": a["calidad"],
                                "Función": a["funcion"]} for a in ac_lab.values()]).set_index("Grado"))
    elegido = st.selectbox("🔊 Escuchar un acorde", list(ac_lab), key="lab_ac",
                           format_func=lambda r: f"{r} · {ac_lab[r]['nombre']} ({ac_lab[r]['funcion']})")
    if st.button("▶️ Tocar acorde", key="lab_ac_play"):
        st.audio(audio.sintetizar_acordes([ac_lab[elegido]["midi"]], 60, 3), format="audio/wav", autoplay=True)

# ============================================================ 3. PIANO
with t_piano:
    st.header("Toca tu melodía")
    izq, der = st.columns([3, 2])
    with izq:
        octava = st.radio("Octava", [3, 4, 5], index=1, horizontal=True)
        st.caption("Haz clic en una tecla: se escucha y se agrega a tu secuencia.")
        blancas = ["Do", "Re", "Mi", "Fa", "Sol", "La", "Si"]
        negras = ["", "Do#", "Re#", "", "Fa#", "Sol#", "La#"]
        cols = st.columns(7)
        for i, n in enumerate(negras):
            if n:
                cols[i].button(f"♯ {n}{octava}", key=f"n_{n}{octava}", on_click=tocar, args=(f"{n}{octava}",))
        cols = st.columns(7)
        for i, n in enumerate(blancas):
            cols[i].button(f"{n}{octava}", key=f"b_{n}{octava}", on_click=tocar, args=(f"{n}{octava}",))
        if st.session_state.ultimo_audio:
            auto = st.session_state.reproducir
            st.session_state.reproducir = False
            st.audio(st.session_state.ultimo_audio, format="audio/wav", autoplay=auto)
    with der:
        st.subheader("Tu secuencia")
        notas = st.session_state.notas
        if notas:
            vista = (["…"] if len(notas) > 40 else []) + notas[-40:]
            st.success(" → ".join(vista))
            st.caption(f"{len(notas)} notas")
        else:
            st.info("Aún no has tocado ninguna nota.")
        b1, b2 = st.columns(2)
        b1.button("↩️ Deshacer", on_click=deshacer, disabled=not notas)
        b2.button("🗑️ Borrar todo", on_click=borrar, disabled=not notas)
        bpm_seq = st.slider("Tempo (BPM)", 60, 200, 110, key="bpm_seq")
        if st.button("▶️ Reproducir mi secuencia", disabled=len(notas) < 1):
            reproductor(audio.sintetizar(notas, bpm_seq), audio.a_midi(notas, bpm_seq), "mi_secuencia")
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.text_area("O pega una secuencia (ej: `Do4 Re4 Mi4, Fa4 → Sol4`)", key="texto_seq", height=90)
        st.button("📥 Cargar texto", on_click=cargar_texto)
        if st.session_state.msg:
            st.warning(st.session_state.msg)
    with c2:
        st.selectbox("O carga una pieza del corpus como tu secuencia", list(CORPUS), key="pieza_sel")
        st.button("📥 Cargar pieza", on_click=cargar_pieza)

# ============================================================ 4. GENERACIÓN
with t_gen:
    st.header("Generación con Markov")
    if not SECS_NOTAS:
        st.warning("Selecciona al menos una pieza del corpus (barra lateral) o toca 2+ notas en el piano.")
    else:
        st.write(f"Modelo de **orden {orden}** sobre **{representacion}**, entrenado con "
                 f"**{len(SECS_NOTAS)} pieza(s)** y **{sum(len(s) for s in SECS_TOK)} tokens**.")
        c1, c2, c3 = st.columns(3)
        longitud = c1.slider("Notas a generar", 4, 64, 24)
        temperatura = c2.slider("Temperatura", 0.2, 2.0, 1.0, 0.1,
                                help="<1 conservador · 1 fiel al modelo · >1 creativo/caótico")
        top_k = c3.slider("Top-k (0 = sin límite)", 0, 12, 0)
        c4, c5, c6 = st.columns(3)
        bpm = c4.slider("Tempo (BPM)", 60, 200, 110)
        semilla_alea = c5.number_input("Semilla aleatoria (0 = distinta cada vez)", 0, 10**6, 0)
        origen = c6.radio("Punto de partida", ["Final de mis notas", "Elegir nota"],
                          disabled=not st.session_state.notas)
        nota_ini = st.selectbox("Nota inicial", NOTAS_ORDENADAS, index=NOTAS_ORDENADAS.index("Do4"),
                                disabled=(origen == "Final de mis notas" and bool(st.session_state.notas)))
        usar_ton = False
        pcs = None
        if representacion == "notas":
            base = st.session_state.notas if len(st.session_state.notas) >= 4 else max(SECS_NOTAS, key=len)
            ton, modo, r, pcs_det = te.detectar_tonalidad(base)
            usar_ton = st.checkbox(f"🎼 Restringir a la tonalidad detectada: {te.nombre_tonalidad(ton, modo)} "
                                   f"(r = {r:.2f})")
            pcs = pcs_det if usar_ton else None

        if st.button("✨ Generar melodía", type="primary"):
            rng = np.random.default_rng(int(semilla_alea) or None)
            if st.session_state.notas and origen == "Final de mis notas":
                semilla = st.session_state.notas[-(orden + 1):]
            else:
                semilla = [nota_ini]
            gen, modelo = mk.generar_melodia(SECS_NOTAS, orden, alpha, representacion, longitud,
                                             temperatura, top_k, semilla, pcs, rng)
            ppl = None
            if representacion == "notas":
                ll, n, _, _ = modelo.evaluar_segmento(semilla + gen, len(semilla))
                ppl = float(np.exp(-ll / n))
            st.session_state.gen = dict(
                notas=gen, semilla=semilla, bpm=bpm, ppl=ppl,
                nov=mk.novedad(semilla + gen, SECS_NOTAS, n=orden + 1),
                wav=audio.sintetizar(gen, bpm), midi=audio.a_midi(gen, bpm))

        g = st.session_state.gen
        if g:
            st.success("**Secuencia generada:** " + " → ".join(g["notas"]))
            m1, m2, m3 = st.columns(3)
            m1.metric("Notas generadas", len(g["notas"]))
            m2.metric("Novedad", f"{g['nov']:.0%}" if not np.isnan(g["nov"]) else "—",
                      help="% de n-gramas (n = orden+1) de la melodía que NO aparecen literalmente en el corpus. "
                           "100% = totalmente original; 0% = copia.")
            m3.metric("Perplejidad bajo el modelo", f"{g['ppl']:.2f}" if g["ppl"] else "—",
                      help="Qué tan probable es la melodía para el propio modelo (solo en modo notas).")
            st.altair_chart(piano_roll(g["notas"]))
            reproductor(g["wav"], g["midi"], "melodia_markov")

# ============================================================ 5. ARMONÍA
with t_arm:
    st.header("Progresiones de acordes con Markov")
    st.markdown("""
Aquí los **estados** de la cadena no son notas sino **acordes** expresados como grados (I, IV, V, vi…).
El modelo aprende, a partir de progresiones famosas, qué acordes suelen seguir a cuáles, y genera progresiones nuevas.
Como son grados, funcionan en **cualquier tonalidad**.""")
    c1, c2, c3 = st.columns(3)
    ton_a = c1.selectbox("Tónica", NOMBRES, key="arm_ton")
    modo_a = c2.radio("Modo", ["mayor", "menor"], horizontal=True, key="arm_modo")
    orden_a = c3.slider("Orden de la cadena", 1, 3, 1, key="arm_ord",
                        help="Con orden 2-3 el modelo recuerda los 2-3 acordes previos y copia más fielmente las progresiones.")
    ac = te.acordes_diatonicos(NOMBRES.index(ton_a), modo_a)
    numerales = list(ac)
    disponibles = list(te.PROGRESIONES[modo_a])
    sel_p = st.multiselect("Progresiones de entrenamiento", disponibles, default=disponibles, key=f"arm_sel_{modo_a}")

    if not sel_p:
        st.warning("Selecciona al menos una progresión.")
    else:
        secs_a = [te.PROGRESIONES[modo_a][k] for k in sel_p]
        modelo_a = mk.ModeloMarkov(orden_a, alpha).entrenar(secs_a, vocab=numerales, clave=numerales.index)
        st.caption(f"Entrenado con {len(secs_a)} progresiones y {sum(len(x) for x in secs_a)} acordes. "
                   f"Usa el α de la barra lateral ({alpha}).")
        d1, d2, d3, d4 = st.columns(4)
        n_ac = d1.slider("Acordes a generar", 4, 32, 8, key="arm_n")
        temp_a = d2.slider("Temperatura", 0.2, 2.0, 1.0, 0.1, key="arm_temp")
        bpm_a = d3.slider("Tempo (BPM)", 50, 160, 90, key="arm_bpm")
        sem_a = d4.number_input("Semilla (0 = aleatoria)", 0, 10**6, 0, key="arm_seed")

        if st.button("✨ Generar progresión", type="primary", key="arm_btn"):
            rng = np.random.default_rng(int(sem_a) or None)
            gen = [numerales[0]] + modelo_a.generar([numerales[0]], n_ac - 1, temp_a, 0, None, rng)
            st.session_state.arm_gen = dict(
                romanos=gen, nombres=[ac[r]["nombre"] for r in gen], funciones=[ac[r]["funcion"] for r in gen],
                tonalidad=f"{ton_a} {modo_a}",
                wav=audio.sintetizar_acordes([ac[r]["midi"] for r in gen], bpm_a, 2))
        g = st.session_state.get("arm_gen")
        if g:
            st.success(f"**{g['tonalidad']}:**  " + "  →  ".join(f"{n} ({r})" for n, r in zip(g["nombres"], g["romanos"])))
            fun = pd.Series(g["funciones"]).value_counts()
            st.caption("Funciones tonales usadas: " + " · ".join(f"{k}: {v}" for k, v in fun.items()))
            st.audio(g["wav"], format="audio/wav")
            st.download_button("⬇️ Descargar WAV", g["wav"], "progresion_markov.wav", "audio/wav", key="arm_dl")

        st.subheader("Lo que aprendió: matriz de transición entre acordes")
        Pa = mk.matriz_transicion(secs_a, numerales, alpha, numerales.index)
        st.altair_chart(heatmap(Pa, "Acorde siguiente", "Acorde actual", "blues", ".2f", altura=300))
        filas = []
        for r in numerales:
            j = Pa.loc[r].idxmax()
            filas.append({"Desde": f"{r} · {ac[r]['nombre']}", "Función": ac[r]["funcion"],
                          "Destino más probable": f"{j} · {ac[j]['nombre']}", "Probabilidad": f"{Pa.loc[r, j]:.0%}"})
        st.dataframe(pd.DataFrame(filas).set_index("Desde"))
        st.caption("Observa cómo el dominante (V) suele resolver a la tónica (I): el modelo redescubre la "
                   "armonía funcional solo contando transiciones.")

        umbral_a = st.slider("Mostrar transiciones con probabilidad ≥", 0.05, 0.6, 0.2, 0.05, key="arm_umb")
        aristas = [f'"{a}" -> "{b}" [label="{Pa.loc[a, b]:.2f}", penwidth={1 + 4 * Pa.loc[a, b]:.2f}];'
                   for a in numerales for b in numerales if Pa.loc[a, b] >= umbral_a]
        if aristas:
            st.graphviz_chart('digraph G { rankdir=LR; node [shape=circle, style=filled, fillcolor="#dbeafe", '
                              'fontname="Helvetica"]; edge [fontsize=9]; ' + " ".join(aristas) + " }")
        else:
            st.info("Ninguna transición supera ese umbral. Bájalo.")

# ============================================================ 5. ANÁLISIS
with t_ana:
    st.header("¿Qué aprendió la cadena?")
    if len(VOCAB) < 2:
        st.warning("Se necesitan al menos 2 estados distintos. Selecciona piezas del corpus o toca más notas.")
    else:
        st.caption("Aquí se muestra la cadena de **orden 1** (matriz de transición) sobre la representación elegida.")
        Pdf = mk.matriz_transicion(SECS_TOK, VOCAB, alpha)
        P = Pdf.values
        pi = mk.distribucion_estacionaria(P)
        H = mk.tasa_entropia(P, pi)
        lam2 = mk.segundo_autovalor(P)
        mezcla = np.log(0.01) / np.log(lam2) if 0 < lam2 < 1 else float("nan")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Estados |S|", len(VOCAB))
        k2.metric("Tasa de entropía", f"{H:.2f} bits", help=f"Máximo posible: log₂|S| = {np.log2(len(VOCAB)):.2f}")
        k3.metric("|λ₂|", f"{lam2:.3f}")
        k4.metric("Pasos para mezclar (1%)", f"{mezcla:.1f}" if not np.isnan(mezcla) else "—",
                  help="Pasos para que la distancia a π sea < 1%: ln(0.01)/ln|λ₂|")

        st.subheader("Matriz de transición P")
        Pdf_vista = Pdf.copy()
        st.altair_chart(heatmap(Pdf_vista, "Siguiente", "Actual", "blues", ".2f",
                                texto=len(VOCAB) <= 16, altura=max(250, 22 * len(VOCAB))))

        cA, cB = st.columns(2)
        with cA:
            st.subheader("Distribución estacionaria π")
            dfpi = pd.DataFrame({"estado": VOCAB, "prob": pi})
            st.altair_chart(alt.Chart(dfpi).mark_bar().encode(
                x=alt.X("estado:O", sort=VOCAB, title=None), y=alt.Y("prob:Q", title="π"),
                tooltip=["estado", alt.Tooltip("prob:Q", format=".3f")]).properties(width="container", height=260))
        with cB:
            st.subheader("Convergencia a π")
            ini = st.selectbox("Estado inicial", VOCAB)
            conv = mk.convergencia(P, pi, VOCAB.index(ini))
            st.line_chart(conv.set_index("paso"), y_label="Distancia de variación total", x_label="Pasos")

        st.subheader("Grafo de la cadena")
        umbral = st.slider("Mostrar transiciones con probabilidad ≥", 0.05, 0.6, 0.2, 0.05)
        aristas = [f'"{a}" -> "{b}" [label="{Pdf.loc[a, b]:.2f}", penwidth={1 + 4 * Pdf.loc[a, b]:.2f}];'
                   for a in VOCAB for b in VOCAB if Pdf.loc[a, b] >= umbral]
        if aristas:
            st.graphviz_chart('digraph G { rankdir=LR; node [shape=circle, style=filled, fillcolor="#dbeafe", '
                              'fontname="Helvetica"]; edge [fontsize=9]; ' + " ".join(aristas) + " }")
        else:
            st.info("Ninguna transición supera ese umbral. Bájalo.")

        st.subheader("Explorador de predicciones (modelo de orden k)")
        ctx_txt = st.text_input("Contexto (notas)", value=" ".join(SECS_NOTAS[0][:orden + 1]))
        ctx_notas, _ = parsear_secuencia(ctx_txt)
        modelo = mk.ModeloMarkov(orden, alpha).entrenar(SECS_TOK, VOCAB)
        ctx_tok = mk.tokenizar(ctx_notas, representacion)
        if ctx_notas:
            p = modelo.dist(ctx_tok)
            dfp = pd.DataFrame({"siguiente": VOCAB, "prob": p}).sort_values("prob", ascending=False)
            st.caption(f"P(siguiente | {' → '.join(ctx_tok[-orden:]) or '∅'})  —  orden {orden}, α = {alpha}")
            st.altair_chart(alt.Chart(dfp.head(12)).mark_bar().encode(
                x=alt.X("siguiente:O", sort=list(dfp.head(12)["siguiente"]), title=None),
                y=alt.Y("prob:Q", title="Probabilidad"),
                tooltip=["siguiente", alt.Tooltip("prob:Q", format=".3f")]).properties(width="container", height=240))
        else:
            st.info("Escribe un contexto válido, por ejemplo `Do4 Re4`.")

# ============================================================ 7. EVALUACIÓN
@st.cache_data(show_spinner="Evaluando modelos…")
def _rejilla(items, representacion, modo, ordenes, alphas):
    piezas = {k: list(v) for k, v in items}
    return mk.rejilla(piezas, ordenes, alphas, representacion, modo)


with t_eval:
    st.header("Evaluación y selección de hiperparámetros")
    st.write("Se compara el **orden k** y el **suavizado α** midiendo cuán bien predice el modelo notas **no vistas**.")
    ev_piezas = piezas_entrenamiento(con_peso=False)
    c1, c2 = st.columns(2)
    modo_ev = c1.radio("Esquema de validación",
                       ["temporal", "pieza"], format_func=lambda x: {
                           "temporal": "Partición temporal (75% inicio → 25% final de cada pieza)",
                           "pieza": "Leave-one-piece-out (entrena con otras piezas, prueba en una nueva)"}[x])
    rep_ev = c2.radio("Representación a evaluar", ["notas", "intervalos"],
                      index=["notas", "intervalos"].index(representacion), horizontal=True)
    if modo_ev == "pieza" and len(ev_piezas) < 2:
        st.warning("Leave-one-piece-out necesita al menos 2 piezas seleccionadas.")
    elif not ev_piezas:
        st.warning("Selecciona piezas del corpus en la barra lateral.")
    else:
        ordenes, alphas = (0, 1, 2, 3, 4), (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
        items = tuple((k, tuple(v)) for k, v in ev_piezas.items())
        res = _rejilla(items, rep_ev, modo_ev, ordenes, alphas).dropna()
        if res.empty:
            st.warning("No hay datos suficientes para evaluar.")
        else:
            mejor = res.loc[res["perplejidad"].idxmin()]
            V, n_test = int(mejor["V"]), int(mejor["n"])
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Mejor configuración", f"k={int(mejor['orden'])}, α={mejor['alpha']}")
            m2.metric("Perplejidad", f"{mejor['perplejidad']:.2f}", delta=f"{mejor['perplejidad'] - V:.1f} vs azar (={V})",
                      delta_color="inverse")
            m3.metric("Accuracy top-1", f"{mejor['top1']:.0%}")
            m4.metric("Accuracy top-3", f"{mejor['top3']:.0%}")
            st.caption(f"Evaluado sobre {n_test} notas de prueba. Perplejidad de un modelo aleatorio uniforme = |V| = {V}.")

            piv = res.pivot(index="orden", columns="alpha", values="perplejidad")
            piv.index = [f"k={o}" for o in piv.index]
            piv.columns = [str(a) for a in piv.columns]
            st.subheader("Perplejidad por (orden, α)  — menor es mejor")
            st.altair_chart(heatmap(piv, "α", "Orden", "yelloworangered", ".1f", altura=240, invertir=True))

            st.subheader("Perplejidad vs. orden")
            sel_alpha = st.select_slider("α", options=list(alphas), value=float(mejor["alpha"]))
            curva = res[res["alpha"] == sel_alpha][["orden", "perplejidad", "top1", "top3"]].set_index("orden")
            st.line_chart(curva[["perplejidad"]], y_label="Perplejidad", x_label="Orden k")
            st.dataframe(curva.rename(columns={"top1": "acc top-1", "top3": "acc top-3"}).round(3))

            st.info("""
**Cómo leerlo.** En la partición temporal suele ganar un orden medio/alto (la pieza se repite a sí misma). En
*leave-one-piece-out* los órdenes altos empeoran: **memorizan** piezas concretas y no generalizan a una pieza nueva
(**sobreajuste**). Un α pequeño agrava ese efecto (probabilidades casi 0 para lo no visto) y un α grande lo suaviza.
Este es el compromiso **sesgo–varianza** en acción. Compara también la representación por *intervalos*, que
comparte estadística entre tonalidades.""")

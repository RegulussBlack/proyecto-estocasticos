"""Corpus de melodías (dominio público) para entrenar el modelo."""

def _m(texto: str):
    return texto.split()

CORPUS = {
    "Estrellita (Twinkle Twinkle)": _m(
        "Do4 Do4 Sol4 Sol4 La4 La4 Sol4 Fa4 Fa4 Mi4 Mi4 Re4 Re4 Do4 "
        "Sol4 Sol4 Fa4 Fa4 Mi4 Mi4 Re4 Sol4 Sol4 Fa4 Fa4 Mi4 Mi4 Re4 "
        "Do4 Do4 Sol4 Sol4 La4 La4 Sol4 Fa4 Fa4 Mi4 Mi4 Re4 Re4 Do4"),
    "Oda a la Alegría (Beethoven)": _m(
        "Mi4 Mi4 Fa4 Sol4 Sol4 Fa4 Mi4 Re4 Do4 Do4 Re4 Mi4 Mi4 Re4 Re4 "
        "Mi4 Mi4 Fa4 Sol4 Sol4 Fa4 Mi4 Re4 Do4 Do4 Re4 Mi4 Re4 Do4 Do4 "
        "Re4 Re4 Mi4 Do4 Re4 Mi4 Fa4 Mi4 Do4 Re4 Mi4 Fa4 Mi4 Re4 Do4 Re4 Sol3 "
        "Mi4 Mi4 Fa4 Sol4 Sol4 Fa4 Mi4 Re4 Do4 Do4 Re4 Mi4 Re4 Do4 Do4"),
    "Hermano Jacobo (Frère Jacques)": _m(
        "Do4 Re4 Mi4 Do4 Do4 Re4 Mi4 Do4 Mi4 Fa4 Sol4 Mi4 Fa4 Sol4 "
        "Sol4 La4 Sol4 Fa4 Mi4 Do4 Sol4 La4 Sol4 Fa4 Mi4 Do4 "
        "Do4 Sol3 Do4 Do4 Sol3 Do4"),
    "Cumpleaños Feliz": _m(
        "Sol4 Sol4 La4 Sol4 Do5 Si4 Sol4 Sol4 La4 Sol4 Re5 Do5 "
        "Sol4 Sol4 Sol5 Mi5 Do5 Si4 La4 Fa5 Fa5 Mi5 Do5 Re5 Do5"),
    "Jingle Bells (estribillo)": _m(
        "Mi4 Mi4 Mi4 Mi4 Mi4 Mi4 Mi4 Sol4 Do4 Re4 Mi4 "
        "Fa4 Fa4 Fa4 Fa4 Fa4 Mi4 Mi4 Mi4 Mi4 Re4 Re4 Mi4 Re4 Sol4"),
    "Para Elisa (Beethoven)": _m(
        "Mi5 Re#5 Mi5 Re#5 Mi5 Si4 Re5 Do5 La4 Do4 Mi4 La4 Si4 Mi4 Sol#4 Si4 Do5 "
        "Mi4 Mi5 Re#5 Mi5 Re#5 Mi5 Si4 Re5 Do5 La4 Do4 Mi4 La4 Si4 Mi4 Do5 Si4 La4"),
    "Canon en Re (Pachelbel)": _m(
        "Fa#5 Mi5 Re5 Do#5 Si4 La4 Si4 Do#5 Re5 Do#5 Si4 La4 Sol4 Fa#4 Sol4 La4 "
        "Fa#5 Mi5 Re5 Do#5 Si4 La4 Si4 Do#5 Re5 Do#5 Si4 La4 Sol4 Fa#4 Sol4 La4"),
}

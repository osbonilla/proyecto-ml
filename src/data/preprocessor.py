"""
Preprocesador de texto para clasificación de intención geoespacial.
Limpieza, normalización y tokenización de consultas en español.
"""

import re
import unicodedata
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional


# Stopwords en español (versión curada para dominio geoespacial)
# Nota: NO eliminamos palabras clave geoespaciales como "zona", "área", "mapa"
SPANISH_STOPWORDS = {
    "a", "al", "algo", "algunas", "algunos", "ante", "antes", "como",
    "con", "contra", "cual", "cuando", "de", "del", "desde", "donde",
    "durante", "e", "el", "ella", "ellas", "ellos", "en", "entre",
    "era", "eras", "erais", "eran", "eres", "es", "esa", "esas",
    "ese", "eso", "esos", "esta", "estas", "este", "esto", "estos",
    "fue", "fui", "gran", "ha", "han", "hay", "he", "hemos",
    "la", "las", "le", "les", "lo", "los", "me", "mi", "mis",
    "muy", "nada", "ni", "no", "nos", "o", "para", "pero", "por",
    "porque", "que", "quien", "quienes", "se", "si", "sin", "sobre",
    "su", "sus", "también", "tan", "te", "tengo", "ti", "tiene",
    "tienen", "todo", "todos", "tu", "tus", "u", "un", "una",
    "unas", "uno", "unos", "usted", "va", "voy", "y", "ya", "yo",
}


class TextPreprocessor:
    """
    Pipeline de preprocesamiento de texto para NLP en español.

    Pasos:
        1. Conversión a minúsculas
        2. Normalización de acentos (opcional)
        3. Eliminación de caracteres especiales
        4. Eliminación de números (opcional)
        5. Eliminación de stopwords
        6. Eliminación de espacios múltiples
    """

    def __init__(
        self,
        remove_accents: bool = False,
        remove_numbers: bool = False,
        remove_stopwords: bool = True,
        min_token_length: int = 2,
    ):
        """
        Args:
            remove_accents: Si True, elimina tildes (ej: análisis → analisis)
            remove_numbers: Si True, elimina dígitos
            remove_stopwords: Si True, elimina stopwords en español
            min_token_length: Longitud mínima de token para conservar
        """
        self.remove_accents = remove_accents
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        self.min_token_length = min_token_length

    def normalize_accents(self, text: str) -> str:
        """Elimina acentos y normaliza caracteres unicode."""
        nfkd = unicodedata.normalize("NFKD", text)
        return "".join(c for c in nfkd if not unicodedata.combining(c))

    def clean_text(self, text: str) -> str:
        """Aplica pipeline completo de limpieza a un texto."""
        if not isinstance(text, str) or text.strip() == "":
            return ""

        # 1. Minúsculas
        text = text.lower().strip()

        # 2. Normalización de acentos (opcional)
        if self.remove_accents:
            text = self.normalize_accents(text)

        # 3. Eliminar caracteres especiales (conservar letras, números, espacios)
        text = re.sub(r"[^a-záéíóúüñA-ZÁÉÍÓÚÜÑ0-9\s]", " ", text)

        # 4. Eliminar números (opcional)
        if self.remove_numbers:
            text = re.sub(r"\d+", " ", text)

        # 5. Normalizar espacios múltiples
        text = re.sub(r"\s+", " ", text).strip()

        # 6. Eliminar stopwords y tokens cortos
        if self.remove_stopwords:
            tokens = text.split()
            tokens = [
                t for t in tokens
                if t not in SPANISH_STOPWORDS and len(t) >= self.min_token_length
            ]
            text = " ".join(tokens)

        return text

    def transform(self, texts: List[str]) -> List[str]:
        """Aplica limpieza a una lista de textos."""
        return [self.clean_text(t) for t in texts]

    def fit_transform(self, texts: List[str]) -> List[str]:
        """Compatible con sklearn Pipeline."""
        return self.transform(texts)

    def get_stats(self, original: List[str], cleaned: List[str]) -> pd.DataFrame:
        """Retorna estadísticas del preprocesamiento."""
        orig_lens = [len(str(t).split()) for t in original]
        clean_lens = [len(str(t).split()) for t in cleaned]
        return pd.DataFrame({
            "metric": ["avg_tokens_before", "avg_tokens_after", "reduction_%"],
            "value": [
                round(np.mean(orig_lens), 2),
                round(np.mean(clean_lens), 2),
                round((1 - np.mean(clean_lens) / np.mean(orig_lens)) * 100, 2),
            ]
        })


def preprocess_dataset(
    input_path: str = "data/raw/dataset_raw.csv",
    output_path: str = "data/processed/dataset_clean.csv",
    **preprocessor_kwargs,
) -> pd.DataFrame:
    """
    Carga el dataset raw, aplica preprocesamiento y guarda el resultado.

    Args:
        input_path: Ruta del dataset crudo
        output_path: Ruta de salida del dataset limpio
        **preprocessor_kwargs: Parámetros para TextPreprocessor

    Returns:
        DataFrame limpio
    """
    print(f"📥 Cargando dataset desde: {input_path}")
    df = pd.read_csv(input_path)

    preprocessor = TextPreprocessor(**preprocessor_kwargs)

    print("🧹 Aplicando preprocesamiento...")
    df["text_clean"] = preprocessor.transform(df["text"].tolist())

    # Eliminar filas vacías tras limpieza
    before = len(df)
    df = df[df["text_clean"].str.strip() != ""].reset_index(drop=True)
    after = len(df)
    if before != after:
        print(f"   ⚠️  Eliminadas {before - after} filas vacías tras limpieza")

    # Estadísticas
    stats = preprocessor.get_stats(df["text"].tolist(), df["text_clean"].tolist())
    print(f"   📊 Stats de preprocesamiento:\n{stats.to_string(index=False)}")

    # Guardar
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"✅ Dataset limpio guardado en: {output_path} ({len(df)} muestras)")

    return df


if __name__ == "__main__":
    df = preprocess_dataset(
        remove_accents=False,
        remove_numbers=False,
        remove_stopwords=True,
    )
    print(df[["text", "text_clean", "intent"]].head(10).to_string())

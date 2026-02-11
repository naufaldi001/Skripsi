import re

SLANG_MAP = {
    "gak": "tidak", "ga": "tidak", "nggak": "tidak", "ngga": "tidak", "tdk": "tidak",
    "bgt": "banget", "mantapp": "mantap", "mantul": "mantap",
    "recomended": "rekomendasi", "recommended": "rekomendasi",
    "worth": "layak", "murmer": "murah", "cepet": "cepat", "lemot": "lambat",
    "yaudah": "ya sudah", "okelah": "oke"
}

def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"(.)\1{2,}", r"\1", text)       # normalisasi huruf berulang
    text = re.sub(r"[^\w\s]", " ", text)           # hapus simbol & emoji

    tokens = text.split()
    tokens = [SLANG_MAP.get(t, t) for t in tokens]

    text = " ".join(tokens)
    text = re.sub(r"\s+", " ", text).strip()
    return text

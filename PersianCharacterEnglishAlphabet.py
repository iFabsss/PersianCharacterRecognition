# PersianCharacterEnglishAlphabet.py

# Each entry: "ClassLabel" -> (english_equivalent, persian_symbol)
PERSIAN_CHAR_MAP = {
    "Alef":     ("A",         "ا"),
    "Aleph":    ("A",         "ا"),   # variant label for همزه / الف
    "Anewfive": ("5",         "۵"),   # Eastern Arabic-Indic digit five
    "Ayin":     ("' / A",     "ع"),
    "Ayn":      ("' / A",     "ع"),   # variant label
    "B":        ("B",         "ب"),
    "Beh":      ("B",         "ب"),   # variant label
    "Che":      ("Ch",        "چ"),
    "D":        ("D",         "د"),
    "Daal":     ("D",         "د"),   # variant label
    "Eight":    ("8",         "۸"),
    "F":        ("F",         "ف"),
    "Feh":      ("F",         "ف"),   # variant label
    "Five":     ("5",         "۵"),
    "Four":     ("4",         "۴"),
    "G":        ("G",         "گ"),
    "Gaf":      ("G",         "گ"),   # variant label
    "Ghayn":    ("Gh",        "غ"),
    "Ghe":      ("Gh",        "غ"),   # variant label
    "Ghyin":    ("Gh",        "غ"),   # variant label
    "H":        ("H",         "ه"),
    "He":       ("H",         "ه"),
    "He jimi":  ("H",         "ح"),   # separate letter — haa-ye jimi
    "Jim":      ("J",         "ج"),
    "K":        ("K",         "ک"),
    "Kaf":      ("K",         "ک"),   # variant label
    "Kh":       ("Kh",        "خ"),
    "Khe":      ("Kh",        "خ"),   # variant label
    "Lam":      ("L",         "ل"),
    "Le":       ("L",         "ل"),   # variant label
    "M":        ("M",         "م"),
    "Mim":      ("M",         "م"),   # variant label
    "N":        ("N",         "ن"),
    "Nine":     ("9",         "۹"),
    "Nun":      ("N",         "ن"),   # variant label
    "One":      ("1",         "۱"),
    "P":        ("P",         "پ"),
    "Peh":      ("P",         "پ"),   # variant label
    "Qaf":      ("Q",         "ق"),
    "R":        ("R",         "ر"),
    "Re":       ("R",         "ر"),   # variant label
    "Sad":      ("S",         "ص"),
    "Se":       ("S",         "ث"),   # ث — distinct from sin/sad
    "Seven":    ("7",         "۷"),
    "Shin":     ("Sh",        "ش"),
    "Sin":      ("S",         "س"),
    "Six":      ("6",         "۶"),
    "T":        ("T",         "ت"),
    "T-long":   ("T",         "ط"),   # ط — emphatic T
    "Taa":      ("T",         "ط"),   # variant label for ط
    "Teh":      ("T",         "ت"),   # variant label for ت
    "Theh":     ("Th / S",    "ث"),   # variant label for ث
    "Three":    ("3",         "۳"),
    "Two":      ("2",         "۲"),
    "V":        ("V",         "و"),
    "Vav":      ("V / W / U", "و"),   # و is multifunctional
    "Yaa":      ("Y / A",     "ی"),   # ی can also be alef maqsura
    "Ye":       ("Y",         "ی"),   # variant label
    "Z":        ("Z",         "ز"),
    "Z-long":   ("Z",         "ظ"),   # ظ — emphatic Z
    "Zaa":      ("Z",         "ظ"),   # variant label for ظ
    "Zaal":     ("Z",         "ذ"),   # ذ
    "Zad":      ("Z",         "ض"),   # ض — emphatic
    "Zal":      ("Z",         "ذ"),   # variant label for ذ
    "Ze":       ("Z",         "ز"),   # variant label
    "Zero":     ("0",         "۰"),
    "Zh":       ("Zh",        "ژ"),
    "Zhe":      ("Zh",        "ژ"),   # variant label
}

# Ambiguous class pairs that are visually similar — used for display hints
AMBIGUOUS_GROUPS = [
    {"One", "Alef", "Aleph"},   # ۱ vs ا
    {"Ayin", "Ayn"},             # same letter, two labels
    {"Ghayn", "Ghe", "Ghyin"},  # same letter, three labels
    {"Z", "Ze"},                 # same letter, two labels
    {"Zaa", "Z-long"},          # same letter, two labels
    {"Zaal", "Zal"},            # same letter, two labels
]

def get_equivalent(label: str) -> str:
    entry = PERSIAN_CHAR_MAP.get(label)
    return entry[0] if entry else "?"

def get_symbol(label: str) -> str:
    entry = PERSIAN_CHAR_MAP.get(label)
    return entry[1] if entry else "?"

def get_ambiguous_hint(label: str) -> str | None:
    """Returns a hint string if this label belongs to a visually similar group."""
    for group in AMBIGUOUS_GROUPS:
        if label in group:
            others = group - {label}
            return f"May also be: {', '.join(sorted(others))}"
    return None
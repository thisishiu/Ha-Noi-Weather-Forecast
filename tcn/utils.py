import unicodedata


def normalize_district(name: str) -> str:
    if not isinstance(name, str):
        return ""
    nfd = unicodedata.normalize("NFD", name)
    cleaned = []
    for ch in nfd:
        if ch in ("Đ",):
            cleaned.append("D")
        elif ch in ("đ",):
            cleaned.append("d")
        elif unicodedata.category(ch) == "Mn":
            continue
        else:
            cleaned.append(ch)
    base = "".join(cleaned)
    base = " ".join(base.split())
    base = base.replace(" ", "_")
    return base


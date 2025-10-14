import unicodedata


def normalize_district(name: str) -> str:
    """Normalize Vietnamese district names: remove diacritics, map Đ/đ to D/d, spaces to underscores.

    Examples:
    - "Ba Đình" -> "Ba_Dinh"
    - "Tây Hồ" -> "Tay_Ho"
    """
    if not isinstance(name, str):
        return ""
    # NFD decomposition to separate base and combining marks
    nfd = unicodedata.normalize("NFD", name)
    # Remove combining marks, map Đ/đ, keep ASCII letters/digits/space/underscore
    cleaned = []
    for ch in nfd:
        if ch in ("Đ",):
            cleaned.append("D")
        elif ch in ("đ",):
            cleaned.append("d")
        elif unicodedata.category(ch) == "Mn":
            # skip diacritics
            continue
        else:
            cleaned.append(ch)
    base = "".join(cleaned)
    # collapse whitespace and replace spaces with underscore
    base = " ".join(base.split())
    base = base.replace(" ", "_")
    return base

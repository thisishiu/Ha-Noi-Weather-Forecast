from __future__ import annotations

import re


def normalize_district(name: str) -> str:
    if name is None:
        return ""
    s = str(name).strip()
    s = s.lower()
    s = s.replace(" ", "_")
    s = re.sub(r"[\s]+", "_", s)
    s = s.replace("đ", "d").replace("Đ", "d")
    s = s.replace("á", "a").replace("à", "a").replace("ả", "a").replace("ã", "a").replace("ạ", "a")
    s = s.replace("ă", "a").replace("ắ", "a").replace("ằ", "a").replace("ẳ", "a").replace("ẵ", "a").replace("ặ", "a")
    s = s.replace("â", "a").replace("ấ", "a").replace("ầ", "a").replace("ẩ", "a").replace("ẫ", "a").replace("ậ", "a")
    s = s.replace("é", "e").replace("è", "e").replace("ẻ", "e").replace("ẽ", "e").replace("ẹ", "e")
    s = s.replace("ê", "e").replace("ế", "e").replace("ề", "e").replace("ể", "e").replace("ễ", "e").replace("ệ", "e")
    s = s.replace("í", "i").replace("ì", "i").replace("ỉ", "i").replace("ĩ", "i").replace("ị", "i")
    s = s.replace("ó", "o").replace("ò", "o").replace("ỏ", "o").replace("õ", "o").replace("ọ", "o")
    s = s.replace("ô", "o").replace("ố", "o").replace("ồ", "o").replace("ổ", "o").replace("ỗ", "o").replace("ộ", "o")
    s = s.replace("ơ", "o").replace("ớ", "o").replace("ờ", "o").replace("ở", "o").replace("ỡ", "o").replace("ợ", "o")
    s = s.replace("ú", "u").replace("ù", "u").replace("ủ", "u").replace("ũ", "u").replace("ụ", "u")
    s = s.replace("ư", "u").replace("ứ", "u").replace("ừ", "u").replace("ử", "u").replace("ữ", "u").replace("ự", "u")
    s = s.replace("ý", "y").replace("ỳ", "y").replace("ỷ", "y").replace("ỹ", "y").replace("ỵ", "y")
    return s

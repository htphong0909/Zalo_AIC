"""Vietnamese Inverse Text Normalization (ITN) and orthographic standardization engine."""
from __future__ import annotations

import re
import unicodedata

# Single-token digit values
_DIGITS: dict[str, int] = {
    "không": 0,
    "linh": 0,
    "lẻ": 0,
    "một": 1,
    "mốt": 1,
    "hai": 2,
    "ba": 3,
    "bốn": 4,
    "tư": 4,
    "năm": 5,
    "lăm": 5,
    "nhăm": 5,
    "sáu": 6,
    "bảy": 7,
    "bẩy": 7,
    "tám": 8,
    "chín": 9,
}

_NUM_WORDS_SET = set(_DIGITS.keys()) | {
    "mười",
    "mươi",
    "chục",
    "trăm",
    "nghìn",
    "ngàn",
    "triệu",
    "tỷ",
}


def _parse_under_thousand(toks: list[str]) -> int | None:
    if not toks:
        return 0
    val = 0

    # Parse hundreds if present
    if "trăm" in toks:
        idx = toks.index("trăm")
        h_toks = toks[:idx]
        if not h_toks:
            h_val = 1
        elif len(h_toks) == 1 and h_toks[0] in _DIGITS:
            h_val = _DIGITS[h_toks[0]]
        else:
            return None
        val += h_val * 100
        toks = toks[idx + 1 :]
        if not toks:
            return val
        if toks[0] in ("linh", "lẻ"):
            toks = toks[1:]

    if not toks:
        return val

    # Parse remaining tens & units
    if len(toks) == 1:
        t = toks[0]
        if t == "mười":
            return val + 10
        if t in _DIGITS:
            return val + _DIGITS[t]
        return None

    if len(toks) == 2:
        t0, t1 = toks[0], toks[1]
        if t0 == "mười" and t1 in _DIGITS:
            return val + 10 + _DIGITS[t1]
        if t1 in ("mươi", "chục") and t0 in _DIGITS:
            return val + _DIGITS[t0] * 10
        return None

    if len(toks) == 3:
        t0, t1, t2 = toks[0], toks[1], toks[2]
        if t1 in ("mươi", "chục") and t0 in _DIGITS and t2 in _DIGITS:
            return val + _DIGITS[t0] * 10 + _DIGITS[t2]
        return None

    return None


def parse_vi_numeral(text_or_tokens: str | list[str]) -> int | None:
    """Parse Vietnamese spoken numeral phrases (from 0 up to billions) into an integer.

    Examples:
        'không' -> 0
        'mười lăm' -> 15
        'hai mươi sáu' -> 26
        'ba mươi' -> 30
        'một trăm hai mươi ba' -> 123
        '15' -> 15
    """
    if isinstance(text_or_tokens, str):
        s = text_or_tokens.strip().lower()
        if s.isdigit():
            return int(s)
        toks = s.split()
    else:
        toks = [t.strip().lower() for t in text_or_tokens if t.strip()]

    if not toks:
        return None

    # Fast path for single token
    if len(toks) == 1:
        t = toks[0]
        if t.isdigit():
            return int(t)
        if t == "mười":
            return 10
        if t in _DIGITS:
            return _DIGITS[t]
        if t == "trăm":
            return 100
        if t in ("nghìn", "ngàn"):
            return 1000
        if t == "triệu":
            return 1000000
        if t == "tỷ":
            return 1000000000
        return None

    total = 0
    rem = list(toks)

    # Billions (tỷ)
    if "tỷ" in rem:
        idx = rem.index("tỷ")
        b_val = _parse_under_thousand(rem[:idx])
        if b_val is None:
            return None
        total += b_val * 1000000000
        rem = rem[idx + 1 :]

    # Millions (triệu)
    if "triệu" in rem:
        idx = rem.index("triệu")
        m_val = _parse_under_thousand(rem[:idx])
        if m_val is None:
            return None
        total += m_val * 1000000
        rem = rem[idx + 1 :]

    # Thousands (nghìn/ngàn)
    if "nghìn" in rem or "ngàn" in rem:
        idx = rem.index("nghìn") if "nghìn" in rem else rem.index("ngàn")
        k_val = _parse_under_thousand(rem[:idx])
        if k_val is None:
            return None
        total += k_val * 1000
        rem = rem[idx + 1 :]

    # Units under a thousand
    if rem:
        u_val = _parse_under_thousand(rem)
        if u_val is None:
            return None
        total += u_val

    return total


# Precompiled regex patterns for inverse text normalization
_NUM_KEM_AFTER = (
    r"(?:mười\s+lăm|mười\s+năm|hai\s+mươi\s+lăm|hai\s+mươi|ba\s+mươi|"
    r"mười\s+một|mười\s+hai|mười\s+ba|mười\s+bốn|mười\s+sáu|mười\s+bảy|mười\s+tám|mười\s+chín|mười|\d+)"
)
_NUM_KEM_BEFORE = (
    r"(?:mười\s+hai|mười\s+một|mười|chín|tám|bảy|bẩy|sáu|năm|bốn|tư|ba|hai|một|\d+)"
)
_RE_KEM = re.compile(rf"\b({_NUM_KEM_BEFORE})\s+giờ\s+kém\s+({_NUM_KEM_AFTER})\b", re.IGNORECASE)

_NUM_DO = (
    r"(?:hai\s+mươi\s+(?:mốt|hai|ba|bốn|tư|lăm|năm|sáu|bảy|tám|chín)|"
    r"ba\s+mươi\s+(?:mốt|hai|ba|bốn|tư|lăm|năm|sáu|bảy|tám|chín)|"
    r"hai\s+mươi|ba\s+mươi|mười\s+sáu|mười\s+bảy|mười\s+tám|mười\s+chín)"
)
_RE_DO = re.compile(rf"\b({_NUM_DO})\s+độ\b", re.IGNORECASE)

_NUM_PCT = (
    r"(?:(?:mười|hai\s+mươi|ba\s+mươi|bốn\s+mươi|năm\s+mươi|sáu\s+mươi|bảy\s+mươi|tám\s+mươi|chín\s+mươi)"
    r"(?:\s+(?:mốt|hai|ba|bốn|tư|lăm|năm|sáu|bảy|tám|chín))?|một\s+trăm|mười\s+lăm)"
)
_RE_PCT = re.compile(rf"\b({_NUM_PCT})\s+phần\s+trăm\b", re.IGNORECASE)

_NUM_TRIEU = (
    r"(?:(?:một|hai|ba|bốn|năm|sáu|bảy|tám|chín)\s+trăm(?:\s+(?:linh|lẻ)\s+(?:một|hai|ba|bốn|lăm|năm|sáu|bảy|tám|chín)|"
    r"\s+(?:mười|hai\s+mươi|ba\s+mươi|bốn\s+mươi|năm\s+mươi|sáu\s+mươi|bảy\s+mươi|tám\s+mươi|chín\s+mươi)"
    r"(?:\s+(?:mốt|hai|ba|bốn|tư|lăm|năm|sáu|bảy|tám|chín))?)?|"
    r"(?:hai\s+mươi|ba\s+mươi|bốn\s+mươi|năm\s+mươi|sáu\s+mươi|bảy\s+mươi|tám\s+mươi|chín\s+mươi)"
    r"(?:\s+(?:mốt|hai|ba|bốn|tư|lăm|năm|sáu|bảy|tám|chín))?)"
)
_RE_TRIEU = re.compile(rf"\b({_NUM_TRIEU})\s+triệu\b", re.IGNORECASE)

# Traditional Vietnamese tone harmonization mapping
_TONE_HARMONIZATION_MAP: dict[str, str] = {
    "hòa": "hoà",
    "hóa": "hoá",
    "xóa": "xoá",
    "hủy": "huỷ",
}

_TONE_PATTERNS: list[tuple[re.Pattern, str]] = []
for _mod, _trad in _TONE_HARMONIZATION_MAP.items():
    _TONE_PATTERNS.append((re.compile(rf"\b{_mod}\b"), _trad))
    _TONE_PATTERNS.append((re.compile(rf"\b{_mod.capitalize()}\b"), _trad.capitalize()))
    _TONE_PATTERNS.append((re.compile(rf"\b{_mod.upper()}\b"), _trad.upper()))



# Dialectal phonological compound normalization
# In Southern Vietnamese spoken speech, initial rhotic /r/ is frequently realized as [g]
# in high-frequency financial collocations ('rút tiền', 'rút tiết kiệm').
# In standard written Vietnamese, 'gút' is strictly an orthographic error in these contexts.
_RE_DIALECT_RUT = re.compile(r"\bgút\s+(tiết\s+kiệm|tiền)\b", re.IGNORECASE)

_EMAIL_TLDS = (
    r"(?:com|vn|net|org|edu|gov|info|biz|co|io|ai|me|app|dev|xyz|test|local|us|uk|jp|kr)"
)
_SPOKEN_EMAIL_RE = re.compile(
    r"(?<![a-z0-9._%+\-@])"
    r"([a-z0-9_%+\-]+(?:\s+(?:chấm|dot|gạch\s+ngang|gạch\s+dưới|gạch)\s+[a-z0-9_%+\-]+)*)"
    r"\s+(?:a[\s\-]*c[oòóỏõọồốổỗộ]ng|a[\s\-]*m[oóòỏõọ]c)\s+"
    r"([a-z0-9\-]+(?:\s+(?:chấm|dot|gạch\s+ngang)\s+[a-z0-9\-]+)*\s+(?:chấm|dot)\s+"
    + _EMAIL_TLDS
    + r"(?:\s+(?:chấm|dot)\s+(?:vn|uk|us|jp|kr))?)"
    r"(?![a-z0-9])",
    re.IGNORECASE,
)


def _collapse_spoken_email_part(part: str, *, allow_underscore: bool) -> str:
    s = re.sub(r"\s+(?:chấm|dot)\s+", ".", part.strip().lower())
    s = re.sub(r"\s+gạch\s+ngang\s+", "-", s)
    if allow_underscore:
        s = re.sub(r"\s+gạch\s+dưới\s+", "_", s)
        s = re.sub(r"\s+gạch\s+", "_", s)
    return re.sub(r"\s+", "", s)


def _normalize_spoken_emails(text: str) -> str:
    """Convert Vietnamese spoken email dictation ('a còng ... chấm ...') into literal addresses."""
    low_check = text.lower()
    if "còng" not in low_check and "cồng" not in low_check and "móc" not in low_check:
        return text

    def _sub_email(m: re.Match) -> str:
        local = _collapse_spoken_email_part(m.group(1), allow_underscore=True)
        domain = _collapse_spoken_email_part(m.group(2), allow_underscore=False)
        return f"{local}@{domain}"

    return _SPOKEN_EMAIL_RE.sub(_sub_email, text)


def inverse_text_normalize(text: str) -> str:
    """Apply Vietnamese Inverse Text Normalization (ITN) and orthographic standardization.

    Converts domain-conditioned spoken Vietnamese numerical constructs into standard written representations:
    1. Time reminders with 'kém': 'chín giờ kém mười lăm' -> '9 giờ kém 15'
    2. Appliance/ambient temperature: 'hai mươi sáu độ' -> '26 độ'
    3. Lighting/intensity percentage: 'ba mươi phần trăm' -> '30 phần trăm'
    4. Multi-digit financial quantities: 'một trăm hai mươi ba triệu' -> '123 triệu'
    5. Traditional orthographic tone harmonization: 'hủy' -> 'huỷ'
    6. Dialectal phonological compound recovery: 'gút tiết kiệm' -> 'rút tiết kiệm'
    """
    if not text:
        return ""

    s = unicodedata.normalize("NFC", str(text))

    # 1. Time expressions with 'kém'
    def _sub_kem(m: re.Match) -> str:
        h_str = m.group(1).strip()
        m_str = m.group(2).strip()
        h_val = parse_vi_numeral(h_str)
        m_val = parse_vi_numeral(m_str)
        if h_val is not None and m_val is not None:
            return f"{h_val} giờ kém {m_val}"
        return m.group(0)

    s = _RE_KEM.sub(_sub_kem, s)

    # 2. Ambient / device temperature
    def _sub_do(m: re.Match) -> str:
        num_str = m.group(1).strip()
        val = parse_vi_numeral(num_str)
        if val is not None and 10 <= val <= 45:
            return f"{val} độ"
        return m.group(0)

    s = _RE_DO.sub(_sub_do, s)

    # 3. Percentage / ratio
    def _sub_pct(m: re.Match) -> str:
        num_str = m.group(1).strip()
        val = parse_vi_numeral(num_str)
        if val is not None and 0 <= val <= 100:
            return f"{val} phần trăm"
        return m.group(0)

    s = _RE_PCT.sub(_sub_pct, s)

    # 4. Multi-digit financial / quantity thresholds before 'triệu'
    def _sub_trieu(m: re.Match) -> str:
        num_str = m.group(1).strip()
        val = parse_vi_numeral(num_str)
        if val is not None and val >= 10:
            return f"{val} triệu"
        return m.group(0)

    s = _RE_TRIEU.sub(_sub_trieu, s)

    # 5. Traditional orthographic tone harmonization
    for pat, rep in _TONE_PATTERNS:
        s = pat.sub(rep, s)

    # 6. Dialectal phonological compound recovery
    s = _RE_DIALECT_RUT.sub(r"rút \1", s)

    # 7. Spoken email dictation ('final a còng example chấm test' -> 'final@example.test')
    s = _normalize_spoken_emails(s)

    return s

"""Arabic text normalization before synthesis: what the model should say for digits and symbols.

The dialect models are trained on transcripts with numbers spelled out, so digits are read as words here:
integers (up to the billions), decimals, percentages, clock times, currency amounts and long digit strings (phone
numbers, digit by digit). Letter variants (Persian keheh / yeh, heh doachashmee) and tatweel are unified, Arabic-Indic
and Persian digits read like Latin ones. Number words take the -ين forms (عشرين، ألفين) that speech uses in most
positions (nominative عشرون / ألفان is right only as a subject); ``style`` "msa" spells the hundreds formally (مئة،
مئتين), "colloquial" as the dialects say them (مية، ميتين). A one-letter proclitic is joined to its number (بـ 3000 ->
بثلاثة آلاف).
"""

from __future__ import annotations

import re

_LETTERS = str.maketrans({"ک": "ك", "ی": "ي", "ھ": "ه", "ـ": None,
                          "٫": ".", "٬": ","})
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

_UNITS = ["صفر", "واحد", "اثنين", "ثلاثة", "أربعة", "خمسة", "ستة", "سبعة", "ثمانية", "تسعة"]
_TEENS = ["عشرة", "أحد عشر", "اثني عشر", "ثلاثة عشر", "أربعة عشر", "خمسة عشر", "ستة عشر", "سبعة عشر",
          "ثمانية عشر", "تسعة عشر"]
_TENS = ["", "", "عشرين", "ثلاثين", "أربعين", "خمسين", "ستين", "سبعين", "ثمانين", "تسعين"]
_HUNDREDS = ["", "مية", "ميتين", "ثلاثمية", "أربعمية", "خمسمية", "ستمية", "سبعمية", "ثمانمية", "تسعمية"]
# (one, two, 3-10 plural)
_SCALES = [(10**9, ("مليار", "مليارين", "مليارات")), (10**6, ("مليون", "مليونين", "ملايين")),
           (10**3, ("ألف", "ألفين", "آلاف"))]
_MSA = {"مية": "مئة", "ميتين": "مئتين", "ثلاثمية": "ثلاثمئة", "أربعمية": "أربعمئة", "خمسمية": "خمسمئة",
        "ستمية": "ستمئة", "سبعمية": "سبعمئة", "ثمانمية": "ثمانمئة", "تسعمية": "تسعمئة"}
_PERCENT = {"colloquial": "بالمية", "msa": "بالمئة"}
_CURRENCY = {"$": "دولار", "usd": "دولار", "€": "يورو", "eur": "يورو", "£": "جنيه إسترليني", "sar": "ريال",
             "aed": "درهم", "egp": "جنيه", "kwd": "دينار", "qar": "ريال قطري"}


def _below_1000(n: int) -> list[str]:
    out = []
    h, rest = divmod(n, 100)
    if h:
        out.append(_HUNDREDS[h])
    if rest >= 20:
        u, t = rest % 10, rest // 10
        out.append(f"{_UNITS[u]} و{_TENS[t]}" if u else _TENS[t])
    elif rest >= 10:
        out.append(_TEENS[rest - 10])
    elif rest:
        out.append(_UNITS[rest])
    return out


def number_words(n: int, style: str = "colloquial") -> str:
    """A non-negative integer as Arabic words (masculine counting form)."""
    if n == 0:
        return _UNITS[0]
    parts = []
    for scale, (one, two, plural) in _SCALES:
        k, n = divmod(n, scale)
        if not k:
            continue
        if k == 1:
            parts.append(one)
        elif k == 2:
            parts.append(two)
        elif k <= 10:
            parts.append(f"{' و'.join(_below_1000(k))} {plural}")
        else:
            parts.append(f"{' و'.join(_below_1000(k)) if k < 1000 else number_words(k, style)} {one}")
    parts += _below_1000(n)
    words = " و".join(parts)
    if style == "msa":
        words = " ".join(_MSA.get(w, w) for w in words.split(" "))
        words = re.sub(r"و(\S+)", lambda m: "و" + _MSA.get(m.group(1), m.group(1)), words)
    return words


def _decimal(int_part: str, frac: str, style: str) -> str:
    words = number_words(int(int_part), style)
    if frac:
        frac_words = number_words(int(frac), style) if not frac.startswith("0") else \
            " ".join(_UNITS[int(d)] for d in frac)
        words += f" فاصلة {frac_words}"
    return words


_NUM = r"\d{1,3}(?:,\d{3})+|\d+"
_PHONE = re.compile(r"(?<![\d.])(?:\+?\d[\d\- ]{6,}\d)(?![\d.])")
_TIME = re.compile(r"(?<!\d)([01]?\d|2[0-3]):([0-5]\d)(?!\d)")
_CUR_BEFORE = re.compile(rf"(\$|€|£|\b(?:USD|EUR|SAR|AED|EGP|KWD|QAR)\b)\s?({_NUM})(?:\.(\d+))?", re.IGNORECASE)
_CUR_AFTER = re.compile(rf"({_NUM})(?:\.(\d+))?\s?(\$|€|£|\b(?:USD|EUR|SAR|AED|EGP|KWD|QAR)\b)", re.IGNORECASE)
_PCT = re.compile(rf"({_NUM})(?:\.(\d+))?\s?[%٪]")
_DEC = re.compile(rf"({_NUM})(?:\.(\d+))?")
_ARABIC_AFTER = re.compile(r"(\d)(?=[ء-ي])")
_PROCLITIC = re.compile(r"(?:(?<=\s)|^)([بلوكف])\s+(?=\d)")  # "بـ 3000" (tatweel gone): "ب 3000" -> "ب3000"


def normalize(text: str, style: str = "colloquial") -> str:
    t = text.translate(_LETTERS).translate(_DIGITS)
    t = _PROCLITIC.sub(r"\1", t)
    t = _ARABIC_AFTER.sub(r"\1 ", t)  # "500متر" -> "500 متر" (a proclitic before the number stays attached)

    def amount(n, frac, unit):
        return f"{_decimal(n.replace(',', ''), frac or '', style)} {_CURRENCY[unit.lower()]}"

    t = _CUR_BEFORE.sub(lambda m: amount(m.group(2), m.group(3), m.group(1)), t)
    t = _CUR_AFTER.sub(lambda m: amount(m.group(1), m.group(2), m.group(3)), t)
    t = _PCT.sub(lambda m: f"{_decimal(m.group(1).replace(',', ''), m.group(2) or '', style)} {_PERCENT.get(style, 'بالمية')}", t)
    t = _TIME.sub(lambda m: number_words(int(m.group(1)), style) + (
        f" و{number_words(int(m.group(2)), style)} دقيقة" if int(m.group(2)) else ""), t)
    t = _PHONE.sub(lambda m: " ".join(_UNITS[int(d)] for d in re.sub(r"\D", "", m.group(0))), t)
    t = _DEC.sub(lambda m: _decimal(m.group(1).replace(",", ""), m.group(2) or "", style), t)
    return re.sub(r"[ \t]+", " ", t).strip()

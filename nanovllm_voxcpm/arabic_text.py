"""Arabic text normalization before synthesis: what the model should read for digits, symbols and spelling variants.

The dialect models are trained on undiacritized transcripts with numbers spelled out (they hold no digits), so the
text is brought to that form; the caller keeps the original. ``style`` "msa" reads formally (مئة، بالمئة، والنصف،
صباحا), "colloquial" as the dialects say it (مية، بالمية، ونص، الصبح). The rules, in the order applied:

- Spelling: Persian / Urdu letter forms (ک ی ھ ہ ە, as the training repo's text.py) and Arabic presentation forms
  (ﻻ، ﷺ) as Arabic letters; tatweel, harakat (tanwin and shadda included; letters stay: شكراً -> شكرا), Quranic marks
  and invisible bidi / zero-width characters removed (the transcripts are undiacritized: harakat in none of 7590
  lahgtna texts, in 1-4 % of the eval, sada and synth texts); Arabic-Indic and Persian digits, ٫ ٬ ٪ as 0-9 . , %.
- Joining: a proclitic (ب ل ك و ف, وب ...) or the article (ال، بال، لل ...) joins the number after it (بـ 3000 ->
  بثلاثة آلاف); any other word glued to digits is split off (رقم5 -> رقم خمسة, 100km -> مية كيلومتر, 5G -> خمسة G).
- Dates d/m/yyyy, d-m-yyyy, d.m.yyyy (also yy with / and -) and yyyy-mm-dd: day, month name, year (12/05/2024 -> اثني
  عشر مايو ألفين وأربعة وعشرين). The Gregorian names (يناير ... ديسمبر) are those of Egypt, the Gulf and pan-Arab
  media, understood in the Levant and Iraq too (where أيار-style names are used alongside).
- Clock times h:mm, h:mm:ss (0-24 h), with ص / م / am / pm (also after الساعة N): 9:00 تسعة, 9:15 تسعة وربع, 9:30
  تسعة ونص, otherwise minutes (and seconds) counted: 9:01 تسعة ودقيقة, 9:02 ودقيقتين, 9:05 وخمس دقايق, 9:25 وخمسة
  وعشرين دقيقة.
- Phone numbers (>= 8 digits) digit by digit: after a +, from a leading 0 (also in groups split by spaces or hyphens),
  or as 3+ hyphen-joined groups (800-123-4567). Other digit groups are separate numbers (100 200 300).
- Signs: - (or −) before a digit at the start or after a space / bracket is ناقص, + زائد (2010-2015 keeps its hyphen).
- Dotted groups: 192.168.1.1, 1.2.3 as numbers joined by نقطة; 1.000.000 as one number.
- Years with an era: 1445هـ -> ... هجري, 2024م -> ... ميلادي.
- Currency before or after the amount ($ US$ € £ ₪, USD EUR GBP SAR AED EGP QAR KWD BHD JOD OMR ILS, ر.س د.إ ج.م
  د.ك), the noun counted: 1 دولار واحد, 2 دولارين, 3-10 ثلاثة دولارات, 11+ أحد عشر دولار, 103 مية وثلاثة دولارات;
  1-2 decimals as the subunit (12.50$ -> اثني عشر دولار وخمسين سنت), other decimals with فاصلة (dinars: 1000 fils).
- Percent after or before the amount (50% / %50).
- Units after a number, singular: km m cm mm kg g mg l ml km/h °C ° GB MB, كم سم مم كغ كجم مل كم/س.
- Fractions a/b: 1/2 1/3 2/3 1/4 3/4 by name (نص، ثلث، ثلثين، ربع، ثلاثة أرباع), others "a على b" (24/7).
- Ordinals 1-100 after the article when it follows a noun of _ORDINAL_NOUNS (الطابق الـ3 -> الطابق الثالث, المرة
  الـ3 -> المرة الثالثة), and القرن N; after other words the article stays on the cardinal (خلال الـ3 أيام -> خلال
  الثلاثة أيام).
- Everything else: integers as cardinals up to the billions (12 digits), longer digit strings and codes with a leading
  zero (007) digit by digit; decimals with فاصلة (also .5 and the decimal comma 12,5), the fraction without trailing
  zeros (12.50 -> اثني عشر فاصلة خمسة), as a number up to 3 digits, longer digit by digit.
- Abbreviations before a name: د. دكتور, أ.د. أستاذ دكتور (not next to a number: 45 د. are minutes).
- Punctuation as the transcripts write it, spaces collapsed: , ? ; -> ، ؟ ؛ (lahgtna: 20210 ، to 131 , ; 2154 ؟, no
  ? ; 7 ؛, no ;).

Number words take the -ين forms (عشرين، ألفين) that speech uses in most positions (nominative عشرون / ألفان is right
only as a subject). Left as written: Latin words and acronyms (GPS, CNN are spelled, NASA, FIFA read as words: no rule
tells them apart), the letter pairs the transcripts keep apart (ى / ي, ة / ه, hamza seats), dates without a year
(12/5 reads as a fraction).
"""

from __future__ import annotations

import re
import unicodedata

_AR = "\u0621-\u063a\u0641-\u064a\u0671-\u06d3\u06fa-\u06fc"  # Arabic-script letters (no tatweel, no marks)
_ARABIC = re.compile("[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufefc]")  # Arabic script
_COMBINING_HAMZA = re.compile("[\u0653-\u0655]")  # madda / hamza above / below: composed with their letter first


def _char_map() -> dict[int, str | None]:
    """One str.translate table: letter variants, presentation forms, digits; marks and invisible characters out."""
    letters = {"ک": "ك", "ی": "ي", "ھ": "ه", "ہ": "ه", "ە": "ه", "٫": ".", "٬": ",", "٪": "%", "−": "-"}
    table: dict[int, str | None] = {ord(k): v for k, v in letters.items()}
    table.update(zip(map(ord, "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹"), "01234567890123456789"))
    drop = [(0x0610, 0x061A), (0x064B, 0x065F), (0x0670, 0x0670), (0x06D6, 0x06ED), (0x0640, 0x0640),  # marks, tatweel
            (0x061C, 0x061C), (0x200B, 0x200F), (0x202A, 0x202E), (0x2066, 0x2069), (0xFEFF, 0xFEFF)]  # invisible
    for lo, hi in drop:
        table.update(dict.fromkeys(range(lo, hi + 1)))
    table.update(dict.fromkeys([0x00A0, *range(0x2000, 0x200B), 0x202F, 0x205F, 0x3000], " "))
    clean = {k: v for k, v in table.items() if v is None or len(v) == 1}
    for cp in [*range(0xFB50, 0xFE00), *range(0xFE70, 0xFEFF)]:  # presentation forms: their NFKC letters
        nfkc = unicodedata.normalize("NFKC", chr(cp))
        if nfkc != chr(cp):
            table[cp] = nfkc.translate(clean).strip() or None
    return table


_CHARS = _char_map()
_MAPPED = re.compile("[" + "".join(map(re.escape, map(chr, sorted(_CHARS)))) + "]")  # text that needs _CHARS
_SPACES = re.compile(r"[ \t]+")

_UNITS = ["صفر", "واحد", "اثنين", "ثلاثة", "أربعة", "خمسة", "ستة", "سبعة", "ثمانية", "تسعة"]
_UNITS_FEM = ["", "", "", "ثلاث", "أربع", "خمس", "ست", "سبع", "ثماني", "تسع", "عشر"]  # 3-10 before a feminine noun
_TEENS = ["عشرة", "أحد عشر", "اثني عشر", "ثلاثة عشر", "أربعة عشر", "خمسة عشر", "ستة عشر", "سبعة عشر",
          "ثمانية عشر", "تسعة عشر"]
_TENS = ["", "", "عشرين", "ثلاثين", "أربعين", "خمسين", "ستين", "سبعين", "ثمانين", "تسعين"]
_HUNDREDS = ["", "مية", "ميتين", "ثلاثمية", "أربعمية", "خمسمية", "ستمية", "سبعمية", "ثمانمية", "تسعمية"]
# (one, two, 3-10 plural)
_SCALES = [(10**9, ("مليار", "مليارين", "مليارات")), (10**6, ("مليون", "مليونين", "ملايين")),
           (10**3, ("ألف", "ألفين", "آلاف"))]
_MSA = {"مية": "مئة", "ميتين": "مئتين", "ثلاثمية": "ثلاثمئة", "أربعمية": "أربعمئة", "خمسمية": "خمسمئة",
        "ستمية": "ستمئة", "سبعمية": "سبعمئة", "ثمانمية": "ثمانمئة", "تسعمية": "تسعمئة"}
_MONTHS = ["يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو", "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر",
           "ديسمبر"]
_ORD = ["", "أول", "ثاني", "ثالث", "رابع", "خامس", "سادس", "سابع", "ثامن", "تاسع", "عاشر"]
_ORD_FEM = ["", "أولى", "ثانية", "ثالثة", "رابعة", "خامسة", "سادسة", "سابعة", "ثامنة", "تاسعة", "عاشرة"]
# nouns a number after the article counts in order (الطابق الـ3: the third floor); feminine: ending in ة, or listed
_ORDINAL_NOUNS = frozenset(
    "الطابق الدور المركز الصف القرن اليوم الأسبوع الشهر العام الموسم الجزء الفصل الباب البند الشوط المستوى العدد "
    "الإصدار المؤتمر المهرجان المرة المرتبة الدورة الحلقة السنة النسخة المباراة الجولة الطبعة المرحلة الخطوة "
    "الصفحة الفقرة المادة الدرجة القمة الليلة الساعة الذكرى".split())
_FEMININE = frozenset({"الذكرى"})
_DEFINITE = re.compile(r"[وف]?(?:[بك]?(?=ال)|ل(?=ل))")  # proclitics before the article (للمرة: ل + المرة)

# style -> word
_PERCENT = {"colloquial": "بالمية", "msa": "بالمئة"}
_QUARTER = {"colloquial": "وربع", "msa": "والربع"}
_HALF = {"colloquial": "ونص", "msa": "والنصف"}
_AM = {"colloquial": "الصبح", "msa": "صباحا"}
_PM = {"colloquial": "المسا", "msa": "مساء"}
# counted nouns: (singular, dual, plural after 3-10, feminine)
_MINUTE = {"colloquial": ("دقيقة", "دقيقتين", "دقايق", True), "msa": ("دقيقة", "دقيقتين", "دقائق", True)}
_SECOND = ("ثانية", "ثانيتين", "ثواني", True)
_CENT = ("سنت", "سنتين", "سنتات", False)
_FILS = ("فلس", "فلسين", "فلوس", False)
_DIRHAM = ("درهم", "درهمين", "دراهم", False)
_DINAR = ("دينار", "دينارين", "دنانير", False)
_DOLLAR = (("دولار", "دولارين", "دولارات", False), _CENT)
_EURO = (("يورو", "اثنين يورو", "يورو", False), _CENT)
_POUND = (("جنيه إسترليني", "جنيهين إسترليني", "جنيهات إسترلينية", False), ("بنس", "بنسين", "بنسات", False))
_RIYAL = (("ريال", "ريالين", "ريالات", False), ("هللة", "هللتين", "هللات", True))
_QATARI = (("ريال قطري", "ريالين قطري", "ريالات قطرية", False), _DIRHAM)
_OMANI = (("ريال عماني", "ريالين عماني", "ريالات عمانية", False), None)  # 1000 baisa: decimals with فاصلة
_EGP = (("جنيه", "جنيهين", "جنيهات", False), ("قرش", "قرشين", "قروش", False))
_SHEKEL = (("شيكل", "شيكلين", "شواكل", False), ("أغورة", "أغورتين", "أغورات", True))
_CURRENCIES = {"$": _DOLLAR, "US$": _DOLLAR, "USD": _DOLLAR, "€": _EURO, "EUR": _EURO, "£": _POUND, "GBP": _POUND,
               "SAR": _RIYAL, "ر.س": _RIYAL, "AED": (_DIRHAM, _FILS), "د.إ": (_DIRHAM, _FILS), "EGP": _EGP,
               "ج.م": _EGP, "QAR": _QATARI, "KWD": (_DINAR, None), "د.ك": (_DINAR, None), "BHD": (_DINAR, None),
               "JOD": (_DINAR, None), "OMR": _OMANI, "₪": _SHEKEL, "ILS": _SHEKEL}
_UNIT_WORDS = {"km/h": "كيلومتر بالساعة", "كم/س": "كيلومتر بالساعة", "km": "كيلومتر", "كم": "كيلومتر",
               "cm": "سنتيمتر", "سم": "سنتيمتر", "mm": "مليمتر", "مم": "مليمتر", "m": "متر",
               "kg": {"colloquial": "كيلو", "msa": "كيلوغرام"}, "كغ": {"colloquial": "كيلو", "msa": "كيلوغرام"},
               "كجم": {"colloquial": "كيلو", "msa": "كيلوغرام"}, "mg": "مليغرام", "g": "غرام", "ml": "مليلتر",
               "مل": "مليلتر", "l": "لتر", "L": "لتر", "°C": "درجة مئوية", "°": "درجة", "GB": "غيغابايت",
               "MB": "ميغابايت"}
_FRACTIONS = {(1, 2): {"colloquial": "نص", "msa": "نصف"}, (1, 3): "ثلث", (2, 3): "ثلثين", (1, 4): "ربع",
              (3, 4): "ثلاثة أرباع"}


def _styled(word, style: str) -> str:
    return word[style] if isinstance(word, dict) else word


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
    """An integer as Arabic words (masculine counting form; negative: ناقص ...)."""
    if n < 0:
        return "ناقص " + number_words(-n, style)
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


def ordinal_words(n: int, feminine: bool = False, style: str = "colloquial") -> str:
    """1-99 as a definite ordinal (الثالث / الثالثة, الحادي والعشرين); other numbers as ال + the cardinal (المية)."""
    if not 1 <= n <= 99:
        return "ال" + number_words(n, style)
    if n <= 10:
        return "ال" + (_ORD_FEM if feminine else _ORD)[n]
    t, u = divmod(n, 10)
    unit = ("حادية" if feminine else "حادي") if u == 1 else (_ORD_FEM if feminine else _ORD)[u]
    if t == 1:
        return f"ال{unit} {'عشرة' if feminine else 'عشر'}"
    return f"ال{unit} وال{_TENS[t]}" if u else "ال" + _TENS[t]


def _count(n: int, noun: tuple, style: str, one: bool = False) -> str:
    """``n`` of a counted noun (singular, dual, plural, feminine): 1 the noun (one: + واحد), 2 its dual, 3-10 the plural
    (ثلاث before a feminine noun, ثلاثة before a masculine one), 103 too (مية وثلاثة دولارات), others the singular."""
    sing, dual, plural, fem = noun
    if n == 1:
        return f"{sing} {'واحدة' if fem else 'واحد'}" if one else sing
    if n == 2:
        return dual
    if 3 <= n <= 10:
        return f"{_UNITS_FEM[n] if fem else _UNITS[n] if n < 10 else 'عشرة'} {plural}"
    return f"{number_words(n, style)} {plural if 3 <= n % 100 <= 10 else sing}"


def _digits(s: str) -> str:
    return " ".join(_UNITS[int(d)] for d in s if d.isdigit())


def _integer(s: str, style: str) -> str:
    """Digits as said: a cardinal up to the billions; longer strings and codes with a leading zero digit by digit."""
    s = s.replace(",", "")
    if len(s) > 12 or (len(s) > 2 and s[0] == "0") or s == "00":
        return _digits(s)
    return number_words(int(s), style)


def _decimal(int_part: str, frac: str, style: str) -> str:
    """int_part فاصلة frac: the fraction without trailing zeros, its leading zeros as صفر, then a number (<= 3 digits)
    or digit by digit."""
    words = _integer(int_part or "0", style)
    if not frac:
        return words
    f = frac.rstrip("0") or "0"
    rest = f.lstrip("0")
    parts = [_UNITS[0]] * (len(f) - len(rest))
    if rest:
        parts.append(number_words(int(rest), style) if len(rest) <= 3 else _digits(rest))
    return f"{words} فاصلة {' '.join(parts)}"


def _money(int_part: str, frac: str | None, cur: str, style: str) -> str:
    noun, sub = _CURRENCIES[cur if cur in _CURRENCIES else cur.upper()]
    s = int_part.replace(",", "")
    frac = frac or ""
    if len(s) > 12:
        return f"{_decimal(s, frac, style)} {noun[0]}"
    n = int(s)
    if not frac.strip("0"):
        return _count(n, noun, style, one=True)
    if sub is None or len(frac) > 2:
        return f"{_decimal(s, frac, style)} {noun[0]}"
    cents = _count(int(frac.ljust(2, "0")), sub, style, one=True)
    return cents if n == 0 else f"{_count(n, noun, style, one=True)} و{cents}"


def _clock(h: int, m: int, s: int | None, suffix: str | None, style: str) -> str:
    minute = _MINUTE[style]
    if s is None:
        words = number_words(h, style)
        if m == 15:
            words += " " + _QUARTER[style]
        elif m == 30:
            words += " " + _HALF[style]
        elif m:
            words += " و" + _count(m, minute, style)
    else:
        parts = [number_words(h, style)] if h else []
        parts += [_count(m, minute, style)] if m else []
        parts += [_count(s, _SECOND, style)] if s else []
        words = " و".join(parts) or _UNITS[0]
    if suffix:
        words += " " + (_AM if suffix[0] in "صAa" else _PM)[style]
    return words


def _alternatives(words) -> str:
    return "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))


_DIGIT = re.compile(r"\d")
# an amount: integer (thousands commas allowed), decimals after a point or a decimal comma. The first digit leads and
# the look-behind follows it (not after [\d.,]): sre scans fast for a leading digit, a leading look-behind is tried at
# every position
_AMT = r"(\d(?<![\d.,]\d)(?:\d{0,2}(?:,\d{3})+(?!\d)|\d*))(?:\.(\d+)|,(\d{1,2})(?![\d,]))?"
_END = r"(?!\d|[.,]\d)"  # the number ends here (a sentence's full stop or comma may follow)
_AMPM = r"ص|م|[AaPp][Mm]|[AaPp]\.[Mm]\."
_CHAIN = r"[وف]?[بلك]|[وف]|[وف]?[بك]?ال|[وف]?لل"  # one-letter proclitics, the article with its proclitics
_PROCLITIC = re.compile(r"([وفبلكا]+)\s+(?=\d)")  # "بـ 3000" (tatweel gone): "ب 3000" -> "ب3000" (_join)
_CHAIN_FULL = re.compile(_CHAIN)
_AR_LETTER = re.compile(f"[{_AR}]")
_AR_GLUED = re.compile(rf"([{_AR}]+)(?=\d)")
_GLUED_AR = re.compile(rf"(\d)(?=[{_AR}])")
_GLUED_LATIN = re.compile(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])")
_DOCTOR = re.compile(rf"(?<![{_AR}.])(أ\.\s*)?د\.\s*(?=((?:أ\.\s*)?د\.|[{_AR}]{{2,}}))")  # before a name: _doctor
_DATE_DMY = re.compile(r"(?<![\d/.\-])(\d{1,2})([/.\-])(\d{1,2})\2(\d{4}|\d{2})(?![\d/\-]|[.,]\d)")
_DATE_YMD = re.compile(r"(?<![\d/.\-])(\d{4})([/.\-])(\d{1,2})\2(\d{1,2})(?![\d/\-]|[.,]\d)")
_TIME = re.compile(rf"(?<![\d:.,])(\d{{1,2}}):(\d{{2}})(?::(\d{{2}}))?(?![\d:]|[.,]\d)"
                   rf"(?:\s?({_AMPM})(?![A-Za-z{_AR}]))?")
_HOUR = re.compile(rf"(الساعة\s?)(\d{{1,2}})\s?({_AMPM})(?![A-Za-z{_AR}])")
_PHONE = re.compile(rf"(?<![\d.,])(?:\+\d[\d\- ]*\d|0\d{{7,}}|0\d{{1,4}}(?:[ \-]\d{{2,4}})+"
                    rf"|\d{{2,4}}(?:-\d{{2,4}}){{2,}}){_END}")  # >= 8 digits: _phone
_SIGN = re.compile(r"(?:(?<=[\s(\[«\"'])|^)([-+])(?=\d)")
_DOTTED = re.compile(rf"(?<![\d.,])\d+(?:\.\d+){{2,}}{_END}")
_ERA = re.compile(rf"(?<![\d.,])(\d{{4}})\s?(ه|م)(?![{_AR}])")
_CODES = "|".join(k for k in _CURRENCIES if k.isalpha() and k.isascii())
_CUR = (rf"(?<![A-Za-z{_AR}])(?:{_alternatives(k for k in _CURRENCIES if not (k.isalpha() and k.isascii()))}"
        rf"|(?i:{_CODES})(?![A-Za-z]))")  # codes in any case (aed100), not inside a word
_CUR_BEFORE = re.compile(rf"({_CUR})\s?{_AMT}{_END}")
_CUR_AFTER = re.compile(rf"{_AMT}\s?({_CUR})(?![{_AR}])")
_PCT_AFTER = re.compile(rf"{_AMT}\s?%")
_PCT_BEFORE = re.compile(rf"%\s?{_AMT}{_END}")
_UNIT = re.compile(rf"{_AMT}\s?({_alternatives(_UNIT_WORDS)})(?![A-Za-z{_AR}\d²³/])")
_FRACTION = re.compile(r"(?<![\d/.,])(\d{1,3})/(\d{1,3})(?![\d/]|[.,]\d)")
_ARTICLE_NUM = re.compile(rf"(?<![{_AR}])([وف]?[بك]?ال|[وف]?لل)(\d+){_END}")
_CENTURY = re.compile(rf"(القرن\s?)(\d{{1,2}}){_END}")
_PREV_WORD = re.compile(rf"([{_AR}]+)\s*$")
_PREV_TOKEN = re.compile(r"(\S+)\s*$")
_NUMBER_WORDS = frozenset(w for words in (_UNITS, _UNITS_FEM, _TEENS, _TENS, _HUNDREDS, _MSA.values(),
                                          *(scale for _, scale in _SCALES)) for word in words for w in word.split())
_LEADING_DOT = re.compile(r"(?:(?<=[\s(])|^)\.(\d+)")
_NUMBER = re.compile(_AMT.replace(r"(?<![\d.,]\d)", r"(?<!\d\d)"))  # after . or , too


def _article_number(m: re.Match, style: str) -> str:
    """ال + number: an ordinal after an ordinal-taking noun (المرة الـ3 -> المرة الثالثة), else the article stays."""
    chain, digits = m.group(1), m.group(2)
    prev = _PREV_WORD.search(m.string, max(0, m.start() - 24), m.start())
    proclitics = prev and _DEFINITE.match(prev.group(1))
    noun = prev.group(1)[proclitics.end():] if proclitics else ""
    if noun.startswith("ل") and not noun.startswith("ال"):
        noun = "ا" + noun  # للمرة -> المرة
    if noun not in _ORDINAL_NOUNS or len(digits) > 3 or not 1 <= int(digits) <= 100:
        return m.group(0)
    words = ordinal_words(int(digits), noun.endswith("ة") or noun in _FEMININE, style)
    return chain[:-2] + ("ل" + words[1:] if chain.endswith("لل") else words)


def has_arabic(text: str) -> bool:
    """Whether ``text`` has Arabic script (letters, marks, tatweel, Arabic-Indic digits): the inference wrappers
    normalize only such text."""
    return _ARABIC.search(text) is not None


def normalize(text: str, style: str = "colloquial") -> str:
    """``text`` as the model should read it (the rules of the module docstring); ``style`` "msa" or "colloquial"."""
    style = "msa" if style == "msa" else "colloquial"
    if _COMBINING_HAMZA.search(text):
        text = unicodedata.normalize("NFC", text)  # ا + hamza above -> أ, before the marks are dropped
    t = text.translate(_CHARS) if _MAPPED.search(text) else text
    if _DIGIT.search(t):
        for gate, rule, repl in _GLUE_RULES:
            if gate.search(t):
                t = rule.sub(repl, t)
        # each rule behind a gate (a substring or a quick pattern that any match holds): most texts skip most rules
        for gate, rule, words in _NUMBER_RULES:
            if (gate in t) if isinstance(gate, str) else gate.search(t):
                t = rule.sub(lambda m: words(m, style), t)
    if "د." in t:  # after the numbers: a number next to it is words by now, as in a second pass
        t = _DOCTOR.sub(_doctor, t)
    t = t.replace(",", "،").replace("?", "؟").replace(";", "؛")
    return (_SPACES.sub(" ", t) if "  " in t or "\t" in t else t).strip()


def _join(m: re.Match) -> str:
    """A proclitic or the article, as a word of its own, joins the number after it."""
    if (m.start() and _AR_LETTER.match(m.string, m.start() - 1)) or not _CHAIN_FULL.fullmatch(m.group(1)):
        return m.group(0)
    return m.group(1)


def _is_number(word: str) -> bool:
    """Digits or a number word (خمسة، وأربعين)."""
    return word[0].isdigit() or word in _NUMBER_WORDS or (word[0] == "و" and word[1:] in _NUMBER_WORDS)


def _doctor(m: re.Match) -> str:
    """د. أحمد -> دكتور أحمد; not next to a number (45 د.: minutes)."""
    prev = _PREV_TOKEN.search(m.string, max(0, m.start() - 24), m.start())
    if (prev and _is_number(prev.group(1))) or _is_number(m.group(2)):
        return m.group(0)
    return "أستاذ دكتور " if m.group(1) else "دكتور "


def _split_word(m: re.Match) -> str:
    """رقم5 -> رقم 5; a proclitic or the article stays on its number (ب5, ال5)."""
    return m.group(1) if _CHAIN_FULL.fullmatch(m.group(1)) else m.group(1) + " "


def _amount(m: re.Match, style: str) -> str:
    """The _AMT groups 1-3 (integer, decimals after a point or a comma) as words."""
    return _decimal(m.group(1), m.group(2) or m.group(3), style)


def _date(m: re.Match, day: str, month: str, year: str, style: str) -> str:
    """Day, month name, year; not a date (13/13/2024, 1.2.24): left to the later rules."""
    d, mo = int(day), int(month)
    if not (1 <= d <= 31 and 1 <= mo <= 12) or (m.group(2) == "." and len(year) == 2):
        return m.group(0)
    return f"{number_words(d, style)} {_MONTHS[mo - 1]} {number_words(int(year), style)}"


def _time(m: re.Match, style: str) -> str:
    h, mi, s = int(m.group(1)), int(m.group(2)), m.group(3) and int(m.group(3))
    if h > 24 or mi > 59 or (s or 0) > 59:
        return m.group(0)
    return _clock(h, mi, s, m.group(4), style)


def _hour(m: re.Match, style: str) -> str:
    """الساعة 8 م: the hour and the time of day."""
    return m.group(1) + _clock(int(m.group(2)), 0, None, m.group(3), style) if int(m.group(2)) <= 24 else m.group(0)


def _phone(m: re.Match, style: str) -> str:
    return _digits(m.group(0)) if sum(c.isdigit() for c in m.group(0)) >= 8 else m.group(0)


def _dotted(s: str, style: str) -> str:
    groups = s.split(".")
    if len(groups[0]) <= 3 and all(len(g) == 3 for g in groups[1:]):  # 1.000.000: thousands separators
        return _integer("".join(groups), style)
    return " نقطة ".join(_integer(g, style) for g in groups)


def _era(m: re.Match, style: str) -> str:
    """1445هـ (1000-1600) hijri, 2024م (1000-2199) gregorian; another number keeps its letter."""
    y, era = int(m.group(1)), m.group(2)
    if era == "ه" and 1000 <= y <= 1600:
        return f"{number_words(y, style)} هجري"
    if era == "م" and 1000 <= y <= 2199:
        return f"{number_words(y, style)} ميلادي"
    return m.group(0)


def _fraction(a: int, b: int, style: str) -> str:
    name = _FRACTIONS.get((a, b))
    return _styled(name, style) if name else f"{number_words(a, style)} على {number_words(b, style)}"


_GLUE_RULES = [  # (gate, rule, replacement)
    (re.compile(r"[وفبلك]\s+\d"), _PROCLITIC, _join),
    (re.compile("[A-Za-z]"), _GLUED_LATIN, " "),
    (re.compile(rf"\d[{_AR}]"), _GLUED_AR, r"\1 "),
    (re.compile(rf"[{_AR}]\d"), _AR_GLUED, _split_word),
]
_CUR_GATE = re.compile(r"[$€£₪A-Za-z]|[رجد]\.[سإمك]")
_NUMBER_RULES = [  # (gate, rule, words(match, style)), in the order of the module docstring
    (re.compile(r"\d[/.\-]\d{1,2}[/.\-]\d"), _DATE_DMY, lambda m, s: _date(m, m.group(1), m.group(3), m.group(4), s)),
    (re.compile(r"\d{4}[/.\-]\d"), _DATE_YMD, lambda m, s: _date(m, m.group(4), m.group(3), m.group(1), s)),
    (":", _TIME, _time),
    ("الساعة", _HOUR, _hour),
    (re.compile(r"\d[\d \-]{6,}\d"), _PHONE, _phone),
    (re.compile(r"[-+]\d"), _SIGN, lambda m, s: "ناقص " if m.group(1) == "-" else "زائد "),
    (re.compile(r"\d\.\d+\.\d"), _DOTTED, lambda m, s: _dotted(m.group(0), s)),
    (re.compile(rf"\d{{4}}\s?[هم](?![{_AR}])"), _ERA, _era),
    (_CUR_GATE, _CUR_BEFORE, lambda m, s: _money(m.group(2), m.group(3) or m.group(4), m.group(1), s)),
    (_CUR_GATE, _CUR_AFTER, lambda m, s: _money(m.group(1), m.group(2) or m.group(3), m.group(4), s)),
    ("%", _PCT_AFTER, lambda m, s: f"{_amount(m, s)} {_PERCENT[s]}"),
    ("%", _PCT_BEFORE, lambda m, s: f"{_amount(m, s)} {_PERCENT[s]}"),
    (re.compile(rf"\d\s?(?:[A-Za-z°]|(?:كم|سم|مم|كغ|كجم|مل)(?![{_AR}]))"), _UNIT,
     lambda m, s: f"{_amount(m, s)} {_styled(_UNIT_WORDS[m.group(4)], s)}"),
    ("/", _FRACTION, lambda m, s: _fraction(int(m.group(1)), int(m.group(2)), s)),
    ("القرن", _CENTURY, lambda m, s: m.group(1) + ordinal_words(int(m.group(2)), False, s)),
    (re.compile(r"(?:ال|لل)\d"), _ARTICLE_NUM, _article_number),
    (re.compile(r"\.\d"), _LEADING_DOT, lambda m, s: _decimal("0", m.group(1), s)),
    (_DIGIT, _NUMBER, _amount),
]

"""nanovllm_voxcpm.arabic_text: one test per rule of its docstring, the eval texts unchanged but for the documented
unifications, the numbers set without a digit left."""

import json
import os
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

import pytest

from nanovllm_voxcpm.arabic_text import has_arabic, normalize, number_words, ordinal_words

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).parent / "data"
EVAL_V1 = Path("/workspace/data/arabic/eval/v1/items.jsonl")  # the full eval set, where present
NUMBERS_V1 = Path("/workspace/voxcpm2-arabic-training/arabic_voxcpm/eval/sets/numbers_v1.jsonl")


def texts(path: Path) -> list[str]:
    return [json.loads(line)["text"] for line in path.open(encoding="utf-8") if line.strip()]


def check(cases, style="colloquial"):
    for text, expected in cases:
        assert normalize(text, style) == expected, text
        assert normalize(expected, style) == expected, f"not idempotent: {text}"


def test_letter_variants():
    check([("کتاب یا ھلا", "كتاب يا هلا"), ("ہ ە", "ه ه"), ("ﻻ", "لا"), ("ﷺ", "صلى الله عليه وسلم"),
           ("مـــرحبا", "مرحبا")])


def test_diacritics_removed_letters_kept():
    check([("شكراً", "شكرا"), ("مُحَمَّدٌ", "محمد"), ("هٰذا", "هذا"),
           ("ا\u0654نا", "أنا"), ("ا\u0653", "آ")])  # combining hamza / madda: composed, not dropped


def test_invisible_characters_removed():
    check([("\u200fمرحبا\u200f", "مرحبا"), ("الساعة\u00a05", "الساعة خمسة"), ("\u202b5\u202c", "خمسة")])


def test_digit_scripts():
    check([("٢٧", "سبعة وعشرين"), ("۴۵", "خمسة وأربعين"), ("١٢٫٥", "اثني عشر فاصلة خمسة"), ("١٬٠٠٠", "ألف"),
           ("٥٠٪", "خمسين بالمية")])


def test_punctuation_unified():
    check([("كيف حالك? تمام, الحمد لله; يلا", "كيف حالك؟ تمام، الحمد لله؛ يلا"), ("شو؟ ،", "شو؟ ،"),
           ("1, 2", "واحد، اثنين")])


def test_whitespace_collapsed():
    check([("  هلا   والله\t", "هلا والله")])


def test_proclitic_and_article_join_number():
    check([("بـ 3000", "بثلاثة آلاف"), ("ب3000", "بثلاثة آلاف"), ("و 5", "وخمسة"), ("وبـ 5", "وبخمسة"),
           ("بالـ 50 مليون", "بالخمسين مليون"), ("ال5", "الخمسة"), ("عندك 5", "عندك خمسة"), ("مال 5", "مال خمسة")])


def test_glued_words_split():
    check([("رقم5", "رقم خمسة"), ("500متر", "خمسمية متر"), ("abc123", "abc مية وثلاثة وعشرين"), ("5G", "خمسة G"),
           ("iPhone15", "iPhone خمسة عشر"), ("v2.0", "v اثنين فاصلة صفر")])


def test_abbreviations():
    check([("د. أحمد", "دكتور أحمد"), ("د.أحمد", "دكتور أحمد"), ("أ.د. سامي", "أستاذ دكتور سامي"),
           ("قال د. سامي", "قال دكتور سامي"), ("ولد. أحمد", "ولد. أحمد"), ("5 د. فقط", "خمسة د. فقط"),
           ("مدة خمسة وأربعين د. والفيلم", "مدة خمسة وأربعين د. والفيلم"), ("د.5", "د.خمسة")])


def test_dates():
    may_12 = "اثني عشر مايو ألفين وأربعة وعشرين"
    check([("12/05/2024", may_12), ("12-5-2024", may_12), ("12.05.2024", may_12), ("2024-05-12", may_12),
           ("2024/5/12", may_12), ("1/1/24", "واحد يناير أربعة وعشرين"), ("يوم 12/05/2024.", f"يوم {may_12}."),
           ("13/13/2024", "ثلاثة عشر/ثلاثة عشر/ألفين وأربعة وعشرين")])  # not a date: numbers


def test_times():
    check([("9:00", "تسعة"), ("9:01", "تسعة ودقيقة"), ("9:02", "تسعة ودقيقتين"), ("9:05", "تسعة وخمس دقايق"),
           ("9:10", "تسعة وعشر دقايق"), ("9:11", "تسعة وأحد عشر دقيقة"), ("9:25", "تسعة وخمسة وعشرين دقيقة"),
           ("7:15", "سبعة وربع"), ("10:30", "عشرة ونص"), ("09:05", "تسعة وخمس دقايق"), ("24:00", "أربعة وعشرين"),
           ("10:30:15", "عشرة وثلاثين دقيقة وخمسة عشر ثانية"), ("0:05:03", "خمس دقايق وثلاث ثواني"),
           ("10:30 ص", "عشرة ونص الصبح"), ("10:30ص", "عشرة ونص الصبح"), ("8:00pm", "ثمانية المسا"),
           ("الساعة 8 م", "الساعة ثمانية المسا"), ("10:30 مساء", "عشرة ونص مساء"), ("25:61", "خمسة وعشرين:واحد وستين")])
    check([("9:05", "تسعة وخمس دقائق"), ("10:30", "عشرة والنصف"), ("7:15 ص", "سبعة والربع صباحا")], "msa")


def test_phone_numbers():
    check([("0551234567", "صفر خمسة خمسة واحد اثنين ثلاثة أربعة خمسة ستة سبعة"),
           ("+966 55 123 4567", "تسعة ستة ستة خمسة خمسة واحد اثنين ثلاثة أربعة خمسة ستة سبعة"),
           ("055 123 4567", "صفر خمسة خمسة واحد اثنين ثلاثة أربعة خمسة ستة سبعة"),
           ("800-123-4567", "ثمانية صفر صفر واحد اثنين ثلاثة أربعة خمسة ستة سبعة")])
    # not phone numbers
    check([("10000000", "عشرة ملايين"), ("100 200 300", "مية ميتين ثلاثمية"),
           ("2010 - 2015", "ألفين وعشرة - ألفين وخمسة عشر"), ("2010-2015", "ألفين وعشرة-ألفين وخمسة عشر")])


def test_signs():
    check([("-5", "ناقص خمسة"), ("الحرارة -5 درجات", "الحرارة ناقص خمسة درجات"), ("(-3)", "(ناقص ثلاثة)"),
           ("−2.5", "ناقص اثنين فاصلة خمسة"), ("+5", "زائد خمسة"), ("-5%", "ناقص خمسة بالمية"),
           ("COVID-19", "COVID-تسعة عشر")])


def test_dotted_groups():
    check([("192.168.1.1", "مية واثنين وتسعين نقطة مية وثمانية وستين نقطة واحد نقطة واحد"),
           ("1.2.3", "واحد نقطة اثنين نقطة ثلاثة"), ("1.000.000", "مليون")])


def test_years_with_era():
    check([("1445هـ", "ألف وأربعمية وخمسة وأربعين هجري"), ("عام 2024م", "عام ألفين وأربعة وعشرين ميلادي"),
           ("2024 م.", "ألفين وأربعة وعشرين ميلادي."), ("سنة 1998", "سنة ألف وتسعمية وثمانية وتسعين"),
           ("5 مرات", "خمسة مرات")])


def test_currency_counted():
    check([("1$", "دولار واحد"), ("2$", "دولارين"), ("3 $", "ثلاثة دولارات"), ("$10", "عشرة دولارات"),
           ("11$", "أحد عشر دولار"), ("$103", "مية وثلاثة دولارات"), ("$1,000,000", "مليون دولار"),
           ("5€", "خمسة يورو"), ("15 £", "خمسة عشر جنيه إسترليني"), ("50₪", "خمسين شيكل"), ("US$100", "مية دولار")])


def test_currency_codes_and_abbreviations():
    check([("AED100", "مية درهم"), ("100AED", "مية درهم"), ("aed 50", "خمسين درهم"), ("SAR 3", "ثلاثة ريالات"),
           ("100 ر.س", "مية ريال"), ("20 ج.م", "عشرين جنيه"), ("3 USDT", "ثلاثة USDT")])


def test_currency_decimals():
    check([("12.50$", "اثني عشر دولار وخمسين سنت"), ("$0.99", "تسعة وتسعين سنت"), ("1.05$", "دولار واحد وخمسة سنتات"),
           ("12.00$", "اثني عشر دولار"), ("2.5 SAR", "ريالين وخمسين هللة"), ("2.5 KWD", "اثنين فاصلة خمسة دينار"),
           ("12,5€", "اثني عشر يورو وخمسين سنت")])


def test_percent():
    check([("50%", "خمسين بالمية"), ("50 %", "خمسين بالمية"), ("%50", "خمسين بالمية"),
           ("3.5%", "ثلاثة فاصلة خمسة بالمية")])
    check([("3.5%", "ثلاثة فاصلة خمسة بالمئة")], "msa")


def test_units():
    check([("100km", "مية كيلومتر"), ("120 km/h", "مية وعشرين كيلومتر بالساعة"), ("5 kg", "خمسة كيلو"),
           ("100m", "مية متر"), ("2.5 l", "اثنين فاصلة خمسة لتر"), ("47°", "سبعة وأربعين درجة"),
           ("30°C", "ثلاثين درجة مئوية"), ("5 كم", "خمسة كيلومتر"), ("10 سم", "عشرة سنتيمتر"),
           ("3 مليون", "ثلاثة مليون"), ("5 كما قال", "خمسة كما قال"), ("كم عمرك", "كم عمرك")])
    check([("5 kg", "خمسة كيلوغرام"), ("5 كجم", "خمسة كيلوغرام")], "msa")


def test_fractions():
    check([("1/2", "نص"), ("1/3", "ثلث"), ("2/3", "ثلثين"), ("1/4", "ربع"), ("3/4", "ثلاثة أرباع"),
           ("24/7", "أربعة وعشرين على سبعة"), ("2/5", "اثنين على خمسة")])
    check([("1/2", "نصف")], "msa")


def test_ordinals():
    check([("الطابق الـ3", "الطابق الثالث"), ("المرة الـ3", "المرة الثالثة"), ("للمرة الـ2", "للمرة الثانية"),
           ("الدورة الـ21", "الدورة الحادية والعشرين"), ("المركز الـ15", "المركز الخامس عشر"),
           ("الذكرى الـ50", "الذكرى الخمسين"), ("اليوم الـ100", "اليوم المية"), ("القرن 21", "القرن الحادي والعشرين"),
           ("في القرن الـ21", "في القرن الحادي والعشرين"), ("الطابق بالـ3", "الطابق بالثالث")])
    # the article on a cardinal: after other words, or a number past 100
    check([("خلال الـ3 أيام", "خلال الثلاثة أيام"), ("مونديال الـ2022", "مونديال الألفين واثنين وعشرين"),
           ("الطابق الـ150", "الطابق المية وخمسين")])
    assert ordinal_words(1) == "الأول" and ordinal_words(1, feminine=True) == "الأولى"
    assert ordinal_words(12, feminine=True) == "الثانية عشرة" and ordinal_words(30) == "الثلاثين"


def test_cardinals():
    cases = {0: "صفر", 11: "أحد عشر", 21: "واحد وعشرين", 100: "مية", 200: "ميتين", 1000: "ألف", 2000: "ألفين",
             3000: "ثلاثة آلاف", 11000: "أحد عشر ألف", 2024: "ألفين وأربعة وعشرين", 10**6: "مليون",
             2 * 10**6: "مليونين", 10**9: "مليار", 123456789012: "مية وثلاثة وعشرين مليار وأربعمية وستة وخمسين "
             "مليون وسبعمية وتسعة وثمانين ألف واثني عشر", -5: "ناقص خمسة"}
    for n, words in cases.items():
        assert number_words(n) == words, n
    assert number_words(250, "msa") == "مئتين وخمسين" and number_words(300, "msa") == "ثلاثمئة"
    check([("200", "مئتين")], "msa")


def test_integers_and_digit_strings():
    check([("1,250", "ألف وميتين وخمسين"), ("05", "خمسة"), ("007", "صفر صفر سبعة"), ("00", "صفر صفر"),
           ("999999999999", number_words(999999999999)),
           ("1234567890123", "واحد اثنين ثلاثة أربعة خمسة ستة سبعة ثمانية تسعة صفر واحد اثنين ثلاثة")])


def test_decimals():
    check([("3.5", "ثلاثة فاصلة خمسة"), (".5", "صفر فاصلة خمسة"), ("12,5", "اثني عشر فاصلة خمسة"),
           ("12.50", "اثني عشر فاصلة خمسة"), ("2.0", "اثنين فاصلة صفر"), ("1.05", "واحد فاصلة صفر خمسة"),
           ("2.125", "اثنين فاصلة مية وخمسة وعشرين"), ("3.14159", "ثلاثة فاصلة واحد أربعة واحد خمسة تسعة"),
           ("1,234.5", "ألف وميتين وأربعة وثلاثين فاصلة خمسة"), ("انتهى 3.", "انتهى ثلاثة.")])


def test_latin_left_alone():
    check([("GPS و CNN", "GPS و CNN"), ("Hello يا صاحبي", "Hello يا صاحبي")])


def test_has_arabic():
    assert has_arabic("هلا") and has_arabic("ﻻ") and has_arabic("ـــ") and has_arabic("٣")
    assert not has_arabic("मीटिंग 9 बजे") and not has_arabic("meet at 9")


def unify(text: str) -> str:
    """The documented unifications alone (letters, presentation forms, marks, punctuation, spaces)."""
    text = unicodedata.normalize("NFC", text)
    text = "".join(unicodedata.normalize("NFKC", c) if 0xFB50 <= ord(c) <= 0xFEFC else c for c in text)
    text = re.sub("[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed\u0640]", "", text)
    text = text.translate(str.maketrans({"ک": "ك", "ی": "ي", "ھ": "ه", "ہ": "ه", "ە": "ه", ",": "،", "?": "؟",
                                         ";": "؛"}))
    return " ".join(text.split())


@pytest.mark.parametrize("path", [DATA / "eval_v1_sample.jsonl", EVAL_V1], ids=["sample", "full"])
def test_eval_texts_only_unified(path):
    """The training-distribution transcripts (no digits) change only by the documented unifications."""
    if not path.exists():
        pytest.skip(f"{path} absent")
    for text in texts(path):
        assert normalize(text) == unify(text), text


@pytest.mark.parametrize("path", [DATA / "numbers_v1.jsonl", NUMBERS_V1], ids=["copy", "training-repo"])
def test_numbers_set_has_no_digits_left(path):
    if not path.exists():
        pytest.skip(f"{path} absent")
    for text in texts(path):
        for style in ("colloquial", "msa"):
            out = normalize(text, style)
            assert not re.search(r"[0-9٠-٩۰-۹]", out), (text, out)


def test_idempotent():
    for text in texts(DATA / "numbers_v1.jsonl") + texts(DATA / "eval_v1_sample.jsonl"):
        once = normalize(text)
        assert normalize(once) == once, text


def test_fast():
    """It runs per request on the server's event loop: well under 1 ms for a 200-character number-heavy sentence."""
    text = ("الموعد يوم 12/05/2024 الساعة 10:30 ص، والسعر 3499 ريال بعد خصم 15% يعني وفرت 12.50$ تقريبا، اتصل على "
            "0551234567 أو زور الطابق الـ3 في العمارة رقم 27، المسافة 45 كم والحرارة -5 درجات.")
    normalize(text)
    n = 200
    start = time.perf_counter()
    for _ in range(n):
        normalize(text)
    assert (time.perf_counter() - start) / n < 1e-3


def test_import_needs_no_engine():
    """The training repo and these tests import the normalizer alone: no torch, no flash-attn."""
    code = ("import sys, nanovllm_voxcpm.arabic_text; "
            "assert not {'torch', 'flash_attn', 'numpy'} & set(sys.modules), sorted(sys.modules)")
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    subprocess.run([sys.executable, "-c", code], check=True, env=env, cwd=ROOT)

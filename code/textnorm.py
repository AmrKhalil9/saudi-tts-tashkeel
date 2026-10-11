"""Arabic text normalization for CER/WER scoring (reported verbatim in the paper's appendix).

Applied identically to reference and ASR hypothesis:
 1. Remove diacritics (U+064B-U+0652, U+0670) and tatweel (U+0640).
 2. Unify letter variants: أ إ آ ٱ -> ا ; ى -> ي ; ة -> ه ; ؤ -> و ; ئ -> ي.
 3. Map Arabic-Indic digits to ASCII; join digit groups separated only by spaces or hyphens (ASR systems
    write "8952 5376" for a spoken 8-digit ID); then expand digit strings to words:
    5+ digits (IDs, phones, order numbers) are read digit by digit, shorter ones as a cardinal (num2words, ar).
 4. Drop Latin-script tokens (code-switched English) from both sides; they are scored separately.
 5. Remove punctuation; collapse whitespace.
CER is computed on the normalized string with spaces removed; WER on whitespace tokens.
"""
import re
from num2words import num2words

DIAC = re.compile(r"[ً-ْٰـ]")
LETTERS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي"})
INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
DIGIT_NAMES = ["صفر", "واحد", "اثنين", "ثلاثه", "اربعه", "خمسه", "سته", "سبعه", "ثمانيه", "تسعه"]


def _num(m):
    s = m.group(0)
    if len(s) >= 5:
        return " " + " ".join(DIGIT_NAMES[int(c)] for c in s) + " "
    return " " + num2words(int(s), lang="ar").replace(" و ", " و") + " "


def norm(t):
    t = DIAC.sub("", t).translate(INDIC)
    t = re.sub(r"\d+(?:[ \-]+\d+)+", lambda m: re.sub(r"[ \-]", "", m.group(0)), t)
    t = re.sub(r"\d+", _num, t)
    t = DIAC.sub("", t).translate(LETTERS)
    t = re.sub(r"[A-Za-z][A-Za-z0-9\-]*", " ", t)
    t = re.sub(r"[^ء-ي\s]", " ", t)
    # case variants of number words are not pronunciation errors (dialect reads the -iin / -ayn form)
    t = re.sub(r"\b(و?)(عشر|ثلاث|اربع|خمس|ست|سبع|ثمان|تسع)ون\b", r"\1\2ين", t)
    t = re.sub(r"\b(و?)اثنان\b", r"\1اثنين", t)
    return re.sub(r"\s+", " ", t).strip()


_DIGIT_WORDS = {w: str(i) for i, w in enumerate(DIGIT_NAMES)}
_DIGIT_WORDS.update({"اثنان": "2", "ثلاث": "3", "اربع": "4", "خمس": "5", "ست": "6", "سبع": "7", "ثمان": "8", "ثماني": "8", "تسع": "9"})


def digits_of(t):
    """The digit sequence a listener would take away: ASCII/Indic digits kept, single-digit number words
    (with or without a leading wa-) mapped to digits, everything else ignored. Used for entity accuracy."""
    t = DIAC.sub("", t).translate(INDIC).translate(LETTERS)
    out = []
    for tok in re.findall(r"\d+|[ء-ي]+", t):
        if tok.isdigit():
            out.append(tok)
        else:
            w = tok[1:] if tok.startswith("و") and tok[1:] in _DIGIT_WORDS else tok
            if w in _DIGIT_WORDS:
                out.append(_DIGIT_WORDS[w])
    return "".join(out)


def _lev(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return d[len(b)]


def cer(ref, hyp):
    r, h = norm(ref).replace(" ", ""), norm(hyp).replace(" ", "")
    return _lev(r, h) / max(1, len(r)), len(r)


def wer(ref, hyp):
    r, h = norm(ref).split(), norm(hyp).split()
    return _lev(r, h) / max(1, len(r)), len(r)


if __name__ == "__main__":
    print(norm("رقم الطلب 30008322، والمبلغ ٣٧٣ ريالًا عبر Apple Pay."))
    print(cer("موعدك يوم 5 مايو", "موعدك يوم خمسة مايو"))
    print(cer("رقم طلبكم هو ثمانية تسعة خمسة اثنين خمسة ثلاثة سبعة ستة", "رقم طلبكم هو 8952 5376"))
    print(digits_of("طلبك رقمو خمسة سبعة أربعة ثلاثة أربعة أربعة سبعة ثمانية"), digits_of("طلبك رقمه 5743 4478"))

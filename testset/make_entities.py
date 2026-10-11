"""Generate the entity-heavy register: 200 sentences, each in a digits form and a spelled-out form.

All numbers are synthetic (seeded RNG). Phone numbers use the 0500 00x xxx test-style block and
national-ID-like numbers start with 1 but are random; none are real.
Output: entities.jsonl with id, register, frame (msa|najdi|hijazi), entity_types, text_digits, text_spelled.
"""
import json, random
from num2words import num2words

rng = random.Random(20261010)
W = lambda n: num2words(n, lang="ar").replace(" و ", " و")


def digits_spoken(s):
    """Read a digit string one digit at a time (how IDs and phones are read on the phone)."""
    names = ["صفر", "واحد", "اثنين", "ثلاثة", "أربعة", "خمسة", "ستة", "سبعة", "ثمانية", "تسعة"]
    return " ".join(names[int(c)] for c in s)


def phone():
    return "05000" + "".join(str(rng.randint(0, 9)) for _ in range(5))


def nid():
    return "1" + "".join(str(rng.randint(0, 9)) for _ in range(9))


def order():
    return "".join(str(rng.randint(0, 9)) for _ in range(8))


MONTHS = ["يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو", "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر"]
NAMES = ["عبدالله العتيبي", "نورة القحطاني", "فهد الشهري", "ريم الدوسري", "سلطان الحربي", "هيفاء الغامدي",
         "تركي المطيري", "لمى الزهراني", "ماجد السبيعي", "أروى الشمري", "بندر العنزي", "جواهر الرشيد"]
ENGLISH = ["Apple Pay", "Wi-Fi", "OTP", "Visa", "PDF", "Bluetooth", "router", "email", "link", "app", "update", "password"]

# (frame, types, template); {N} amount, {P} phone, {I} id, {O} order, {D} day, {M} month, {Y} year,
# {H} hour, {MIN} minutes, {NAME} name, {EN} English word
T = [
    ("msa", "amount", "المبلغ المستحق على حسابكم هو {N} ريالًا."),
    ("msa", "amount", "تم خصم {N} ريالًا من رصيدكم."),
    ("msa", "amount,date", "يرجى سداد مبلغ {N} ريالًا قبل {D} {M}."),
    ("msa", "phone", "سيتم التواصل معكم على الرقم {P}."),
    ("msa", "id", "يرجى التأكد من رقم الهوية {I}."),
    ("msa", "order", "رقم طلبكم هو {O}، يرجى الاحتفاظ به."),
    ("msa", "date,time", "موعدكم يوم {D} {M} {Y} الساعة {H}."),
    ("msa", "time", "تبدأ الصيانة الساعة {H} وتستمر {MIN} دقيقة."),
    ("msa", "name", "يسعدنا خدمتكم يا أستاذ {NAME}."),
    ("msa", "name,order", "تم تسجيل الطلب رقم {O} باسم {NAME}."),
    ("msa", "english", "يمكنكم الدفع عبر {ENPAY} بكل سهولة."),
    ("msa", "english", "يرجى إدخال رمز {ENCODE} المرسل إلى جوالكم."),
    ("najdi", "amount", "عليك {N} ريال، تبي تسددها الحين؟"),
    ("najdi", "amount", "رجعنا لك {N} ريال على البطاقة."),
    ("najdi", "phone", "بنتصل عليك على الرقم {P}، صح؟"),
    ("najdi", "id", "عطني رقم الهوية الله يحفظك، {I} صح؟"),
    ("najdi", "order", "طلبك رقمه {O}، وبيوصلك بكرة."),
    ("najdi", "date,time", "موعدك يوم {D} {M} الساعة {H}."),
    ("najdi", "name", "هلا أستاذ {NAME}، وش أقدر أخدمك؟"),
    ("najdi", "english", "فعّل {ENSET} من الإعدادات وبيزبط معك."),
    ("hijazi", "amount", "عليك {N} ريال، تبغى تسددها دحين؟"),
    ("hijazi", "phone", "حنتصل عليك على الرقم {P}، صح؟"),
    ("hijazi", "id", "ممكن رقم الهوية؟ {I} صح؟"),
    ("hijazi", "order", "طلبك رقمو {O}، وحيوصلك بكرة."),
    ("hijazi", "date,time", "موعدك يوم {D} {M} الساعة {H}."),
    ("hijazi", "name", "أهلين أستاذة {NAME}، إيش أقدر أساعدك؟"),
    ("hijazi", "english", "شغّل {ENSET} من الإعدادات وحيمشي معاك."),
]


def render(tpl, v, spelled):
    out = tpl
    if spelled:
        out = out.replace("{N}", W(v["N"])).replace("{P}", digits_spoken(v["P"])).replace("{I}", digits_spoken(v["I"]))
        out = out.replace("{O}", digits_spoken(v["O"])).replace("{D}", W(v["D"])).replace("{Y}", W(v["Y"]))
        out = out.replace("{H}", W(v["H"])).replace("{MIN}", W(v["MIN"]))
    else:
        for k in ("N", "P", "I", "O", "D", "Y", "H", "MIN"):
            out = out.replace("{%s}" % k, str(v[k]))
    return (out.replace("{M}", v["M"]).replace("{NAME}", v["NAME"]).replace("{ENPAY}", v["ENPAY"])
            .replace("{ENCODE}", v["ENCODE"]).replace("{ENSET}", v["ENSET"]))


rows = []
i = 0
while len(rows) < 200:
    frame, types, tpl = T[i % len(T)]
    i += 1
    v = {"N": rng.choice([rng.randint(15, 999), rng.randint(1000, 25000)]), "P": phone(), "I": nid(), "O": order(),
         "D": rng.randint(1, 28), "M": rng.choice(MONTHS), "Y": rng.choice([2026, 2027]), "H": rng.randint(1, 12),
         "MIN": rng.choice([15, 20, 30, 45, 90]), "NAME": rng.choice(NAMES), "ENPAY": rng.choice(["Apple Pay", "Visa", "Mada Pay", "STC Pay"]),
         "ENCODE": rng.choice(["OTP", "PIN", "QR"]), "ENSET": rng.choice(["Wi-Fi", "Bluetooth", "Face ID", "5G"])}
    rows.append({"id": f"ent{len(rows)+1:03d}", "register": "entity", "frame": frame, "entity_types": types,
                 "text_digits": render(tpl, v, False), "text_spelled": render(tpl, v, True)})

with open("entities.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(rows), "rows;", rows[0], rows[3], rows[16], sep="\n")

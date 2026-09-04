# -*- coding: utf-8 -*-
"""
Is the input filter's 99.76% real?

The filter is trained with positives = Arabic medical Q&A and negatives = Arabic product
reviews + news. Both negative sources are DECLARATIVE prose, while every positive is a
QUESTION. So the filter can score 99.76% on its own test split by partly learning
"is this a question?" instead of "is this medical?" — and that shortcut collapses on
question-form non-medical text, which is exactly what a user types into a symptom box.

This script probes the filter with held-out, question-form non-medical text (never seen in
training) and reports the real rejection rate on it. Writes models/ar/filter_robustness.json.

Run:  python check_filter_robustness.py
"""
import os, json, time
import ar_service
from ar_utils import normalize_ar

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "models", "ar", "filter_robustness.json")

# Question-form NON-medical Arabic text — the distribution the negatives never covered.
HARD_NEGATIVES = [
    "ما هي عاصمة فرنسا وكم عدد سكانها",
    "كيف اتعلم البرمجة بلغة بايثون من الصفر وما هي افضل الدورات",
    "ما هو افضل هاتف يمكن شراؤه في حدود الف ريال",
    "كيف اطبخ الرز مع الدجاج بطريقة لذيذة وسريعة في البيت",
    "من فاز بكاس العالم لكرة القدم هذا العام ومن سجل الهدف",
    "ما هي افضل طريقة لتعلم اللغة الانجليزية بسرعة",
    "كم تبعد مدينة الرياض عن مدينة جدة بالسيارة",
    "ما هي شروط الحصول على رخصة قيادة جديدة",
    "كيف اصلح مكيف السيارة اذا توقف عن التبريد",
    "ما هو سعر صرف الدولار مقابل الريال اليوم",
    "متى يبدا الدوام الدراسي في المدارس هذا العام",
    "كيف افتح حساب بنكي جديد وما هي الاوراق المطلوبة",
]

# Real medical complaints — must still PASS (guard against over-rejecting).
POSITIVES = [
    "اعاني من حرقان شديد عند التبول مع الم اسفل البطن ورغبة متكررة في التبول",
    "عندي الم في الصدر وضيق في التنفس عند بذل الجهد مع خفقان في القلب",
    "عندي بقع حمراء منتشرة وحكة على الذراعين والجلد جاف ومتشقق",
    "اشعر بحرقة في المعدة بعد الاكل مع ارتجاع حمضي وطعم مر في الحلق",
    "الم شديد ومستمر في الرقبة والكتف ينتشر الى الذراع مع تنميل في الاصابع",
    "ابني عمره سنتين عنده اسهال شديد وقيء متكرر منذ يومين مع حرارة",
]


def main():
    assert ar_service.load(), "Arabic system not loaded — train it first."
    f = ar_service._filter
    assert f is not None, "No filter.joblib — run ar_filter.py first."
    model, th = f["model"], f.get("threshold", 0.5)

    def p_medical(t):
        return float(model.predict_proba([normalize_ar(t)])[0][1])

    neg_scores = [p_medical(t) for t in HARD_NEGATIVES]
    pos_scores = [p_medical(t) for t in POSITIVES]
    neg_rejected = sum(1 for s in neg_scores if s < th)
    pos_passed = sum(1 for s in pos_scores if s >= th)

    print(f"{'hard NON-medical (question form)':52} {'P(medical)':>10}  verdict")
    print("-" * 78)
    for t, s in zip(HARD_NEGATIVES, neg_scores):
        print(f"  {t[:50]:50} {s:10.3f}  {'REJECT ok' if s < th else 'LEAKED THROUGH'}")
    print(f"\n{'real medical complaints':52} {'P(medical)':>10}  verdict")
    print("-" * 78)
    for t, s in zip(POSITIVES, pos_scores):
        print(f"  {t[:50]:50} {s:10.3f}  {'pass ok' if s >= th else 'WRONGLY REJECTED'}")

    claimed = None
    fm = os.path.join(HERE, "models", "ar", "filter_metrics.json")
    if os.path.exists(fm):
        claimed = json.load(open(fm, encoding="utf-8")).get("nonmedical_rejection_rate")

    report = {
        "question": "Does the filter's reported non-medical rejection rate hold on question-form non-medical text?",
        "why_it_matters": ("Training negatives were reviews + news (declarative) while all positives were "
                           "questions, so the reported rate may reflect a question/statement shortcut."),
        "claimed_nonmedical_rejection_rate_on_own_testset": claimed,
        "measured_rejection_rate_on_hard_question_negatives": round(neg_rejected / len(HARD_NEGATIVES), 4),
        "n_hard_negatives": len(HARD_NEGATIVES),
        "hard_negatives_leaked_through": len(HARD_NEGATIVES) - neg_rejected,
        "medical_pass_rate": round(pos_passed / len(POSITIVES), 4),
        "threshold": th,
        "leaked_examples": [t for t, s in zip(HARD_NEGATIVES, neg_scores) if s >= th],
        "mitigation_in_place": ("The router abstains below 30% confidence, so leaked non-medical text still "
                                "surfaces a low-confidence warning rather than a confident diagnosis."),
        "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    json.dump(report, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n" + "=" * 78)
    print(f"  claimed rejection (own test set) : {claimed}")
    print(f"  MEASURED on question-form negatives: {report['measured_rejection_rate_on_hard_question_negatives']} "
          f"({report['hard_negatives_leaked_through']}/{len(HARD_NEGATIVES)} leaked through)")
    print(f"  medical pass rate (must stay high): {report['medical_pass_rate']}")
    print("=" * 78)
    print("FILTER_ROBUSTNESS_DONE")


if __name__ == "__main__":
    main()

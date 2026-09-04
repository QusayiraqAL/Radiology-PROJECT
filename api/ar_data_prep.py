# -*- coding: utf-8 -*-
"""
Phase 1 — Aggregate & clean real Arabic medical data (>=200k genuine entries).

Real HuggingFace sources:
  * hajerbchn/arabic-medical-qa      (~144k Arabic Q, 20 specialties)   -> router training
  * madilcy/...MERGED-MAQA...        (Arabic QA w/ specialty)           -> router training
  * Ahmed-Selem/Shifaa_...           (~55k, fine 'Hierarchical Diagnosis')-> per-category models
  * MKamil/arabic_medical_50k        (~49k Arabic medical text)         -> filter positives / corpus

Outputs (data/arabic/):
  * router_data.parquet   text, category            (>=200k category-labelled -> specialty router)
  * labeled_shifaa.parquet text, category, diagnosis (fine per-category models)
  * corpus.parquet        text, source              (medical positives for the input filter)
  * stats.json
"""
import os
import re
import json
import pandas as pd
from huggingface_hub import hf_hub_download

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "data", "arabic")
os.makedirs(OUT, exist_ok=True)

# Arabic combining marks / harakat / tanwin / shadda / sukun / superscript-alef / tatweel
_AR_DIAC = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭـ]")
def normalize_ar(s):
    if not isinstance(s, str):
        return ""
    s = _AR_DIAC.sub("", s)
    s = (s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
           .replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي").replace("ة", "ه"))
    s = re.sub("[^؀-ۿ0-9\\s]", " ", s)   # keep Arabic block + digits
    return re.sub(r"\s+", " ", s).strip()

# ---- canonical specialty taxonomy (20) ----
CANON = {
    "general_surgery": "جراحة عامة", "endocrine": "الغدد الصماء",
    "internal_medicine": "الأمراض الباطنية", "respiratory": "الجهاز التنفسي",
    "cardiology": "القلب والشرايين", "general_medicine": "الطب العام",
    "gynecology": "أمراض النساء", "hematology": "أمراض الدم",
    "dermatology": "الأمراض الجلدية", "cosmetic_surgery": "جراحة التجميل",
    "gastroenterology": "الجهاز الهضمي", "ent": "الأنف والأذن والحنجرة",
    "urology": "المسالك البولية والتناسلية", "ophthalmology": "أمراض العيون",
    "musculoskeletal": "العضلات والعظام والمفاصل", "psych_neuro": "النفسية والعصبية",
    "sexual_health": "الصحة الجنسية", "oncology": "الأورام",
    "pediatrics": "طب الأطفال", "dentistry": "طب الأسنان",
}
# raw arabic specialty (from hajerbchn / MAQA) -> canonical key  (matched on normalized text)
_RAW2CANON = {
    "جراحة عامة": "general_surgery", "امراض الغدد الصماء": "endocrine",
    "امراض باطنية": "internal_medicine", "امراض الجهاز التنفسي": "respiratory",
    "امراض القلب و الشرايين": "cardiology", "امراض القلب والشرايين": "cardiology",
    "الطب العام": "general_medicine", "امراض نسائية": "gynecology", "امراض الدم": "hematology",
    "الامراض الجلدية": "dermatology", "جراحة تجميل": "cosmetic_surgery",
    "امراض الجهاز الهضمي": "gastroenterology", "انف اذن وحنجرة": "ent",
    "امراض المسالك البولية والتناسلية": "urology", "امراض العيون": "ophthalmology",
    "امراض العضلات والعظام و المفاصل": "musculoskeletal", "امراض العضلات والعظام والمفاصل": "musculoskeletal",
    "امراض نفسية وعصبية": "psych_neuro", "الامراض الجنسية": "sexual_health",
    "الاورام الخبيثة والحميدة": "oncology", "امراض الاطفال": "pediatrics", "طب الاسنان": "dentistry",
}
RAW2CANON = {normalize_ar(k): v for k, v in _RAW2CANON.items()}

SHIFAA = "Ahmed-Selem/Shifaa_Arabic_Medical_Consultations"
SHIFAA_FILES = {
    "Blood_Diseases_and_Oncology": "hematology", "Bone_Diseases": "musculoskeletal",
    "Dermatological_Diseases": "dermatology", "Endocrine_and_Hormonal_Diseases": "endocrine",
    "General_Surgery_and_Cosmetic_Surgery": "general_surgery", "Head_Diseases": "ent",
    "Internal_Medicine_and_Respiratory_Diseases": "internal_medicine",
    "Medical_Affairs_and_Miscellaneous_Issues": "general_medicine",
    "Muscular_Diseases": "musculoskeletal", "Nervous_System_Diseases": "psych_neuro",
    "Obstetrics_and_Gynecology": "gynecology", "Pharmaceuticals_and_Preparations": "general_medicine",
    "Physical_Health": "general_medicine", "Urinary_System_Diseases_and_Others": "urology",
    "alternative_Medicine": "general_medicine", "pediatrics": "pediatrics",
}


def parse_diag(h):
    if not isinstance(h, str):
        return None
    parts = [p.strip() for p in h.split("-") if p.strip()]
    return parts[1] if len(parts) >= 2 else (parts[0] if parts else None)


def load_shifaa():
    rows = []
    for fname, canon in SHIFAA_FILES.items():
        p = hf_hub_download(SHIFAA, f"{fname}.csv", repo_type="dataset")
        df = pd.read_csv(p)
        text = (df.get("Question Title", "").fillna("") + " . " + df.get("Question", "").fillna("")).astype(str)
        diag = df.get("Hierarchical Diagnosis", pd.Series([None] * len(df))).map(parse_diag)
        rows.append(pd.DataFrame({"text": text, "category": canon, "diagnosis": diag}))
    return pd.concat(rows, ignore_index=True)


def load_hajerbchn():
    p = hf_hub_download("hajerbchn/arabic-medical-qa", "arabic-medical-qa-cleaned.csv", repo_type="dataset")
    df = pd.read_csv(p, usecols=["question", "category"]).dropna(subset=["question", "category"])
    df["canon"] = df["category"].map(lambda c: RAW2CANON.get(normalize_ar(c)))
    df = df.dropna(subset=["canon"])
    return pd.DataFrame({"text": df["question"].astype(str), "category": df["canon"], "source": "hajerbchn"})


def load_maqa():
    p = hf_hub_download("madilcy/arabic-medical-qa-MERGED-MAQA-MMMLU-MI",
                        "data/arabic-medical-qa-MERGED-MAQA-MMMLU-MI.json", repo_type="dataset")
    with open(p, encoding="utf-8") as f:
        try:
            data = json.load(f)
            if isinstance(data, dict):
                data = list(data.values())
        except Exception:
            f.seek(0); data = [json.loads(l) for l in f if l.strip()]
    recs = []
    for r in data:
        q = r.get("question") or r.get("q_body")
        if q:
            recs.append({"text": str(q), "category": RAW2CANON.get(normalize_ar(str(r.get("category") or ""))), "source": "maqa"})
    return pd.DataFrame(recs)


def load_mkamil():
    import pyarrow.parquet as pq
    p = hf_hub_download("MKamil/arabic_medical_50k", "data/train-00000-of-00001.parquet", repo_type="dataset")
    t = pq.read_table(p).to_pydict()
    return pd.DataFrame({"text": [str(x) for x in (t.get("input_text") or [])], "category": None, "source": "mkamil"})


def norm_col(df, min_len=15):
    df = df.copy()
    df["text"] = df["text"].map(normalize_ar)
    df = df[df["text"].str.len() >= min_len].drop_duplicates(subset="text")
    return df


def main():
    print("[load] shifaa / hajerbchn / maqa / mkamil ...")
    shifaa = norm_col(load_shifaa())
    shifaa_lab = shifaa[shifaa["diagnosis"].notna()].copy()
    haj = norm_col(load_hajerbchn())
    maqa = norm_col(load_maqa())
    mk = norm_col(load_mkamil())
    print(f"[load] shifaa={len(shifaa)} (labeled={len(shifaa_lab)})  hajerbchn={len(haj)}  maqa={len(maqa)}  mkamil={len(mk)}")

    # ---- router training data: all category-labelled Arabic text (canonical taxonomy) ----
    router = pd.concat([
        haj[["text", "category"]],
        maqa.dropna(subset=["category"])[["text", "category"]],
        shifaa[["text", "category"]],
    ], ignore_index=True).drop_duplicates(subset="text")
    router = router[router["category"].isin(CANON)]
    router.to_parquet(os.path.join(OUT, "router_data.parquet"), index=False)

    # ---- fine per-category diagnosis data (Shifaa) ----
    shifaa_lab[["text", "category", "diagnosis"]].to_parquet(os.path.join(OUT, "labeled_shifaa.parquet"), index=False)

    # ---- full medical corpus (filter positives) ----
    corpus = pd.concat([router[["text"]], mk[["text"]], shifaa_lab[["text"]]], ignore_index=True).drop_duplicates(subset="text")
    corpus.to_parquet(os.path.join(OUT, "corpus.parquet"), index=False)

    rc = router["category"].value_counts().to_dict()
    dc = shifaa_lab.groupby("category")["diagnosis"].nunique().to_dict()
    stats = {
        "total_genuine_entries": int(len(corpus)),
        "router_labeled_entries": int(len(router)),
        "shifaa_diagnosis_labeled": int(len(shifaa_lab)),
        "n_categories": int(router["category"].nunique()),
        "router_per_category": {k: int(v) for k, v in rc.items()},
        "shifaa_diagnoses_per_category": {k: int(v) for k, v in dc.items()},
    }
    json.dump(stats, open(os.path.join(OUT, "stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("\n========== PHASE 1 RESULT ==========")
    print(f"  TOTAL genuine Arabic medical entries (corpus): {len(corpus):,}")
    print(f"  Router category-labelled training entries    : {len(router):,}  across {router['category'].nunique()} specialties")
    print(f"  Shifaa diagnosis-labelled (fine models)      : {len(shifaa_lab):,}")
    print(f"  >=200,000 target: {'MET' if len(corpus) >= 200000 else 'NOT MET ('+format(len(corpus),',')+')'}")
    print("  router per-specialty counts:")
    for k, v in sorted(rc.items(), key=lambda x: -x[1]):
        print(f"    {v:6}  {k:18} ({dc.get(k,0)} fine diagnoses in Shifaa)")
    print("PHASE1_DONE")


if __name__ == "__main__":
    main()

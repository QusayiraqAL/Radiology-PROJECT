# -*- coding: utf-8 -*-
"""Shared metadata for the text-based (symptom -> disease) system.
Single source of truth for category grouping and Arabic labels; imported by
both the trainer (train_text.py) and the server (main.py)."""

# disease -> medical category (6 categories over 22 real diagnoses)
CATEGORY_MAP = {
    # respiratory
    "bronchial asthma": "respiratory",
    "common cold": "respiratory",
    "pneumonia": "respiratory",
    # infectious / systemic
    "dengue": "infectious",
    "malaria": "infectious",
    "typhoid": "infectious",
    "chicken pox": "infectious",
    "urinary tract infection": "infectious",
    # dermatological / allergic
    "fungal infection": "dermatological",
    "impetigo": "dermatological",
    "psoriasis": "dermatological",
    "drug reaction": "dermatological",
    "allergy": "dermatological",
    # gastrointestinal
    "gastroesophageal reflux disease": "gastrointestinal",
    "peptic ulcer disease": "gastrointestinal",
    "jaundice": "gastrointestinal",
    # cardio-metabolic / vascular
    "diabetes": "cardio_metabolic",
    "hypertension": "cardio_metabolic",
    "varicose veins": "cardio_metabolic",
    # musculoskeletal / neurological
    "arthritis": "musculoskeletal_neuro",
    "cervical spondylosis": "musculoskeletal_neuro",
    "migraine": "musculoskeletal_neuro",
}

CATEGORY_AR = {
    "respiratory": "الجهاز التنفسي",
    "infectious": "الأمراض المعدية",
    "dermatological": "الجلدية والحساسية",
    "gastrointestinal": "الجهاز الهضمي",
    "cardio_metabolic": "القلب والأيض والأوعية",
    "musculoskeletal_neuro": "العضلي الهيكلي والعصبي",
}

DISEASE_AR = {
    "allergy": "حساسية",
    "arthritis": "التهاب المفاصل",
    "bronchial asthma": "الربو الشعبي",
    "cervical spondylosis": "التهاب فقرات الرقبة",
    "chicken pox": "جدري الماء",
    "common cold": "نزلة برد",
    "dengue": "حمّى الضنك",
    "diabetes": "داء السكري",
    "drug reaction": "تفاعل دوائي",
    "fungal infection": "عدوى فطرية",
    "gastroesophageal reflux disease": "ارتجاع المريء",
    "hypertension": "ارتفاع ضغط الدم",
    "impetigo": "القوباء (تقيّح جلدي)",
    "jaundice": "اليرقان",
    "malaria": "الملاريا",
    "migraine": "الشقيقة (الصداع النصفي)",
    "peptic ulcer disease": "قرحة هضمية",
    "pneumonia": "التهاب رئوي",
    "psoriasis": "الصدفية",
    "typhoid": "حمّى التيفوئيد",
    "urinary tract infection": "عدوى المسالك البولية",
    "varicose veins": "دوالي الأوردة",
}

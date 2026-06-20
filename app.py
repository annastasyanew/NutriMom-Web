import json
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from flask import Flask, jsonify, redirect, render_template, request, url_for


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

MODEL_PATH = MODEL_DIR / "nutrimom_xgboost_model.json"
IMPUTER_PATH = MODEL_DIR / "nutrimom_imputer_ai.pkl"
FOOD_PATH = MODEL_DIR / "nutrimom_food_knowledge_base.csv"
CONFIG_PATH = MODEL_DIR / "nutrimom_system_config.json"

NUMERIC_FIELDS = {
    "Age": ("Usia", 10, 60),
    "SystolicBP": ("Tekanan darah sistolik", 60, 220),
    "DiastolicBP": ("Tekanan darah diastolik", 40, 140),
    "BS": ("Gula darah", 0, 30),
    "BodyTemp": ("Suhu tubuh", 90, 110),
    "BMI": ("BMI", 10, 60),
    "HeartRate": ("Denyut jantung", 30, 180),
}
BOOLEAN_FIELDS = {
    "PreviousComplications": "Riwayat komplikasi",
    "PreexistingDiabetes": "Diabetes sebelum hamil",
    "GestationalDiabetes": "Diabetes gestasional",
    "MentalHealth": "Gangguan kesehatan mental",
}
RISK_LABELS = {
    "age_risk": "Risiko usia",
    "bp_risk": "Risiko tekanan darah",
    "glucose_risk": "Risiko gula darah/diabetes",
    "bmi_risk": "Risiko BMI",
    "temperature_risk": "Risiko suhu tubuh",
    "heart_rate_risk": "Risiko denyut jantung",
    "complication_risk": "Risiko riwayat komplikasi",
    "mental_health_risk": "Risiko kesehatan mental",
}
CRITERIA_LABELS = {
    "protein_g": "Protein",
    "fiber_g": "Serat",
    "iron_mg": "Zat besi",
    "calcium_mg": "Kalsium",
    "sugar_g": "Gula",
    "sodium_mg": "Natrium",
    "calories": "Kalori",
}

app = Flask(__name__, template_folder=".")
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "nutrimom-prototype")
logging.basicConfig(level=logging.INFO)


def load_resources():
    required = [MODEL_PATH, IMPUTER_PATH, FOOD_PATH, CONFIG_PATH]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"File model tidak ditemukan: {', '.join(missing)}")

    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    model = xgb.Booster()
    model.load_model(MODEL_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    foods = pd.read_csv(FOOD_PATH)

    feature_cols = config["ai_module"]["features"]
    food_criteria = config["spk_module"]["food_criteria"]
    missing_food_cols = [col for col in ["food_name", *food_criteria] if col not in foods]
    if missing_food_cols:
        raise ValueError(
            f"Kolom knowledge base tidak lengkap: {', '.join(missing_food_cols)}"
        )

    if "food_display_name" not in foods.columns:
        foods["food_display_name"] = foods["food_name"]
    if "preparation_method" not in foods.columns:
        foods["preparation_method"] = "unspecified"

    foods["food_display_name"] = foods["food_display_name"].fillna(foods["food_name"])
    foods["preparation_method"] = foods["preparation_method"].fillna("unspecified")
    foods["food_label"] = np.where(
        foods["preparation_method"].astype(str).str.lower().eq("unspecified"),
        foods["food_display_name"],
        foods["food_display_name"].astype(str)
        + " ("
        + foods["preparation_method"].astype(str)
        + ")",
    )

    foods[food_criteria] = foods[food_criteria].apply(pd.to_numeric, errors="coerce")
    foods = foods.dropna(subset=["food_name", *food_criteria]).copy()
    return config, model, imputer, foods


SYSTEM_CONFIG, MODEL, IMPUTER, FOOD_DATA = load_resources()
FEATURE_COLS = SYSTEM_CONFIG["ai_module"]["features"]
CLASS_NAMES = SYSTEM_CONFIG["ai_module"]["classes"]
FOOD_CRITERIA = SYSTEM_CONFIG["spk_module"]["food_criteria"]


def rising(value, start, end):
    if value <= start:
        return 0.0
    if value >= end:
        return 1.0
    return float((value - start) / (end - start))


def falling(value, start, end):
    if value <= start:
        return 1.0
    if value >= end:
        return 0.0
    return float((end - value) / (end - start))


def outside_risk(value, low_danger, low_safe, high_safe, high_danger):
    low_risk = falling(value, low_danger, low_safe)
    high_risk = rising(value, high_safe, high_danger)
    return max(low_risk, high_risk)


def calculate_fuzzy_risk_scores(patient_data):
    diabetes_risk = max(
        patient_data["PreexistingDiabetes"],
        patient_data["GestationalDiabetes"],
    )
    return {
        "age_risk": outside_risk(patient_data["Age"], 16, 20, 34, 40),
        "bp_risk": max(
            outside_risk(patient_data["SystolicBP"], 80, 100, 130, 160),
            outside_risk(patient_data["DiastolicBP"], 50, 65, 85, 105),
        ),
        "glucose_risk": max(rising(patient_data["BS"], 6.0, 9.0), diabetes_risk),
        "bmi_risk": outside_risk(patient_data["BMI"], 16, 18.5, 25, 32),
        "temperature_risk": outside_risk(
            patient_data["BodyTemp"], 94, 97, 99.5, 103
        ),
        "heart_rate_risk": outside_risk(
            patient_data["HeartRate"], 45, 60, 100, 130
        ),
        "complication_risk": float(patient_data["PreviousComplications"]),
        "mental_health_risk": float(patient_data["MentalHealth"]),
    }


def get_dominant_risk_factors(fuzzy_scores):
    dominant = [
        {"key": key, "label": RISK_LABELS[key], "score": score}
        for key, score in fuzzy_scores.items()
        if score >= 0.5
    ]
    return sorted(dominant, key=lambda item: item["score"], reverse=True)


def predict_risk(patient_data):
    frame = pd.DataFrame([patient_data], columns=FEATURE_COLS)
    imputed = IMPUTER.transform(frame)
    imputed_frame = pd.DataFrame(imputed, columns=FEATURE_COLS)
    probabilities = np.asarray(
        MODEL.predict(xgb.DMatrix(imputed_frame, feature_names=FEATURE_COLS))
    ).reshape(-1)

    if len(probabilities) != len(CLASS_NAMES):
        raise ValueError("Jumlah kelas model tidak cocok dengan konfigurasi sistem.")

    best_index = int(np.argmax(probabilities))
    return {
        "prediction": CLASS_NAMES[best_index],
        "confidence": float(probabilities[best_index]),
        "probabilities": {
            class_name: float(probability)
            for class_name, probability in zip(CLASS_NAMES, probabilities)
        },
    }


def adaptive_food_weights(patient_data, fuzzy_scores):
    weights = {
        "protein_g": 0.15,
        "fiber_g": 0.15,
        "iron_mg": 0.13,
        "calcium_mg": 0.13,
        "sugar_g": 0.12,
        "sodium_mg": 0.12,
        "calories": 0.10,
    }

    bp_risk = fuzzy_scores["bp_risk"]
    glucose_risk = fuzzy_scores["glucose_risk"]
    bmi_risk = fuzzy_scores["bmi_risk"]

    # Make the dominant maternal risks meaningfully change the SAW priorities.
    weights["sodium_mg"] += 0.50 * bp_risk
    weights["sugar_g"] += 0.40 * glucose_risk
    weights["fiber_g"] += 0.25 * max(glucose_risk, bmi_risk)
    weights["protein_g"] += 0.18 * fuzzy_scores["complication_risk"]
    weights["iron_mg"] += 0.20 * max(
        fuzzy_scores["age_risk"], fuzzy_scores["complication_risk"]
    )
    weights["calcium_mg"] += 0.15 * fuzzy_scores["age_risk"]

    if patient_data["BMI"] < 18.5:
        weights["calories"] += 0.45 * bmi_risk
        weights["protein_g"] += 0.20 * bmi_risk
    elif patient_data["BMI"] >= 25:
        weights["calories"] += 0.40 * bmi_risk
        weights["protein_g"] += 0.10 * bmi_risk

    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}


def prepare_food_candidates(food_data):
    foods = food_data.copy()
    unsafe_name_pattern = (
        r"\braw\b|alcohol|beer|wine|liquor|cocktail|soda|soft drink|"
        r"energy drink|candy|chocolate|ice cream|frozen yogurt|cake|cookie|"
        r"brownie|donut|sweetened|syrup|pudding|dessert|\bbar\b|\bbars\b|"
        r"burger|burrito|jerky|crumbles|sandwich|wrap|pizza|hot dog|"
        r"sausage|deli meat|ready-to-eat|snack mix|breaded|fried|pretzel|"
        r"\bchips\b|cheese sticks|supplement|"
        r"meal replacement|protein powder|whey protein|milkshake|"
        r"infant formula|baby\s*food|freeze-dried|dehydrated|powder|"
        r"\bcrude\b|\bspice\b|\bseasoning\b|\bextract\b|\bspleen\b|"
        r"\bliver\b|\bbrain\b|beef kidney|pork kidney|lamb kidney|veal kidney|"
        r"variety meats|by-products"
    )
    foods = foods[
        ~foods["food_name"].str.contains(
            unsafe_name_pattern, case=False, regex=True, na=False
        )
    ]
    if "health_score" in foods:
        foods = foods[pd.to_numeric(foods["health_score"], errors="coerce") >= 50]
    if "food_type" in foods:
        relevant_types = {
            "Vegetables",
            "Fruits",
            "Grains",
            "Dairy",
            "Seafood",
            "Meat & Poultry",
        }
        foods = foods[foods["food_type"].isin(relevant_types)]

    return foods[
        (foods["calories"].between(20, 700))
        & (foods["sodium_mg"] <= 1500)
        & (foods["sugar_g"] <= 40)
    ].copy()

def apply_condition_based_food_filter(food_data, patient_data, fuzzy_scores):
    """
    Filter makanan berdasarkan kondisi maternal.
    Tujuannya agar rekomendasi berubah sesuai profil risiko ibu.
    """

    df = food_data.copy()

    glucose_risk = fuzzy_scores.get("glucose_risk", 0)
    bp_risk = fuzzy_scores.get("bp_risk", 0)

    bmi = patient_data.get("BMI", np.nan)

    # Risiko gula tinggi: batasi gula
    if glucose_risk >= 0.5:
        df = df[df["sugar_g"] <= 8].copy()

    if glucose_risk >= 0.75:
        df = df[df["sugar_g"] <= 5].copy()

    # Risiko tekanan darah tinggi: batasi sodium
    if bp_risk >= 0.5:
        df = df[df["sodium_mg"] <= 300].copy()

    if bp_risk >= 0.75:
        df = df[df["sodium_mg"] <= 150].copy()

    # BMI tinggi: batasi kalori dan gula
    if not pd.isna(bmi) and bmi >= 30:
        df = df[
            (df["calories"] <= 350) &
            (df["sugar_g"] <= 10)
        ].copy()

    # BMI rendah: pilih makanan dengan energi dan protein cukup
    if not pd.isna(bmi) and bmi < 18.5:
        df = df[
            (df["calories"] >= 100) &
            (df["protein_g"] >= 3)
        ].copy()

    exclude_keywords = [
        "alcohol", "beer", "wine", "liquor",
        "candy", "soda", "soft drink",
        "raw egg", "raw meat", "raw fish",
        "supplement", "protein powder", "whey protein",
        "meal replacement", "infant formula", "babyfood", "baby food"
    ]

    pattern = "|".join(exclude_keywords)

    df = df[
        ~df["food_name"].astype(str).str.lower().str.contains(pattern, na=False)
    ].copy()

    # Jika filter terlalu ketat, pakai data awal agar sistem tetap bisa memberi rekomendasi
    if len(df) < 20:
        return food_data.copy()

    return df

def rank_foods_saw(food_data, patient_data, fuzzy_scores, top_n=5):
    foods = prepare_food_candidates(food_data)
    foods = apply_condition_based_food_filter(
        foods,
        patient_data,
        fuzzy_scores,
    )
    weights = adaptive_food_weights(patient_data, fuzzy_scores)
    criteria_types = {
        "protein_g": "benefit",
        "fiber_g": "benefit",
        "iron_mg": "benefit",
        "calcium_mg": "benefit",
        "sugar_g": "cost",
        "sodium_mg": "cost",
        "calories": "benefit" if patient_data["BMI"] < 18.5 else "cost",
    }

    normalized = pd.DataFrame(index=foods.index)
    for criterion in FOOD_CRITERIA:
        values = foods[criterion].astype(float)
        percentile_rank = values.rank(method="average", pct=True)
        if criteria_types[criterion] == "benefit":
            normalized[criterion] = percentile_rank
        else:
            normalized[criterion] = 1.0 - percentile_rank

    foods["saw_score"] = sum(
        normalized[criterion] * weights[criterion] for criterion in FOOD_CRITERIA
    )
    foods = foods.sort_values(["saw_score", "health_score"], ascending=False)

    # Keep recommendations varied instead of returning five near-identical foods.
    selected_indices = []
    seen_types = set()
    for index, food in foods.iterrows():
        food_type = food.get("food_type", "")
        if food_type not in seen_types:
            selected_indices.append(index)
            seen_types.add(food_type)
        if len(selected_indices) == top_n:
            break
    if len(selected_indices) < top_n:
        selected_indices.extend(
            index for index in foods.index if index not in selected_indices
        )

    top_foods = foods.loc[selected_indices[:top_n]].copy()
    columns = [
        "food_name",
        "food_display_name",
        "preparation_method",
        "food_label",
        "food_category",
        "food_type",
        "food_group_clean",
        *FOOD_CRITERIA,
        "health_score",
        "saw_score",
    ]
    records = top_foods[[col for col in columns if col in top_foods]].to_dict("records")
    records = [
        {**record, "food_score": record.get("saw_score", 0)}
        for record in records
    ]
    return records, weights, criteria_types


def print_recommendation_debug(
    patient_data, fuzzy_scores, food_weights, criteria_types, top_foods
):
    if os.environ.get("DEBUG_RECOMMENDATIONS") != "1":
        return

    print("PATIENT DATA:", patient_data, flush=True)
    print("FUZZY SCORES:", fuzzy_scores, flush=True)
    print("FOOD WEIGHTS:", food_weights, flush=True)
    print("CRITERIA TYPES:", criteria_types, flush=True)
    print("FULL TOP FOODS:", flush=True)
    for food in top_foods:
        print(food, flush=True)


def build_food_reason(weights, criteria_types):
    priority = sorted(weights.items(), key=lambda item: item[1], reverse=True)[:3]
    reasons = []
    for criterion, _ in priority:
        direction = "tinggi" if criteria_types[criterion] == "benefit" else "lebih rendah"
        reasons.append(f"{direction} {CRITERIA_LABELS[criterion].lower()}")
    return ", ".join(reasons)


def build_final_recommendation(prediction, dominant_factors, food_reason):
    if prediction == "High Risk":
        opening = (
            "Model mendeteksi risiko maternal tinggi. Prioritaskan konsultasi dengan "
            "dokter atau bidan dan lakukan pemantauan kondisi secara lebih ketat."
        )

    elif prediction == "Mid Risk":
        opening = (
            "Model mendeteksi risiko maternal sedang. Lakukan pemantauan lanjutan "
            "dan perhatikan faktor risiko yang dominan."
        )

    elif prediction == "Low Risk":
        opening = (
            "Model mendeteksi risiko maternal rendah. Tetap lakukan pemeriksaan "
            "kehamilan rutin dan pertahankan pola makan seimbang."
        )

    else:
        opening = (
            "Hasil prediksi belum dapat dikategorikan secara jelas. Lakukan pemeriksaan "
            "ulang input dan konsultasikan kondisi kepada tenaga kesehatan."
        )

    if dominant_factors:
        factors = ", ".join(item["label"].lower() for item in dominant_factors[:3])
        factor_text = f" Faktor yang paling perlu diperhatikan: {factors}."
    else:
        factor_text = " Tidak ada faktor fuzzy dominan dengan skor minimal 0,5."

    return (
        f"{opening}{factor_text} Rekomendasi makanan diprioritaskan berdasarkan "
        f"kriteria {food_reason}. Sistem ini hanya alat bantu edukatif dan tidak "
        f"menggantikan konsultasi dokter atau ahli gizi."
    )


def serialize_foods(top_foods):
    return [
        {
            "food_display_name": item.get(
                "food_display_name", item.get("food_name", "Nutrition item")
            ),
            "food_label": item.get("food_label", item.get("food_name", "")),
            "food_type": item.get("food_type", "Nutrition item"),
            "calories": float(item.get("calories", 0) or 0),
            "protein_g": float(item.get("protein_g", 0) or 0),
            "fiber_g": float(item.get("fiber_g", 0) or 0),
            "sugar_g": float(item.get("sugar_g", 0) or 0),
            "sodium_mg": float(item.get("sodium_mg", 0) or 0),
            "iron_mg": float(item.get("iron_mg", 0) or 0),
            "calcium_mg": float(item.get("calcium_mg", 0) or 0),
            "saw_score": float(item.get("saw_score", item.get("food_score", 0)) or 0),
        }
        for item in top_foods
    ]


def build_result_payload(ai_result, fuzzy_scores, dominant_factors, top_foods, final_recommendation):
    return {
        "ai_result": {
            "prediction": ai_result["prediction"],
            "confidence": float(ai_result["confidence"]),
        },
        "fuzzy_scores": {key: float(value) for key, value in fuzzy_scores.items()},
        "dominant_factors": [
            {
                "label": item["label"],
                "score": float(item["score"]),
            }
            for item in dominant_factors
        ],
        "top_foods": serialize_foods(top_foods),
        "final_recommendation": final_recommendation,
    }
    
def parse_patient_form(form):
    patient_data = {}
    errors = []

    for field, (label, minimum, maximum) in NUMERIC_FIELDS.items():
        raw_value = form.get(field, "")
        if raw_value is None:
            raw_value = ""
        raw_value = str(raw_value).strip()
        if not raw_value:
            errors.append(f"{label} wajib diisi.")
            continue
        try:
            value = float(raw_value)
        except ValueError:
            errors.append(f"{label} harus berupa angka.")
            continue
        if not minimum <= value <= maximum:
            errors.append(f"{label} harus berada pada rentang {minimum}-{maximum}.")
        patient_data[field] = value

    for field, label in BOOLEAN_FIELDS.items():
        raw_value = form.get(field, "")
        if raw_value is None:
            raw_value = ""
        raw_value = str(raw_value).strip()
        if raw_value not in {"0", "1"}:
            errors.append(f"{label} harus dipilih.")
        else:
            patient_data[field] = float(raw_value)

    return patient_data, errors


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/index.html")
def index_file_page():
    return render_template("index.html")


@app.get("/predict")
def predict_page():
    return render_template("predict.html", errors=[], values={})


@app.get("/predict.html")
def predict_file_page():
    return render_template("predict.html", errors=[], values={})


@app.get("/result.html")
def result_file_page():
    return redirect(url_for("predict_file_page"))


@app.get("/healthz")
def health_check():
    return jsonify(
        {
            "status": "ok",
            "model_loaded": MODEL is not None,
            "food_rows": int(len(FOOD_DATA)),
        }
    )


@app.get("/templates/index.html")
def legacy_index_file_page():
    return redirect(url_for("index"))


@app.get("/templates/predict.html")
def legacy_predict_file_page():
    return redirect(url_for("predict_file_page"))


@app.get("/templates/result.html")
def legacy_result_file_page():
    return redirect(url_for("predict_file_page"))


@app.post("/predict")
def predict():
    patient_data, errors = parse_patient_form(request.form)
    if errors:
        return render_template(
            "predict.html", errors=errors, values=request.form.to_dict()
        ), 400

    try:
        ai_result = predict_risk(patient_data)
        fuzzy_scores = calculate_fuzzy_risk_scores(patient_data)
        dominant_factors = get_dominant_risk_factors(fuzzy_scores)
        top_foods, food_weights, criteria_types = rank_foods_saw(
            FOOD_DATA, patient_data, fuzzy_scores
        )
        print_recommendation_debug(
            patient_data,
            fuzzy_scores,
            food_weights,
            criteria_types,
            top_foods,
        )
        food_reason = build_food_reason(food_weights, criteria_types)
        final_recommendation = build_final_recommendation(
            ai_result["prediction"], dominant_factors, food_reason
        )
    except Exception:
        app.logger.exception("Gagal memproses rekomendasi")
        return render_template(
            "predict.html",
            errors=["Sistem gagal memproses data. Periksa file model dan coba lagi."],
            values=request.form.to_dict(),
        ), 500

    return render_template(
        "result.html",
        result_payload=build_result_payload(
            ai_result,
            fuzzy_scores,
            dominant_factors,
            top_foods,
            final_recommendation,
        ),
        patient_data=patient_data,
        ai_result=ai_result,
        fuzzy_scores=fuzzy_scores,
        risk_labels=RISK_LABELS,
        dominant_factors=dominant_factors,
        top_foods=top_foods,
        food_weights=food_weights,
        criteria_types=criteria_types,
        criteria_labels=CRITERIA_LABELS,
        food_reason=food_reason,
        final_recommendation=final_recommendation,
        disclaimer=SYSTEM_CONFIG["disclaimer"],
    )


@app.post("/api/predict")
def api_predict():
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"errors": ["Payload JSON tidak ditemukan."]}), 400

    patient_data, errors = parse_patient_form(payload)
    if errors:
        return jsonify({"errors": errors}), 400

    try:
        ai_result = predict_risk(patient_data)
        fuzzy_scores = calculate_fuzzy_risk_scores(patient_data)
        dominant_factors = get_dominant_risk_factors(fuzzy_scores)
        top_foods, food_weights, criteria_types = rank_foods_saw(
            FOOD_DATA, patient_data, fuzzy_scores
        )
        food_reason = build_food_reason(food_weights, criteria_types)
        final_recommendation = build_final_recommendation(
            ai_result["prediction"], dominant_factors, food_reason
        )
    except Exception:
        app.logger.exception("Gagal memproses rekomendasi API")
        return jsonify(
            {"errors": ["Sistem gagal memproses data. Periksa file model dan coba lagi."]}
        ), 500

    response_foods = [
        {
            "food_name": item.get("food_name", ""),
            "food_display_name": item.get("food_display_name", item.get("food_name", "")),
            "preparation_method": item.get("preparation_method", "unspecified"),
            "food_label": item.get("food_label", item.get("food_name", "")),
            "food_type": item.get("food_type", ""),
            "food_group_clean": item.get("food_group_clean", ""),
            "calories": float(item.get("calories", 0) or 0),
            "protein": float(item.get("protein_g", 0) or 0),
            "protein_g": float(item.get("protein_g", 0) or 0),
            "fiber": float(item.get("fiber_g", 0) or 0),
            "fiber_g": float(item.get("fiber_g", 0) or 0),
            "sugar": float(item.get("sugar_g", 0) or 0),
            "sugar_g": float(item.get("sugar_g", 0) or 0),
            "sodium_mg": float(item.get("sodium_mg", 0) or 0),
            "iron": float(item.get("iron_mg", 0) or 0),
            "iron_mg": float(item.get("iron_mg", 0) or 0),
            "calcium": float(item.get("calcium_mg", 0) or 0),
            "calcium_mg": float(item.get("calcium_mg", 0) or 0),
            "food_score": float(item.get("saw_score", 0) or 0),
        }
        for item in top_foods
    ]

    return jsonify(
        {
            "ai_prediction": ai_result["prediction"],
            "ai_confidence": round(ai_result["confidence"], 2),
            "dominant_risks": [item["label"] for item in dominant_factors],
            "top_foods": response_foods,
            "final_recommendation": final_recommendation,
            "probabilities": ai_result["probabilities"],
            "disclaimer": SYSTEM_CONFIG["disclaimer"],
        }
    )


@app.errorhandler(404)
def not_found(_error):
    return render_template(
        "index.html",
        errors=["Halaman yang Anda cari tidak ditemukan."],
        values={},
    ), 404


@app.errorhandler(405)
def method_not_allowed(_error):
    return redirect(url_for("index"))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from flask import Flask, redirect, render_template, request, url_for


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

app = Flask(__name__)
app.config["SECRET_KEY"] = "nutrimom-prototype"
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
        "protein_g": 0.18,
        "fiber_g": 0.18,
        "iron_mg": 0.15,
        "calcium_mg": 0.14,
        "sugar_g": 0.13,
        "sodium_mg": 0.12,
        "calories": 0.10,
    }

    bp_risk = fuzzy_scores["bp_risk"]
    glucose_risk = fuzzy_scores["glucose_risk"]
    bmi_risk = fuzzy_scores["bmi_risk"]

    weights["sodium_mg"] += 0.16 * bp_risk
    weights["sugar_g"] += 0.16 * glucose_risk
    weights["fiber_g"] += 0.08 * max(glucose_risk, bmi_risk)
    weights["protein_g"] += 0.05 * fuzzy_scores["complication_risk"]
    weights["iron_mg"] += 0.05 * max(fuzzy_scores["age_risk"], fuzzy_scores["complication_risk"])
    weights["calcium_mg"] += 0.04 * fuzzy_scores["age_risk"]
    weights["calories"] += 0.10 * bmi_risk

    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}


def prepare_food_candidates(food_data):
    foods = food_data.copy()
    unsafe_name_pattern = (
        r"\braw\b|supplement|meal replacement|protein powder|whey protein|"
        r"milkshake|infant formula|baby food"
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


def rank_foods_saw(food_data, patient_data, fuzzy_scores, top_n=5):
    foods = prepare_food_candidates(food_data)
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
        if criteria_types[criterion] == "benefit":
            maximum = values.max()
            normalized[criterion] = values / maximum if maximum > 0 else 0.0
        else:
            value_range = values.max() - values.min()
            normalized[criterion] = (
                (values.max() - values) / value_range if value_range > 0 else 1.0
            )

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
        "food_category",
        "food_type",
        *FOOD_CRITERIA,
        "saw_score",
    ]
    records = top_foods[[col for col in columns if col in top_foods]].to_dict("records")
    return records, weights, criteria_types


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
    else:
        opening = (
            "Model mendeteksi risiko maternal rendah. Tetap lakukan pemeriksaan "
            "kehamilan rutin dan pertahankan pola makan seimbang."
        )

    if dominant_factors:
        factors = ", ".join(item["label"].lower() for item in dominant_factors[:3])
        factor_text = f" Faktor yang paling perlu diperhatikan: {factors}."
    else:
        factor_text = " Tidak ada faktor fuzzy dominan dengan skor minimal 0,5."

    return (
        f"{opening}{factor_text} Rekomendasi makanan diprioritaskan berdasarkan "
        f"kriteria {food_reason}."
    )


def parse_patient_form(form):
    patient_data = {}
    errors = []

    for field, (label, minimum, maximum) in NUMERIC_FIELDS.items():
        raw_value = form.get(field, "").strip()
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
        raw_value = form.get(field, "").strip()
        if raw_value not in {"0", "1"}:
            errors.append(f"{label} harus dipilih.")
        else:
            patient_data[field] = float(raw_value)

    return patient_data, errors


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        return predict()
    return render_template("index.html", errors=[], values={})


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return redirect(url_for("index"))

    patient_data, errors = parse_patient_form(request.form)
    if errors:
        return render_template(
            "index.html", errors=errors, values=request.form.to_dict()
        ), 400

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
        app.logger.exception("Gagal memproses rekomendasi")
        return render_template(
            "index.html",
            errors=["Sistem gagal memproses data. Periksa file model dan coba lagi."],
            values=request.form.to_dict(),
        ), 500

    return render_template(
        "result.html",
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
    app.run(host="127.0.0.1", port=5000, debug=False)

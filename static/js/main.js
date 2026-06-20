// Setup API endpoint configuration for Flask integration
const apiConfig = {
  endpoint: "/api/predict",
  timeout: 12000,
};

const resultStorageKey = "pregnanutri_result_payload";
const flaskOrigin = "http://127.0.0.1:5000";
const riskLabels = {
  age_risk: "Risiko usia",
  bp_risk: "Risiko tekanan darah",
  glucose_risk: "Risiko gula darah/diabetes",
  bmi_risk: "Risiko BMI",
  temperature_risk: "Risiko suhu tubuh",
  heart_rate_risk: "Risiko denyut jantung",
  complication_risk: "Risiko riwayat komplikasi",
  mental_health_risk: "Risiko kesehatan mental",
};
const visibleRiskItems = [
  ["bp_risk", "Blood Pressure Risk"],
  ["glucose_risk", "Glucose Risk"],
  ["bmi_risk", "BMI Risk"],
  ["temperature_risk", "Temperature Risk"],
  ["heart_rate_risk", "Heart Rate Risk"],
];
const staticFoodBase = [
  {
    food_display_name: "Collards",
    food_label: "Collards (cooked, boiled, drained, without salt)",
    food_type: "Vegetables",
    calories: 33,
    protein_g: 2.7,
    fiber_g: 4,
    sugar_g: 0.4,
    sodium_mg: 15,
    iron_mg: 1.1,
    calcium_mg: 141,
    saw_score: 0.75,
  },
  {
    food_display_name: "Bulgur",
    food_label: "Bulgur (cooked)",
    food_type: "Grains",
    calories: 83,
    protein_g: 3.1,
    fiber_g: 4.5,
    sugar_g: 0.1,
    sodium_mg: 5,
    iron_mg: 1,
    calcium_mg: 10,
    saw_score: 0.67,
  },
  {
    food_display_name: "Tuna skipjack",
    food_label: "Fish, tuna, skipjack, fresh (cooked, dry heat)",
    food_type: "Seafood",
    calories: 132,
    protein_g: 28.2,
    fiber_g: 0,
    sugar_g: 0,
    sodium_mg: 47,
    iron_mg: 1.6,
    calcium_mg: 37,
    saw_score: 0.62,
  },
  {
    food_display_name: "Chicken gizzard",
    food_label: "Chicken gizzard (cooked, simmered)",
    food_type: "Meat & Poultry",
    calories: 154,
    protein_g: 30.4,
    fiber_g: 0,
    sugar_g: 0,
    sodium_mg: 56,
    iron_mg: 3.2,
    calcium_mg: 17,
    saw_score: 0.6,
  },
  {
    food_display_name: "Plain yogurt",
    food_label: "Plain yogurt, low fat",
    food_type: "Dairy",
    calories: 63,
    protein_g: 5.3,
    fiber_g: 0,
    sugar_g: 7,
    sodium_mg: 70,
    iron_mg: 0.1,
    calcium_mg: 183,
    saw_score: 0.57,
  },
];

const resolveBackendUrl = (path) => {
  if (window.location.port === "5500") {
    return `${flaskOrigin}${path}`;
  }

  return path;
};

const isLiveServerPage = () =>
  window.location.hostname === "127.0.0.1" && window.location.port === "5500";

const isBackendServedPage = () =>
  window.location.protocol !== "file:" && !isLiveServerPage();

const buildRequestBody = (form) => ({
  Age: Number(form.Age.value),
  SystolicBP: Number(form.SystolicBP.value),
  DiastolicBP: Number(form.DiastolicBP.value),
  BS: Number(form.BS.value),
  BodyTemp: Number(form.BodyTemp.value),
  BMI: Number(form.BMI.value),
  HeartRate: Number(form.HeartRate.value),
  PreviousComplications: Number(form.PreviousComplications.value),
  PreexistingDiabetes: Number(form.PreexistingDiabetes.value),
  GestationalDiabetes: Number(form.GestationalDiabetes.value),
  MentalHealth: Number(form.MentalHealth.value),
});

const predictWithApi = async (payload) => {
  const response = await fetch(resolveBackendUrl(apiConfig.endpoint), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const data = await response.json();
    throw new Error(data.errors ? data.errors.join(" ") : "Gagal memproses request.");
  }

  return response.json();
};

const showErrorToast = (message) => {
  const existing = document.querySelector(".app-toast");
  if (existing) existing.remove();

  const toast = document.createElement("div");
  toast.className = "app-toast";
  toast.textContent = message;
  document.body.appendChild(toast);

  requestAnimationFrame(() => toast.classList.add("visible"));
  setTimeout(() => {
    toast.classList.remove("visible");
    setTimeout(() => toast.remove(), 300);
  }, 4200);
};

const initPageAnimations = () => {
  const elements = document.querySelectorAll(".fade-in");
  elements.forEach((element, index) => {
    element.style.animationDelay = `${index * 0.12}s`;
  });
};

const initFoodToggle = () => {
  const buttons = document.querySelectorAll('.food-toggle-btn');
  buttons.forEach((button) => {
    const card = button.closest('.food-card');
    if (!card) return;
    const details = card.querySelector('.food-details');
    button.addEventListener('click', (event) => {
      event.preventDefault();
      const expanded = details.classList.toggle('open');
      button.setAttribute('aria-expanded', expanded);
      details.setAttribute('aria-hidden', String(!expanded));
      button.textContent = expanded ? 'Hide details' : 'View details';
      if (expanded) {
        details.style.maxHeight = `${details.scrollHeight}px`;
      } else {
        details.style.maxHeight = null;
      }
    });
  });
};

const rising = (value, start, end) => {
  if (value <= start) return 0;
  if (value >= end) return 1;
  return (value - start) / (end - start);
};

const falling = (value, start, end) => {
  if (value <= start) return 1;
  if (value >= end) return 0;
  return (end - value) / (end - start);
};

const outsideRisk = (value, lowDanger, lowSafe, highSafe, highDanger) =>
  Math.max(falling(value, lowDanger, lowSafe), rising(value, highSafe, highDanger));

const calculateFuzzyScores = (patientData) => {
  const diabetesRisk = Math.max(
    patientData.PreexistingDiabetes,
    patientData.GestationalDiabetes
  );

  return {
    age_risk: outsideRisk(patientData.Age, 16, 20, 34, 40),
    bp_risk: Math.max(
      outsideRisk(patientData.SystolicBP, 80, 100, 130, 160),
      outsideRisk(patientData.DiastolicBP, 50, 65, 85, 105)
    ),
    glucose_risk: Math.max(rising(patientData.BS, 6.0, 9.0), diabetesRisk),
    bmi_risk: outsideRisk(patientData.BMI, 16, 18.5, 25, 32),
    temperature_risk: outsideRisk(patientData.BodyTemp, 94, 97, 99.5, 103),
    heart_rate_risk: outsideRisk(patientData.HeartRate, 45, 60, 100, 130),
    complication_risk: patientData.PreviousComplications,
    mental_health_risk: patientData.MentalHealth,
  };
};

const getDominantFactors = (fuzzyScores) =>
  Object.entries(fuzzyScores)
    .filter(([, score]) => score >= 0.5)
    .map(([key, score]) => ({ key, label: riskLabels[key], score }))
    .sort((a, b) => b.score - a.score);

const titleCase = (value) =>
  String(value)
    .toLowerCase()
    .replace(/\b\w/g, (letter) => letter.toUpperCase());

const estimatePrediction = (fuzzyScores) => {
  const maxRisk = Math.max(...Object.values(fuzzyScores));
  const dominantCount = Object.values(fuzzyScores).filter((score) => score >= 0.5).length;

  if (maxRisk >= 0.85 || dominantCount >= 3) {
    return { prediction: "High Risk", confidence: Math.max(0.82, maxRisk) };
  }

  if (maxRisk >= 0.5 || dominantCount >= 1) {
    return { prediction: "Mid Risk", confidence: Math.max(0.68, maxRisk) };
  }

  return { prediction: "Low Risk", confidence: 0.76 };
};

const rankStaticFoods = (patientData, fuzzyScores) => {
  const glucoseRisk = fuzzyScores.glucose_risk;
  const bpRisk = fuzzyScores.bp_risk;
  const bmiRisk = fuzzyScores.bmi_risk;

  return staticFoodBase
    .map((food) => {
      let score = food.saw_score;
      score += glucoseRisk >= 0.5 ? Math.max(0, (8 - food.sugar_g) / 100) : 0;
      score += bpRisk >= 0.5 ? Math.max(0, (120 - food.sodium_mg) / 1000) : 0;
      score += bmiRisk >= 0.5 && patientData.BMI >= 25 ? Math.max(0, (180 - food.calories) / 1000) : 0;
      return { ...food, saw_score: Math.min(score, 0.98) };
    })
    .sort((a, b) => b.saw_score - a.saw_score)
    .slice(0, 5);
};

const buildStaticRecommendation = (prediction, dominantFactors) => {
  const factorText = dominantFactors.length
    ? dominantFactors.slice(0, 3).map((factor) => factor.label.toLowerCase()).join(", ")
    : "tidak ada faktor fuzzy dominan";

  if (prediction === "High Risk") {
    return `Estimasi menunjukkan risiko maternal tinggi. Faktor yang perlu diperhatikan: ${factorText}. Prioritaskan konsultasi dengan dokter atau bidan dan lakukan pemantauan kondisi secara lebih ketat.`;
  }

  if (prediction === "Mid Risk") {
    return `Estimasi menunjukkan risiko maternal sedang. Faktor yang perlu diperhatikan: ${factorText}. Lakukan pemantauan lanjutan dan pertahankan pola makan seimbang.`;
  }

  return "Estimasi menunjukkan risiko maternal rendah. Tetap lakukan pemeriksaan kehamilan rutin dan pertahankan pola makan seimbang.";
};

const buildStaticResultPayload = (form) => {
  const patientData = buildRequestBody(form);
  const fuzzyScores = calculateFuzzyScores(patientData);
  const dominantFactors = getDominantFactors(fuzzyScores);
  const aiResult = estimatePrediction(fuzzyScores);
  const topFoods = rankStaticFoods(patientData, fuzzyScores);

  return {
    ai_result: aiResult,
    fuzzy_scores: fuzzyScores,
    dominant_factors: dominantFactors,
    top_foods: topFoods,
    final_recommendation: buildStaticRecommendation(aiResult.prediction, dominantFactors),
  };
};

const getServerResultPayload = () => {
  const element = document.querySelector("#serverResultPayload");
  if (!element) return null;

  const text = element.textContent.trim();
  if (!text || text.startsWith("{{")) return null;

  try {
    const payload = JSON.parse(text);
    return Object.keys(payload).length ? payload : null;
  } catch (_error) {
    return null;
  }
};

const getStoredResultPayload = () => {
  try {
    const stored = sessionStorage.getItem(resultStorageKey);
    return stored ? JSON.parse(stored) : null;
  } catch (_error) {
    return null;
  }
};

const setText = (selector, value) => {
  const element = document.querySelector(selector);
  if (element) element.textContent = value;
};

const scorePercent = (score) => `${Math.round((Number(score) || 0) * 100)}%`;

const renderRiskGrid = (fuzzyScores = {}) => {
  const grid = document.querySelector("#riskGrid");
  if (!grid) return;

  grid.innerHTML = visibleRiskItems
    .map(([key, label]) => {
      const score = Number(fuzzyScores[key]) || 0;
      return `
        <div class="risk-item hover-card">
          <div class="risk-meta">
            <div>
              <small>${label}</small>
              <strong>${scorePercent(score)}</strong>
            </div>
            <span class="risk-badge">${score.toFixed(2)}</span>
          </div>
          <div class="progress-bar soft"><span style="width: ${score * 100}%;"></span></div>
        </div>
      `;
    })
    .join("");
};

const renderDominantFactors = (dominantFactors = []) => {
  const chart = document.querySelector("#dominantChart");
  const empty = document.querySelector("#dominantEmpty");
  if (!chart) return;

  empty.hidden = dominantFactors.length > 0;
  chart.innerHTML = dominantFactors
    .map((factor, index) => {
      const icons = ["bi-person-heart", "bi-droplet-half", "bi-speedometer2", "bi-heart-pulse"];
      return `
        <div class="dominant-bar">
          <div class="dominant-topline">
            <span class="dominant-icon"><i class="bi ${icons[index % icons.length]}"></i></span>
            <div class="dominant-copy">
              <span>${factor.label}</span>
              <small>${factor.score >= 0.85 ? "Critical attention" : "Needs attention"}</small>
            </div>
            <strong>${scorePercent(factor.score)}</strong>
          </div>
          <div class="progress-bar slim"><span style="width: ${factor.score * 100}%;"></span></div>
        </div>
      `;
    })
    .join("");
};

const renderFoods = (topFoods = []) => {
  const grid = document.querySelector("#foodsGrid");
  if (!grid) return;

  grid.innerHTML = topFoods
    .map((food, index) => `
      <article class="food-card hover-card">
        <div class="food-card-top">
          <div class="food-card-title-group">
            <span class="food-rank">#${String(index + 1).padStart(2, "0")}</span>
            <div>
              <h3>${titleCase(food.food_display_name || "Nutrition item")}</h3>
              <div class="food-badges">
                <span class="food-category-badge">${food.food_type || "Nutrition item"}</span>
              </div>
            </div>
          </div>
          <span class="badge-pill">Score ${(Number(food.saw_score) || 0).toFixed(2)}</span>
        </div>
        <p class="food-description">${food.food_label || "Nutrisi berkualitas untuk mendukung kesehatan ibu dan janin."}</p>
        <div class="food-card-footer">
          <button type="button" class="btn btn-soft btn-block food-toggle-btn" aria-expanded="false">View details</button>
        </div>
        <div class="food-details" aria-hidden="true">
          <div class="food-card-stats">
            <div class="food-card-stat"><strong>${Math.round(food.calories || 0)}</strong><small>Calories</small></div>
            <div class="food-card-stat"><strong>${Number(food.protein_g || 0).toFixed(1)}g</strong><small>Protein</small></div>
            <div class="food-card-stat"><strong>${Number(food.fiber_g || 0).toFixed(1)}g</strong><small>Fiber</small></div>
            <div class="food-card-stat"><strong>${Number(food.sugar_g || 0).toFixed(1)}g</strong><small>Sugar</small></div>
            <div class="food-card-stat"><strong>${Math.round(food.sodium_mg || 0)}mg</strong><small>Sodium</small></div>
            <div class="food-card-stat"><strong>${Number(food.iron_mg || 0).toFixed(1)}mg</strong><small>Iron</small></div>
            <div class="food-card-stat"><strong>${Math.round(food.calcium_mg || 0)}mg</strong><small>Calcium</small></div>
          </div>
        </div>
      </article>
    `)
    .join("");

  initFoodToggle();
};

const renderResultPage = () => {
  if (!document.querySelector(".result-page")) return;

  const payload = getServerResultPayload() || getStoredResultPayload();
  if (!payload) return;

  const prediction = payload.ai_result?.prediction || "Low Risk";
  const confidence = Number(payload.ai_result?.confidence) || 0;
  const summaryCard = document.querySelector("#summaryCard");
  if (summaryCard) {
    summaryCard.classList.remove("badge-high", "badge-mid", "badge-low");
    summaryCard.classList.add(
      prediction === "High Risk" ? "badge-high" : prediction === "Mid Risk" ? "badge-mid" : "badge-low"
    );
  }

  setText('[data-result="prediction"]', prediction);
  setText('[data-result="confidence"]', scorePercent(confidence));
  const confidenceBar = document.querySelector('[data-result-style="confidenceWidth"]');
  if (confidenceBar) confidenceBar.style.width = `${confidence * 100}%`;

  renderRiskGrid(payload.fuzzy_scores);
  renderDominantFactors(payload.dominant_factors);
  renderFoods(payload.top_foods);
  setText(
    '[data-result="dominantSummary"]',
    payload.dominant_factors?.length
      ? payload.dominant_factors.map((factor) => factor.label).join(", ")
      : "Tidak ada faktor risiko dominan."
  );
  setText('[data-result="finalRecommendation"]', payload.final_recommendation || "");
};

const initPredictionForm = () => {
  const form = document.querySelector("#predictionForm");
  if (!form) return;

  if (isBackendServedPage()) {
    form.setAttribute("action", "/predict");
    form.setAttribute("method", "post");
    return;
  }

  form.setAttribute("action", "result.html");
  form.setAttribute("method", "get");
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    if (!form.reportValidity()) return;

    const payload = buildStaticResultPayload(form);
    sessionStorage.setItem(resultStorageKey, JSON.stringify(payload));
    window.location.href = "result.html";
  });
};

window.addEventListener("DOMContentLoaded", () => {
  initPageAnimations();
  initFoodToggle();
  initPredictionForm();
  renderResultPage();
});

window.PregnaNutriApi = {
  predictWithApi,
  buildRequestBody,
};

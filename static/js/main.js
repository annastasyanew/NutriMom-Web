// Setup API endpoint configuration for Flask integration
const apiConfig = {
  endpoint: "/api/predict",
  timeout: 12000,
};

const apiConfig = {
  endpoint: "/api/predict",
  timeout: 12000,
};

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
  const response = await fetch(apiConfig.endpoint, {
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

window.addEventListener("DOMContentLoaded", () => {
  initPageAnimations();
});

window.PregnaNutriApi = {
  predictWithApi,
  buildRequestBody,
};

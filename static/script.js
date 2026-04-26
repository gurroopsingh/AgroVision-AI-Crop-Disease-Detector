const form = document.getElementById("predict-form");
const input = document.getElementById("image-input");
const preview = document.getElementById("preview");
const previewPlaceholder = document.getElementById("preview-placeholder");
const loading = document.getElementById("loading");
const resultBox = document.getElementById("result");
const predictedClassEl = document.getElementById("predicted-class");
const confidenceEl = document.getElementById("confidence");
const confidenceBar = document.getElementById("confidence-bar");
const certaintyBadge = document.getElementById("certainty-badge");
const descriptionEl = document.getElementById("description");
const treatmentEl = document.getElementById("treatment");
const preventionEl = document.getElementById("prevention");
const apiError = document.getElementById("api-error");
const fileError = document.getElementById("file-error");
const predictBtn = document.getElementById("predict-btn");

const allowedTypes = ["image/jpeg", "image/png", "image/webp", "image/bmp"];

function resetMessages() {
  apiError.textContent = "";
  fileError.textContent = "";
  apiError.classList.add("hidden");
  fileError.classList.add("hidden");
}

function confidenceToCertainty(confidence) {
  if (confidence > 0.9) return "HIGH";
  if (confidence > 0.7) return "MEDIUM";
  return "LOW";
}

function applyCertaintyBadge(confidence) {
  const level = confidenceToCertainty(confidence);
  certaintyBadge.textContent = level;
  certaintyBadge.classList.remove("certainty-high", "certainty-medium", "certainty-low");
  if (level === "HIGH") certaintyBadge.classList.add("certainty-high");
  else if (level === "MEDIUM") certaintyBadge.classList.add("certainty-medium");
  else certaintyBadge.classList.add("certainty-low");
}

input.addEventListener("change", () => {
  resetMessages();
  resultBox.classList.add("hidden");

  const file = input.files?.[0];
  if (!file) {
    preview.classList.add("hidden");
    previewPlaceholder.classList.remove("hidden");
    preview.src = "";
    return;
  }

  if (!allowedTypes.includes(file.type)) {
    input.value = "";
    preview.classList.add("hidden");
    previewPlaceholder.classList.remove("hidden");
    preview.src = "";
    fileError.textContent = "Unsupported file format. Please upload JPG, PNG, WEBP, or BMP.";
    fileError.classList.remove("hidden");
    return;
  }

  preview.src = URL.createObjectURL(file);
  preview.classList.remove("hidden");
  previewPlaceholder.classList.add("hidden");
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  resetMessages();
  resultBox.classList.add("hidden");

  const file = input.files?.[0];
  if (!file) {
    fileError.textContent = "Please select an image before predicting.";
    fileError.classList.remove("hidden");
    return;
  }

  if (!allowedTypes.includes(file.type)) {
    fileError.textContent = "Unsupported file format. Please upload JPG, PNG, WEBP, or BMP.";
    fileError.classList.remove("hidden");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  loading.classList.remove("hidden");
  predictBtn.disabled = true;

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Prediction failed.");
    }

    const confidenceValue = Number(data.confidence);
    const confidencePercent = (confidenceValue * 100).toFixed(2);
    predictedClassEl.textContent = data.predicted_class;
    confidenceEl.textContent = `${confidencePercent}%`;
    confidenceBar.style.width = "0%";
    requestAnimationFrame(() => {
      confidenceBar.style.width = `${confidencePercent}%`;
    });
    applyCertaintyBadge(confidenceValue);

    descriptionEl.textContent =
      data.description || "No disease description is available yet for this class.";
    treatmentEl.textContent =
      data.treatment || "Treatment guidance is not available yet. Consult a local agronomist.";
    preventionEl.textContent =
      data.prevention || "Prevention tips are not available yet. Follow standard crop hygiene practices.";

    resultBox.classList.remove("hidden");
  } catch (error) {
    apiError.textContent = error.message || "Prediction request failed.";
    apiError.classList.remove("hidden");
  } finally {
    loading.classList.add("hidden");
    predictBtn.disabled = false;
  }
});

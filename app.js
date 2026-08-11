// ── Model parameters (Trained Classifier) ──────────────────
const MODEL = {
  // "primary" shown by default, "advanced" shown in dropdown
  primary: [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean"
  ],
  advanced: [
    "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se",
    "concave_points_se", "symmetry_se", "fractal_dimension_se"
  ],
  labels: {
    "radius_mean":            "Mean Radius (mm)",
    "texture_mean":           "Mean Texture",
    "perimeter_mean":         "Mean Perimeter (mm)",
    "area_mean":              "Mean Area (mm²)",
    "smoothness_mean":        "Mean Smoothness",
    "compactness_mean":       "Mean Compactness",
    "concavity_mean":         "Mean Concavity",
    "concave_points_mean":    "Mean Concave Points",
    "symmetry_mean":          "Mean Symmetry",
    "fractal_dimension_mean": "Mean Fractal Dim.",
    "radius_se":              "Radius SE",
    "texture_se":             "Texture SE",
    "perimeter_se":           "Perimeter SE",
    "area_se":                "Area SE",
    "smoothness_se":          "Smoothness SE",
    "compactness_se":         "Compactness SE",
    "concavity_se":           "Concavity SE",
    "concave_points_se":      "Concave Points SE",
    "symmetry_se":            "Symmetry SE",
    "fractal_dimension_se":   "Fractal Dim. SE"
  },
  intercept: -0.5256839982761035,
  // coefficients aligned to features in order: primary + advanced
  // (only 10 primary+mean features used for inference, same as trained model)
  // The 10 trained features map exactly to this list:
  trainedFeatures: [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave_points_mean", "symmetry_mean", "fractal_dimension_mean"
  ],
  coefficients: [
    1.0518048043681918,
    1.4898594141110502,
    0.9236708353618363,
    1.2206644250617806,
    1.1441284004537842,
    -0.2803856291609571,
    0.8804866253880916,
    1.6624827936124507,
    0.431303418553719,
    -0.4120452571077299
  ],
  stats: {
    "radius_mean":            { min: 6.98,   max: 28.11,  mean: 14.166, std: 3.575,   step: 0.1   },
    "texture_mean":           { min: 9.71,   max: 39.28,  mean: 19.418, std: 4.286,   step: 0.1   },
    "perimeter_mean":         { min: 43.79,  max: 188.5,  mean: 92.216, std: 24.690,  step: 0.5   },
    "area_mean":              { min: 143.5,  max: 2501.0, mean: 659.578,std: 360.022, step: 5.0   },
    "smoothness_mean":        { min: 0.052,  max: 0.163,  mean: 0.096,  std: 0.014,   step: 0.001 },
    "compactness_mean":       { min: 0.019,  max: 0.345,  mean: 0.104,  std: 0.054,   step: 0.001 },
    "concavity_mean":         { min: 0.0,    max: 0.427,  mean: 0.089,  std: 0.082,   step: 0.001 },
    "concave_points_mean":    { min: 0.0,    max: 0.201,  mean: 0.049,  std: 0.040,   step: 0.001 },
    "symmetry_mean":          { min: 0.106,  max: 0.304,  mean: 0.181,  std: 0.028,   step: 0.001 },
    "fractal_dimension_mean": { min: 0.050,  max: 0.097,  mean: 0.063,  std: 0.007,   step: 0.001 },
    // SE features — approximate stats from dataset
    "radius_se":              { min: 0.11,   max: 2.87,   mean: 0.405,  std: 0.277,   step: 0.01  },
    "texture_se":             { min: 0.36,   max: 4.88,   mean: 1.217,  std: 0.552,   step: 0.01  },
    "perimeter_se":           { min: 0.76,   max: 21.98,  mean: 2.867,  std: 2.022,   step: 0.1   },
    "area_se":                { min: 6.80,   max: 542.2,  mean: 40.337, std: 45.491,  step: 0.5   },
    "smoothness_se":          { min: 0.002,  max: 0.031,  mean: 0.007,  std: 0.003,   step: 0.001 },
    "compactness_se":         { min: 0.002,  max: 0.135,  mean: 0.025,  std: 0.018,   step: 0.001 },
    "concavity_se":           { min: 0.0,    max: 0.396,  mean: 0.032,  std: 0.030,   step: 0.001 },
    "concave_points_se":      { min: 0.0,    max: 0.053,  mean: 0.012,  std: 0.006,   step: 0.001 },
    "symmetry_se":            { min: 0.008,  max: 0.079,  mean: 0.020,  std: 0.008,   step: 0.001 },
    "fractal_dimension_se":   { min: 0.001,  max: 0.030,  mean: 0.004,  std: 0.004,   step: 0.001 }
  },
  // Extreme samples from the actual dataset
  samples: {
    // Most benign: smallest radius, low concavity
    benign: {
      "radius_mean": 6.981, "texture_mean": 13.43, "perimeter_mean": 43.79, "area_mean": 143.5,
      "smoothness_mean": 0.1170, "compactness_mean": 0.07568, "concavity_mean": 0.0,
      "concave_points_mean": 0.0, "symmetry_mean": 0.1930, "fractal_dimension_mean": 0.07818,
      "radius_se": 0.284, "texture_se": 0.920, "perimeter_se": 1.935, "area_se": 14.96,
      "smoothness_se": 0.011, "compactness_se": 0.021, "concavity_se": 0.023,
      "concave_points_se": 0.009, "symmetry_se": 0.026, "fractal_dimension_se": 0.007
    },
    // Most malignant: largest radius, high concavity
    malignant: {
      "radius_mean": 28.11, "texture_mean": 29.40, "perimeter_mean": 188.5, "area_mean": 2501.0,
      "smoothness_mean": 0.1275, "compactness_mean": 0.2736, "concavity_mean": 0.4268,
      "concave_points_mean": 0.2012, "symmetry_mean": 0.2650, "fractal_dimension_mean": 0.07781,
      "radius_se": 2.873, "texture_se": 4.885, "perimeter_se": 21.98, "area_se": 542.2,
      "smoothness_se": 0.031, "compactness_se": 0.135, "concavity_se": 0.396,
      "concave_points_se": 0.053, "symmetry_se": 0.079, "fractal_dimension_se": 0.030
    },
    // Average case: dataset means across all samples
    average: {
      "radius_mean": 14.166, "texture_mean": 19.418, "perimeter_mean": 92.216, "area_mean": 659.578,
      "smoothness_mean": 0.096, "compactness_mean": 0.104, "concavity_mean": 0.089,
      "concave_points_mean": 0.049, "symmetry_mean": 0.181, "fractal_dimension_mean": 0.063,
      "radius_se": 0.405, "texture_se": 1.217, "perimeter_se": 2.867, "area_se": 40.337,
      "smoothness_se": 0.007, "compactness_se": 0.025, "concavity_se": 0.032,
      "concave_points_se": 0.012, "symmetry_se": 0.020, "fractal_dimension_se": 0.004
    }
  }
};

// ── DOM refs ──────────────────────────────────────────────────────────────
const resultCard      = document.getElementById("result-card");
const badge           = document.getElementById("prediction-badge");
const probMalignantTx = document.getElementById("prob-malignant-text");
const probBenignTx    = document.getElementById("prob-benign-text");
const barMalignant    = document.getElementById("bar-malignant");
const barBenign       = document.getElementById("bar-benign");
const primaryCont     = document.getElementById("primary-sliders");
const advancedCont    = document.getElementById("advanced-sliders");

// ── Build a slider card ───────────────────────────────────────────────────
function buildSlider(id, container) {
  const s    = MODEL.stats[id];
  const lbl  = MODEL.labels[id] || id;
  const decs = s.step < 0.01 ? 4 : s.step < 1 ? 3 : 1;
  const def  = s.mean.toFixed(decs);

  const card = document.createElement("div");
  card.className = "slider-card";
  card.innerHTML = `
    <div class="slider-header">
      <label>${lbl}</label>
      <input type="number" id="num-${id}" value="${def}" min="${s.min}" max="${s.max}" step="${s.step}">
    </div>
    <input type="range" id="sl-${id}" value="${def}" min="${s.min}" max="${s.max}" step="${s.step}">
    <div class="slider-footer">
      <span>${s.min}</span>
      <span>${s.max}</span>
    </div>
  `;
  container.appendChild(card);

  const range = document.getElementById(`sl-${id}`);
  const num   = document.getElementById(`num-${id}`);

  range.addEventListener("input", e => { num.value = e.target.value; classify(); });
  num.addEventListener("input",   e => { range.value = e.target.value; classify(); });
}

// ── Render all sliders ────────────────────────────────────────────────────
MODEL.primary.forEach(id => buildSlider(id, primaryCont));
MODEL.advanced.forEach(id => buildSlider(id, advancedCont));

// ── Load a sample into all controls ──────────────────────────────────────
function loadSample(sampleObj) {
  const allIds = [...MODEL.primary, ...MODEL.advanced];
  allIds.forEach(id => {
    const val = sampleObj[id];
    if (val === undefined) return;
    const s    = MODEL.stats[id];
    const decs = s.step < 0.01 ? 4 : s.step < 1 ? 3 : 1;
    const fmt  = parseFloat(val).toFixed(decs);
    const num   = document.getElementById(`num-${id}`);
    const range = document.getElementById(`sl-${id}`);
    if (num)   num.value   = fmt;
    if (range) range.value = fmt;
  });
  classify();
}

// ── Preset buttons ────────────────────────────────────────────────────────
document.getElementById("btn-benign").addEventListener("click",    () => loadSample(MODEL.samples.benign));
document.getElementById("btn-malignant").addEventListener("click", () => loadSample(MODEL.samples.malignant));
document.getElementById("btn-average").addEventListener("click",   () => loadSample(MODEL.samples.average));

// ── Classification ────────────────────────────────────────────────────────
function classify() {
  let logit = MODEL.intercept;

  MODEL.trainedFeatures.forEach((id, i) => {
    const num = document.getElementById(`num-${id}`);
    const val = parseFloat(num?.value);
    const s   = MODEL.stats[id];
    const v   = isNaN(val) ? s.mean : val;
    logit += ((v - s.mean) / s.std) * MODEL.coefficients[i];
  });

  const pMal = 1 / (1 + Math.exp(-logit));
  const pBen = 1 - pMal;
  const malPct = (pMal * 100).toFixed(1);
  const benPct = (pBen * 100).toFixed(1);

  probMalignantTx.textContent = `${malPct}%`;
  probBenignTx.textContent    = `${benPct}%`;
  barMalignant.style.width    = `${malPct}%`;
  barBenign.style.width       = `${benPct}%`;

  if (pMal >= 0.5) {
    badge.textContent = "MALIGNANT";
    badge.className   = "prediction-badge is-malignant";
    resultCard.className = "result-card state-malignant";
  } else {
    badge.textContent = "BENIGN";
    badge.className   = "prediction-badge is-benign";
    resultCard.className = "result-card state-benign";
  }
}

// ── Boot with most malignant sample ──────────────────────────────────────
loadSample(MODEL.samples.malignant);

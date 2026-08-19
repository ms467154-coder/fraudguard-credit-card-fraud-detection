const state = { features: [], threshold: 0.5 };
const featureGrid = document.querySelector('#featureGrid');
const form = document.querySelector('#predictionForm');
const threshold = document.querySelector('#threshold');
const thresholdValue = document.querySelector('#thresholdValue');
const emptyState = document.querySelector('#emptyState');
const results = document.querySelector('#results');
const errorState = document.querySelector('#errorState');
const modelList = document.querySelector('#modelList');
const apiStatus = document.querySelector('#apiStatus');

function showError(message) {
  errorState.textContent = message;
  errorState.classList.remove('hidden');
  results.classList.add('hidden');
  emptyState.classList.add('hidden');
}

function renderInputs() {
  featureGrid.innerHTML = state.features.map((feature) => `<div class="field"><label for="${feature}">${feature}</label><input id="${feature}" name="${feature}" type="number" step="any" inputmode="decimal" placeholder="0.00" required /></div>`).join('');
}

function renderRegistry(models) {
  modelList.innerHTML = models.map(model => `<div class="model-row"><span>${model.name}</span><span class="loaded">${model.loaded ? '● Loaded' : '○ Missing'}</span></div>`).join('');
}

async function loadRegistry() {
  try {
    const response = await fetch('/api/models');
    if (!response.ok) throw new Error('Registry unavailable');
    const data = await response.json();
    state.features = data.features || [];
    state.threshold = data.threshold || 0.5;
    threshold.value = state.threshold;
    thresholdValue.textContent = Number(state.threshold).toFixed(2);
    renderInputs();
    renderRegistry(data.models || []);
    apiStatus.textContent = `${(data.models || []).filter(model => model.loaded).length} models online`;
  } catch (error) {
    apiStatus.textContent = 'Service unavailable';
    apiStatus.style.color = 'var(--red)';
    modelList.innerHTML = '<span class="muted">Unable to reach the model registry.</span>';
    showError('The inference service could not be reached. Please confirm the API is running and try again.');
  }
}

threshold.addEventListener('input', () => { thresholdValue.textContent = Number(threshold.value).toFixed(2); });

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const button = form.querySelector('button');
  const original = button.innerHTML;
  const values = Object.fromEntries(new FormData(form).entries());
  delete values.threshold;
  const features = Object.fromEntries(Object.entries(values).map(([key, value]) => [key, Number(value)]));
  if (Object.values(features).some(value => !Number.isFinite(value))) {
    showError('Please provide a valid number for every feature before running the analysis.');
    return;
  }
  button.disabled = true;
  button.innerHTML = '<span>Analyzing…</span><span>↗</span>';
  errorState.classList.add('hidden');
  try {
    const response = await fetch('/api/predict', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ features, threshold: Number(threshold.value) }) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Prediction failed');
    const average = data.predictions.reduce((sum, item) => sum + item.fraud_probability, 0) / data.predictions.length;
    const isRisk = average >= Number(threshold.value);
    results.innerHTML = `<div class="result-summary"><div class="summary-title"><div><span class="risk-label ${isRisk ? 'risk' : ''}">${isRisk ? 'Review recommended' : 'Likely legitimate'}</span><div class="risk-score">${(average * 100).toFixed(1)}%</div><span class="muted">Ensemble fraud probability</span></div><span class="panel-index">${isRisk ? '!' : '✓'}</span></div><div class="score-bar"><div class="score-fill" style="width:${average * 100}%"></div></div></div><div class="model-results">${data.predictions.map(item => `<div class="model-result"><div><strong>${item.model}</strong><br /><span>${item.interpretation}</span></div><b class="${item.predicted_class ? 'risk' : ''}">${(item.fraud_probability * 100).toFixed(1)}%</b></div>`).join('')}</div>`;
    emptyState.classList.add('hidden');
    errorState.classList.add('hidden');
    results.classList.remove('hidden');
  } catch (error) {
    showError(error.message || 'Analysis failed. Please review the transaction fields and try again.');
  } finally {
    button.disabled = false;
    button.innerHTML = original;
  }
});

loadRegistry();

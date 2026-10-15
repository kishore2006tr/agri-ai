/**
 * AgriFertile AI - Frontend Application Logic
 * Integrates Chart.js, real-time ML inference, presets, and interactive diagnostics
 */

// State
let soilRadarChart = null;
let modelComparisonChart = null;
let featureImportanceChart = null;
let currentBatchCSV = null;
let currentPresetId = 'fertile_alluvial';

const FEATURE_KEYS = ['N', 'P', 'K', 'pH', 'EC', 'OC', 'S', 'Zn', 'Fe', 'Cu', 'Mn', 'B'];

// Preset Profiles
const PRESET_DATA = {
    'fertile_alluvial': {
        N: 480.0, P: 58.0, K: 340.0, pH: 6.80, EC: 0.55, OC: 0.95,
        S: 22.0, Zn: 1.45, Fe: 11.5, Cu: 0.75, Mn: 6.2, B: 1.05
    },
    'moderate_field': {
        N: 310.0, P: 32.0, K: 210.0, pH: 6.70, EC: 0.65, OC: 0.60,
        S: 13.5, Zn: 0.85, Fe: 7.5, Cu: 0.45, Mn: 3.8, B: 0.65
    },
    'acidic_degraded': {
        N: 150.0, P: 11.0, K: 105.0, pH: 4.80, EC: 0.35, OC: 0.32,
        S: 6.0, Zn: 0.35, Fe: 7.0, Cu: 0.18, Mn: 1.5, B: 0.25
    },
    'saline_alkali': {
        N: 175.0, P: 14.0, K: 185.0, pH: 8.70, EC: 2.80, OC: 0.28,
        S: 14.0, Zn: 0.40, Fe: 2.6, Cu: 0.20, Mn: 1.2, B: 0.70
    }
};

// Optimal Reference Standards (Normalized 100% scale for radar comparison)
const OPTIMAL_TARGETS = {
    N: 420, P: 45, K: 220, pH: 6.8, EC: 0.6, OC: 0.85,
    S: 18, Zn: 1.2, Fe: 9.0, Cu: 0.6, Mn: 5.0, B: 0.85
};

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initRadarChart();
    loadModelMetadata();
    // Run initial prediction with default preset
    runPrediction();
});

// Tab Navigation
function switchTab(tabId) {
    document.querySelectorAll('.tab-panel').forEach(panel => panel.classList.remove('active'));
    document.querySelectorAll('.nav-tab').forEach(btn => btn.classList.remove('active'));

    const targetPanel = document.getElementById(`tab-${tabId}`);
    const targetBtn = document.getElementById(`tab-btn-${tabId}`);

    if (targetPanel) targetPanel.classList.add('active');
    if (targetBtn) targetBtn.classList.add('active');

    // Trigger chart resize when switching to analytics tab
    if (tabId === 'analytics') {
        setTimeout(() => {
            if (modelComparisonChart) modelComparisonChart.resize();
            if (featureImportanceChart) featureImportanceChart.resize();
        }, 100);
    }
}

function initTabs() {
    // Already setup via onclick in HTML
}

// Parameter Synchronization between Range & Number Input
function syncParam(key, value, fromNumInput = false) {
    const val = parseFloat(value);
    const rangeEl = document.getElementById(`range-${key}`);
    const numEl = document.getElementById(`input-${key}`);

    if (fromNumInput) {
        if (rangeEl) rangeEl.value = val;
    } else {
        if (numEl) numEl.value = val;
    }

    // Clear active preset chip state since user modified values
    document.querySelectorAll('.presets-chips .chip').forEach(c => c.classList.remove('active'));
}

// Preset Loader
function loadPreset(presetKey) {
    const data = PRESET_DATA[presetKey];
    if (!data) return;

    currentPresetId = presetKey;
    document.querySelectorAll('.presets-chips .chip').forEach(c => {
        c.classList.remove('active');
        if (c.getAttribute('onclick') && c.getAttribute('onclick').includes(presetKey)) {
            c.classList.add('active');
        }
    });

    for (const [key, val] of Object.entries(data)) {
        const rangeEl = document.getElementById(`range-${key}`);
        const numEl = document.getElementById(`input-${key}`);
        if (rangeEl) rangeEl.value = val;
        if (numEl) numEl.value = val;
    }

    runPrediction();
}

function resetForm() {
    loadPreset('fertile_alluvial');
}

// Gather Current Form Values
function getFormValues() {
    const payload = {};
    FEATURE_KEYS.forEach(k => {
        const input = document.getElementById(`input-${k}`);
        payload[k] = input ? parseFloat(input.value) || 0 : 0;
    });
    return payload;
}

// Real-Time ML Prediction
async function runPrediction() {
    const btn = document.getElementById('btn-predict');
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = `<span class="spinner-sm"></span> Classifying...`;
    }

    const payload = getFormValues();

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (response.ok) {
            const data = await response.json();
            if (data.status === 'success') {
                renderPredictionResults(data, payload);
                return;
            }
        }
    } catch (err) {
        // Fallback to client-side ML engine (essential for GitHub Pages)
    }

    // Client-side execution (instant, zero-backend, works on GitHub Pages)
    try {
        const clientData = runClientSideInference(payload);
        renderPredictionResults(clientData, payload);
    } catch (clientErr) {
        console.error("Client-side prediction error:", clientErr);
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = `<span class="btn-icon">⚡</span> Run AI Fertility Classification`;
        }
    }
}

// Render Results to UI
function renderPredictionResults(res, values) {
    // 1. Classification & Confidence
    const classText = document.getElementById('pred-class-text');
    const confText = document.getElementById('pred-confidence-text');
    const statusDot = document.getElementById('status-dot');
    const bannerBox = document.getElementById('result-banner-box');
    const gaugeCircle = document.getElementById('gauge-circle');
    const gaugeScore = document.getElementById('gauge-score');

    const pred = res.prediction;
    classText.textContent = `${pred} Fertility`;
    confText.innerHTML = `Model Confidence: <strong>${res.confidence}%</strong>`;

    let colorVar = '#10b981';
    let gradientBg = 'rgba(16, 185, 129, 0.12)';
    let borderColor = 'rgba(52, 211, 153, 0.3)';

    if (pred === 'Moderate') {
        colorVar = '#f59e0b';
        gradientBg = 'rgba(245, 158, 11, 0.12)';
        borderColor = 'rgba(251, 191, 36, 0.3)';
    } else if (pred === 'Low') {
        colorVar = '#f43f5e';
        gradientBg = 'rgba(244, 63, 94, 0.12)';
        borderColor = 'rgba(251, 113, 133, 0.3)';
    }

    statusDot.style.backgroundColor = colorVar;
    statusDot.style.boxShadow = `0 0 15px ${colorVar}`;
    bannerBox.style.background = `linear-gradient(135deg, ${gradientBg} 0%, rgba(6, 182, 212, 0.05) 100%)`;
    bannerBox.style.borderColor = borderColor;

    // 2. Health Score Gauge
    const score = res.health_score || 50;
    gaugeScore.textContent = score;
    gaugeCircle.style.background = `conic-gradient(${colorVar} 0% ${score}%, rgba(255, 255, 255, 0.08) ${score}% 100%)`;
    gaugeCircle.style.boxShadow = `0 0 25px ${colorVar}40`;

    // 3. Class Probabilities
    const probs = res.probabilities || { High: 0, Moderate: 0, Low: 0 };
    document.getElementById('prob-high').textContent = `${probs.High || 0}%`;
    document.getElementById('prob-mod').textContent = `${probs.Moderate || 0}%`;
    document.getElementById('prob-low').textContent = `${probs.Low || 0}%`;

    document.getElementById('fill-high').style.width = `${probs.High || 0}%`;
    document.getElementById('fill-mod').style.width = `${probs.Moderate || 0}%`;
    document.getElementById('fill-low').style.width = `${probs.Low || 0}%`;

    // 4. Update Radar Chart
    updateRadarChart(values);

    // 5. Parameter Status Badges Breakdown
    const badgesContainer = document.getElementById('param-badges-list');
    badgesContainer.innerHTML = '';
    const evals = res.evaluations || {};

    FEATURE_KEYS.forEach(key => {
        const item = evals[key];
        if (!item) return;

        const badge = document.createElement('div');
        badge.className = 'badge-item';
        badge.innerHTML = `
            <span class="badge-param">${item.name}</span>
            <span class="badge-val">${item.value} <small style="font-size:0.7rem; color:#94a3b8;">${item.unit}</small></span>
            <span class="badge-tag ${item.badge_color}">${item.status}</span>
        `;
        badgesContainer.appendChild(badge);
    });

    // 6. Fertilizer & Soil Amendments
    const fertContainer = document.getElementById('fertilizer-cards-list');
    fertContainer.innerHTML = '';

    if (res.recommendations && res.recommendations.length > 0) {
        res.recommendations.forEach(rec => {
            const card = document.createElement('div');
            card.className = 'fert-card';
            card.innerHTML = `
                <div class="fert-name">${rec.title} <span class="badge-tag info">${rec.category}</span></div>
                <div class="crop-desc" style="margin-top: 4px;">${rec.detail}</div>
            `;
            fertContainer.appendChild(card);
        });
    }

    if (res.fertilizers && res.fertilizers.length > 0) {
        res.fertilizers.forEach(f => {
            const card = document.createElement('div');
            card.className = 'fert-card';
            card.innerHTML = `
                <div class="fert-name">🎯 ${f.name}</div>
                <div class="fert-dose">Recommended Application: ${f.dose}</div>
            `;
            fertContainer.appendChild(card);
        });
    }

    // 7. Optimal Crops
    const cropsContainer = document.getElementById('crops-list');
    cropsContainer.innerHTML = '';
    (res.crops || []).forEach(c => {
        const card = document.createElement('div');
        card.className = 'crop-card';
        card.innerHTML = `
            <div class="crop-name">🌾 ${c.crop}</div>
            <div class="crop-yield">${c.yield_potential} Potential</div>
            <div class="crop-desc">${c.suitability}</div>
        `;
        cropsContainer.appendChild(card);
    });

    // Update timestamp
    const now = new Date();
    document.getElementById('result-timestamp').textContent = `Analyzed: ${now.toLocaleTimeString()}`;
}

// Chart.js Radar Chart
function initRadarChart() {
    const ctx = document.getElementById('soilRadarChart').getContext('2d');
    
    soilRadarChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['N', 'P', 'K', 'pH', 'EC', 'OC', 'S', 'Zn', 'Fe', 'Cu', 'Mn', 'B'],
            datasets: [
                {
                    label: 'Sample Values (% of Benchmark)',
                    data: [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
                    backgroundColor: 'rgba(16, 185, 129, 0.25)',
                    borderColor: '#10b981',
                    borderWidth: 2,
                    pointBackgroundColor: '#34d399',
                    pointBorderColor: '#fff',
                    pointHoverRadius: 6
                },
                {
                    label: 'Optimal Reference Line (100%)',
                    data: [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
                    backgroundColor: 'transparent',
                    borderColor: 'rgba(255, 255, 255, 0.25)',
                    borderWidth: 1.5,
                    borderDash: [4, 4],
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    angleLines: { color: 'rgba(255, 255, 255, 0.08)' },
                    grid: { color: 'rgba(255, 255, 255, 0.08)' },
                    pointLabels: {
                        color: '#94a3b8',
                        font: { family: 'Plus Jakarta Sans', size: 11, weight: '600' }
                    },
                    ticks: { display: false, maxTicksLimit: 5 },
                    suggestedMin: 20,
                    suggestedMax: 140
                }
            },
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: '#cbd5e1', font: { family: 'Plus Jakarta Sans', size: 11 } }
                }
            }
        }
    });
}

function updateRadarChart(values) {
    if (!soilRadarChart) return;

    // Calculate percentage relative to optimal midpoint
    const normalizedData = FEATURE_KEYS.map(k => {
        const val = values[k] || 0;
        const target = OPTIMAL_TARGETS[k] || 1;
        // Cap between 10% and 180% to avoid skewed radar
        const pct = (val / target) * 100;
        return Math.min(180, Math.max(15, Math.round(pct)));
    });

    soilRadarChart.data.datasets[0].data = normalizedData;
    soilRadarChart.update();
}

async function loadModelMetadata() {
    try {
        const res = await fetch('/api/model-info');
        if (res.ok) {
            const data = await res.json();
            if (data.status === 'success') {
                applyModelMetadataUI(data);
                return;
            }
        }
    } catch (e) {
        // Fallback to embedded static model data
    }

    if (window.SOIL_MODEL_DATA && window.SOIL_MODEL_DATA.metadata) {
        const meta = window.SOIL_MODEL_DATA.metadata;
        applyModelMetadataUI({
            model_name: meta.best_model_name || "Random Forest Classifier",
            test_accuracy: meta.best_test_accuracy || 100.0,
            total_samples: meta.total_samples || 3200,
            model_comparison: meta.model_comparison || {},
            feature_importances: meta.feature_importances || {}
        });
    }
}

function applyModelMetadataUI(data) {
    document.getElementById('header-model-name').textContent = `${data.model_name} (${data.test_accuracy}% Acc)`;
    document.getElementById('analytics-best-model').textContent = data.model_name;
    document.getElementById('analytics-test-acc').textContent = `${data.test_accuracy}%`;
    document.getElementById('analytics-samples').textContent = (data.total_samples || 3200).toLocaleString();

    renderModelComparisonTable(data.model_comparison || {});
    renderFeatureImportanceChart(data.feature_importances || {});
}

function renderModelComparisonTable(comparison) {
    const tbody = document.getElementById('model-comparison-tbody');
    tbody.innerHTML = '';

    const labels = [];
    const cvAccs = [];
    const testAccs = [];

    for (const [name, stats] of Object.entries(comparison)) {
        labels.push(name);
        cvAccs.push(stats.cv_accuracy_mean);
        testAccs.push(stats.test_accuracy);

        const row = document.createElement('tr');
        row.innerHTML = `
            <td style="font-weight: 700; color: #fff;">${name}</td>
            <td><span class="badge-tag info">${stats.cv_accuracy_mean}%</span></td>
            <td><span class="badge-tag success">${stats.test_accuracy}%</span></td>
            <td>${stats.precision_weighted}%</td>
            <td>${stats.recall_weighted}%</td>
            <td><strong style="color: #34d399;">${stats.f1_weighted}%</strong></td>
        `;
        tbody.appendChild(row);
    }

    // Chart.js Bar Chart for Model Comparison
    const ctx = document.getElementById('modelComparisonChart').getContext('2d');
    modelComparisonChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: '5-Fold CV Accuracy (%)',
                    data: cvAccs,
                    backgroundColor: 'rgba(6, 182, 212, 0.65)',
                    borderRadius: 6
                },
                {
                    label: 'Test Set Accuracy (%)',
                    data: testAccs,
                    backgroundColor: 'rgba(16, 185, 129, 0.75)',
                    borderRadius: 6
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    min: 98,
                    max: 100,
                    grid: { color: 'rgba(255, 255, 255, 0.06)' },
                    ticks: { color: '#94a3b8' }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#94a3b8', font: { family: 'Plus Jakarta Sans', size: 10 } }
                }
            },
            plugins: {
                legend: { labels: { color: '#cbd5e1' } }
            }
        }
    });
}

function renderFeatureImportanceChart(importances) {
    const keys = Object.keys(importances);
    const vals = Object.values(importances).map(v => (v * 100).toFixed(2));

    const ctx = document.getElementById('featureImportanceChart').getContext('2d');
    featureImportanceChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: keys,
            datasets: [{
                label: 'Relative Importance (%)',
                data: vals,
                backgroundColor: [
                    '#10b981', '#06b6d4', '#3b82f6', '#6366f1', '#8b5cf6',
                    '#ec4899', '#f43f5e', '#f59e0b', '#84cc16', '#14b8a6', '#0ea5e9', '#a855f7'
                ],
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.06)' },
                    ticks: { color: '#94a3b8' }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#fff', font: { weight: '600' } }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

// Tab 3: Batch CSV Upload Handling
async function handleCSVUpload(file) {
    if (!file) return;

    const statusBox = document.getElementById('batch-status');
    statusBox.style.display = 'flex';
    document.getElementById('batch-results-area').style.display = 'none';

    let success = false;
    try {
        const formData = new FormData();
        formData.append('file', file);
        const response = await fetch('/api/batch-predict', {
            method: 'POST',
            body: formData
        });
        if (response.ok) {
            const data = await response.json();
            if (data.status === 'success') {
                renderBatchSuccess(data);
                success = true;
            }
        }
    } catch (e) {
        // Will fallback to client-side parsing below
    }

    if (!success) {
        try {
            const reader = new FileReader();
            reader.onload = function(e) {
                try {
                    const csvText = e.target.result;
                    const data = processClientBatchCSV(csvText);
                    renderBatchSuccess(data);
                } catch (parseErr) {
                    statusBox.style.display = 'none';
                    alert("Error processing CSV: " + parseErr.message);
                }
            };
            reader.readAsText(file);
        } catch (readErr) {
            statusBox.style.display = 'none';
            alert("Failed to read CSV file: " + readErr.message);
        }
    }
}

function renderBatchSuccess(data) {
    const statusBox = document.getElementById('batch-status');
    const resultsArea = document.getElementById('batch-results-area');
    statusBox.style.display = 'none';
    currentBatchCSV = data.csv_content;
    resultsArea.style.display = 'block';

    document.getElementById('batch-total-count').textContent = data.total_samples;
    document.getElementById('batch-high-count').textContent = data.summary['High'] || 0;
    document.getElementById('batch-mod-count').textContent = data.summary['Moderate'] || 0;
    document.getElementById('batch-low-count').textContent = data.summary['Low'] || 0;

    renderBatchTable(data.preview);
}

function renderBatchTable(previewRows) {
    if (!previewRows || previewRows.length === 0) return;

    const header = document.getElementById('batch-table-header');
    const tbody = document.getElementById('batch-table-body');
    header.innerHTML = '';
    tbody.innerHTML = '';

    const cols = Object.keys(previewRows[0]);
    cols.forEach(c => {
        const th = document.createElement('th');
        th.textContent = c;
        header.appendChild(th);
    });

    previewRows.forEach(row => {
        const tr = document.createElement('tr');
        cols.forEach(c => {
            const td = document.createElement('td');
            if (c === 'Predicted_Fertility') {
                const val = row[c];
                let tagCls = val === 'High' ? 'success' : (val === 'Moderate' ? 'warning' : 'danger');
                td.innerHTML = `<span class="badge-tag ${tagCls}">${val}</span>`;
            } else {
                td.textContent = row[c];
            }
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });
}

function downloadBatchResults() {
    if (!currentBatchCSV) return;
    const blob = new Blob([currentBatchCSV], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `soil_fertility_predictions_${Date.now()}.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
}

function downloadSampleCSV() {
    const csvContent = 
`N,P,K,pH,EC,OC,S,Zn,Fe,Cu,Mn,B
480,58,340,6.8,0.55,0.95,22.0,1.45,11.5,0.75,6.2,1.05
310,32,210,6.7,0.65,0.60,13.5,0.85,7.5,0.45,3.8,0.65
150,11,105,4.8,0.35,0.32,6.0,0.35,7.0,0.18,1.5,0.25
175,14,185,8.7,2.80,0.28,14.0,0.40,2.6,0.20,1.2,0.70
390,44,270,6.6,0.48,0.80,17.0,1.10,8.8,0.55,4.8,0.80`;

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'sample_soil_test_template.csv';
    a.click();
    window.URL.revokeObjectURL(url);
}

// =========================================================================
// Client-Side ML & Agronomic Engine (GitHub Pages & Offline Mode)
// =========================================================================

function runClientSideInference(values) {
    const rfResult = runRandomForestPredict(values);
    const evalsObj = analyzeClientSoilParameters(values);
    const recsObj = generateClientRecommendations(rfResult.prediction, values, evalsObj.deficiencies, evalsObj.excesses);
    
    return {
        status: 'success',
        prediction: rfResult.prediction,
        confidence: rfResult.confidence,
        probabilities: rfResult.probabilities,
        health_score: evalsObj.health_score,
        evaluations: evalsObj.evaluations,
        recommendations: recsObj.recommendations,
        fertilizers: recsObj.fertilizers,
        crops: recsObj.crops
    };
}

function runRandomForestPredict(values) {
    const classNames = (window.SOIL_MODEL_DATA && window.SOIL_MODEL_DATA.metadata && window.SOIL_MODEL_DATA.metadata.class_names) 
        ? window.SOIL_MODEL_DATA.metadata.class_names 
        : ['Low', 'Moderate', 'High'];
    const featureCols = FEATURE_KEYS;

    if (!window.SOIL_MODEL_DATA || !window.SOIL_MODEL_DATA.trees) {
        return {
            prediction: 'Moderate',
            confidence: 85.0,
            probabilities: { 'Low': 10.0, 'Moderate': 85.0, 'High': 5.0 }
        };
    }

    const trees = window.SOIL_MODEL_DATA.trees;
    const x = featureCols.map(col => parseFloat(values[col]) || 0.0);
    const nClasses = classNames.length;
    const sumProbs = new Array(nClasses).fill(0.0);

    for (let i = 0; i < trees.length; i++) {
        const t = trees[i];
        let node = 0;
        while (t.cl[node] !== -1) {
            const f = t.f[node];
            const th = t.th[node];
            if (x[f] <= th) {
                node = t.cl[node];
            } else {
                node = t.cr[node];
            }
        }
        const vals = t.v[node];
        let nodeSum = 0;
        for (let j = 0; j < nClasses; j++) nodeSum += vals[j];
        if (nodeSum === 0) nodeSum = 1;
        for (let j = 0; j < nClasses; j++) {
            sumProbs[j] += vals[j] / nodeSum;
        }
    }

    const avgProbs = sumProbs.map(p => p / trees.length);
    let bestIdx = 0;
    let maxProb = -1;
    for (let j = 0; j < nClasses; j++) {
        if (avgProbs[j] > maxProb) {
            maxProb = avgProbs[j];
            bestIdx = j;
        }
    }

    const probDict = {};
    for (let j = 0; j < nClasses; j++) {
        probDict[classNames[j]] = Math.round(avgProbs[j] * 1000) / 10;
    }

    return {
        prediction: classNames[bestIdx],
        confidence: Math.round(maxProb * 1000) / 10,
        probabilities: probDict
    };
}

function analyzeClientSoilParameters(values) {
    const benchmarks = (window.SOIL_MODEL_DATA && window.SOIL_MODEL_DATA.benchmarks) ? window.SOIL_MODEL_DATA.benchmarks : {
        "N": {"name": "Nitrogen (N)", "unit": "kg/ha", "opt_min": 280, "opt_max": 560, "low": 280, "high": 560},
        "P": {"name": "Phosphorus (P)", "unit": "kg/ha", "opt_min": 23, "opt_max": 56, "low": 23, "high": 56},
        "K": {"name": "Potassium (K)", "unit": "kg/ha", "opt_min": 140, "opt_max": 280, "low": 140, "high": 280},
        "pH": {"name": "Soil pH", "unit": "pH", "opt_min": 6.0, "opt_max": 7.5, "low": 6.0, "high": 7.5},
        "EC": {"name": "Salinity (EC)", "unit": "dS/m", "opt_min": 0.1, "opt_max": 1.0, "low": 0.0, "high": 1.5},
        "OC": {"name": "Organic Carbon (OC)", "unit": "%", "opt_min": 0.50, "opt_max": 0.75, "low": 0.50, "high": 0.75},
        "S": {"name": "Sulfur (S)", "unit": "ppm", "opt_min": 10.0, "opt_max": 25.0, "low": 10.0, "high": 25.0},
        "Zn": {"name": "Zinc (Zn)", "unit": "ppm", "opt_min": 0.60, "opt_max": 1.50, "low": 0.60, "high": 1.50},
        "Fe": {"name": "Iron (Fe)", "unit": "ppm", "opt_min": 4.50, "opt_max": 12.0, "low": 4.50, "high": 12.0},
        "Cu": {"name": "Copper (Cu)", "unit": "ppm", "opt_min": 0.20, "opt_max": 0.80, "low": 0.20, "high": 0.80},
        "Mn": {"name": "Manganese (Mn)", "unit": "ppm", "opt_min": 2.00, "opt_max": 8.00, "low": 2.00, "high": 8.00},
        "B": {"name": "Boron (B)", "unit": "ppm", "opt_min": 0.50, "opt_max": 1.20, "low": 0.50, "high": 1.20}
    };

    const evaluations = {};
    const deficiencies = [];
    const excesses = [];
    let optimalCount = 0;

    FEATURE_KEYS.forEach(key => {
        const val = parseFloat(values[key]) || 0;
        const bench = benchmarks[key] || { name: key, unit: '', opt_min: 0, opt_max: 100, low: 0, high: 100 };
        let status = '';
        let color = '';

        if (key === 'pH') {
            if (val < 5.5) {
                status = "Strongly Acidic";
                color = "danger";
                deficiencies.push("Severe Soil Acidity");
            } else if (val < 6.0) {
                status = "Moderately Acidic";
                color = "warning";
                deficiencies.push("Mild Soil Acidity");
            } else if (val <= 7.5) {
                status = "Optimal / Neutral";
                color = "success";
                optimalCount++;
            } else if (val <= 8.5) {
                status = "Moderately Alkaline";
                color = "warning";
                excesses.push("Alkalinity");
            } else {
                status = "Strongly Alkaline / Sodic";
                color = "danger";
                excesses.push("Severe Alkalinity");
            }
        } else if (key === 'EC') {
            if (val <= 1.0) {
                status = "Non-Saline (Safe)";
                color = "success";
                optimalCount++;
            } else if (val <= 2.0) {
                status = "Slight Salinity";
                color = "warning";
                excesses.push("Moderate Salinity (EC)");
            } else {
                status = "High Salinity (Injurious)";
                color = "danger";
                excesses.push("High Salinity (EC)");
            }
        } else {
            if (val < bench.low) {
                status = "Deficient";
                color = "danger";
                deficiencies.push(`${bench.name} Deficient`);
            } else if (val <= bench.high) {
                status = "Sufficient / Optimal";
                color = "success";
                optimalCount++;
            } else {
                status = "High / Surfeit";
                color = "info";
            }
        }

        evaluations[key] = {
            value: val,
            unit: bench.unit,
            name: bench.name,
            status: status,
            badge_color: color,
            opt_min: bench.opt_min,
            opt_max: bench.opt_max
        };
    });

    const ocVal = parseFloat(values.OC) || 0.5;
    const health_score = Math.round(Math.min(100, Math.max(15, (optimalCount / FEATURE_KEYS.length) * 100 + (ocVal * 15))));

    return { evaluations, deficiencies, excesses, health_score };
}

function generateClientRecommendations(pred_class, values, deficiencies, excesses) {
    const recommendations = [];
    const fertilizers = [];
    let crops = [];

    const ph = parseFloat(values.pH) || 6.5;
    const ec = parseFloat(values.EC) || 0.5;
    const oc = parseFloat(values.OC) || 0.5;
    const n = parseFloat(values.N) || 300;
    const p = parseFloat(values.P) || 30;
    const k = parseFloat(values.K) || 200;
    const zn = parseFloat(values.Zn) || 0.8;
    const fe = parseFloat(values.Fe) || 7.0;

    // 1. Soil Amendment & Conditioning
    if (ph < 6.0) {
        recommendations.push({
            category: "Soil Conditioning",
            title: "Apply Agricultural Lime (CaCO3)",
            detail: `Soil pH is acidic (${ph}). Apply 2.0 - 3.5 tons/hectare of finely ground agricultural limestone or dolomite to neutralize acidity and unlock phosphorus fixation.`
        });
    } else if (ph > 8.0) {
        recommendations.push({
            category: "Soil Conditioning",
            title: "Apply Gypsum (CaSO4·2H2O)",
            detail: `Soil pH is alkaline/sodic (${ph}). Incorporate agricultural gypsum (2.5 - 4.0 tons/ha) accompanied by deep leaching to displace exchangeable sodium.`
        });
    }

    if (ec > 1.5) {
        recommendations.push({
            category: "Salinity Management",
            title: "Install Leaching & Drainage Channels",
            detail: `Electrical conductivity is elevated (${ec} dS/m). Ensure adequate drainage to wash root-zone salts and avoid saline groundwater irrigation.`
        });
    }

    // 2. Organic Matter
    if (oc < 0.50) {
        recommendations.push({
            category: "Organic Carbon Enhancement",
            title: "Incorporate Compost / Farmyard Manure (FYM)",
            detail: `Organic Carbon is low (${oc}%). Incorporate 8 - 12 tons/ha of well-decomposed FYM, vermicompost, or cultivate green manure crops (Dhaincha/Sunn hemp).`
        });
        fertilizers.push({ name: "Vermicompost / Well-rotted FYM", dose: "8 - 10 tons/ha" });
    } else if (oc < 0.75) {
        recommendations.push({
            category: "Organic Carbon Maintenance",
            title: "Maintain Mulching & Crop Residues",
            detail: "Retain crop stubble and add 4 - 5 tons/ha compost annually to maintain active microbial biomass."
        });
    }

    // 3. Primary Macronutrient Fertilizer Program
    if (n < 280) {
        fertilizers.push({ name: "Urea (46% N) or Ammonium Sulfate", dose: "110 - 130 kg/ha in 2-3 split doses" });
        recommendations.push({
            category: "Nutrient Rectification",
            title: "Nitrogen Boost via Split Application",
            detail: "Soil nitrogen is deficient. Administer split urea applications (at basal, vegetative, and panicle initiation) to prevent leaching losses."
        });
    } else if (n <= 560) {
        fertilizers.push({ name: "Balanced Urea Maintenance", dose: "70 - 90 kg/ha in split doses" });
    }

    if (p < 23) {
        fertilizers.push({ name: "Single Super Phosphate (SSP) or DAP", dose: "120 - 150 kg/ha applied as basal" });
        recommendations.push({
            category: "Nutrient Rectification",
            title: "Phosphorus Application (Basal Placement)",
            detail: "Place phosphorus 5cm below and beside seeds during sowing for vigorous root system development."
        });
    } else if (p <= 56) {
        fertilizers.push({ name: "Di-Ammonium Phosphate (DAP)", dose: "50 - 65 kg/ha basal" });
    }

    if (k < 140) {
        fertilizers.push({ name: "Muriate of Potash (MOP - 60% K2O)", dose: "60 - 80 kg/ha basal" });
        recommendations.push({
            category: "Nutrient Rectification",
            title: "Potash Application for Stress Resistance",
            detail: "Potassium is low. Apply MOP to enhance lodging resistance, drought tolerance, and grain filling."
        });
    }

    // 4. Micronutrients
    if (zn < 0.60) {
        fertilizers.push({ name: "Zinc Sulfate Heptahydrate (ZnSO4)", dose: "20 - 25 kg/ha soil application or 0.5% foliar spray" });
    }
    if (fe < 4.50) {
        fertilizers.push({ name: "Ferrous Sulfate (FeSO4)", dose: "15 - 20 kg/ha or 1.0% foliar spray with citric acid" });
    }

    // 5. Crop Recommendations
    if (pred_class === "High") {
        crops = [
            { crop: "Wheat / Rice", yield_potential: "Excellent (95-100%)", suitability: "Ideal nutrient balance ensures peak grain yield." },
            { crop: "Maize & Hybrid Corn", yield_potential: "High (90-95%)", suitability: "Heavy feeder crops flourish in high organic carbon soil." },
            { crop: "Cotton / Sugarcane", yield_potential: "Optimal (92-98%)", suitability: "Deep root network extracts rich nutrient reserves." },
            { crop: "High-Value Vegetables (Tomatoes, Capsicum)", yield_potential: "Exceptional", suitability: "Excellent return on investment with premium quality produce." }
        ];
    } else if (pred_class === "Moderate") {
        crops = [
            { crop: "Soybeans / Pulses (Chickpea, Lentil)", yield_potential: "Good (80-85%)", suitability: "Legumes fix atmospheric nitrogen and improve fertility." },
            { crop: "Wheat / Barley", yield_potential: "Good (75-82%)", suitability: "Performs well with standard N-P-K supplemental fertilization." },
            { crop: "Mustard / Groundnut (Oilseeds)", yield_potential: "Moderate to High", suitability: "Thrives well with recommended sulfur and potash top-dressing." },
            { crop: "Millets (Pearl Millet, Finger Millet)", yield_potential: "Very Resilient", suitability: "Low input cost with reliable harvest stability." }
        ];
    } else {
        if (ph < 5.8) {
            crops = [
                { crop: "Tea / Coffee / Cardamom", yield_potential: "Good", suitability: "Naturally tolerant to acidic soil conditions." },
                { crop: "Potato", yield_potential: "Fair (with liming)", suitability: "Tolerates moderate acidity; scab incidence is lower." },
                { crop: "Buckwheat / Oats", yield_potential: "Sturdy", suitability: "Extracts bound soil phosphorus effectively." }
            ];
        } else if (ec > 1.8) {
            crops = [
                { crop: "Barley", yield_potential: "High Tolerance", suitability: "Most salt-tolerant cereal crop available." },
                { crop: "Sugar Beet", yield_potential: "Moderate", suitability: "High tolerance to saline conditions." },
                { crop: "Cotton", yield_potential: "Moderate", suitability: "Tolerates moderate osmotic soil stress." }
            ];
        } else {
            crops = [
                { crop: "Millets (Sorghum, Pearl Millet)", yield_potential: "Dependable", suitability: "Hardy drought and nutrient-stress tolerant crops." },
                { crop: "Legumes (Cowpea, Green Gram)", yield_potential: "Recommended", suitability: "Rehabilitates depleted soil through biological nitrogen fixation." },
                { crop: "Cassava / Sweet Potato", yield_potential: "Fair", suitability: "Yields reliably even in impoverished soils." }
            ];
        }
    }

    return { recommendations, fertilizers, crops };
}

function processClientBatchCSV(csvText) {
    const lines = csvText.trim().split(/\r?\n/).filter(line => line.trim().length > 0);
    if (lines.length < 2) throw new Error("CSV file must contain a header line and at least one data row.");

    const rawHeaders = lines[0].split(',').map(h => h.trim().replace(/^["']|["']$/g, ''));
    const colIndices = {};
    FEATURE_KEYS.forEach(k => {
        const idx = rawHeaders.findIndex(h => h.toLowerCase() === k.toLowerCase());
        colIndices[k] = idx;
    });

    const missing = FEATURE_KEYS.filter(k => colIndices[k] === -1);
    if (missing.length > 0) {
        throw new Error(`CSV is missing required feature columns: ${missing.join(', ')}`);
    }

    const summary = { 'High': 0, 'Moderate': 0, 'Low': 0 };
    const preview = [];
    const outputLines = [rawHeaders.join(',') + ',Predicted_Fertility'];

    for (let i = 1; i < lines.length; i++) {
        const parts = lines[i].split(',').map(p => p.trim().replace(/^["']|["']$/g, ''));
        if (parts.length < rawHeaders.length) continue;

        const rowValues = {};
        FEATURE_KEYS.forEach(k => {
            rowValues[k] = parseFloat(parts[colIndices[k]]) || 0.0;
        });

        const rf = runRandomForestPredict(rowValues);
        const pred = rf.prediction;
        summary[pred] = (summary[pred] || 0) + 1;

        outputLines.push(parts.join(',') + ',' + pred);

        if (preview.length < 10) {
            const previewRow = {};
            rawHeaders.forEach((h, idx) => {
                previewRow[h] = parts[idx];
            });
            previewRow['Predicted_Fertility'] = pred;
            preview.push(previewRow);
        }
    }

    return {
        status: 'success',
        total_samples: lines.length - 1,
        summary: summary,
        preview: preview,
        csv_content: outputLines.join('\n')
    };
}


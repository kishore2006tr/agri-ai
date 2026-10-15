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

        const data = await response.json();

        if (data.status === 'success') {
            renderPredictionResults(data, payload);
        } else {
            console.error("Prediction error:", data.message);
        }
    } catch (err) {
        console.error("API request failed:", err);
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

// Load Model Metadata & Render Tab 2 Analytics
async function loadModelMetadata() {
    try {
        const res = await fetch('/api/model-info');
        const data = await res.json();

        if (data.status === 'success') {
            document.getElementById('header-model-name').textContent = `${data.model_name} (${data.test_accuracy}% Acc)`;
            document.getElementById('analytics-best-model').textContent = data.model_name;
            document.getElementById('analytics-test-acc').textContent = `${data.test_accuracy}%`;
            document.getElementById('analytics-samples').textContent = data.total_samples.toLocaleString();

            renderModelComparisonTable(data.model_comparison);
            renderFeatureImportanceChart(data.feature_importances);
        }
    } catch (e) {
        console.error("Failed to load model metadata:", e);
    }
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
    const resultsArea = document.getElementById('batch-results-area');

    statusBox.style.display = 'flex';
    resultsArea.style.display = 'none';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/api/batch-predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        statusBox.style.display = 'none';

        if (data.status === 'success') {
            currentBatchCSV = data.csv_content;
            resultsArea.style.display = 'block';

            document.getElementById('batch-total-count').textContent = data.total_samples;
            document.getElementById('batch-high-count').textContent = data.summary['High'] || 0;
            document.getElementById('batch-mod-count').textContent = data.summary['Moderate'] || 0;
            document.getElementById('batch-low-count').textContent = data.summary['Low'] || 0;

            renderBatchTable(data.preview);
        } else {
            alert("Error processing CSV: " + (data.error || data.message));
        }
    } catch (e) {
        statusBox.style.display = 'none';
        alert("Failed to upload CSV file: " + e.message);
    }
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

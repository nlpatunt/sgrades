// ===== API Configuration =====
const API_BASE_URL = '';  // Empty string = same domain
console.log("BESESR JS CSV Workflow loaded");

// ===== DOM Elements =====
const elements = {
    // Stats
    totalDatasets: document.getElementById('total-datasets'),
    totalModels: document.getElementById('total-models'),
    totalEvaluations: document.getElementById('total-evaluations'),
    
    // Datasets
    datasetsGrid: document.getElementById('datasets-grid'),
    downloadAllBtn: document.getElementById('download-all-btn'),
    
    // CSV Submission
    outputForm: document.getElementById('output-submit-form'),
    datasetSelect: document.getElementById('dataset-name'),
    submitResult: document.getElementById('submit-result'),
    
    // Leaderboard
    metricSelector: document.getElementById('metric-selector'),
    refreshLeaderboard: document.getElementById('refresh-leaderboard'),
    leaderboardTable: document.getElementById('leaderboard-table')
};

// ===== Utility Functions =====
function showLoading(element, message = 'Loading...') {
    element.innerHTML = `<div class="loading"><i class="fas fa-spinner fa-spin"></i> ${message}</div>`;
}

function showError(element, message) {
    element.innerHTML = `
        <div class="error" style="text-align: center; padding: 3rem; color: var(--error);">
            <i class="fas fa-exclamation-triangle" style="font-size: 2rem; margin-bottom: 1rem; display: block;"></i>
            <h3>Error Loading Data</h3>
            <p>${message}</p>
            <button onclick="location.reload()" class="btn btn-outline" style="margin-top: 1rem;">
                <i class="fas fa-redo"></i> Retry
            </button>
        </div>
    `;
}

function showEmptyState(element, title, message, actionHtml = '') {
    element.innerHTML = `
        <div class="empty-state" style="text-align: center; padding: 4rem; color: var(--text-secondary);">
            <i class="fas fa-inbox" style="font-size: 3rem; margin-bottom: 1rem; display: block; opacity: 0.5;"></i>
            <h3 style="margin-bottom: 1rem; color: var(--text-primary);">${title}</h3>
            <p style="margin-bottom: 2rem; max-width: 400px; margin-left: auto; margin-right: auto;">${message}</p>
            ${actionHtml}
        </div>
    `;
}

// ===== API Functions =====
async function fetchAPI(endpoint) {
    try {
        const response = await fetch(`${API_BASE_URL}${endpoint}`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

// ===== Download Functions =====
async function downloadFile(url, suggestedName) {
    try {
        console.log(`📥 Downloading: ${url}`);
        const res = await fetch(`${API_BASE_URL}${url}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        
        const blob = await res.blob();
        const a = document.createElement('a');
        const href = URL.createObjectURL(blob);
        a.href = href;
        a.download = suggestedName;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(href);
        
        console.log(`✅ Downloaded: ${suggestedName}`);
    } catch (err) {
        alert('Download failed: ' + err.message);
        console.error('Download failed:', err);
    }
}

function downloadAllDatasets() {
    downloadFile('/api/datasets/download/all', 'besesr_datasets.zip');
}

function downloadSingleDataset(datasetName) {
    downloadFile(`/api/datasets/download/${encodeURIComponent(datasetName)}`, `${datasetName}.csv`);
}

// ===== Load Platform Stats =====
async function loadPlatformStats() {
    try {
        const stats = await fetchAPI('/api/leaderboard/stats');
        
        // Update stats with animation
        if (elements.totalModels) {
            animateNumber(elements.totalModels, stats.total_models_submitted || 0);
        }
        if (elements.totalEvaluations) {
            animateNumber(elements.totalEvaluations, stats.total_evaluations_completed || 0);
        }
        
        console.log('✅ Platform stats loaded');
    } catch (error) {
        console.error('Failed to load stats:', error);
        if (elements.totalModels) elements.totalModels.textContent = '0';
        if (elements.totalEvaluations) elements.totalEvaluations.textContent = '0';
    }
}

function animateNumber(element, targetValue, duration = 1000) {
    const startValue = 0;
    const startTime = performance.now();
    
    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        const currentValue = Math.floor(startValue + (targetValue - startValue) * progress);
        element.textContent = currentValue;
        
        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }
    
    requestAnimationFrame(update);
}

// ===== Load Datasets =====
async function loadDatasets() {
    if (!elements.datasetsGrid) return;

    showLoading(elements.datasetsGrid, 'Loading datasets...');

    try {
        // ✅ Fixed endpoint
        const data = await fetchAPI('/api/datasets/');

        if (!data.datasets || data.datasets.length === 0) {
            showEmptyState(
                elements.datasetsGrid,
                'No Datasets Available',
                'No datasets have been configured yet. Please check back later.'
            );
            return;
        }

        // ✅ Updated dataset cards with download buttons
        const datasetsHTML = data.datasets.map((dataset, index) => `
            <div class="dataset-card fade-in" style="animation-delay: ${index * 0.1}s;">
                <h3>${dataset.name}</h3>
                <p>${dataset.description}</p>

                <div class="dataset-info">
                    <span>
                        <i class="fas fa-database"></i>
                        Score Range: ${dataset.score_range[0]}-${dataset.score_range[1]}
                    </span>
                    <span>
                        <i class="fas fa-columns"></i>
                        ${dataset.split}
                    </span>
                </div>

                <div class="dataset-metrics">
                    <div class="metric-tag">Essays: ${dataset.essay_column}</div>
                    <div class="metric-tag">Scores: ${dataset.score_column}</div>
                    <div class="metric-tag">Prompts: ${dataset.prompt_column}</div>
                </div>

                <div class="dataset-actions">
                    <button onclick="downloadSingleDataset('${dataset.name}')" class="btn btn-sm btn-primary">
                        <i class="fas fa-download"></i> Download CSV
                    </button>
                    <button onclick="showDatasetSample('${dataset.name}')" class="btn btn-sm btn-outline">
                        <i class="fas fa-eye"></i> Preview
                    </button>
                </div>
            </div>
        `).join('');

        elements.datasetsGrid.innerHTML = datasetsHTML;

        // ✅ Update total count and populate dataset selector
        if (elements.totalDatasets) {
            animateNumber(elements.totalDatasets, data.total_count || data.datasets.length);
        }

        // ✅ Populate dataset dropdown for submission form
        if (elements.datasetSelect) {
            const options = data.datasets.map(dataset => 
                `<option value="${dataset.name}">${dataset.name}</option>`
            ).join('');
            elements.datasetSelect.innerHTML = '<option value="">-- Choose a dataset --</option>' + options;
        }

        console.log(`✅ Loaded ${data.datasets.length} datasets successfully`);

    } catch (error) {
        showError(elements.datasetsGrid, 'Failed to load datasets. Please check the API connection.');
        console.error('Failed to load datasets:', error);
    }
}

// ===== Dataset Sample Preview =====
async function showDatasetSample(datasetName) {
    try {
        console.log(`🔍 Loading sample for ${datasetName}`);
        const sample = await fetchAPI(`/api/datasets/${datasetName}/sample?size=3`);
        
        if (sample.essays && sample.essays.length > 0) {
            const sampleHTML = sample.essays.map(essay => `
                <div style="border: 1px solid var(--border-light); padding: 1rem; margin: 0.5rem 0; border-radius: 6px;">
                    <div style="font-weight: 600; color: var(--primary-blue);">${essay.essay_id}</div>
                    <div style="margin: 0.5rem 0; font-size: 0.9rem; color: var(--text-secondary);">
                        <strong>Prompt:</strong> ${essay.prompt}
                    </div>
                    <div style="margin: 0.5rem 0; color: var(--text-primary);">
                        ${essay.essay_preview}
                    </div>
                    <div style="font-size: 0.8rem; color: var(--text-muted);">
                        <strong>Human Score:</strong> ${essay.human_score} | 
                        <strong>Words:</strong> ${essay.word_count}
                    </div>
                </div>
            `).join('');
            
            alert(`Sample Essays from ${datasetName}:\n\n${sample.essays.map(e => `${e.essay_id}: ${e.essay_preview.substring(0, 100)}...`).join('\n\n')}`);
        } else {
            alert(`No sample data available for ${datasetName}`);
        }
    } catch (error) {
        alert(`Failed to load sample for ${datasetName}: ${error.message}`);
        console.error('Sample load failed:', error);
    }
}

// ===== CSV Submission Form =====
function setupCSVSubmissionForm() {
    if (!elements.outputForm) return;
    
    elements.outputForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const submitBtn = this.querySelector('.btn-large');
        const originalText = submitBtn.innerHTML;
        
        // Show loading state
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Uploading Results...';
        elements.submitResult.style.display = 'none';
        
        try {
            // ✅ Create FormData for file upload
            const formData = new FormData(this);
            
            // ✅ Submit to CSV upload endpoint
            const response = await fetch('/submissions/upload-single-result', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (!response.ok) {
                throw new Error(result.detail || `HTTP ${response.status}`);
            }
            
            // Show success
            elements.submitResult.className = 'submit-result success';
            elements.submitResult.innerHTML = `
                <h3><i class="fas fa-check-circle"></i> Results Submitted Successfully!</h3>
                <p><strong>Submission ID:</strong> <code>${result.submission_id}</code></p>
                <p>Your results for "${result.model_name}" on dataset "${result.dataset_name}" have been evaluated!</p>
                <div style="background: var(--bg-light); padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                    <p><strong>Quadratic Weighted Kappa:</strong> ${result.evaluation_results.quadratic_weighted_kappa.toFixed(3)}</p>
                    <p><strong>Pearson Correlation:</strong> ${result.evaluation_results.pearson_correlation.toFixed(3)}</p>
                    <p><strong>Essays Evaluated:</strong> ${result.evaluation_results.essays_evaluated}</p>
                    <p><strong>Match Rate:</strong> ${(result.evaluation_results.match_rate * 100).toFixed(1)}%</p>
                </div>
                <div style="margin-top: 1.5rem;">
                    <a href="#leaderboard" class="btn btn-primary">
                        <i class="fas fa-trophy"></i> View Leaderboard
                    </a>
                </div>
            `;
            elements.submitResult.style.display = 'block';
            
            // Reset form
            this.reset();
            
            // Scroll to result
            elements.submitResult.scrollIntoView({ behavior: 'smooth', block: 'center' });
            
            // Refresh data
            setTimeout(() => {
                loadPlatformStats();
                loadLeaderboard();
            }, 1000);
            
            console.log(`✅ CSV results submitted successfully: ${result.submission_id}`);
            
        } catch (error) {
            // Show error
            elements.submitResult.className = 'submit-result error';
            elements.submitResult.innerHTML = `
                <h3><i class="fas fa-exclamation-circle"></i> Submission Failed</h3>
                <p><strong>Error:</strong> ${error.message}</p>
                <p>Please check your CSV file format and try again.</p>
                <div style="margin-top: 1rem;">
                    <a href="/submissions/template" target="_blank" class="btn btn-outline">
                        <i class="fas fa-file-csv"></i> View CSV Template
                    </a>
                </div>
            `;
            elements.submitResult.style.display = 'block';
            elements.submitResult.scrollIntoView({ behavior: 'smooth', block: 'center' });
            
            console.error('CSV submission failed:', error);
        } finally {
            // Reset button
            submitBtn.disabled = false;
            submitBtn.innerHTML = originalText;
        }
    });
}

// ===== Load Leaderboard =====
async function loadLeaderboard() {
    if (!elements.leaderboardTable) return;
    
    showLoading(elements.leaderboardTable, 'Loading leaderboard...');
    
    try {
        const metric = elements.metricSelector?.value || 'quadratic_weighted_kappa';
        // ✅ Fixed endpoint
        const data = await fetchAPI(`/api/leaderboard/?metric=${metric}&limit=20`);
        
        if (!data.leaderboard || data.leaderboard.length === 0) {
            showEmptyState(
                elements.leaderboardTable,
                'No Models on Leaderboard Yet',
                'Be the first to submit CSV results to appear on the leaderboard!',
                '<a href="#submit" class="btn btn-primary">Submit Your Results</a>'
            );
            return;
        }
        
        const tableHTML = `
            <table>
                <thead>
                    <tr>
                        <th style="width: 60px;">Rank</th>
                        <th>Model</th>
                        <th>Submitter</th>
                        <th>QWK Score</th>
                        <th>Pearson Score</th>
                        <th>Datasets</th>
                        <th>Essays</th>
                        <th style="width: 140px;">Submitted</th>
                    </tr>
                </thead>
                <tbody>
                    ${data.leaderboard.map((entry, index) => `
                        <tr style="animation: fadeIn 0.6s ease-out ${index * 0.1}s both;">
                            <td>
                                <div class="rank-badge ${getRankClass(entry.rank)}">
                                    ${entry.rank}
                                </div>
                            </td>
                            <td>
                                <div>
                                    <strong style="color: var(--text-primary);">${entry.model_name}</strong>
                                    <br>
                                    <small style="color: var(--text-muted); font-family: monospace;">${entry.model_id}</small>
                                </div>
                            </td>
                            <td style="color: var(--text-secondary);">${entry.submitter_name}</td>
                            <td>
                                <span class="score-badge ${getScoreClass(entry.avg_quadratic_weighted_kappa)}" 
                                      title="${getScoreDescription(entry.avg_quadratic_weighted_kappa)}">
                                    ${formatScore(entry.avg_quadratic_weighted_kappa)}
                                </span>
                            </td>
                            <td>
                                <span class="score-badge ${getScoreClass(entry.avg_pearson_correlation)}"
                                      title="${getScoreDescription(entry.avg_pearson_correlation)}">
                                    ${formatScore(entry.avg_pearson_correlation)}
                                </span>
                            </td>
                            <td style="text-align: center;">
                                ${entry.datasets_completed}/${entry.total_datasets}
                            </td>
                            <td style="text-align: center;">
                                ${entry.total_essays_evaluated || 0}
                            </td>
                            <td>
                                <small style="color: var(--text-muted);">
                                    ${formatDate(entry.submission_time)}
                                </small>
                            </td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;
        
        elements.leaderboardTable.innerHTML = tableHTML;
        
        console.log(`✅ Loaded leaderboard with ${data.leaderboard.length} entries`);
        
    } catch (error) {
        showError(elements.leaderboardTable, 'Failed to load leaderboard. Please check the API connection.');
        console.error('Failed to load leaderboard:', error);
    }
}

// ===== Utility Functions =====
function formatDate(dateString) {
    if (!dateString) return 'N/A';
    return new Date(dateString).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

function formatScore(score, decimals = 3) {
    if (score === null || score === undefined || isNaN(score)) return 'N/A';
    return Number(score).toFixed(decimals);
}

function getScoreClass(score) {
    if (score === null || score === undefined || isNaN(score)) return 'score-poor';
    if (score >= 0.8) return 'score-excellent';
    if (score >= 0.6) return 'score-good';
    if (score >= 0.4) return 'score-fair';
    return 'score-poor';
}

function getRankClass(rank) {
    if (rank === 1) return 'rank-1';
    if (rank === 2) return 'rank-2';
    if (rank === 3) return 'rank-3';
    return 'rank-other';
}

function getScoreDescription(score) {
    if (score === null || score === undefined || isNaN(score)) return 'No data';
    if (score >= 0.8) return 'Excellent';
    if (score >= 0.6) return 'Good';
    if (score >= 0.4) return 'Fair';
    return 'Needs improvement';
}

// ===== Event Listeners =====
function setupEventListeners() {
    // Download all datasets button
    if (elements.downloadAllBtn) {
        elements.downloadAllBtn.addEventListener('click', (e) => {
            e.preventDefault();
            downloadAllDatasets();
        });
    }
    
    // Metric selector change
    if (elements.metricSelector) {
        elements.metricSelector.addEventListener('change', () => {
            console.log('Metric changed, reloading leaderboard...');
            loadLeaderboard();
        });
    }
    
    // Refresh leaderboard button
    if (elements.refreshLeaderboard) {
        elements.refreshLeaderboard.addEventListener('click', () => {
            console.log('Manual leaderboard refresh requested...');
            loadLeaderboard();
            loadPlatformStats();
            
            // Visual feedback
            const button = elements.refreshLeaderboard;
            const icon = button.querySelector('i');
            if (icon) {
                icon.classList.add('fa-spin');
                setTimeout(() => icon.classList.remove('fa-spin'), 1000);
            }
        });
    }
}

// ===== Smooth Scrolling =====
function setupSmoothScrolling() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                const headerOffset = 80;
                const elementPosition = target.getBoundingClientRect().top;
                const offsetPosition = elementPosition + window.pageYOffset - headerOffset;
                
                window.scrollTo({
                    top: offsetPosition,
                    behavior: 'smooth'
                });
            }
        });
    });
}

// ===== Initialize Application =====
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 BESESR CSV Frontend Initialized');
    console.log('📡 API Base URL:', API_BASE_URL);
    
    // Setup all functionality
    setupSmoothScrolling();
    setupCSVSubmissionForm();
    setupEventListeners();
    
    // Load initial data
    console.log('📊 Loading initial data...');
    loadPlatformStats();
    loadDatasets();
    loadLeaderboard();
    
    // Auto-refresh data periodically
    setInterval(() => {
        console.log('🔄 Auto-refreshing data...');
        loadPlatformStats();
        loadLeaderboard();
    }, 60000);
    
    console.log('✅ BESESR Platform ready!');
});

// Add these functions to your app.js (replace the existing setupCSVSubmissionForm function)

// ===== Setup Form Handlers =====
function setupCSVSubmissionForm() {
    // Setup benchmark form (main form)
    const benchmarkForm = document.getElementById('benchmark-submit-form');
    if (benchmarkForm) {
        benchmarkForm.addEventListener('submit', handleBenchmarkSubmission);
    }
    
    // Setup single dataset form (for testing)
    const singleForm = document.getElementById('single-submit-form');
    if (singleForm) {
        singleForm.addEventListener('submit', handleSingleSubmission);
    }
}

// ===== Handle Complete Benchmark Submission =====
async function handleBenchmarkSubmission(e) {
    e.preventDefault();
    
    const submitBtn = this.querySelector('.btn-large');
    const originalText = submitBtn.innerHTML;
    const submitResult = document.getElementById('submit-result');
    
    // Show loading state
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing Complete Benchmark...';
    submitResult.style.display = 'none';
    
    try {
        // Create FormData for ZIP upload
        const formData = new FormData(this);
        
        // Check file size (optional)
        const zipFile = formData.get('results_zip');
        if (zipFile && zipFile.size > 50 * 1024 * 1024) { // 50MB limit
            throw new Error('ZIP file too large. Maximum size is 50MB.');
        }
        
        // Submit to benchmark endpoint
        const response = await fetch('/submissions/upload-benchmark-results', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.detail || `HTTP ${response.status}`);
        }
        
        // Show success with benchmark results
        submitResult.className = 'submit-result success';
        submitResult.innerHTML = `
            <h3><i class="fas fa-trophy"></i> Complete Benchmark Submitted Successfully!</h3>
            <p><strong>Submission ID:</strong> <code>${result.submission_id}</code></p>
            <p>Your model "${result.model_name}" has been evaluated across all datasets!</p>
            
            <div style="background: var(--bg-light); padding: 1.5rem; border-radius: 8px; margin: 1rem 0;">
                <h4 style="margin-bottom: 1rem; color: var(--primary-blue);">🏆 Benchmark Results</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
                    <div>
                        <strong>Average QWK:</strong><br>
                        <span style="font-size: 1.2em; color: var(--success);">${result.benchmark_results.avg_quadratic_weighted_kappa.toFixed(3)}</span>
                    </div>
                    <div>
                        <strong>Average Pearson:</strong><br>
                        <span style="font-size: 1.2em; color: var(--success);">${result.benchmark_results.avg_pearson_correlation.toFixed(3)}</span>
                    </div>
                    <div>
                        <strong>Completion Rate:</strong><br>
                        <span style="font-size: 1.2em; color: var(--success);">${result.benchmark_results.completion_rate.toFixed(1)}%</span>
                    </div>
                    <div>
                        <strong>Total Essays:</strong><br>
                        <span style="font-size: 1.2em; color: var(--primary-blue);">${result.benchmark_results.total_essays_evaluated}</span>
                    </div>
                </div>
                
                <div style="margin-top: 1rem;">
                    <strong>Datasets Completed:</strong> ${result.benchmark_results.datasets_completed}/${result.benchmark_results.total_datasets}
                    ${result.failed_datasets && result.failed_datasets.length > 0 ? 
                        `<br><small style="color: var(--warning);">Failed: ${result.failed_datasets.join(', ')}</small>` : 
                        '<br><span style="color: var(--success);">✅ All datasets processed successfully!</span>'
                    }
                </div>
            </div>
            
            <div style="margin-top: 1.5rem;">
                <a href="#leaderboard" class="btn btn-primary">
                    <i class="fas fa-trophy"></i> View Leaderboard Position
                </a>
                <button onclick="showDetailedResults('${result.submission_id}')" class="btn btn-outline" style="margin-left: 1rem;">
                    <i class="fas fa-chart-bar"></i> View Detailed Results
                </button>
            </div>
        `;
        submitResult.style.display = 'block';
        
        // Reset form
        this.reset();
        
        // Scroll to result
        submitResult.scrollIntoView({ behavior: 'smooth', block: 'center' });
        
        // Refresh leaderboard
        setTimeout(() => {
            loadPlatformStats();
            loadLeaderboard();
        }, 1000);
        
        console.log(`✅ Complete benchmark submitted successfully: ${result.submission_id}`);
        
    } catch (error) {
        // Show error
        submitResult.className = 'submit-result error';
        submitResult.innerHTML = `
            <h3><i class="fas fa-exclamation-circle"></i> Benchmark Submission Failed</h3>
            <p><strong>Error:</strong> ${error.message}</p>
            <p>Please check your ZIP file structure and try again.</p>
            <div style="margin-top: 1rem;">
                <a href="/submissions/benchmark-template" target="_blank" class="btn btn-outline">
                    <i class="fas fa-file-archive"></i> View ZIP Template
                </a>
                <a href="#datasets" class="btn btn-outline" style="margin-left: 1rem;">
                    <i class="fas fa-download"></i> Download All Datasets
                </a>
            </div>
        `;
        submitResult.style.display = 'block';
        submitResult.scrollIntoView({ behavior: 'smooth', block: 'center' });
        
        console.error('Benchmark submission failed:', error);
    } finally {
        // Reset button
        submitBtn.disabled = false;
        submitBtn.innerHTML = originalText;
    }
}

// ===== Handle Single Dataset Submission (for testing) =====
async function handleSingleSubmission(e) {
    e.preventDefault();
    
    const submitBtn = this.querySelector('button[type="submit"]');
    const originalText = submitBtn.innerHTML;
    const submitResult = document.getElementById('submit-result');
    
    // Show loading state
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Testing...';
    submitResult.style.display = 'none';
    
    try {
        // Create FormData for single file upload
        const formData = new FormData(this);
        
        // Submit to single result endpoint
        const response = await fetch('/submissions/upload-single-result', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.detail || `HTTP ${response.status}`);
        }
        
        // Show success (but note it's not on leaderboard)
        submitResult.className = 'submit-result success';
        submitResult.innerHTML = `
            <h3><i class="fas fa-check-circle"></i> Test Submission Successful!</h3>
            <p><strong>⚠️ Note:</strong> This is a test submission and will NOT appear on the leaderboard.</p>
            <p>Dataset: <strong>${result.dataset_name}</strong> | Model: <strong>${result.model_name}</strong></p>
            <div style="background: var(--bg-light); padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                <p><strong>QWK:</strong> ${result.evaluation_results.quadratic_weighted_kappa.toFixed(3)}</p>
                <p><strong>Pearson:</strong> ${result.evaluation_results.pearson_correlation.toFixed(3)}</p>
                <p><strong>Essays Evaluated:</strong> ${result.evaluation_results.essays_evaluated}</p>
                <p><strong>Match Rate:</strong> ${(result.evaluation_results.match_rate * 100).toFixed(1)}%</p>
            </div>
            <div style="margin-top: 1.5rem;">
                <p style="color: var(--warning); font-weight: 600;">
                    <i class="fas fa-info-circle"></i> 
                    To appear on the leaderboard, submit a complete ZIP with all 12 datasets above.
                </p>
            </div>
        `;
        submitResult.style.display = 'block';
        
        // Reset form
        this.reset();
        
        // Scroll to result
        submitResult.scrollIntoView({ behavior: 'smooth', block: 'center' });
        
        console.log(`✅ Test submission completed: ${result.submission_id}`);
        
    } catch (error) {
        // Show error
        submitResult.className = 'submit-result error';
        submitResult.innerHTML = `
            <h3><i class="fas fa-exclamation-circle"></i> Test Submission Failed</h3>
            <p><strong>Error:</strong> ${error.message}</p>
            <div style="margin-top: 1rem;">
                <a href="/submissions/template" target="_blank" class="btn btn-outline">
                    <i class="fas fa-file-csv"></i> View CSV Template
                </a>
            </div>
        `;
        submitResult.style.display = 'block';
        
        console.error('Test submission failed:', error);
    } finally {
        // Reset button
        submitBtn.disabled = false;
        submitBtn.innerHTML = originalText;
    }
}

// ===== Show Detailed Results Modal =====
async function showDetailedResults(submissionId) {
    try {
        const response = await fetch(`/submissions/submission-status/${submissionId}`);
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || 'Failed to load detailed results');
        }
        
        // Create a modal or detailed view
        alert(`Detailed results for submission ${submissionId}:\n\n${JSON.stringify(data, null, 2)}`);
        
    } catch (error) {
        alert(`Failed to load detailed results: ${error.message}`);
    }
}

// ===== Update the datasets loading to populate single form dropdown =====
async function loadDatasets() {
    if (!elements.datasetsGrid) return;

    showLoading(elements.datasetsGrid, 'Loading datasets...');

    try {
        const data = await fetchAPI('/api/datasets/');

        if (!data.datasets || data.datasets.length === 0) {
            showEmptyState(
                elements.datasetsGrid,
                'No Datasets Available',
                'No datasets have been configured yet. Please check back later.'
            );
            return;
        }

        // Create dataset cards
        const datasetsHTML = data.datasets.map((dataset, index) => `
            <div class="dataset-card fade-in" style="animation-delay: ${index * 0.1}s;">
                <h3>${dataset.name}</h3>
                <p>${dataset.description}</p>

                <div class="dataset-info">
                    <span>
                        <i class="fas fa-database"></i>
                        Score Range: ${dataset.score_range[0]}-${dataset.score_range[1]}
                    </span>
                    <span>
                        <i class="fas fa-columns"></i>
                        ${dataset.split}
                    </span>
                </div>

                <div class="dataset-metrics">
                    <div class="metric-tag">Essays: ${dataset.essay_column}</div>
                    <div class="metric-tag">Scores: ${dataset.score_column}</div>
                    <div class="metric-tag">Prompts: ${dataset.prompt_column}</div>
                </div>

                <div class="dataset-actions">
                    <button onclick="downloadSingleDataset('${dataset.name}')" class="btn btn-sm btn-primary">
                        <i class="fas fa-download"></i> Download CSV
                    </button>
                    <button onclick="showDatasetSample('${dataset.name}')" class="btn btn-sm btn-outline">
                        <i class="fas fa-eye"></i> Preview
                    </button>
                </div>
            </div>
        `).join('');

        elements.datasetsGrid.innerHTML = datasetsHTML;

        // Update total count
        if (elements.totalDatasets) {
            animateNumber(elements.totalDatasets, data.total_count || data.datasets.length);
        }

        // Populate single dataset dropdown for testing form
        const singleDatasetSelect = document.getElementById('single-dataset-name');
        if (singleDatasetSelect) {
            const options = data.datasets.map(dataset => 
                `<option value="${dataset.name}">${dataset.name}</option>`
            ).join('');
            singleDatasetSelect.innerHTML = '<option value="">-- Choose dataset --</option>' + options;
        }

        console.log(`✅ Loaded ${data.datasets.length} datasets successfully`);

    } catch (error) {
        showError(elements.datasetsGrid, 'Failed to load datasets. Please check the API connection.');
        console.error('Failed to load datasets:', error);
    }
}

// ===== Update Leaderboard to Show Only Complete Benchmarks =====
async function loadLeaderboard() {
    if (!elements.leaderboardTable) return;
    
    showLoading(elements.leaderboardTable, 'Loading leaderboard...');
    
    try {
        const metric = elements.metricSelector?.value || 'quadratic_weighted_kappa';
        const data = await fetchAPI(`/api/leaderboard/?metric=${metric}&limit=20`);
        
        if (!data.leaderboard || data.leaderboard.length === 0) {
            showEmptyState(
                elements.leaderboardTable,
                'No Complete Benchmarks Yet',
                'Be the first to submit a complete ZIP with all 12 datasets to appear on the leaderboard!',
                '<a href="#submit" class="btn btn-primary"><i class="fas fa-upload"></i> Submit Complete Benchmark</a>'
            );
            return;
        }
        
        // Filter to show only complete benchmarks (10+ datasets)
        const completeBenchmarks = data.leaderboard.filter(entry => 
            (entry.datasets_completed >= 10) || (entry.dataset_name === 'BENCHMARK_COMPLETE')
        );
        
        if (completeBenchmarks.length === 0) {
            showEmptyState(
                elements.leaderboardTable,
                'No Complete Benchmarks Yet',
                'Only submissions with results for 10+ datasets appear on the leaderboard.',
                '<a href="#submit" class="btn btn-primary"><i class="fas fa-upload"></i> Submit Complete Benchmark</a>'
            );
            return;
        }
        
        const tableHTML = `
            <div style="margin-bottom: 1rem; padding: 1rem; background: var(--bg-light); border-radius: 8px;">
                <p><strong>🏆 Leaderboard Rules:</strong> Only complete benchmarks with 10+ datasets are ranked here. 
                Single dataset submissions are for testing only.</p>
            </div>
            <table>
                <thead>
                    <tr>
                        <th style="width: 60px;">Rank</th>
                        <th>Model</th>
                        <th>Submitter</th>
                        <th>Avg QWK</th>
                        <th>Avg Pearson</th>
                        <th>Datasets</th>
                        <th>Essays</th>
                        <th style="width: 140px;">Submitted</th>
                    </tr>
                </thead>
                <tbody>
                    ${completeBenchmarks.map((entry, index) => `
                        <tr style="animation: fadeIn 0.6s ease-out ${index * 0.1}s both;">
                            <td>
                                <div class="rank-badge ${getRankClass(index + 1)}">
                                    ${index + 1}
                                </div>
                            </td>
                            <td>
                                <div>
                                    <strong style="color: var(--text-primary);">${entry.model_name || 'Unknown Model'}</strong>
                                    <br>
                                    <small style="color: var(--text-muted); font-family: monospace;">${entry.submission_id}</small>
                                </div>
                            </td>
                            <td style="color: var(--text-secondary);">${entry.submitter_name}</td>
                            <td>
                                <span class="score-badge ${getScoreClass(entry.avg_quadratic_weighted_kappa || entry.quadratic_weighted_kappa)}" 
                                      title="${getScoreDescription(entry.avg_quadratic_weighted_kappa || entry.quadratic_weighted_kappa)}">
                                    ${formatScore(entry.avg_quadratic_weighted_kappa || entry.quadratic_weighted_kappa)}
                                </span>
                            </td>
                            <td>
                                <span class="score-badge ${getScoreClass(entry.avg_pearson_correlation || entry.pearson_correlation)}"
                                      title="${getScoreDescription(entry.avg_pearson_correlation || entry.pearson_correlation)}">
                                    ${formatScore(entry.avg_pearson_correlation || entry.pearson_correlation)}
                                </span>
                            </td>
                            <td style="text-align: center;">
                                <span style="color: var(--success); font-weight: 600;">
                                    ${entry.datasets_completed || 12}/12
                                </span>
                            </td>
                            <td style="text-align: center;">
                                ${entry.total_essays_evaluated || entry.essays_evaluated || 0}
                            </td>
                            <td>
                                <small style="color: var(--text-muted);">
                                    ${formatDate(entry.submission_time)}
                                </small>
                            </td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;
        
        elements.leaderboardTable.innerHTML = tableHTML;
        
        console.log(`✅ Loaded leaderboard with ${completeBenchmarks.length} complete benchmarks`);
        
    } catch (error) {
        showError(elements.leaderboardTable, 'Failed to load leaderboard. Please check the API connection.');
        console.error('Failed to load leaderboard:', error);
    }
}

// Make functions globally available
window.showDetailedResults = showDetailedResults;

// ===== Global Functions =====
window.downloadSingleDataset = downloadSingleDataset;
window.downloadAllDatasets = downloadAllDatasets;
window.showDatasetSample = showDatasetSample;
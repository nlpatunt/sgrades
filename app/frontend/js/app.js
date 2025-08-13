// ===== API Configuration =====
const API_BASE_URL = '';  // Empty string = same domain
console.log("BESESR 15-Dataset Batch System loaded");

// ===== DOM Elements =====
const elements = {
    // Stats
    totalDatasets: document.getElementById('total-datasets'),
    totalModels: document.getElementById('total-models'),
    totalEvaluations: document.getElementById('total-evaluations'),
    
    // Datasets
    datasetsGrid: document.getElementById('datasets-grid'),
    downloadAllBtn: document.getElementById('download-all-btn'),
    
    // Batch Submission
    benchmarkForm: document.getElementById('benchmark-submit-form'),
    fileDropArea: document.getElementById('file-drop-area'),
    csvFilesInput: document.getElementById('csv-files'),
    fileList: document.getElementById('file-list'),
    fileItems: document.getElementById('file-items'),
    fileCount: document.getElementById('file-count'),
    submitBenchmarkBtn: document.getElementById('submit-benchmark-btn'),
    validateFilesBtn: document.getElementById('validate-files-btn'),
    submitResult: document.getElementById('submit-result'),
    
    // Individual Testing
    singleTestForm: document.getElementById('single-test-form'),
    testDataset: document.getElementById('test-dataset'),
    
    // Leaderboard
    metricSelector: document.getElementById('metric-selector'),
    refreshLeaderboard: document.getElementById('refresh-leaderboard'),
    leaderboardTable: document.getElementById('leaderboard-table')
};

// ===== File Management =====
let uploadedFiles = new Map(); // dataset_name -> file
const requiredDatasets = [
    'ASAP-AES', 'ASAP-SAS', 'ASAP2', 'ASAP_plus_plus', 'rice_chem',
    'CSEE', 'EFL', 'grade_like_a_human_dataset_os', 'persuade_2',
    'SciEntSBank', 'BEEtlE', 'automatic_short_answer_grading_mohlar',
    'dataset_13', 'dataset_14', 'dataset_15'  // Update these with real names
];

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
    downloadFile('/api/datasets/download/all', 'besesr_15_datasets.zip');
}

// ===== Load Platform Stats =====
async function loadPlatformStats() {
    try {
        const stats = await fetchAPI('/api/leaderboard/stats'); 
        
        // Update stats with animation
        if (elements.totalModels) {
            animateNumber(elements.totalModels, stats.total_complete_benchmarks || 0);
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
        console.log("📊 Loading datasets from API...");
        const data = await fetchAPI('/submissions/template');
        console.log("📊 Received data:", data);

        if (!data.available_datasets || data.available_datasets.length === 0) {
            showEmptyState(
                elements.datasetsGrid,
                'No Datasets Available',
                'No datasets have been configured yet. Please check back later.'
            );
            return;
        }

        // Create dataset cards
        const datasetsHTML = data.available_datasets.map((datasetName, index) => `
            <div class="dataset-card fade-in" style="animation-delay: ${index * 0.1}s;">
                <h3>${datasetName}</h3>
                <p>Dataset for automatic essay scoring evaluation</p>

                <div class="dataset-info">
                    <span>
                        <i class="fas fa-database"></i>
                        Train/Val/Test splits available
                    </span>
                </div>

                <div class="dataset-actions">
                    <button onclick="downloadSingleDataset('${datasetName}')" class="btn btn-sm btn-primary">
                        <i class="fas fa-download"></i> Download
                    </button>
                </div>
            </div>
        `).join('');

        elements.datasetsGrid.innerHTML = datasetsHTML;

        // Populate the individual testing dropdown
        if (elements.testDataset) {
            console.log("📋 Populating test dataset dropdown...");
            const options = data.available_datasets.map(datasetName => 
                `<option value="${datasetName}">${datasetName}</option>`
            ).join('');
            elements.testDataset.innerHTML = '<option value="">-- Choose dataset --</option>' + options;
        }

        console.log(`✅ Loaded ${data.available_datasets.length} datasets successfully`);

    } catch (error) {
        console.error('❌ Failed to load datasets:', error);
        showError(elements.datasetsGrid, 'Failed to load datasets. Please check the API connection.');
    }
}

// ===== File Upload Handling =====
function setupFileUpload() {
    if (!elements.fileDropArea || !elements.csvFilesInput) return;

    // Drag and drop handlers
    elements.fileDropArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        elements.fileDropArea.classList.add('dragover');
    });

    elements.fileDropArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        elements.fileDropArea.classList.remove('dragover');
    });

    elements.fileDropArea.addEventListener('drop', (e) => {
        e.preventDefault();
        elements.fileDropArea.classList.remove('dragover');
        const files = Array.from(e.dataTransfer.files);
        handleFileSelection(files);
    });

    // Click to upload
    elements.fileDropArea.addEventListener('click', () => {
        elements.csvFilesInput.click();
    });

    // File input change
    elements.csvFilesInput.addEventListener('change', (e) => {
        const files = Array.from(e.target.files);
        handleFileSelection(files);
    });
}

function handleFileSelection(files) {
    console.log(`📁 Selected ${files.length} files`);
    
    // Filter only CSV files
    const csvFiles = files.filter(file => file.name.endsWith('.csv'));
    
    if (csvFiles.length !== files.length) {
        alert(`Only CSV files are allowed. ${files.length - csvFiles.length} non-CSV files were ignored.`);
    }

    // Process each CSV file
    csvFiles.forEach(file => {
        const datasetName = extractDatasetName(file.name);
        if (datasetName) {
            uploadedFiles.set(datasetName, file);
            console.log(`✅ Added ${datasetName}: ${file.name}`);
        } else {
            console.warn(`⚠️ Could not determine dataset for file: ${file.name}`);
        }
    });

    updateFileDisplay();
    updateSubmitButton();
}

function extractDatasetName(filename) {
    // Remove .csv extension
    const nameWithoutExt = filename.replace(/\.csv$/i, '');
    
    // Try to match with known dataset names
    for (const dataset of requiredDatasets) {
        if (nameWithoutExt.includes(dataset) || dataset.includes(nameWithoutExt)) {
            return dataset;
        }
    }
    
    // If no match, use the filename as dataset name
    return nameWithoutExt;
}

function updateFileDisplay() {
    if (!elements.fileList || !elements.fileItems || !elements.fileCount) return;

    const fileCount = uploadedFiles.size;
    elements.fileCount.textContent = fileCount;

    if (fileCount === 0) {
        elements.fileList.style.display = 'none';
        return;
    }

    elements.fileList.style.display = 'block';

    // Create file list HTML
    const fileHTML = Array.from(uploadedFiles.entries()).map(([dataset, file]) => `
        <div class="file-item">
            <div class="file-info">
                <i class="fas fa-file-csv file-icon"></i>
                <span class="file-name">${dataset}</span>
                <span class="file-size">(${(file.size / 1024).toFixed(1)} KB)</span>
            </div>
            <button type="button" class="file-remove" onclick="removeFile('${dataset}')">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `).join('');

    elements.fileItems.innerHTML = fileHTML;

    // Update file drop area appearance
    if (fileCount === 15) {
        elements.fileDropArea.classList.add('has-files');
        elements.fileDropArea.querySelector('.upload-text').innerHTML = 
            '<strong>✅ All 15 datasets ready for submission!</strong>';
    } else {
        elements.fileDropArea.classList.remove('has-files');
        elements.fileDropArea.querySelector('.upload-text').innerHTML = 
            `<strong>📁 ${fileCount}/15 datasets uploaded - Add ${15 - fileCount} more</strong>`;
    }
}

function removeFile(datasetName) {
    uploadedFiles.delete(datasetName);
    updateFileDisplay();
    updateSubmitButton();
}

function updateSubmitButton() {
    if (!elements.submitBenchmarkBtn || !elements.validateFilesBtn) return;

    const fileCount = uploadedFiles.size;
    const isComplete = fileCount === 15;

    elements.submitBenchmarkBtn.disabled = !isComplete;
    elements.validateFilesBtn.disabled = fileCount === 0;

    if (isComplete) {
        elements.submitBenchmarkBtn.innerHTML = '<i class="fas fa-rocket"></i> Submit Complete Benchmark (15/15)';
    } else {
        elements.submitBenchmarkBtn.innerHTML = `<i class="fas fa-upload"></i> Need ${15 - fileCount} more datasets`;
    }
}

// ===== Form Submission =====
function setupBenchmarkSubmission() {
    if (!elements.benchmarkForm) return;

    elements.benchmarkForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        if (uploadedFiles.size !== 15) {
            alert('Please upload all 15 datasets before submitting.');
            return;
        }

        const submitBtn = elements.submitBenchmarkBtn;
        const originalText = submitBtn.innerHTML;
        
        // Show loading state
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing Complete Benchmark...';
        elements.submitResult.style.display = 'none';

        try {
            // Create FormData with all files and form data
            const formData = new FormData(this);
            
            // Add all uploaded files
            uploadedFiles.forEach((file, datasetName) => {
                formData.append('csv_files', file, `${datasetName}.csv`);
            });

            console.log(`🚀 Submitting complete benchmark with ${uploadedFiles.size} datasets`);

            // Submit to batch processing endpoint
            const response = await fetch('/submissions/upload-complete-benchmark', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.detail || `HTTP ${response.status}`);
            }

            // Show success
            showBenchmarkSuccess(result);
            
            // Reset form
            this.reset();
            uploadedFiles.clear();
            updateFileDisplay();
            updateSubmitButton();
            
            // Refresh leaderboard
            setTimeout(() => {
                loadPlatformStats();
                loadLeaderboard();
            }, 1000);

        } catch (error) {
            showBenchmarkError(error.message);
        } finally {
            // Reset button
            submitBtn.disabled = false;
            submitBtn.innerHTML = originalText;
        }
    });
}

function showBenchmarkSuccess(result) {
    elements.submitResult.className = 'submit-result success';
    elements.submitResult.innerHTML = `
        <h3><i class="fas fa-trophy"></i> Complete Benchmark Submitted Successfully!</h3>
        <p><strong>Researcher:</strong> ${result.submitter_name}</p>
        <p><strong>Model:</strong> ${result.model_name}</p>
        
        <div style="background: var(--bg-light); padding: 1.5rem; border-radius: 8px; margin: 1rem 0;">
            <h4 style="margin-bottom: 1rem; color: var(--primary-blue);">🏆 Aggregate Results</h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
                <div>
                    <strong>Average QWK:</strong><br>
                    <span style="font-size: 1.2em; color: var(--success);">${result.avg_quadratic_weighted_kappa?.toFixed(3) || 'Calculating...'}</span>
                </div>
                <div>
                    <strong>Average Pearson:</strong><br>
                    <span style="font-size: 1.2em; color: var(--success);">${result.avg_pearson_correlation?.toFixed(3) || 'Calculating...'}</span>
                </div>
                <div>
                    <strong>Datasets Processed:</strong><br>
                    <span style="font-size: 1.2em; color: var(--primary-blue);">${result.datasets_processed || 15}/15</span>
                </div>
                <div>
                    <strong>Total Essays:</strong><br>
                    <span style="font-size: 1.2em; color: var(--primary-blue);">${result.total_essays_evaluated || 'Calculating...'}</span>
                </div>
            </div>
        </div>
        
        <div style="margin-top: 1.5rem;">
            <a href="#leaderboard" class="btn btn-primary">
                <i class="fas fa-trophy"></i> View Leaderboard Position
            </a>
        </div>
    `;
    elements.submitResult.style.display = 'block';
    elements.submitResult.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function showBenchmarkError(message) {
    elements.submitResult.className = 'submit-result error';
    elements.submitResult.innerHTML = `
        <h3><i class="fas fa-exclamation-circle"></i> Benchmark Submission Failed</h3>
        <p><strong>Error:</strong> ${message}</p>
        <p>Please check your files and try again.</p>
        <div style="margin-top: 1rem;">
            <a href="#datasets" class="btn btn-outline">
                <i class="fas fa-download"></i> Download Datasets Again
            </a>
        </div>
    `;
    elements.submitResult.style.display = 'block';
    elements.submitResult.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// ===== Individual Testing =====
function setupIndividualTesting() {
    if (!elements.singleTestForm) return;

    elements.singleTestForm.addEventListener('submit', async function(e) {
        e.preventDefault();

        const submitBtn = this.querySelector('button[type="submit"]');
        const originalText = submitBtn.innerHTML;

        submitBtn.disabled = true;
        submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Testing...';

        try {
            const formData = new FormData(this);
            
            // Add required fields for single upload
            formData.append('model_name', 'Test Model');
            formData.append('submitter_name', 'Test User');
            formData.append('submitter_email', 'test@example.com');

            const response = await fetch('/submissions/upload-single-result', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.detail || `HTTP ${response.status}`);
            }

            // Show test result
            alert(`Test Result for ${result.dataset_name}:\nQWK: ${result.evaluation_results.quadratic_weighted_kappa.toFixed(3)}\nNote: This is for testing only and does not appear on the leaderboard.`);
            
            this.reset();

        } catch (error) {
            alert(`Test failed: ${error.message}`);
        } finally {
            submitBtn.disabled = false;
            submitBtn.innerHTML = originalText;
        }
    });
}

// ===== Load Leaderboard =====
async function loadLeaderboard() {
    if (!elements.leaderboardTable) return;
    
    showLoading(elements.leaderboardTable, 'Loading complete benchmark leaderboard...');
    
    try {
        const metric = elements.metricSelector?.value || 'avg_quadratic_weighted_kappa';
        const data = await fetchAPI(`/submissions/leaderboard?metric=${metric}&limit=20`);
        
        if (!data || data.length === 0) {
            showEmptyState(
                elements.leaderboardTable,
                'No Complete Benchmarks Yet',
                'Be the first to submit all 15 datasets to appear on the leaderboard!',
                '<a href="#submit" class="btn btn-primary"><i class="fas fa-rocket"></i> Submit Complete Benchmark</a>'
            );
            return;
        }
        
        const tableHTML = `
            <div style="margin-bottom: 1rem; padding: 1rem; background: var(--bg-light); border-radius: 8px;">
                <p><strong>🏆 Complete Benchmarks Only:</strong> Only researchers who submitted all 15 datasets appear here.</p>
            </div>
            <table>
                <thead>
                    <tr>
                        <th style="width: 60px;">Rank</th>
                        <th>Researcher</th>
                        <th>Model</th>
                        <th>Avg QWK</th>
                        <th>Avg Pearson</th>
                        <th>Datasets</th>
                        <th>Essays</th>
                        <th style="width: 140px;">Submitted</th>
                    </tr>
                </thead>
                <tbody>
                    ${data.map((entry, index) => `
                        <tr style="animation: fadeIn 0.6s ease-out ${index * 0.1}s both;">
                            <td>
                                <div class="rank-badge ${getRankClass(entry.rank || index + 1)}">
                                    ${entry.rank || index + 1}
                                </div>
                            </td>
                            <td style="color: var(--text-primary);">
                                <strong>${entry.submitter_name || 'Unknown'}</strong>
                            </td>
                            <td style="color: var(--text-secondary);">
                                ${entry.model_name || 'Model Description'}
                            </td>
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
                                <span style="color: var(--success); font-weight: 600;">
                                    15/15
                                </span>
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
        
        console.log(`✅ Loaded leaderboard with ${data.length} complete benchmarks`);
        
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
        day: 'numeric'
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

    // File validation button
    if (elements.validateFilesBtn) {
        elements.validateFilesBtn.addEventListener('click', async () => {
            await validateUploadedFiles();
        });
    }
}

async function validateUploadedFiles() {
    if (uploadedFiles.size === 0) {
        alert('No files to validate. Please upload CSV files first.');
        return;
    }

    console.log(`🔍 Validating ${uploadedFiles.size} uploaded files...`);
    
    let validFiles = 0;
    let errors = [];

    for (const [dataset, file] of uploadedFiles) {
        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch('/submissions/validate-csv', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.valid) {
                validFiles++;
                console.log(`✅ ${dataset}: Valid (${result.row_count} rows)`);
            } else {
                errors.push(`${dataset}: ${result.error}`);
                console.log(`❌ ${dataset}: ${result.error}`);
            }
        } catch (error) {
            errors.push(`${dataset}: Validation failed - ${error.message}`);
        }
    }

    // Show validation results
    if (errors.length === 0) {
        alert(`✅ All ${validFiles} files are valid and ready for submission!`);
    } else {
        alert(`Validation Results:\n✅ Valid: ${validFiles}\n❌ Errors: ${errors.length}\n\nErrors:\n${errors.join('\n')}`);
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
    console.log('🚀 BESESR 15-Dataset Batch System Initialized');
    
    // Setup all functionality
    setupSmoothScrolling();
    setupFileUpload();
    setupBenchmarkSubmission();
    setupIndividualTesting();
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
    
    console.log('✅ BESESR 15-Dataset Batch Platform ready!');
});

// ===== Global Functions =====
window.removeFile = removeFile;
window.downloadSingleDataset = (datasetName) => {
    downloadFile(`/api/datasets/download/${encodeURIComponent(datasetName)}`, `${datasetName}.csv`);
};
// ===== Enhanced BESESR Platform with Dynamic Dataset Loading =====
const API_BASE_URL = '';  // Empty string = same domain
console.log("Enhanced BESESR Platform loaded");

// ===== Global State =====
let uploadedDatasets = new Map(); // dataset_name -> { file, status, result }
let availableDatasets = []; // Will be populated from API
let totalDatasetsCount = 0;
let datasetFormats = {}; // Will store format info from API

// ===== DOM Elements =====
const elements = {
    // Stats
    totalDatasets: document.getElementById('total-datasets'),
    totalDatasetsHero: document.getElementById('total-datasets-hero'),
    totalModels: document.getElementById('total-models'),
    totalEvaluations: document.getElementById('total-evaluations'),
    
    // Datasets
    datasetsGrid: document.getElementById('datasets-grid'),
    downloadAllBtn: document.getElementById('download-all-btn'),
    
    // Enhanced Upload Elements
    datasetSelector: document.getElementById('dataset-selector'),
    datasetFile: document.getElementById('dataset-file'),
    uploadSingleBtn: document.getElementById('upload-single-btn'),
    formatHelper: document.getElementById('format-helper'),
    formatDatasetName: document.getElementById('format-dataset-name'),
    formatDetails: document.getElementById('format-details'),
    uploadedCount: document.getElementById('uploaded-count'),
    overallProgressFill: document.getElementById('overall-progress-fill'),
    overallProgressText: document.getElementById('overall-progress-text'),
    submitCount: document.getElementById('submit-count'),
    submitTotal: document.getElementById('submit-total'),
    datasetsProgressGrid: document.getElementById('datasets-progress-grid'),
    validateAllBtn: document.getElementById('validate-all-btn'),
    
    // Forms
    benchmarkForm: document.getElementById('benchmark-submit-form'),
    submitBenchmarkBtn: document.getElementById('submit-benchmark-btn'),
    submitResult: document.getElementById('submit-result'),
    
    // Individual Testing
    singleTestForm: document.getElementById('single-test-form'),
    testDataset: document.getElementById('test-dataset'),
    
    // Leaderboard
    metricSelector: document.getElementById('metric-selector'),
    refreshLeaderboard: document.getElementById('refresh-leaderboard'),
    leaderboardTable: document.getElementById('leaderboard-table')
};

// ===== Utility Functions =====
function showLoading(element, message = 'Loading...') {
    if (!element) return;
    element.innerHTML = '<div class="loading"><i class="fas fa-spinner fa-spin"></i> ' + message + '</div>';
}

function showError(element, message) {
    if (!element) return;
    var errorHtml = '<div class="error" style="text-align: center; padding: 3rem; color: var(--error);">';
    errorHtml += '<i class="fas fa-exclamation-triangle" style="font-size: 2rem; margin-bottom: 1rem; display: block;"></i>';
    errorHtml += '<h3>Error Loading Data</h3>';
    errorHtml += '<p>' + message + '</p>';
    errorHtml += '<button onclick="location.reload()" class="btn btn-outline" style="margin-top: 1rem;">';
    errorHtml += '<i class="fas fa-redo"></i> Retry';
    errorHtml += '</button>';
    errorHtml += '</div>';
    element.innerHTML = errorHtml;
}

function showEmptyState(element, title, message, actionHtml = '') {
    if (!element) return;
    var emptyHtml = '<div class="empty-state" style="text-align: center; padding: 4rem; color: var(--text-secondary);">';
    emptyHtml += '<i class="fas fa-inbox" style="font-size: 3rem; margin-bottom: 1rem; display: block; opacity: 0.5;"></i>';
    emptyHtml += '<h3 style="margin-bottom: 1rem; color: var(--text-primary);">' + title + '</h3>';
    emptyHtml += '<p style="margin-bottom: 2rem; max-width: 400px; margin-left: auto; margin-right: auto;">' + message + '</p>';
    emptyHtml += actionHtml;
    emptyHtml += '</div>';
    element.innerHTML = emptyHtml;
}

// ===== API Functions =====
async function fetchAPI(endpoint) {
    try {
        const response = await fetch(API_BASE_URL + endpoint);
        if (!response.ok) {
            throw new Error('HTTP ' + response.status + ': ' + response.statusText);
        }
        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

// ===== Dataset Loading Functions =====
async function loadAvailableDatasets() {
    try {
        console.log('🔄 Loading available datasets from API...');
        
        let datasets = [];
        let formats = {};
        
        try {
            // Primary method: Get datasets from datasets API
            const response = await fetchAPI('/api/datasets/');
            if (response.datasets && Array.isArray(response.datasets)) {
                datasets = response.datasets.map(d => d.name);
                console.log('✅ Loaded datasets from /api/datasets/');
            }
        } catch (error) {
            console.log('⚠️ Primary datasets API failed, trying template endpoint...');
            
            try {
                // Fallback method: Get from template
                const templateResponse = await fetchAPI('/api/submissions/template');
                if (templateResponse.datasets && typeof templateResponse.datasets === 'object') {
                    datasets = Object.keys(templateResponse.datasets);
                    console.log('✅ Loaded datasets from /api/submissions/template');
                }
            } catch (templateError) {
                console.log('⚠️ All endpoints failed, using minimal fallback...');
                datasets = ['ASAP-AES', 'BEEtlE_2way', 'CSEE']; // Minimal fallback
            }
        }
        
        // Load format information for each dataset
        for (const dataset of datasets) {
            try {
                const formatResponse = await fetchAPI('/api/submissions/format/' + encodeURIComponent(dataset));
                if (formatResponse && !formatResponse.error) {
                    formats[dataset] = formatResponse;
                }
            } catch (error) {
                // Use fallback format
                formats[dataset] = getDatasetFormatFallback(dataset);
            }
        }
        
        availableDatasets = datasets;
        datasetFormats = formats;
        totalDatasetsCount = datasets.length;
        
        console.log('✅ Loaded ' + datasets.length + ' datasets with formats');
        
        // Update UI elements
        updateDatasetCounts();
        populateDatasetDropdowns();
        initializeProgressGrid();
        
        return datasets;
        
    } catch (error) {
        console.error('❌ Failed to load datasets:', error);
        
        // Use minimal fallback
        availableDatasets = ['ASAP-AES', 'BEEtlE_2way', 'CSEE'];
        totalDatasetsCount = availableDatasets.length;
        
        updateDatasetCounts();
        populateDatasetDropdowns();
        initializeProgressGrid();
        
        return availableDatasets;
    }
}

function getDatasetFormatFallback(datasetName) {
    const fallbackFormats = {
        'ASAP-AES': { 
            required_columns: ['essay_id', 'domain1_score'], 
            score_range: [0, 60], 
            score_type: 'float',
            primary_score_column: 'domain1_score'
        },
        'ASAP2': { 
            required_columns: ['essay_id', 'score'], 
            score_range: [0, 60], 
            score_type: 'float',
            primary_score_column: 'score'
        },
        'BEEtlE_2way': { 
            required_columns: ['question_id', 'label'], 
            score_range: [0, 1], 
            score_type: 'int',
            primary_score_column: 'label'
        },
        'BEEtlE_3way': { 
            required_columns: ['question_id', 'label'], 
            score_range: [0, 2], 
            score_type: 'int',
            primary_score_column: 'label'
        },
        'CSEE': { 
            required_columns: ['essay_id', 'overall_score'], 
            score_range: [0, 100], 
            score_type: 'float',
            primary_score_column: 'overall_score'
        }
    };
    
    // Default format for grade_like_a_human datasets
    if (datasetName.includes('grade_like_a_human_dataset_os')) {
        return { 
            required_columns: ['id', 'score_1'], 
            score_range: [0, 100], 
            score_type: 'float',
            primary_score_column: 'score_1'
        };
    }
    
    // Default format for Rice_Chem datasets
    if (datasetName.includes('Rice_Chem')) {
        return { 
            required_columns: ['sis_id', 'Score'], 
            score_range: [0, 100], 
            score_type: 'float',
            primary_score_column: 'Score'
        };
    }
    
    // Return known format or default
    return fallbackFormats[datasetName] || { 
        required_columns: ['essay_id', 'predicted_score'], 
        score_range: [0, 100], 
        score_type: 'float',
        primary_score_column: 'predicted_score'
    };
}

function updateDatasetCounts() {
    if (elements.totalDatasets) {
        elements.totalDatasets.textContent = totalDatasetsCount;
    }
    if (elements.totalDatasetsHero) {
        elements.totalDatasetsHero.textContent = totalDatasetsCount;
    }
    if (elements.submitTotal) {
        elements.submitTotal.textContent = totalDatasetsCount;
    }
}

function populateDatasetDropdowns() {
    console.log('📋 Populating dataset dropdowns...');
    
    // Populate main dataset selector
    if (elements.datasetSelector) {
        populateDropdown(elements.datasetSelector, true);
    }
    
    // Populate test dataset selector
    if (elements.testDataset) {
        populateDropdown(elements.testDataset, false);
    }
}

function populateDropdown(selectElement, showDetails = false) {
    if (!selectElement || !availableDatasets.length) return;
    
    // Categorize datasets
    const categories = {
        main: [],
        grading: [],
        chemistry: [],
        other: []
    };
    
    availableDatasets.forEach(dataset => {
        if (dataset.includes('grade_like_a_human')) {
            categories.grading.push(dataset);
        } else if (dataset.includes('Rice_Chem')) {
            categories.chemistry.push(dataset);
        } else if (['ASAP-AES', 'ASAP2', 'ASAP-SAS', 'ASAP_plus_plus', 'CSEE', 'BEEtlE_2way', 'BEEtlE_3way', 'SciEntSBank_2way', 'SciEntSBank_3way', 'EFL', 'persuade_2'].some(main => dataset.includes(main))) {
            categories.main.push(dataset);
        } else {
            categories.other.push(dataset);
        }
    });
    
    // Build dropdown HTML
    let optionsHTML = '<option value="">-- Choose a dataset --</option>';
    
    // Main datasets
    if (categories.main.length > 0) {
        optionsHTML += '<optgroup label="📚 Main Datasets">';
        categories.main.sort().forEach(dataset => {
            const format = datasetFormats[dataset] || getDatasetFormatFallback(dataset);
            const displayText = showDetails 
                ? dataset + ' (' + format.required_columns.join(', ') + ')'
                : dataset;
            optionsHTML += '<option value="' + dataset + '">' + displayText + '</option>';
        });
        optionsHTML += '</optgroup>';
    }
    
    // Grading datasets
    if (categories.grading.length > 0) {
        optionsHTML += '<optgroup label="🎓 Grade Like Human (OS Questions)">';
        categories.grading.sort().forEach(dataset => {
            const qNumber = dataset.split('_').pop();
            const format = datasetFormats[dataset] || getDatasetFormatFallback(dataset);
            const displayText = showDetails 
                ? 'Q' + qNumber.toUpperCase() + ' - OS Concepts (' + format.required_columns.join(', ') + ')'
                : 'Q' + qNumber.toUpperCase() + ' - OS Concepts';
            optionsHTML += '<option value="' + dataset + '">' + displayText + '</option>';
        });
        optionsHTML += '</optgroup>';
    }
    
    // Chemistry datasets
    if (categories.chemistry.length > 0) {
        optionsHTML += '<optgroup label="🧪 Rice Chemistry Questions">';
        categories.chemistry.sort().forEach(dataset => {
            const qNumber = dataset.split('_').pop();
            const format = datasetFormats[dataset] || getDatasetFormatFallback(dataset);
            const displayText = showDetails 
                ? 'Chemistry ' + qNumber + ' (' + format.required_columns.join(', ') + ')'
                : 'Chemistry ' + qNumber;
            optionsHTML += '<option value="' + dataset + '">' + displayText + '</option>';
        });
        optionsHTML += '</optgroup>';
    }
    
    // Other datasets
    if (categories.other.length > 0) {
        optionsHTML += '<optgroup label="📄 Other Datasets">';
        categories.other.sort().forEach(dataset => {
            const format = datasetFormats[dataset] || getDatasetFormatFallback(dataset);
            const displayText = showDetails 
                ? dataset + ' (' + format.required_columns.join(', ') + ')'
                : dataset;
            optionsHTML += '<option value="' + dataset + '">' + displayText + '</option>';
        });
        optionsHTML += '</optgroup>';
    }
    
    selectElement.innerHTML = optionsHTML;
    console.log('✅ Populated dropdown with ' + availableDatasets.length + ' datasets');
}

function initializeProgressGrid() {
    if (!elements.datasetsProgressGrid || !availableDatasets.length) return;
    
    console.log('📊 Initializing progress grid...');
    
    var gridHTML = '';
    availableDatasets.forEach(dataset => {
        const format = datasetFormats[dataset] || getDatasetFormatFallback(dataset);
        gridHTML += '<div class="dataset-progress-item" id="progress-' + dataset + '">';
        gridHTML += '<div class="dataset-item-header">';
        gridHTML += '<span class="dataset-name">' + dataset + '</span>';
        gridHTML += '<span class="dataset-status status-pending" id="status-' + dataset + '">Pending</span>';
        gridHTML += '</div>';
        gridHTML += '<div class="dataset-details" id="details-' + dataset + '">';
        gridHTML += format.required_columns.join(', ') + ' | Range: ' + format.score_range[0] + '-' + format.score_range[1];
        gridHTML += '</div>';
        gridHTML += '</div>';
    });
    
    elements.datasetsProgressGrid.innerHTML = gridHTML;
    updateOverallProgress();
}

// ===== Enhanced Upload Functions =====
function setupEnhancedUpload() {
    if (!elements.datasetSelector) return;
    
    console.log('🔧 Setting up enhanced upload functionality...');
    
    // Dataset selection change handler
    elements.datasetSelector.addEventListener('change', handleDatasetSelection);
    
    // File selection handler
    elements.datasetFile.addEventListener('change', handleFileSelection);
    
    // Upload button handler
    elements.uploadSingleBtn.addEventListener('click', handleSingleUpload);
    
    // Validate all button handler
    if (elements.validateAllBtn) {
        elements.validateAllBtn.addEventListener('click', validateAllFiles);
    }
}

function handleDatasetSelection() {
    const selectedDataset = elements.datasetSelector.value;
    const hasFile = elements.datasetFile.files.length > 0;
    
    // Update upload button state
    elements.uploadSingleBtn.disabled = !selectedDataset || !hasFile;
    
    // Show format helper
    if (selectedDataset) {
        showFormatHelper(selectedDataset);
    } else {
        hideFormatHelper();
    }
}

function handleFileSelection() {
    const selectedDataset = elements.datasetSelector.value;
    const hasFile = elements.datasetFile.files.length > 0;
    
    // Update upload button state
    elements.uploadSingleBtn.disabled = !selectedDataset || !hasFile;
}

function showFormatHelper(datasetName) {
    if (!elements.formatHelper) return;
    
    const format = datasetFormats[datasetName] || getDatasetFormatFallback(datasetName);
    
    elements.formatDatasetName.textContent = datasetName;
    
    const exampleData = generateExampleCSV(datasetName, format);
    elements.formatDetails.textContent = exampleData;
    
    elements.formatHelper.style.display = 'block';
}

function hideFormatHelper() {
    if (elements.formatHelper) {
        elements.formatHelper.style.display = 'none';
    }
}

function generateExampleCSV(datasetName, format) {
    const columns = format.required_columns;
    
    let example = columns.join(',') + '\n';
    example += datasetName + '_test_0,' + (format.score_type === 'int' ? '1' : '3.5') + '\n';
    example += datasetName + '_test_1,' + (format.score_type === 'int' ? '0' : '4.2');
    
    return example;
}

async function handleSingleUpload() {
    const selectedDataset = elements.datasetSelector.value;
    const file = elements.datasetFile.files[0];
    
    if (!selectedDataset || !file) {
        alert('Please select both a dataset and a file.');
        return;
    }
    
    console.log('📤 Uploading ' + file.name + ' for ' + selectedDataset + '...');
    
    // Update UI
    elements.uploadSingleBtn.disabled = true;
    elements.uploadSingleBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Uploading...';
    
    // Update progress item
    updateDatasetProgress(selectedDataset, 'uploading', 'Uploading file...');
    
    try {
        // Validate file first
        const isValid = await validateSingleFile(selectedDataset, file);
        if (!isValid) {
            throw new Error('File validation failed');
        }
        
        // Store file
        uploadedDatasets.set(selectedDataset, {
            file: file,
            status: 'uploaded',
            timestamp: new Date().toISOString()
        });
        
        // Update UI
        updateDatasetProgress(selectedDataset, 'uploaded', 'Uploaded: ' + file.name);
        updateOverallProgress();
        updateSubmitButton();
        
        // Reset form
        elements.datasetSelector.value = '';
        elements.datasetFile.value = '';
        hideFormatHelper();
        
        console.log('✅ Successfully uploaded ' + selectedDataset);
        
    } catch (error) {
        console.error('❌ Upload failed for ' + selectedDataset + ':', error);
        updateDatasetProgress(selectedDataset, 'error', 'Error: ' + error.message);
        alert('Upload failed: ' + error.message);
    } finally {
        // Reset button
        elements.uploadSingleBtn.disabled = true;
        elements.uploadSingleBtn.innerHTML = '<i class="fas fa-upload"></i> Upload';
    }
}

async function validateSingleFile(datasetName, file) {
    try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('dataset_name', datasetName);
        
        const response = await fetch('/api/submissions/validate-csv', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (!result.valid) {
            throw new Error(result.errors ? result.errors.join(', ') : 'Validation failed');
        }
        
        return true;
    } catch (error) {
        console.error('Validation error:', error);
        throw error;
    }
}

function updateDatasetProgress(datasetName, status, message) {
    const progressItem = document.getElementById('progress-' + datasetName);
    const statusElement = document.getElementById('status-' + datasetName);
    const detailsElement = document.getElementById('details-' + datasetName);
    
    if (progressItem) {
        progressItem.className = 'dataset-progress-item ' + status;
        
        if (status === 'uploading') {
            progressItem.classList.add('upload-animation');
        } else {
            progressItem.classList.remove('upload-animation');
        }
    }
    
    if (statusElement) {
        statusElement.className = 'dataset-status status-' + status;
        statusElement.textContent = status.charAt(0).toUpperCase() + status.slice(1);
    }
    
    if (detailsElement) {
        detailsElement.textContent = message;
    }
}

function updateOverallProgress() {
    const uploadedCount = uploadedDatasets.size;
    const percentage = totalDatasetsCount > 0 ? (uploadedCount / totalDatasetsCount) * 100 : 0;
    
    if (elements.uploadedCount) {
        elements.uploadedCount.textContent = uploadedCount;
    }
    
    if (elements.overallProgressFill) {
        elements.overallProgressFill.style.width = percentage + '%';
    }
    
    if (elements.overallProgressText) {
        elements.overallProgressText.textContent = Math.round(percentage) + '%';
    }
    
    if (elements.submitCount) {
        elements.submitCount.textContent = uploadedCount;
    }
}

function updateSubmitButton() {
    if (!elements.submitBenchmarkBtn) return;
    
    const uploadedCount = uploadedDatasets.size;
    const isComplete = uploadedCount === totalDatasetsCount && totalDatasetsCount > 0;
    
    elements.submitBenchmarkBtn.disabled = !isComplete;
    
    if (isComplete) {
        elements.submitBenchmarkBtn.innerHTML = '<i class="fas fa-rocket"></i> Submit Complete Benchmark (' + uploadedCount + '/' + totalDatasetsCount + ')';
    } else {
        elements.submitBenchmarkBtn.innerHTML = '<i class="fas fa-upload"></i> Upload ' + (totalDatasetsCount - uploadedCount) + ' more datasets (' + uploadedCount + '/' + totalDatasetsCount + ')';
    }
    
    // Update validate button
    if (elements.validateAllBtn) {
        elements.validateAllBtn.disabled = uploadedCount === 0;
    }
}

async function validateAllFiles() {
    if (uploadedDatasets.size === 0) {
        alert('No files to validate. Please upload some datasets first.');
        return;
    }
    
    console.log('🔍 Validating ' + uploadedDatasets.size + ' uploaded files...');
    
    elements.validateAllBtn.disabled = true;
    elements.validateAllBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Validating...';
    
    let validFiles = 0;
    let errors = [];
    
    for (const [datasetName, uploadInfo] of uploadedDatasets) {
        try {
            updateDatasetProgress(datasetName, 'uploading', 'Validating...');
            
            const isValid = await validateSingleFile(datasetName, uploadInfo.file);
            if (isValid) {
                validFiles++;
                updateDatasetProgress(datasetName, 'uploaded', 'Valid: ' + uploadInfo.file.name);
            }
        } catch (error) {
            errors.push(datasetName + ': ' + error.message);
            updateDatasetProgress(datasetName, 'error', 'Invalid: ' + error.message);
        }
    }
    
    // Show validation results
    if (errors.length === 0) {
        alert('✅ All ' + validFiles + ' files are valid and ready for submission!');
    } else {
        const errorMessage = 'Validation Results:\n✅ Valid: ' + validFiles + '\n❌ Errors: ' + errors.length + '\n\nErrors:\n' + errors.slice(0, 5).join('\n') + (errors.length > 5 ? '\n... and more' : '');
        alert(errorMessage);
    }
    
    elements.validateAllBtn.disabled = uploadedDatasets.size === 0;
    elements.validateAllBtn.innerHTML = '<i class="fas fa-check-circle"></i> Validate All Files';
}

// ===== Download Functions =====
async function downloadFile(url, suggestedName) {
    try {
        console.log('📥 Downloading: ' + url);
        const res = await fetch(API_BASE_URL + url);
        if (!res.ok) throw new Error('HTTP ' + res.status);
        
        const blob = await res.blob();
        const a = document.createElement('a');
        const href = URL.createObjectURL(blob);
        a.href = href;
        a.download = suggestedName;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(href);
        
        console.log('✅ Downloaded: ' + suggestedName);
    } catch (err) {
        alert('Download failed: ' + err.message);
        console.error('Download failed:', err);
    }
}

function downloadAllDatasets() {
    downloadFile('/api/datasets/download/all', 'besesr_all_datasets.zip');
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

// ===== Load Datasets Grid =====
async function loadDatasets() {
    if (!elements.datasetsGrid) return;

    showLoading(elements.datasetsGrid, 'Loading datasets...');

    try {
        console.log("📊 Loading datasets for grid display...");

        if (!availableDatasets.length) {
            showEmptyState(
                elements.datasetsGrid,
                'No Datasets Available',
                'No datasets have been configured yet. Please check back later.'
            );
            return;
        }

        // Create dataset cards
        var datasetsHTML = '';
        availableDatasets.forEach((datasetName, index) => {
            const animationDelay = index * 0.1;
            datasetsHTML += '<div class="dataset-card fade-in" style="animation-delay: ' + animationDelay + 's;">';
            datasetsHTML += '<h3>' + datasetName + '</h3>';
            datasetsHTML += '<p>Train/validation/test splits with essay data</p>';
            datasetsHTML += '<div class="dataset-info">';
            datasetsHTML += '<span><i class="fas fa-database"></i> Original format preserved</span>';
            datasetsHTML += '</div>';
            datasetsHTML += '<div class="dataset-actions">';
            datasetsHTML += '<button onclick="downloadSingleDataset(\'' + datasetName + '\')" class="btn btn-sm btn-primary">';
            datasetsHTML += '<i class="fas fa-download"></i> Download';
            datasetsHTML += '</button>';
            datasetsHTML += '</div>';
            datasetsHTML += '</div>';
        });

        elements.datasetsGrid.innerHTML = datasetsHTML;

        console.log('✅ Loaded ' + availableDatasets.length + ' datasets in grid');

    } catch (error) {
        console.error('❌ Failed to load datasets grid:', error);
        showError(elements.datasetsGrid, 'Failed to load datasets. Please check the API connection.');
    }
}

// ===== FINAL FIXED: loadLeaderboard() function =====
async function loadLeaderboard() {
    if (!elements.leaderboardTable) return;
    
    showLoading(elements.leaderboardTable, 'Loading complete benchmark leaderboard...');
    
    try {
        const metric = elements.metricSelector?.value || 'avg_quadratic_weighted_kappa';
        
        console.log('🔍 Debug: Loading leaderboard with metric:', metric);
        
        const data = await fetchAPI('/api/submissions/leaderboard?metric=' + metric + '&limit=20');
        
        console.log('🔍 Debug: Full API response:', data);
        
        if (!data || !data.rankings || data.rankings.length === 0) {
            const actionHtml = '<a href="#submit" class="btn btn-primary"><i class="fas fa-rocket"></i> Submit Complete Benchmark</a>';
            showEmptyState(
                elements.leaderboardTable,
                'No Complete Benchmarks Yet',
                'Be the first to submit all datasets to appear on the leaderboard!',
                actionHtml
            );
            return;
        }
        
        // Debug: Check first entry metrics
        if (data.rankings && data.rankings.length > 0) {
            console.log('🔍 Debug: First ranking entry ALL metrics:', {
                qwk: data.rankings[0].avg_quadratic_weighted_kappa,
                pearson: data.rankings[0].avg_pearson_correlation,
                f1: data.rankings[0].avg_f1_score,
                precision: data.rankings[0].avg_precision,
                recall: data.rankings[0].avg_recall,
                mae: data.rankings[0].avg_mae
            });
        }
        
        // ✅ FIXED: Build table with proper structure and ALL 6 metrics
        var tableHTML = '<div style="margin-bottom: 1rem; padding: 1rem; background: var(--bg-light); border-radius: 8px;">';
        tableHTML += '<p><strong>🏆 Complete Benchmarks Only:</strong> Only researchers who submitted all ' + totalDatasetsCount + ' datasets appear here.</p>';
        tableHTML += '</div>';
        
        // ✅ FIXED: Proper overflow container with table
        tableHTML += '<div style="overflow-x: auto; -webkit-overflow-scrolling: touch;">';
        tableHTML += '<table style="width: 100%; min-width: 1400px; border-collapse: collapse; font-size: 0.9rem;">';
        
        // ✅ FIXED: Complete table header with ALL 6 metrics
        tableHTML += '<thead>';
        tableHTML += '<tr style="background: var(--bg-light);">';
        tableHTML += '<th style="padding: 1rem 0.75rem; border: 1px solid var(--border-light); text-align: center; min-width: 60px;">Rank</th>';
        tableHTML += '<th style="padding: 1rem 0.75rem; border: 1px solid var(--border-light); text-align: left; min-width: 150px;">Researcher</th>';
        tableHTML += '<th style="padding: 1rem 0.75rem; border: 1px solid var(--border-light); text-align: left; min-width: 120px;">Model</th>';
        tableHTML += '<th style="padding: 1rem 0.75rem; border: 1px solid var(--border-light); text-align: center; min-width: 100px;">Avg QWK</th>';
        tableHTML += '<th style="padding: 1rem 0.75rem; border: 1px solid var(--border-light); text-align: center; min-width: 100px;">Avg Pearson</th>';
        tableHTML += '<th style="padding: 1rem 0.75rem; border: 1px solid var(--border-light); text-align: center; min-width: 100px;">Avg F1</th>';
        tableHTML += '<th style="padding: 1rem 0.75rem; border: 1px solid var(--border-light); text-align: center; min-width: 100px;">Avg Precision</th>';
        tableHTML += '<th style="padding: 1rem 0.75rem; border: 1px solid var(--border-light); text-align: center; min-width: 100px;">Avg Recall</th>';
        tableHTML += '<th style="padding: 1rem 0.75rem; border: 1px solid var(--border-light); text-align: center; min-width: 100px;">Avg MAE</th>';
        tableHTML += '<th style="padding: 1rem 0.75rem; border: 1px solid var(--border-light); text-align: center; min-width: 80px;">Datasets</th>';
        tableHTML += '<th style="padding: 1rem 0.75rem; border: 1px solid var(--border-light); text-align: center; min-width: 80px;">Essays</th>';
        tableHTML += '<th style="padding: 1rem 0.75rem; border: 1px solid var(--border-light); text-align: center; min-width: 100px;">Submitted</th>';
        tableHTML += '</tr>';
        tableHTML += '</thead>';
        
        // ✅ FIXED: Table body with ALL metric data
        tableHTML += '<tbody>';
        
        data.rankings.forEach((entry, index) => {
            const rank = entry.rank || (index + 1);
            const rankClass = getRankClass(rank);
            
            // ✅ FIXED: Extract ALL 6 metrics from API response
            const qwk = entry.avg_quadratic_weighted_kappa || 0;
            const pearson = entry.avg_pearson_correlation || 0;
            const f1 = entry.avg_f1_score || 0;
            const precision = entry.avg_precision || 0;
            const recall = entry.avg_recall || 0;
            const mae = entry.avg_mae || 0;
            
            // ✅ FIXED: Style classes for each metric
            const qwkClass = getScoreClass(qwk);
            const pearsonClass = getScoreClass(pearson);
            const f1Class = getScoreClass(f1);
            const precisionClass = getScoreClass(precision);
            const recallClass = getScoreClass(recall);
            const maeClass = getScoreClass(mae < 0.5 ? 0.8 : mae < 1.0 ? 0.6 : 0.3); // MAE: lower is better
            
            // Format scores
            const qwkScore = formatScore(qwk);
            const pearsonScore = formatScore(pearson);
            const f1Score = formatScore(f1);
            const precisionScore = formatScore(precision);
            const recallScore = formatScore(recall);
            const maeScore = formatScore(mae);
            
            // Other data
            const submitterName = entry.model_name || 'Unknown';
            const modelName = entry.model_description || entry.dataset_name || 'Model Description';
            const totalEssays = entry.total_essays_evaluated || 0;
            const submissionDate = formatDate(entry.upload_date);
            const datasetsCompleted = entry.total_datasets_submitted || 0;
            
            // ✅ FIXED: Build complete table row with ALL 6 metrics
            tableHTML += '<tr style="border-bottom: 1px solid var(--border-light);">';
            tableHTML += '<td style="padding: 1rem 0.75rem; text-align: center;"><div class="rank-badge ' + rankClass + '">' + rank + '</div></td>';
            tableHTML += '<td style="padding: 1rem 0.75rem; color: var(--text-primary);"><strong>' + submitterName + '</strong></td>';
            tableHTML += '<td style="padding: 1rem 0.75rem; color: var(--text-secondary); font-size: 0.9em;">' + modelName + '</td>';
            
            // ✅ FIXED: ALL 6 metric columns with proper styling and tooltips
            tableHTML += '<td style="padding: 1rem 0.75rem; text-align: center;"><span class="score-badge ' + qwkClass + '" title="Quadratic Weighted Kappa: ' + getScoreDescription(qwk) + '">' + qwkScore + '</span></td>';
            tableHTML += '<td style="padding: 1rem 0.75rem; text-align: center;"><span class="score-badge ' + pearsonClass + '" title="Pearson Correlation: ' + getScoreDescription(pearson) + '">' + pearsonScore + '</span></td>';
            tableHTML += '<td style="padding: 1rem 0.75rem; text-align: center;"><span class="score-badge ' + f1Class + '" title="F1 Score: ' + getScoreDescription(f1) + '">' + f1Score + '</span></td>';
            tableHTML += '<td style="padding: 1rem 0.75rem; text-align: center;"><span class="score-badge ' + precisionClass + '" title="Precision: ' + getScoreDescription(precision) + '">' + precisionScore + '</span></td>';
            tableHTML += '<td style="padding: 1rem 0.75rem; text-align: center;"><span class="score-badge ' + recallClass + '" title="Recall: ' + getScoreDescription(recall) + '">' + recallScore + '</span></td>';
            tableHTML += '<td style="padding: 1rem 0.75rem; text-align: center;"><span class="score-badge ' + maeClass + '" title="Mean Absolute Error: ' + (mae < 0.5 ? 'Excellent' : mae < 1.0 ? 'Good' : 'Fair') + ' (Lower is Better)">' + maeScore + '</span></td>';
            
            // Other columns
            tableHTML += '<td style="padding: 1rem 0.75rem; text-align: center;"><span style="color: var(--success); font-weight: 600;">' + datasetsCompleted + '/' + totalDatasetsCount + '</span></td>';
            tableHTML += '<td style="padding: 1rem 0.75rem; text-align: center;">' + totalEssays + '</td>';
            tableHTML += '<td style="padding: 1rem 0.75rem; text-align: center;"><small style="color: var(--text-muted);">' + submissionDate + '</small></td>';
            tableHTML += '</tr>';
            
            // Debug log for each entry
            console.log('🔍 Debug entry ' + index + ' (' + submitterName + '):', {
                qwk: qwk,
                pearson: pearson,
                f1: f1,
                precision: precision,
                recall: recall,
                mae: mae
            });
        });
        
        tableHTML += '</tbody>';
        tableHTML += '</table>';
        tableHTML += '</div>'; // Close overflow container
        
        // ✅ Add summary statistics if available
        if (data.summary_stats) {
            tableHTML += '<div style="margin-top: 1.5rem; padding: 1rem; background: var(--bg-light); border-radius: 8px;">';
            tableHTML += '<h4 style="margin-bottom: 1rem;">📊 Platform Summary</h4>';
            tableHTML += '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem;">';
            tableHTML += '<div><strong>Total Researchers:</strong><br><span style="font-size: 1.2em; color: var(--primary-blue);">' + (data.summary_stats.total_researchers || 0) + '</span></div>';
            tableHTML += '<div><strong>Complete Benchmarks:</strong><br><span style="font-size: 1.2em; color: var(--success);">' + (data.summary_stats.complete_benchmarks || 0) + '</span></div>';
            tableHTML += '<div><strong>Avg QWK:</strong><br><span style="font-size: 1.2em; color: var(--primary-blue);">' + formatScore(data.summary_stats.avg_quadratic_weighted_kappa || 0) + '</span></div>';
            tableHTML += '<div><strong>Avg F1:</strong><br><span style="font-size: 1.2em; color: var(--primary-blue);">' + formatScore(data.summary_stats.avg_f1_score || 0) + '</span></div>';
            tableHTML += '<div><strong>Avg Precision:</strong><br><span style="font-size: 1.2em; color: var(--primary-blue);">' + formatScore(data.summary_stats.avg_precision || 0) + '</span></div>';
            tableHTML += '<div><strong>Avg Recall:</strong><br><span style="font-size: 1.2em; color: var(--primary-blue);">' + formatScore(data.summary_stats.avg_recall || 0) + '</span></div>';
            tableHTML += '</div>';
            tableHTML += '</div>';
        }
        
        // ✅ FIXED: Set the complete HTML
        elements.leaderboardTable.innerHTML = tableHTML;
        
        console.log('✅ FINAL SUCCESS: Loaded leaderboard with ' + data.rankings.length + ' complete benchmarks and ALL 6 METRICS');
        console.log('✅ Table should now show: Rank | Researcher | Model | QWK | Pearson | F1 | Precision | Recall | MAE | Datasets | Essays | Submitted');
        
    } catch (error) {
        showError(elements.leaderboardTable, 'Failed to load leaderboard. Please check the API connection.');
        console.error('❌ Failed to load leaderboard:', error);
    }
}

// ===== Form Submission =====
function setupBenchmarkSubmission() {
    if (!elements.benchmarkForm) return;

    elements.benchmarkForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        if (uploadedDatasets.size !== totalDatasetsCount) {
            alert('Please upload all ' + totalDatasetsCount + ' datasets before submitting.');
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
            uploadedDatasets.forEach((uploadInfo, datasetName) => {
                formData.append('files', uploadInfo.file, datasetName + '.csv');
            });

            console.log('🚀 Submitting complete benchmark with ' + uploadedDatasets.size + ' datasets');

            // Submit to batch processing endpoint
            const response = await fetch('/api/submissions/upload-batch', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.detail || 'HTTP ' + response.status);
            }

            // Show success
            showBenchmarkSuccess(result);
            
            // Reset form
            this.reset();
            uploadedDatasets.clear();
            initializeProgressGrid();
            updateOverallProgress();
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
    
    var successHTML = '<h3><i class="fas fa-trophy"></i> Complete Benchmark Submitted Successfully!</h3>';
    successHTML += '<p><strong>Researcher:</strong> ' + (result.model_name || 'Unknown') + '</p>';
    successHTML += '<p><strong>Model:</strong> ' + (result.model_name || 'Unknown') + '</p>';
    
    successHTML += '<div style="background: var(--bg-light); padding: 1.5rem; border-radius: 8px; margin: 1rem 0;">';
    successHTML += '<h4 style="margin-bottom: 1rem; color: var(--primary-blue);">🏆 Submission Results</h4>';
    successHTML += '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">';
    successHTML += '<div><strong>Files Uploaded:</strong><br>';
    successHTML += '<span style="font-size: 1.2em; color: var(--success);">' + (result.successful_uploads || 0) + '/' + (result.total_files || 0) + '</span></div>';
    successHTML += '<div><strong>Success Rate:</strong><br>';
    successHTML += '<span style="font-size: 1.2em; color: var(--success);">' + (result.success_rate || 0) + '%</span></div>';
    successHTML += '<div><strong>Upload Date:</strong><br>';
    successHTML += '<span style="font-size: 1.2em; color: var(--primary-blue);">' + formatDate(result.upload_date) + '</span></div>';
    successHTML += '</div>';
    successHTML += '</div>';
    
    successHTML += '<div style="margin-top: 1.5rem;">';
    successHTML += '<a href="#leaderboard" class="btn btn-primary">';
    successHTML += '<i class="fas fa-trophy"></i> View Leaderboard Position';
    successHTML += '</a>';
    successHTML += '</div>';
    
    elements.submitResult.innerHTML = successHTML;
    elements.submitResult.style.display = 'block';
    elements.submitResult.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function showBenchmarkError(message) {
    elements.submitResult.className = 'submit-result error';
    
    var errorHTML = '<h3><i class="fas fa-exclamation-circle"></i> Benchmark Submission Failed</h3>';
    errorHTML += '<p><strong>Error:</strong> ' + message + '</p>';
    errorHTML += '<p>Please check your files and try again.</p>';
    errorHTML += '<div style="margin-top: 1rem;">';
    errorHTML += '<a href="#datasets" class="btn btn-outline">';
    errorHTML += '<i class="fas fa-download"></i> Download Datasets Again';
    errorHTML += '</a>';
    errorHTML += '</div>';
    
    elements.submitResult.innerHTML = errorHTML;
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
            formData.append('contact_email', 'test@example.com');

            const response = await fetch('/api/submissions/upload-single', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.detail || result.message || 'HTTP ' + response.status);
            }

            // Show test result
            const alertMessage = 'Test Upload Successful for ' + result.dataset + '!\nRows: ' + result.row_count + '\nNote: This is for testing only.';
            alert(alertMessage);
            
            this.reset();

        } catch (error) {
            alert('Test failed: ' + error.message);
        } finally {
            submitBtn.disabled = false;
            submitBtn.innerHTML = originalText;
        }
    });
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
document.addEventListener('DOMContentLoaded', async function() {
    console.log('🚀 Enhanced BESESR Platform Initialized');
    
    // Setup basic functionality
    setupSmoothScrolling();
    setupEventListeners();
    
    console.log('📊 Loading datasets and initializing platform...');
    
    try {
        // Load datasets first (this is crucial for everything else)
        await loadAvailableDatasets();
        
        // Setup enhanced upload after datasets are loaded
        setupEnhancedUpload();
        setupBenchmarkSubmission();
        setupIndividualTesting();
        
        // Load other data
        loadPlatformStats();
        loadDatasets(); // For the grid display
        loadLeaderboard();
        
        console.log('✅ Enhanced BESESR Platform ready!');
        
    } catch (error) {
        console.error('❌ Failed to initialize platform:', error);
        
        // Still try to setup basic functionality
        setupEnhancedUpload();
        setupBenchmarkSubmission();
        setupIndividualTesting();
        
        // Show error message to user
        if (elements.datasetsGrid) {
            showError(elements.datasetsGrid, 'Failed to load platform data. Some features may not work correctly.');
        }
    }
    
    // Auto-refresh data periodically
    setInterval(() => {
        console.log('🔄 Auto-refreshing data...');
        loadPlatformStats();
        loadLeaderboard();
    }, 60000);
});

// ===== Global Functions =====
window.downloadSingleDataset = (datasetName) => {
    downloadFile('/api/datasets/download/' + encodeURIComponent(datasetName), datasetName + '_splits.zip');
};

// ===== Debug Functions (for development) =====
window.debugBesesr = {
    getUploadedDatasets: () => Array.from(uploadedDatasets.entries()),
    getAvailableDatasets: () => availableDatasets,
    getDatasetFormats: () => datasetFormats,
    getTotalCount: () => totalDatasetsCount,
    clearUploads: () => {
        uploadedDatasets.clear();
        initializeProgressGrid();
        updateOverallProgress();
        updateSubmitButton();
        console.log('✅ Cleared all uploads');
    },
    simulateUpload: (datasetName) => {
        if (availableDatasets.includes(datasetName)) {
            uploadedDatasets.set(datasetName, {
                file: new File(['test'], 'test.csv', { type: 'text/csv' }),
                status: 'uploaded',
                timestamp: new Date().toISOString()
            });
            updateDatasetProgress(datasetName, 'uploaded', 'Simulated upload');
            updateOverallProgress();
            updateSubmitButton();
            console.log('✅ Simulated upload for ' + datasetName);
        } else {
            console.log('❌ Dataset ' + datasetName + ' not found');
        }
    }
};
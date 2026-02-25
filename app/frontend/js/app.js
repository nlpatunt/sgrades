// ===== Enhanced S-GRADES Platform with Dynamic Dataset Loading =====
const API_BASE_URL = '';  // Empty string = same domain
console.log("Enhanced S-GRADES Platform loaded");

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

    // Individual Testing (add these)
    singleTestForm: document.getElementById('single-test-form'),
    testDataset: document.getElementById('test-dataset'),
    testFile: document.getElementById('test-file'),

    // Leaderboard
    metricSelector: document.getElementById('metric-selector'),
    refreshLeaderboard: document.getElementById('refresh-leaderboard'),
    leaderboardTable: document.getElementById('leaderboard-table')
};
console.log("DEBUG: Form element check:", {
    formElement: elements.benchmarkForm,
    formExists: !!elements.benchmarkForm,
    formId: elements.benchmarkForm ? elements.benchmarkForm.id : 'NOT FOUND'
});

// Dataset display name mapping - maps D_ names to user-friendly names
const DATASET_DISPLAY_NAMES = {
    'D_ASAP-AES': 'ASAP-AES: Automated Essay Scoring (8 sets with varying ranges)',
    'D_ASAP2': 'ASAP2: Essay Scoring Dataset',
    'D_ASAP_plus_plus': 'ASAP++: Enhanced Essay Dataset',
    'D_ASAP-SAS': 'ASAP-SAS: Short Answer Scoring',
    'D_BEEtlE_2way': 'BEEtlE: 2-Way Classification',
    'D_BEEtlE_3way': 'BEEtlE: 3-Way Classification',
    'D_CSEE': 'CSEE: Computer Science Essays',
    'D_EFL': 'EFL: English as Foreign Language',
    'D_Mohlar': 'Mohlar: Short Answer Grading',
    'D_SciEntSBank_2way': 'SciEntSBank: 2-Way Science QA',
    'D_SciEntSBank_3way': 'SciEntSBank: 3-Way Science QA',
    'D_Ielts_Writing_Dataset': 'IELTS Writing Assessment',
    'D_Ielst_Writing_Task_2_Dataset': 'IELTS Writing Task 2',
    'D_persuade_2': 'Persuade Essays v2',
    'D_Regrading_Dataset_J2C': 'Regrading Dataset J2C',
    'D_grade_like_a_human_dataset_os_q1': 'OS Concepts: Question 1',
    'D_grade_like_a_human_dataset_os_q2': 'OS Concepts: Question 2',
    'D_grade_like_a_human_dataset_os_q3': 'OS Concepts: Question 3',
    'D_grade_like_a_human_dataset_os_q4': 'OS Concepts: Question 4',
    'D_grade_like_a_human_dataset_os_q5': 'OS Concepts: Question 5',
    'D_Rice_Chem_Q1': 'Chemistry: Question 1',
    'D_Rice_Chem_Q2': 'Chemistry: Question 2',
    'D_Rice_Chem_Q3': 'Chemistry: Question 3',
    'D_Rice_Chem_Q4': 'Chemistry: Question 4'
};

// Helper function to get display name
function getDisplayName(datasetName) {
    return DATASET_DISPLAY_NAMES[datasetName] || datasetName;
}

function getScoreRange(datasetName) {
    const scoreRanges = {
        // ASAP-AES splits
        'D_ASAP-AES_Set1': '2-12',
        'D_ASAP-AES_Set2_Domain1': '1-6',
        'D_ASAP-AES_Set2_Domain2': '1-4',
        'D_ASAP-AES_Set3': '0-3',
        'D_ASAP-AES_Set4': '0-3',
        'D_ASAP-AES_Set5': '0-4',
        'D_ASAP-AES_Set6': '0-4',
        'D_ASAP-AES_Set7': '0-30',
        'D_ASAP-AES_Set8': '0-60',

        // Other datasets with correct ranges
        'D_BEEtlE_2way': 'Correct/Incorrect',
        'D_BEEtlE_3way': 'Correct/Incorrect/Contradictory',
        'D_SciEntSBank_2way': 'Correct/Incorrect',
        'D_SciEntSBank_3way': 'Correct/Incorrect/Contradictory',
        'D_grade_like_a_human_dataset_os_q1': '0-19',
        'D_grade_like_a_human_dataset_os_q2': '0-16',
        'D_grade_like_a_human_dataset_os_q3': '0-15',
        'D_grade_like_a_human_dataset_os_q4': '0-16',
        'D_grade_like_a_human_dataset_os_q5': '0-27',
        'D_Rice_Chem_Q1': '0-8',
        'D_Rice_Chem_Q2': '0-8',
        'D_Rice_Chem_Q3': '0-9',
        'D_Rice_Chem_Q4': '0-8',
        // ... other datasets
    };
    return scoreRanges[datasetName] || 'Variable';
}

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
        const response = await fetch(endpoint);
        if (!response.ok) {
            throw new Error('HTTP ' + response.status + ': ' + response.statusText);
        }
        return await response.json();
    } catch (error) {
        console.error('API Error: <a>', error);
        throw error;
    }
}

async function loadAvailableDatasets() {
    try {
        console.log('Loading available datasets from API...');

        let datasets = [];

        try {
            const response = await fetchAPI(API_BASE_URL + '/api/available-datasets');
	    console.log(response);
            if (response.datasets && Array.isArray(response.datasets)) {
                // Extract just the names from the dataset objects
                datasets = response.datasets.map(dataset => dataset.name);
                console.log('Loaded from available-datasets endpoint');
            }
        } catch (availableError) {
            console.log('Available-datasets failed, using hardcoded fallback');
            datasets = [
                'D_ASAP-AES', 'D_ASAP2', 'D_ASAP_plus_plus', 'D_ASAP-SAS', 'D_BEEtlE_2way', 'D_BEEtlE_3way',
                'D_SciEntSBank_2way', 'D_SciEntSBank_3way', 'D_CSEE', 'D_EFL', 'D_Mohlar',
                'D_Regrading_Dataset_J2C', 'D_Ielts_Writing_Dataset', 'D_Ielst_Writing_Task_2_Dataset',
                'D_persuade_2', 'D_grade_like_a_human_dataset_os_q1', 'D_grade_like_a_human_dataset_os_q2',
                'D_grade_like_a_human_dataset_os_q3', 'D_grade_like_a_human_dataset_os_q4',
                'D_grade_like_a_human_dataset_os_q5', 'D_Rice_Chem_Q1', 'D_Rice_Chem_Q2', 'D_Rice_Chem_Q3', 'D_Rice_Chem_Q4'
            ];
        }

        availableDatasets = datasets;
        totalDatasetsCount = datasets.length;

        updateDatasetCounts();
        populateDatasetDropdowns();
        initializeProgressGrid();

        return datasets;

    } catch (error) {
        console.error('Complete failure loading datasets:', error);
        availableDatasets = ['D_ASAP-AES', 'D_BEEtlE_2way', 'D_CSEE'];
        totalDatasetsCount = availableDatasets.length;
        updateDatasetCounts();
        populateDatasetDropdowns();
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
            const displayName = getDisplayName(dataset);
            const displayText = showDetails
                ? displayName + ' (' + format.required_columns.join(', ') + ')'
                : displayName;
            optionsHTML += '<option value="' + dataset + '">' + displayText + '</option>'; // value=D_ name, text=friendly
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

    // Special case for ASAP-AES
    if (datasetName === 'D_ASAP-AES') {
        const exampleData = `essay_id,domain1_score
essay_1_set1,8.5
essay_2_set7,25.0
essay_3_set2,4.0
essay_4_set3,2.0
essay_5_set8,45.5

# Note: ASAP-AES contains 8 sets with different score ranges:
# Set 1: 2-12 (Persuasive essays)
# Set 2: 1-6 (Persuasive with source material)  
# Sets 3-4: 0-3 (Source-dependent responses)
# Sets 5-6: 0-4 (Source-dependent responses)
# Set 7: 0-30 (Narrative essays)
# Set 8: 0-60 (Narrative essays)
# 
# Click "Score Range Info" button above for detailed breakdown
# Ensure your predicted scores match the appropriate range for each set`;

        elements.formatDetails.textContent = exampleData;
    } else {
        const exampleData = generateExampleCSV(datasetName, format);
        elements.formatDetails.textContent = exampleData;
    }

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

async function testFileValidation(datasetName, file) {
    try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('dataset_name', datasetName);

        const response = await fetch(API_BASE_URL + '/api/submissions/test-single-dataset', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (!result.success) {
            throw new Error(result.error || 'Test failed');
        }

        return true;
    } catch (error) {
        throw error;
    }
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
        const isValid = await testFileValidation(selectedDataset, file);
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

    console.log('🔍 Testing ' + uploadedDatasets.size + ' uploaded files...');

    elements.validateAllBtn.disabled = true;
    elements.validateAllBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Testing...';

    let validFiles = 0;
    let errors = [];

    for (const [datasetName, uploadInfo] of uploadedDatasets) {
        try {
            updateDatasetProgress(datasetName, 'uploading', 'Testing...');

            const isValid = await testFileValidation(datasetName, uploadInfo.file);
            if (isValid) {
                validFiles++;
                updateDatasetProgress(datasetName, 'uploaded', 'Valid: ' + uploadInfo.file.name);
            }
        } catch (error) {
            errors.push(datasetName + ': ' + error.message);
            updateDatasetProgress(datasetName, 'error', 'Invalid: ' + error.message);
        }
    }

    // Show results...
    elements.validateAllBtn.innerHTML = '<i class="fas fa-check-circle"></i> Test All Files';
}

// ===== Download Functions =====
async function downloadFile(url, suggestedName) {
    try {
        console.log('📥 Downloading: ' + url);
        const res = await fetch(url);
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
        alert('Download failed:' + err.message+'::'+url+'::'+suggestedName);
        console.error('Download failed:', err);
    }
}

function downloadAllDatasets() {
    downloadFile(API_BASE_URL + '/api/datasets/download/all', 'sgrades_all_datasets.zip');
}

// ===== Load Platform Stats =====
async function loadPlatformStats() {
    try {
        const stats = await fetchAPI(API_BASE_URL + '/api/leaderboard/stats');

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
            const displayName = getDisplayName(datasetName);
            const scoreRange = getScoreRange(datasetName);

            // Generate appropriate description based on dataset type
            const getDescription = (name) => {
                if (name.includes('BEEtlE')) return 'Student answer classification dataset';
                if (name.includes('Rice_Chem')) return 'Chemistry question scoring';
                if (name.includes('grade_like_a_human')) return 'Operating systems grading';
                if (name.includes('SciEntSBank')) return 'Science answer evaluation';
                if (name.includes('IELTS') || name.includes('Ielts') || name.includes('Ielst')) return 'IELTS writing assessment';
                if (name.includes('ASAP')) return 'Automated essay scoring';
                if (name.includes('persuade')) return 'Persuasive essay evaluation';
                if (name.includes('CSEE')) return 'Computer science essays';
                return 'Essay scoring dataset';
            };

            datasetsHTML += '<div class="dataset-card fade-in" style="animation-delay: ' + animationDelay + 's;">';
            datasetsHTML += '<h3>' + displayName + '</h3>';
            datasetsHTML += '<p>' + getDescription(datasetName) + '</p>';
            datasetsHTML += '<div class="dataset-info">';
            datasetsHTML += '<span><i class="fas fa-database"></i> Original format preserved</span>';
            datasetsHTML += '<span><i class="fas fa-chart-line"></i> Score Range: ' + scoreRange + '</span>';
            datasetsHTML += '</div>';
            datasetsHTML += '<div class="dataset-actions">';
            datasetsHTML += '<button onclick="showScoreRangeModal(\'' + datasetName + '\')" class="btn btn-sm btn-outline" style="margin-right: 0.5rem;">';
            datasetsHTML += '<i class="fas fa-info-circle"></i> Score Range Info';
            datasetsHTML += '</button>';
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

async function loadTestDatasets() {
    try {
        const response = await fetchAPI(API_BASE_URL + '/api/available-datasets');
        const data = response;

        const select = elements.testDataset;
        if (!select) return;

        select.innerHTML = '<option value="">-- Choose dataset --</option>';

        // Process the different data structure
        data.datasets.forEach(dataset => {
            const option = document.createElement('option');
            option.value = dataset.name;
            option.textContent = dataset.name;
            select.appendChild(option);
        });

        console.log('Loaded test datasets');
    } catch (error) {
        console.error('Failed to load test datasets:', error);
    }
}

async function loadTestDatasetInfo(datasetName) {
    const infoDiv = document.getElementById('test-dataset-info');
    const detailsDiv = document.getElementById('test-dataset-details');

    if (!datasetName) {
        if (infoDiv) infoDiv.style.display = 'none';
        return;
    }

    try {
        const response = await fetchAPI(`${API_BASE_URL}/api/output-submissions/dataset-info/${datasetName}`);
        const data = await response.json();

        if (data.error) {
            if (detailsDiv) detailsDiv.innerHTML = `<div class="error">Error: ${data.error}</div>`;
        } else {
            const verification = data.content_verification ? ' (with content verification)' : '';
            if (detailsDiv) {
                detailsDiv.innerHTML = `
                    <strong>Required columns:</strong> ${data.required_columns.join(', ')}<br>
                    <strong>Score column:</strong> ${data.primary_score_column}<br>
                    <strong>Matching:</strong> ${data.matching_method}${verification}<br>
                    <strong>Ground truth examples:</strong> ${data.ground_truth.total_examples || 'Unknown'}
                `;
            }
        }

        if (infoDiv) infoDiv.style.display = 'block';

    } catch (error) {
        console.error('Failed to load dataset info:', error);
        if (detailsDiv) {
            detailsDiv.innerHTML = '<div class="error">Could not load dataset information</div>';
        }
        if (infoDiv) infoDiv.style.display = 'block';
    }
}


function updateTestSubmitButton() {
    const testDatasetSelect = document.getElementById('test-dataset');
    const testFileInput = document.getElementById('test-file');
    const testSubmitBtn = document.getElementById('test-submit-btn');

    if (!testDatasetSelect || !testFileInput || !testSubmitBtn) {
        console.log('Missing elements for test button update');
        return;
    }

    const datasetSelected = testDatasetSelect.value;
    const fileSelected = testFileInput.files.length > 0;

    console.log('Button update:', { datasetSelected, fileSelected });

    testSubmitBtn.disabled = !datasetSelected || !fileSelected;

    if (datasetSelected && fileSelected) {
        testSubmitBtn.innerHTML = '<i class="fas fa-flask"></i> Test Single Dataset';
        testSubmitBtn.classList.remove('btn-disabled');
    } else {
        testSubmitBtn.innerHTML = '<i class="fas fa-flask"></i> Select dataset and file';
        testSubmitBtn.classList.add('btn-disabled');
    }
}

async function handleTestSubmission(event) {
    event.preventDefault();
    console.log('Test submission started');

    const datasetSelect = document.getElementById('test-dataset');
    const fileInput = document.getElementById('test-file');
    const testSubmitBtn = document.getElementById('test-submit-btn');

    if (!datasetSelect?.value || !fileInput?.files[0]) {
        alert('Please select a dataset and upload a CSV file');
        return;
    }

    const formData = new FormData();
    formData.append('dataset_name', datasetSelect.value);
    formData.append('file', fileInput.files[0]);

    const originalText = testSubmitBtn.innerHTML;
    testSubmitBtn.disabled = true;
    testSubmitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Testing...';

    try {
        const response = await fetch(API_BASE_URL + '/api/submissions/test-single-dataset', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        console.log('Test result:', result);

        if (response.ok && result.success) {
            // Store the latest result
            lastTestResult = {
                dataset: datasetSelect.value,
                filename: fileInput.files[0].name,
                evaluation: result.evaluation,
                // keep metrics from the root of the response
                evaluation_metrics: result.evaluation_metrics || result.metrics?.all_metrics || result.metrics,
                validation: result.validation,
                success: result.success,
                timestamp: new Date()
            };

            // Update the results table
            updateIndividualTestResults(lastTestResult);

            alert('Test completed! Results shown in table below.');
        } else {
            alert(`Test failed: ${result.error || result.detail || 'Unknown error'}`);
        }

    } catch (error) {
        console.error('Testing failed:', error);
        alert('Network error: ' + error.message);
    } finally {
        testSubmitBtn.disabled = false;
        testSubmitBtn.innerHTML = originalText;
        updateTestSubmitButton();
    }
}

function updateIndividualTestResults(testResult) {
    const resultsDiv = document.getElementById('individual-test-results');
    const resultsBody = document.getElementById('individual-test-results-body');

    if (!resultsDiv || !resultsBody) return;

    // Show the results section
    resultsDiv.style.display = 'block';

    // Extract metrics from the evaluation
    const metrics =
        testResult.evaluation_metrics ||
        testResult.metrics ||
        (testResult.evaluation && testResult.evaluation.metrics) ||
        {};
    const qwk = metrics.quadratic_weighted_kappa ?? 'N/A';
    const pearson = metrics.pearson_correlation ?? 'N/A';
    const f1 = metrics.f1_score ?? 'N/A';
    const precision = metrics.precision ?? 'N/A';
    const recall = metrics.recall ?? 'N/A';
    const mae = metrics.mean_absolute_error ?? 'N/A';

    const formatScore = (score) => {
        if (score === null || score === undefined) return 'N/A';
        if (score === 'N/A') return 'N/A';
        return Number(score).toFixed(3);
    };

    const status = (testResult.validation?.valid ?? testResult.success ?? false)
        ? '<span style="color: green;">✅ Success</span>'
        : '<span style="color: red;">❌ Failed</span>';

    const rowHTML = `
        <tr style="border-bottom: 1px solid var(--border-light);">
            <td style="padding: 0.75rem;">${testResult.dataset}</td>
            <td style="padding: 0.75rem;">${testResult.filename}</td>
            <td style="padding: 0.75rem; text-align: center;">${formatScore(qwk)}</td>
            <td style="padding: 0.75rem; text-align: center;">${formatScore(pearson)}</td>
            <td style="padding: 0.75rem; text-align: center;">${formatScore(f1)}</td>
            <td style="padding: 0.75rem; text-align: center;">${formatScore(precision)}</td>
            <td style="padding: 0.75rem; text-align: center;">${formatScore(recall)}</td>
            <td style="padding: 0.75rem; text-align: center;">${formatScore(mae)}</td>
            <td style="padding: 0.75rem; text-align: center;">${status}</td>
        </tr>
    `;

    // Replace the table body content (always shows only the latest result)
    resultsBody.innerHTML = rowHTML;

    console.log('✅ Updated individual test results table');
}

function showTestResults(result) {
    const testResults = document.getElementById('test-results');
    const testResultsTitle = document.getElementById('test-results-title');
    const testResultsSubtitle = document.getElementById('test-results-subtitle');
    const testPerformanceItems = document.getElementById('test-performance-items');
    const testMetricsGrid = document.getElementById('test-metrics-grid');
    const testDetailsGrid = document.getElementById('test-details-grid');
    const testPerformanceSummary = document.getElementById('test-performance-summary');
    const testEvaluationDetails = document.getElementById('test-evaluation-details');
    const testWarnings = document.getElementById('test-warnings');

    if (testResults) testResults.style.display = 'block';
    if (testResultsTitle) testResultsTitle.textContent = `Test Results: ${result.dataset}`;
    if (testResultsSubtitle) {
        testResultsSubtitle.textContent =
            `File: ${result.filename} | ${result.evaluation.matched_examples} examples evaluated`;
    }

    // Performance Summary
    if (testPerformanceItems && result.performance_summary) {
        testPerformanceItems.innerHTML = '';

        Object.entries(result.performance_summary).forEach(([key, value]) => {
            const item = document.createElement('div');
            item.className = 'performance-item';

            const badgeClass = getBadgeClass(value);
            item.innerHTML = `
                <span class="detail-label">${formatLabel(key)}</span>
                <span class="performance-badge ${badgeClass}">${value}</span>
            `;
            testPerformanceItems.appendChild(item);
        });

        if (testPerformanceSummary) testPerformanceSummary.style.display = 'block';
    }

    // Primary Metrics
    if (testMetricsGrid && result.metrics) {
        testMetricsGrid.innerHTML = '';

        Object.entries(result.metrics.primary_metrics).forEach(([key, value]) => {
            const card = document.createElement('div');
            card.className = 'metric-card';
            card.innerHTML = `
                <h6>${key}</h6>
                <div class="metric-value">${value}</div>
            `;
            testMetricsGrid.appendChild(card);
        });

        // Additional metrics
        Object.entries(result.metrics.additional_metrics).forEach(([key, value]) => {
            const card = document.createElement('div');
            card.className = 'metric-card';
            card.innerHTML = `
                <h6>${key}</h6>
                <div class="metric-value">${value}</div>
            `;
            testMetricsGrid.appendChild(card);
        });
    }

    // Evaluation Details
    if (testDetailsGrid && result.evaluation) {
        testDetailsGrid.innerHTML = '';

        const details = [
            ['Matched Examples', result.evaluation.matched_examples],
            ['Total Predictions', result.evaluation.total_predictions],
            ['Matching Method', result.evaluation.matching_method],
            ['Score Column', result.evaluation.score_column],
            ['Prediction Range', `${result.evaluation.score_range_pred[0]} - ${result.evaluation.score_range_pred[1]}`],
            ['Ground Truth Range', `${result.evaluation.score_range_true[0]} - ${result.evaluation.score_range_true[1]}`]
        ];

        details.forEach(([label, value]) => {
            const item = document.createElement('div');
            item.className = 'detail-item';
            item.innerHTML = `
                <div class="detail-label">${label}:</div>
                <div class="detail-value">${value}</div>
            `;
            testDetailsGrid.appendChild(item);
        });

        if (testEvaluationDetails) testEvaluationDetails.style.display = 'block';
    }

    // Show warnings if any
    if (testWarnings && result.validation?.warnings?.length > 0) {
        testWarnings.innerHTML = `
            <strong>Warnings:</strong><br>${result.validation.warnings.join('<br>')}
        `;
        testWarnings.style.display = 'block';
    }
}

function showTestError(message, details = null) {
    const testResults = document.getElementById('test-results');
    const testWarnings = document.getElementById('test-warnings');

    if (testResults) testResults.style.display = 'none';

    let errorHtml = `<div class="error"><strong>Error:</strong> ${message}`;
    if (details && details.validation && details.validation.errors) {
        errorHtml += `<br><br><strong>Validation errors:</strong><br>${details.validation.errors.join('<br>')}`;
    }
    errorHtml += '</div>';

    if (testWarnings) {
        testWarnings.innerHTML = errorHtml;
        testWarnings.style.display = 'block';
    }
}

function showTestLoading() {
    const testLoading = document.getElementById('test-loading');
    const testResults = document.getElementById('test-results');
    const testWarnings = document.getElementById('test-warnings');

    if (testLoading) testLoading.style.display = 'block';
    if (testResults) testResults.style.display = 'none';
    if (testWarnings) testWarnings.style.display = 'none';
}

function hideTestLoading() {
    const testLoading = document.getElementById('test-loading');
    if (testLoading) testLoading.style.display = 'none';
}

function formatLabel(key) {
    return key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

function getBadgeClass(value) {
    const lowerValue = value.toLowerCase();
    if (lowerValue.includes('excellent')) return 'badge-excellent';
    if (lowerValue.includes('good')) return 'badge-good';
    if (lowerValue.includes('fair')) return 'badge-fair';
    return 'badge-poor';
}

// ===== FINAL FIXED: loadLeaderboard() function =====
async function loadLeaderboard() {
    if (!elements.leaderboardTable) return;

    showLoading(elements.leaderboardTable, 'Loading complete benchmark leaderboard...');

    try {
        const metric = elements.metricSelector?.value || 'avg_quadratic_weighted_kappa';

        console.log('🔍 Debug: Loading leaderboard with metric:', metric);

        const data = await fetchAPI(`${API_BASE_URL}/api/submissions/leaderboard-cached?metric=${metric}&limit=20`);

        console.log('🔍 Debug: Full API response:', data);

        console.log('Debug: API response summary_stats:', data.summary_stats);

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

        var tableHTML = '<div style="margin-bottom: 1rem; padding: 1rem; background: var(--bg-light); border-radius: 8px;">';
        tableHTML += '<p><strong>🏆 Complete Benchmarks Only:</strong> Only researchers who submitted all ' + totalDatasetsCount + ' datasets appear here.</p>';
        tableHTML += '</div>';

        tableHTML += '<div style="overflow-x: auto; -webkit-overflow-scrolling: touch;">';
        tableHTML += '<table style="width: 100%; min-width: 1400px; border-collapse: collapse; font-size: 0.9rem;">';

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
            const mae = entry.avg_mean_absolute_error || 0;

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

            // Other data - using correct field mapping
            const submitterName = entry.contact_email || 'Research Team';  //
            const totalEssays = entry.total_submissions || 0;
            const modelName = entry.model_name || 'Unknown';
            const submissionDate = formatDate(entry.last_updated || new Date().toISOString());
            const datasetsCompleted = entry.unique_datasets_count || 0;

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

        if (data.summary_stats) {
            tableHTML += '<div style="margin-top: 1.5rem; padding: 1rem; background: var(--bg-light); border-radius: 8px;">';
            tableHTML += '<h4 style="margin-bottom: 1rem;">📊 Platform Summary</h4>';
            tableHTML += '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem;">';
            tableHTML += '<div><strong>Total Researchers:</strong><br><span style="font-size: 1.2em; color: var(--primary-blue);">' + (data.summary_stats.total_researchers || 0) + '</span></div>';
            tableHTML += '<div><strong>Complete Benchmarks:</strong><br><span style="font-size: 1.2em; color: var(--success);">' + (data.summary_stats.complete_benchmarks || 0) + '</span></div>';

            // Use .mean to access the actual values
            if (data.summary_stats.avg_quadratic_weighted_kappa && data.summary_stats.avg_quadratic_weighted_kappa.mean !== undefined) {
                tableHTML += '<div><strong>Avg QWK:</strong><br><span style="font-size: 1.2em; color: var(--primary-blue);">' + formatScore(data.summary_stats.avg_quadratic_weighted_kappa.mean) + '</span></div>';
            }

            if (data.summary_stats.avg_f1_score && data.summary_stats.avg_f1_score.mean !== undefined) {
                tableHTML += '<div><strong>Avg F1:</strong><br><span style="font-size: 1.2em; color: var(--primary-blue);">' + formatScore(data.summary_stats.avg_f1_score.mean) + '</span></div>';
            }

            if (data.summary_stats.avg_precision && data.summary_stats.avg_precision.mean !== undefined) {
                tableHTML += '<div><strong>Avg Precision:</strong><br><span style="font-size: 1.2em; color: var(--primary-blue);">' + formatScore(data.summary_stats.avg_precision.mean) + '</span></div>';
            }

            if (data.summary_stats.avg_recall && data.summary_stats.avg_recall.mean !== undefined) {
                tableHTML += '<div><strong>Avg Recall:</strong><br><span style="font-size: 1.2em; color: var(--primary-blue);">' + formatScore(data.summary_stats.avg_recall.mean) + '</span></div>';
            }

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
    console.log("Form element found:", elements.benchmarkForm);
    if (!elements.benchmarkForm) {
        console.log("ERROR: benchmarkForm element not found!");
        return;
    }

    elements.benchmarkForm.addEventListener('submit', async function (e) {
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
            const datasetNamesArray = Array.from(uploadedDatasets.keys());

            // Add all uploaded files
            uploadedDatasets.forEach((uploadInfo, datasetName) => {
                formData.append('files', uploadInfo.file);
            });

            datasetNamesArray.forEach(name => {
                formData.append('dataset_names', name);
            });

            console.log('🚀 Submitting complete benchmark with ' + uploadedDatasets.size + ' datasets');

            // Submit to batch processing endpoint
            const response = await fetch(API_BASE_URL + '/api/submissions/upload-batch', {
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

    const totalFiles = result.total_files || 0;
    const successfulUploads = result.successful_uploads || 0;
    const successRate = totalFiles > 0 ? Math.round((successfulUploads / totalFiles) * 100) : 0;

    var successHTML = '<h3><i class="fas fa-trophy"></i> Complete Benchmark Submitted Successfully!</h3>';
    successHTML += '<p><strong>Researcher:</strong> ' + (result.model_name || 'Unknown') + '</p>';
    successHTML += '<p><strong>Model:</strong> ' + (result.model_name || 'Unknown') + '</p>';

    successHTML += '<div style="background: var(--bg-light); padding: 1.5rem; border-radius: 8px; margin: 1rem 0;">';
    successHTML += '<h4 style="margin-bottom: 1rem; color: var(--primary-blue);">🏆 Submission Results</h4>';
    successHTML += '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">';
    successHTML += '<div><strong>Files Uploaded:</strong><br>';
    successHTML += '<span style="font-size: 1.2em; color: var(--success);">' + + successfulUploads + '/' + totalFiles + '</span></div>';
    successHTML += '<div><strong>Success Rate:</strong><br>';
    successHTML += '<span style="font-size: 1.2em; color: var(--success);">' + successRate + '%</span></div>';
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

function setupIndividualTesting() {
    console.log('Setting up individual testing...');

    if (!elements.singleTestForm) {
        console.log('Single test form not found');
        return;
    }

    // Load test datasets
    loadTestDatasets();

    // Set up event listeners with error checking
    const testDatasetSelect = document.getElementById('test-dataset');
    const testFileInput = document.getElementById('test-file');

    if (testDatasetSelect) {
        testDatasetSelect.addEventListener('change', (e) => {
            console.log('Dataset changed:', e.target.value);
            updateTestSubmitButton();
        });
    } else {
        console.log('Test dataset select not found');
    }

    if (testFileInput) {
        testFileInput.addEventListener('change', (e) => {
            console.log('File changed:', e.target.files.length);
            updateTestSubmitButton();
        });
    } else {
        console.log('Test file input not found');
    }

    // Form submission handler
    elements.singleTestForm.addEventListener('submit', handleTestSubmission);

    // Initial button state
    updateTestSubmitButton();

    console.log('Individual testing setup complete');
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

async function loadMethodologyLeaderboard(methodology, tableElementId) {
    const tableElement = document.getElementById(tableElementId);
    if (!tableElement) return;

    showLoading(tableElement, `Loading ${methodology} leaderboard...`);

    try {
        const data = await fetchAPI(`${API_BASE_URL}/api/submissions/leaderboard?methodology=${methodology}&limit=20`);

        if (!data || !data.rankings || data.rankings.length === 0) {
            const actionHtml = `<a href="#submit" class="btn btn-primary"><i class="fas fa-rocket"></i> Submit ${methodology} Benchmark</a>`;
            showEmptyState(
                tableElement,
                `No ${methodology} Benchmarks Yet`,
                `Be the first to submit a ${methodology} benchmark!`,
                actionHtml
            );
            return;
        }

        buildLeaderboardTable(data, tableElement, methodology);

    } catch (error) {
        showError(tableElement, `Failed to load ${methodology} leaderboard.`);
        console.error(`Failed to load ${methodology} leaderboard:`, error);
    }
}

function loadAllLeaderboards() {
    loadMethodologyLeaderboard('zero-shot', 'leaderboard-zero-shot-table');
    loadMethodologyLeaderboard('few-shot', 'leaderboard-few-shot-table');
    loadMethodologyLeaderboard('fully-trained', 'leaderboard-fully-trained-table');
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

function showScoreRangeModal(datasetName) {
    // Special handling for ASAP-AES to show all 8 sets
    let contentHTML = '';

    if (datasetName === 'ASAP-AES' || datasetName === 'D_ASAP-AES') {
        const asapSets = [
            { set: 1, range: '2-12', description: 'Persuasive essays about a specific topic' },
            { set: 2, range: '1-6', description: 'Persuasive essays with source material' },
            { set: 3, range: '0-3', description: 'Source-dependent responses' },
            { set: 4, range: '0-3', description: 'Source-dependent responses' },
            { set: 5, range: '0-4', description: 'Source-dependent responses' },
            { set: 6, range: '0-4', description: 'Source-dependent responses' },
            { set: 7, range: '0-30', description: 'Narrative essays' },
            { set: 8, range: '0-60', description: 'Narrative essays' }
        ];

        // Build the 8 sets breakdown for the description
        let setsHTML = '';
        asapSets.forEach(set => {
            setsHTML += `
                <div style="padding: 0.5rem; background: #f8f9fa; border-radius: 4px; margin-bottom: 0.25rem; border-left: 3px solid #1976d2;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <strong style="color: #1976d2;">Set ${set.set}</strong>
                        <span style="background: #e3f2fd; color: #1565c0; padding: 0.15rem 0.4rem; border-radius: 3px; font-weight: 600; font-size: 0.85rem;">
                            ${set.range}
                        </span>
                    </div>
                    <div style="color: #666; font-size: 0.85rem; margin-top: 0.15rem;">
                        ${set.description}
                    </div>
                </div>
            `;
        });

        contentHTML = `
            <div style="background: #e3f2fd; border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem;">
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <i class="fas fa-chart-line" style="color: #1976d2; margin-right: 0.5rem;"></i>
                    <span style="font-weight: 600; color: #1976d2;">Score Range</span>
                </div>
                <p style="font-size: 1.5rem; font-weight: bold; color: #1976d2; margin: 0;">
                    Variable by Set (8 total sets)
                </p>
            </div>
            
            <div style="margin-bottom: 1.5rem;">
                <h4 style="margin-bottom: 1rem; color: var(--text-primary);">Description</h4>
                <p style="color: var(--text-secondary); line-height: 1.5; margin-bottom: 1rem;">
                    ASAP-AES contains 8 essay sets with different scoring scales, ranging from binary classification to extended narrative scoring. Each set targets different writing skills and assessment objectives:
                </p>
                <div style="max-height: 300px; overflow-y: auto; padding-right: 0.5rem;">
                    ${setsHTML}
                </div>
            </div>
            
            <div style="background: #f5f5f5; border-radius: 8px; padding: 1rem; margin-bottom: 1.5rem;">
                <h4 style="margin-bottom: 0.5rem; color: var(--text-primary);">Task Type</h4>
                <p style="color: var(--text-secondary); margin: 0; font-size: 0.9rem;">
                    Regression Task - Holistic essay scoring with different scales per essay set
                </p>
            </div>
        `;
    } else {
        // Regular single-range display for other datasets
        const scoreInfo = getDetailedScoreInfo(datasetName);
        contentHTML = `
            <div style="background: #e3f2fd; border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem;">
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <i class="fas fa-chart-line" style="color: #1976d2; margin-right: 0.5rem;"></i>
                    <span style="font-weight: 600; color: #1976d2;">Score Range</span>
                </div>
                <p style="font-size: 1.5rem; font-weight: bold; color: #1976d2; margin: 0;">
                    ${scoreInfo.range}
                </p>
            </div>
            
            <div style="margin-bottom: 1.5rem;">
                <h4 style="margin-bottom: 0.5rem; color: var(--text-primary);">Description</h4>
                <p style="color: var(--text-secondary); line-height: 1.5; margin: 0;">
                    ${scoreInfo.description}
                </p>
            </div>
            
            <div style="background: #f5f5f5; border-radius: 8px; padding: 1rem; margin-bottom: 1.5rem;">
                <h4 style="margin-bottom: 0.5rem; color: var(--text-primary);">Task Type</h4>
                <p style="color: var(--text-secondary); margin: 0; font-size: 0.9rem;">
                    ${scoreInfo.taskType}
                </p>
            </div>
        `;
    }

    // Create modal HTML (rest of your modal code stays the same)
    const modalHTML = `
        <div class="score-range-modal-backdrop" onclick="closeScoreRangeModal()" style="
            position: fixed; top: 0; left: 0; width: 100%; height: 100%; 
            background: rgba(0,0,0,0.5); display: flex; align-items: center; 
            justify-content: center; z-index: 10000;
        ">
            <div class="score-range-modal" onclick="event.stopPropagation()" style="
                background: white; max-width: 600px; width: 90%; 
                border-radius: 12px; padding: 2rem; box-shadow: 0 20px 40px rgba(0,0,0,0.15);
                animation: modalFadeIn 0.3s ease; max-height: 80vh; overflow-y: auto;
            ">
                <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 1.5rem;">
                    <h3 style="margin: 0; color: var(--text-primary);">
                        Score Range: ${getDisplayName(datasetName)}
                    </h3>
                    <button onclick="closeScoreRangeModal()" style="
                        background: none; border: none; font-size: 1.5rem; 
                        color: var(--text-secondary); cursor: pointer; padding: 0;
                        width: 30px; height: 30px; display: flex; align-items: center; justify-content: center;
                    ">&times;</button>
                </div>
                
                ${contentHTML}
                
                <button onclick="closeScoreRangeModal()" style="
                    width: 100%; background: var(--primary-blue); color: white; 
                    border: none; padding: 0.75rem 1rem; border-radius: 6px; 
                    cursor: pointer; font-weight: 600;
                ">Close</button>
            </div>
        </div>
    `;

    const modalContainer = document.createElement('div');
    modalContainer.id = 'score-range-modal-container';
    modalContainer.innerHTML = modalHTML;
    document.body.appendChild(modalContainer);
}

function closeScoreRangeModal() {
    const modalContainer = document.getElementById('score-range-modal-container');
    if (modalContainer) {
        modalContainer.remove();
    }
}

// Add CSS animation for modal fade-in effect (add this to your CSS file or in a <style> tag)
const modalCSS = `
@keyframes modalFadeIn {
    from {
        opacity: 0;
        transform: scale(0.9);
    }
    to {
        opacity: 1;
        transform: scale(1);
    }
}
`;

// Inject the CSS if it doesn't exist
if (!document.querySelector('#modal-animations')) {
    const style = document.createElement('style');
    style.id = 'modal-animations';
    style.textContent = modalCSS;
    document.head.appendChild(style);
}

function getDetailedScoreInfo(datasetName) {
    const scoreDetails = {
        'D_ASAP-AES': {
            range: '2-12 (Variable by set)',
            description: 'ASAP-AES contains multiple essay sets with different scoring scales. Set 1 uses 2-12 for persuasive essays, while other sets range from 0-3 for source-dependent responses up to 0-60 for narrative essays.',
            taskType: 'Regression Task - Holistic essay scoring with different scales per essay set'
        },
        'D_ASAP2': {
            range: '0-4',
            description: 'ASAP 2.0 Dataset for automated essay scoring with holistic scores from 0 to 4.',
            taskType: 'Regression Task - Holistic essay scoring'
        },
        'D_ASAP-SAS': {
            range: '0-3',
            description: 'ASAP Short Answer Scoring dataset with scores ranging from 0 to 3 for short response evaluation.',
            taskType: 'Regression Task - Short answer scoring'
        },
        'D_ASAP_plus_plus': {
            range: '0-60',
            description: 'ASAP++ Enhanced Dataset with extended scoring range from 0 to 60 for comprehensive essay evaluation.',
            taskType: 'Regression Task - Enhanced essay scoring with extended range'
        },
        'D_CSEE': {
            range: '0-16',
            description: 'Computer Science Essay Evaluation Dataset with scores from 0 to 16 for CS-related essays.',
            taskType: 'Regression Task - Computer science essay evaluation'
        },
        'D_BEEtlE_2way': {
            range: 'Correct/Incorrect',
            description: 'Binary classification for student answers in science education: Incorrect (0) for wrong answers and Correct (1) for accurate responses.',
            taskType: 'Classification Task - Binary categories'
        },
        'D_BEEtlE_3way': {
            range: 'Correct/Incorrect/Contradictory',
            description: 'Three-way classification for student answers: Incorrect (0) for wrong answers, Contradictory (1) for answers that contradict known facts, and Correct (2) for accurate responses.',
            taskType: 'Classification Task - Three discrete categories'
        },
        'D_SciEntSBank_2way': {
            range: 'Correct/Incorrect',
            description: 'Binary classification for science answers: Incorrect (0) for wrong answers and Correct (1) for accurate responses.',
            taskType: 'Classification Task - Binary science answer evaluation'
        },
        'D_SciEntSBank_3way': {
            range: 'Correct/Incorrect/Contradictory',
            description: 'Three-way classification for science answers: Incorrect (0) for wrong answers, Contradictory (1) for contradictory responses, and Correct (2) for accurate answers.',
            taskType: 'Classification Task - Three-way science answer evaluation'
        },
        'D_Mohlar': {
            range: '0-5',
            description: 'Automatic Short Answer Grading dataset with scores from 0 to 5 for short answer evaluation.',
            taskType: 'Regression Task - Short answer grading'
        },
        'D_Ielts_Writing_Dataset': {
            range: '1-9',
            description: 'IELTS Writing Assessment Dataset with band scores from 1 to 9 following official IELTS scoring criteria.',
            taskType: 'Regression Task - IELTS band scoring'
        },
        'D_Ielst_Writing_Task_2_Dataset': {
            range: '1-9',
            description: 'IELTS Writing Task 2 Dataset with band scores from 1 to 9 for argumentative and discursive essays.',
            taskType: 'Regression Task - IELTS Task 2 band scoring'
        },
        'D_grade_like_a_human_dataset_os_q1': {
            range: '0-19',
            description: 'Operating Systems Question 1 with comprehensive point allocation from 0 to 19. Scores reflect understanding of OS concepts with detailed grading criteria.',
            taskType: 'Regression Task - Academic grading with multiple evaluation criteria'
        },
        'D_grade_like_a_human_dataset_os_q2': {
            range: '0-16',
            description: 'Operating Systems Question 2 with point allocation from 0 to 16. Evaluates understanding of OS principles and implementation.',
            taskType: 'Regression Task - Academic grading for OS concepts'
        },
        'D_grade_like_a_human_dataset_os_q3': {
            range: '0-15',
            description: 'Operating Systems Question 3 with scoring from 0 to 15. Assesses comprehension of advanced OS topics.',
            taskType: 'Regression Task - Academic grading for advanced OS topics'
        },
        'D_grade_like_a_human_dataset_os_q4': {
            range: '0-16',
            description: 'Operating Systems Question 4 with point allocation from 0 to 16. Evaluates practical OS knowledge and application.',
            taskType: 'Regression Task - Academic grading for practical OS knowledge'
        },
        'D_grade_like_a_human_dataset_os_q5': {
            range: '0-27',
            description: 'Operating Systems Question 5 with comprehensive scoring from 0 to 27. Covers complex OS scenarios and problem-solving.',
            taskType: 'Regression Task - Academic grading for complex OS scenarios'
        },
        'D_persuade_2': {
            range: '1-6',
            description: 'Persuasive Essays Dataset v2 with holistic scores from 1 to 6 for argumentative and persuasive writing.',
            taskType: 'Regression Task - Persuasive essay evaluation'
        },
        'D_Regrading_Dataset_J2C': {
            range: '0-8',
            description: 'Regrading Dataset J2C with scores from 0 to 8 for student answer evaluation and grading consistency studies.',
            taskType: 'Regression Task - Grading consistency evaluation'
        },
        'D_Rice_Chem_Q1': {
            range: '0-8',
            description: 'Rice Chemistry Question 1 with detailed rubric scoring from 0 to 8. Points awarded for correct chemical concepts, proper reasoning, and complete explanations.',
            taskType: 'Regression Task - Partial credit scoring with detailed chemistry rubric'
        },
        'D_Rice_Chem_Q2': {
            range: '0-8',
            description: 'Rice Chemistry Question 2 with scoring from 0 to 8. Evaluates understanding of chemical principles and problem-solving approaches.',
            taskType: 'Regression Task - Chemistry problem-solving evaluation'
        },
        'D_Rice_Chem_Q3': {
            range: '0-9',
            description: 'Rice Chemistry Question 3 with point allocation from 0 to 9. Assesses advanced chemistry concepts and analytical thinking.',
            taskType: 'Regression Task - Advanced chemistry concept evaluation'
        },
        'D_Rice_Chem_Q4': {
            range: '0-8',
            description: 'Rice Chemistry Question 4 with scoring from 0 to 8. Evaluates chemical reasoning and application of theoretical knowledge.',
            taskType: 'Regression Task - Chemical reasoning and application assessment'
        }
    };

    const info = scoreDetails[datasetName];
    if (info) return info;

    // Fallback for datasets not in detailed list
    const basicRange = getScoreRange(datasetName);
    const isClassification = basicRange.includes('/') || basicRange.toLowerCase().includes('correct');

    return {
        range: basicRange,
        description: `This dataset uses ${isClassification ? 'categorical scoring' : 'numerical scoring'} for evaluation. Please refer to the dataset documentation for detailed scoring criteria.`,
        taskType: isClassification ? 'Classification Task - Discrete categories' : 'Regression Task - Numerical scoring'
    };
}

const style = document.createElement('style');
style.textContent = `
@keyframes modalFadeIn {
    from { opacity: 0; transform: scale(0.9); }
    to { opacity: 1; transform: scale(1); }
}
`;
document.head.appendChild(style);

document.addEventListener('DOMContentLoaded', async function () {
    console.log('🚀 Enhanced S-GRADES Platform Initialized');

    try {
        setupSmoothScrolling();
        setupEventListeners();

        console.log('📊 Loading datasets and initializing platform...');

        await loadAvailableDatasets();

        // Only call functions that exist and wrap in try-catch
        try {
            if (typeof setupEnhancedUpload === 'function') {
                setupEnhancedUpload();
            }
        } catch (error) {
            console.warn('setupEnhancedUpload failed:', error);
        }

        try {
            if (typeof setupBenchmarkSubmission === 'function') {
                setupBenchmarkSubmission();
            }
        } catch (error) {
            console.warn('setupBenchmarkSubmission failed:', error);
        }

        try {
            if (typeof setupIndividualTesting === 'function') {
                setupIndividualTesting();
            }
        } catch (error) {
            console.warn('setupIndividualTesting failed:', error);
        }

        console.log('✅ Enhanced S-GRADES Platform ready!');

    } catch (error) {
        console.error('❌ Failed to initialize platform:', error);

        if (elements.datasetsGrid) {
            showError(elements.datasetsGrid, 'Failed to load platform data. Some features may not work correctly.');
        }
    }

    // Always try to load essential data, regardless of initialization errors
    try {
        loadPlatformStats();
        loadDatasets();
        loadLeaderboard();
        if (typeof loadAllLeaderboards === 'function') {
            loadAllLeaderboards();
        }
    } catch (error) {
        console.error('Failed to load essential data:', error);
    }

    // Auto-refresh interval
    setInterval(() => {
        console.log('🔄 Auto-refreshing data...');
        try {
            loadPlatformStats();
            loadLeaderboard();
        } catch (error) {
            console.error('Auto-refresh failed:', error);
        }
    }, 60000);
});

window.downloadSingleDataset = (datasetName) => {
    downloadFile(`${API_BASE_URL}/api/datasets/download/${encodeURIComponent(datasetName)}`, `${datasetName}_splits.zip`);
};

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

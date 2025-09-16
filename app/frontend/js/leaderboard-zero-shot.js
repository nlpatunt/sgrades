// Zero-shot Leaderboard JavaScript
const API_BASE_URL = '';

// DOM Elements
const elements = {
    leaderboardTable: document.getElementById('leaderboard-zero-shot-table'),
    metricSelector: document.getElementById('metric-selector'),
    refreshLeaderboard: document.getElementById('refresh-leaderboard')
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

// ===== Utility Functions for Display =====
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

// ===== Load Zero-shot Leaderboard =====
async function loadZeroShotLeaderboard() {
    if (!elements.leaderboardTable) return;
    
    showLoading(elements.leaderboardTable, 'Loading zero-shot leaderboard...');
    
    try {
        const metric = elements.metricSelector?.value || 'avg_quadratic_weighted_kappa';
        const data = await fetchAPI(`/api/submissions/leaderboard?methodology=zero-shot&metric=${metric}&min_datasets=20&limit=20`);
        
        console.log('Debug: API response:', data);
        
        if (!data || !data.rankings || data.rankings.length === 0) {
            const actionHtml = '<a href="index.html#submit" class="btn btn-primary"><i class="fas fa-rocket"></i> Submit Zero-shot Benchmark</a>';
            showEmptyState(
                elements.leaderboardTable,
                'No Zero-shot Benchmarks Yet',
                'Be the first to submit a zero-shot benchmark to appear on this leaderboard!',
                actionHtml
            );
            return;
        }
        
        // Build leaderboard table
        buildLeaderboardTable(data, elements.leaderboardTable);
        
        console.log('Success: Loaded zero-shot leaderboard with ' + data.rankings.length + ' entries');
        
    } catch (error) {
        showError(elements.leaderboardTable, 'Failed to load zero-shot leaderboard. Please check the API connection.');
        console.error('Failed to load zero-shot leaderboard:', error);
    }
}

// ===== Build Leaderboard Table =====
function buildLeaderboardTable(data, tableElement) {
    // Build table with proper structure
    var tableHTML = '<div style="margin-bottom: 1rem; padding: 1rem; background: var(--bg-light); border-radius: 8px;">';
    tableHTML += '<p><strong>Zero-shot Models:</strong> These models use only prompts without any training on target datasets.</p>';
    tableHTML += '</div>';
    
    // Proper overflow container with table
    tableHTML += '<div style="overflow-x: auto; -webkit-overflow-scrolling: touch;">';
    tableHTML += '<table style="width: 100%; min-width: 1400px; border-collapse: collapse; font-size: 0.9rem;">';
    
    // Complete table header
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
    
    // Table body
    tableHTML += '<tbody>';
    
    data.rankings.forEach((entry, index) => {
        const rank = entry.rank || (index + 1);
        const rankClass = getRankClass(rank);
        
        // Extract all 6 metrics
        const qwk = entry.avg_quadratic_weighted_kappa || 0;
        const pearson = entry.avg_pearson_correlation || 0;
        const f1 = entry.avg_f1_score || 0;
        const precision = entry.avg_precision || 0;
        const recall = entry.avg_recall || 0;
        const mae = entry.avg_mean_absolute_error || 0;
        
        // Style classes for each metric
        const qwkClass = getScoreClass(qwk);
        const pearsonClass = getScoreClass(pearson);
        const f1Class = getScoreClass(f1);
        const precisionClass = getScoreClass(precision);
        const recallClass = getScoreClass(recall);
        const maeClass = getScoreClass(mae < 0.5 ? 0.8 : mae < 1.0 ? 0.6 : 0.3);
        
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
        const totalDatasetsCount = 25; // This should match your actual count
        
        // Build complete table row
        tableHTML += '<tr style="border-bottom: 1px solid var(--border-light);">';
        tableHTML += '<td style="padding: 1rem 0.75rem; text-align: center;"><div class="rank-badge ' + rankClass + '">' + rank + '</div></td>';
        tableHTML += '<td style="padding: 1rem 0.75rem; color: var(--text-primary);"><strong>' + submitterName + '</strong></td>';
        tableHTML += '<td style="padding: 1rem 0.75rem; color: var(--text-secondary); font-size: 0.9em;">' + modelName + '</td>';
        
        // All 6 metric columns
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
    });
    
    tableHTML += '</tbody>';
    tableHTML += '</table>';
    tableHTML += '</div>'; // Close overflow container
    
    // Set the complete HTML
    tableElement.innerHTML = tableHTML;
}

// ===== Event Listeners =====
function setupEventListeners() {
    if (elements.metricSelector) {
        elements.metricSelector.addEventListener('change', loadZeroShotLeaderboard);
    }
    
    if (elements.refreshLeaderboard) {
        elements.refreshLeaderboard.addEventListener('click', () => {
            console.log('Manual leaderboard refresh requested...');
            loadZeroShotLeaderboard();
            
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

// ===== Initialize =====
document.addEventListener('DOMContentLoaded', function() {
    console.log('Zero-shot Leaderboard Page Initialized');
    setupEventListeners();
    loadZeroShotLeaderboard();
    
    // Auto-refresh every 5 minutes
    setInterval(loadZeroShotLeaderboard, 300000);
});
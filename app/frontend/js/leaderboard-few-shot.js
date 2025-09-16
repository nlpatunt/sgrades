// Few-shot Leaderboard JavaScript
const API_BASE_URL = '';

// DOM Elements
const elements = {
    leaderboardTable: document.getElementById('leaderboard-few-shot-table'),
    metricSelector: document.getElementById('metric-selector'),
    refreshLeaderboard: document.getElementById('refresh-leaderboard')
};

// API Functions
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

// Utility Functions
function showLoading(element, message = 'Loading...') {
    if (!element) return;
    element.innerHTML = '<div class="loading"><i class="fas fa-spinner fa-spin"></i> ' + message + '</div>';
}

function showError(element, message) {
    if (!element) return;
    element.innerHTML = '<div class="error" style="text-align: center; padding: 3rem; color: var(--error);">' +
        '<i class="fas fa-exclamation-triangle" style="font-size: 2rem; margin-bottom: 1rem; display: block;"></i>' +
        '<h3>Error Loading Data</h3><p>' + message + '</p></div>';
}

function showEmptyState(element, title, message, actionHtml = '') {
    if (!element) return;
    element.innerHTML = '<div class="empty-state" style="text-align: center; padding: 4rem; color: var(--text-secondary);">' +
        '<i class="fas fa-inbox" style="font-size: 3rem; margin-bottom: 1rem; display: block; opacity: 0.5;"></i>' +
        '<h3 style="margin-bottom: 1rem; color: var(--text-primary);">' + title + '</h3>' +
        '<p style="margin-bottom: 2rem;">' + message + '</p>' + actionHtml + '</div>';
}

// Load Few-shot Leaderboard
async function loadFewShotLeaderboard() {
    if (!elements.leaderboardTable) return;
    
    showLoading(elements.leaderboardTable, 'Loading few-shot leaderboard...');
    
    try {
        const metric = elements.metricSelector?.value || 'avg_quadratic_weighted_kappa';
        const data = await fetchAPI('/api/submissions/leaderboard?methodology=few-shot&metric=' + metric + '&limit=20');
        
        if (!data || !data.rankings || data.rankings.length === 0) {
            const actionHtml = '<a href="index.html#submit" class="btn btn-primary"><i class="fas fa-rocket"></i> Submit Few-shot Benchmark</a>';
            showEmptyState(
                elements.leaderboardTable,
                'No Few-shot Benchmarks Yet',
                'Be the first to submit a few-shot benchmark to appear on this leaderboard!',
                actionHtml
            );
            return;
        }
        
        // Build leaderboard table (reuse your existing table building logic)
        buildLeaderboardTable(data, elements.leaderboardTable);
        
    } catch (error) {
        showError(elements.leaderboardTable, 'Failed to load few-shot leaderboard. Please check the API connection.');
        console.error('Failed to load few-shot leaderboard:', error);
    }
}

// Table building function (copy from your main app.js)
function buildLeaderboardTable(data, tableElement) {
    // Copy your existing table building logic from app.js
    // This should be the same logic from your loadLeaderboard function
    // but applied to the few-shot specific data
}

// Event Listeners
function setupEventListeners() {
    if (elements.metricSelector) {
        elements.metricSelector.addEventListener('change', loadFewShotLeaderboard);
    }
    
    if (elements.refreshLeaderboard) {
        elements.refreshLeaderboard.addEventListener('click', () => {
            loadFewShotLeaderboard();
            const button = elements.refreshLeaderboard;
            const icon = button.querySelector('i');
            if (icon) {
                icon.classList.add('fa-spin');
                setTimeout(() => icon.classList.remove('fa-spin'), 1000);
            }
        });
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    console.log('Few-shot Leaderboard Page Initialized');
    setupEventListeners();
    loadFewShotLeaderboard();
    
    // Auto-refresh every 5 minutes
    setInterval(loadFewShotLeaderboard, 300000);
});
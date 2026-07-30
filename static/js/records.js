// Escape before interpolating anything user-supplied into innerHTML.
// Filenames are chosen by whoever uploads the file and are stored verbatim,
// so a file named e.g. `<img src=x onerror=...>.jpg` was executing in every
// viewer's browser -- including an admin's, who sees every user's rows.
// Container codes and usernames go through the same path.
function esc(value) {
    return String(value == null ? '' : value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function toIST(isoString) {
    if (!isoString) return '';
    return new Date(isoString).toLocaleString('en-IN', {
        timeZone: 'Asia/Kolkata',
        year: 'numeric', month: 'short', day: 'numeric',
        hour: '2-digit', minute: '2-digit', second: '2-digit',
        hour12: false
    }) + ' IST';
}

function toISTDate(isoString) {
    if (!isoString) return '';
    return new Date(isoString).toLocaleDateString('en-IN', {
        timeZone: 'Asia/Kolkata',
        year: 'numeric', month: 'short', day: 'numeric'
    });
}

const recordsConfig = {
    apiBaseUrl: '/main',
    pageSize: 10,
    currentPage: 1,
    totalRecords: 0,
    allRecords: [],
    currentSort: { column: null, direction: 'asc' },
    selectedRecords: []
};

function initRecordsPage() {
    loadRecords();
    loadUserFilterOptions();
    setupEventListeners();
}

function loadUserFilterOptions() {
    fetch(`${recordsConfig.apiBaseUrl}/users`)
        .then(response => response.json())
        .then(data => {
            const userSelect = document.getElementById('userFilter');
            (data.users || []).forEach(u => {
                const option = document.createElement('option');
                option.value = u.id;
                option.textContent = u.username;
                userSelect.appendChild(option);
            });
        })
        .catch(error => {
            console.error('Error loading users for filter:', error);
        });
}

function loadRecords() {
    showLoading(true);
    const url = `${recordsConfig.apiBaseUrl}/records`;
    
    fetch(url)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showError(data.error);
                return;
            }
            
            recordsConfig.allRecords = data.records || [];
            recordsConfig.totalRecords = recordsConfig.allRecords.length;
            applyFiltersAndPagination();
            showLoading(false);
        })
        .catch(error => {
            showError('Failed to load records');
            console.error('Error loading records:', error);
            showLoading(false);
        });
}

function applyFiltersAndPagination() {
    let filteredRecords = [...recordsConfig.allRecords];
    
    filteredRecords = _applyFilters(filteredRecords);
    filteredRecords = sortRecords(filteredRecords);
    
    recordsConfig.totalRecords = filteredRecords.length;
    const startIndex = (recordsConfig.currentPage - 1) * recordsConfig.pageSize;
    const endIndex = startIndex + recordsConfig.pageSize;
    const paginatedRecords = filteredRecords.slice(startIndex, endIndex);
    
    renderRecordsTable(paginatedRecords);
    renderPagination(filteredRecords.length);
    updateTableInfo(filteredRecords.length);
}

function _applyFilters(records) {
    const searchInput = document.getElementById('searchInput').value.toLowerCase();
    const dateFrom = document.getElementById('dateFrom').value;
    const dateTo = document.getElementById('dateTo').value;
    const containerCode = (document.getElementById('containerCodeInput') || {}).value || '';
    const minConfEl = document.getElementById('minConfidence');
    const minConf = minConfEl && minConfEl.value !== '' ? parseFloat(minConfEl.value) / 100 : null;
    const userFilterEl = document.getElementById('userFilter');
    const userFilter = userFilterEl && userFilterEl.value !== 'all' ? userFilterEl.value : null;

    return records.filter(record => {
        const matchesSearch = searchInput === '' || record.filename.toLowerCase().includes(searchInput);
        const matchesDateFrom = !dateFrom || new Date(record.created_at) >= new Date(dateFrom);
        const matchesDateTo = !dateTo || new Date(record.created_at) <= new Date(dateTo + 'T23:59:59');
        const matchesCode = containerCode === '' ||
            (record.container_codes || []).some(c => c.toLowerCase().includes(containerCode.toLowerCase()));
        const matchesConf = minConf === null ||
            (record.max_confidence !== undefined && record.max_confidence >= minConf);
        const matchesUser = userFilter === null || String(record.user_id) === userFilter;
        return matchesSearch && matchesDateFrom && matchesDateTo && matchesCode && matchesConf && matchesUser;
    });
}

function sortRecords(records) {
    const sortSelect = document.getElementById('sortSelect').value;
    let sorted = [...records];
    
    switch (sortSelect) {
        case 'created_at_desc':
            sorted.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
            break;
        case 'created_at_asc':
            sorted.sort((a, b) => new Date(a.created_at) - new Date(b.created_at));
            break;
        case 'filename_asc':
            sorted.sort((a, b) => a.filename.localeCompare(b.filename));
            break;
        case 'filename_desc':
            sorted.sort((a, b) => b.filename.localeCompare(a.filename));
            break;
    }
    
    return sorted;
}

function renderRecordsTable(records) {
    const tbody = document.getElementById('recordsBody');
    
    if (records.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="8" class="empty-state">
                    <i class="fas fa-inbox"></i> No records found
                </td>
            </tr>
        `;
        return;
    }
    
    tbody.innerHTML = records.map(record => {
        const createdDate = toISTDate(record.created_at);
        const formattedSize = formatFileSize(record.image_size);
        const status = 'completed';
        
        return `
            <tr>
                <td><input type="checkbox" class="record-checkbox" value="${record.id}" onchange="toggleRecordSelection(this)"></td>
                <td>${esc(record.filename)}</td>
                <td>${esc(createdDate)}</td>
                <td>${esc(formattedSize)}</td>
                <td>
                    ${record.detections_count || 0}
                    ${(record.container_codes || []).filter(Boolean).length
                        ? `<br><small>${esc(record.container_codes.filter(Boolean).join(', '))}</small>`
                        : ''}
                </td>
                <td>${esc(record.username || 'Unknown')}</td>
                <td><span class="status-badge status-${status}">${status}</span></td>
                <td>
                    <button class="table-action-btn" onclick="viewRecordDetails(${record.id})">View</button>
                    <button class="table-action-btn" onclick="saveToDatabase(${record.id})">Save</button>
                </td>
            </tr>
        `;
    }).join('');
}

function renderPagination(totalFiltered) {
    const pagination = document.getElementById('pagination');
    const totalPages = Math.ceil(totalFiltered / recordsConfig.pageSize);
    
    if (totalPages <= 1) {
        pagination.innerHTML = '';
        return;
    }
    
    let paginationHTML = '';
    
    paginationHTML += `
        <button class="pagination-btn" ${recordsConfig.currentPage === 1 ? 'disabled' : ''} onclick="changePage(${recordsConfig.currentPage - 1})">
            Previous
        </button>
    `;
    
    const startPage = Math.max(1, recordsConfig.currentPage - 2);
    const endPage = Math.min(totalPages, startPage + 4);
    
    for (let i = startPage; i <= endPage; i++) {
        paginationHTML += `
            <button class="pagination-btn ${i === recordsConfig.currentPage ? 'active' : ''}" onclick="changePage(${i})">${i}</button>
        `;
    }
    
    paginationHTML += `
        <button class="pagination-btn" ${recordsConfig.currentPage === totalPages ? 'disabled' : ''} onclick="changePage(${recordsConfig.currentPage + 1})">
            Next
        </button>
    `;
    
    pagination.innerHTML = paginationHTML;
}

function changePage(page) {
    recordsConfig.currentPage = page;
    applyFiltersAndPagination();
}

function updateTableInfo(totalFiltered) {
    const start = (recordsConfig.currentPage - 1) * recordsConfig.pageSize + 1;
    const end = Math.min(start + recordsConfig.pageSize - 1, totalFiltered);
    
    if (totalFiltered === 0) {
        return;
    }
    
    const info = document.querySelector('.table-header .table-title');
    info.textContent = `Showing ${start}-${end} of ${totalFiltered} records`;
}

function toggleSelectAll() {
    const selectAllCheckbox = document.getElementById('selectAllCheckbox');
    const recordCheckboxes = document.querySelectorAll('.record-checkbox');
    
    recordCheckboxes.forEach(checkbox => {
        checkbox.checked = selectAllCheckbox.checked;
        const recordId = parseInt(checkbox.value);
        
        if (selectAllCheckbox.checked) {
            if (!recordsConfig.selectedRecords.includes(recordId)) {
                recordsConfig.selectedRecords.push(recordId);
            }
        } else {
            const index = recordsConfig.selectedRecords.indexOf(recordId);
            if (index > -1) {
                recordsConfig.selectedRecords.splice(index, 1);
            }
        }
    });
}

function toggleRecordSelection(checkbox) {
    const recordId = parseInt(checkbox.value);
    
    if (checkbox.checked) {
        if (!recordsConfig.selectedRecords.includes(recordId)) {
            recordsConfig.selectedRecords.push(recordId);
        }
    } else {
        const index = recordsConfig.selectedRecords.indexOf(recordId);
        if (index > -1) {
            recordsConfig.selectedRecords.splice(index, 1);
        }
    }
    
    const selectAllCheckbox = document.getElementById('selectAllCheckbox');
    const allCheckboxes = document.querySelectorAll('.record-checkbox');
    const allChecked = Array.from(allCheckboxes).every(cb => cb.checked);
    selectAllCheckbox.checked = allChecked;
}

function viewRecordDetails(recordId) {
    const url = `${recordsConfig.apiBaseUrl}/records/${recordId}`;
    
    fetch(url)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showError(data.error);
                return;
            }
            
            renderRecordModal(data.record);
            showModal();
        })
        .catch(error => {
            showError('Failed to load record details');
            console.error('Error loading record:', error);
        });
}

function renderRecordModal(record) {
    const modalContent = document.getElementById('modalContent');

    const detectionsHtml = (record.detections || []).map((d, i) => `
        <div class="detail-item">
            <div class="detail-label">Detection ${i + 1}:</div>
            <div class="detail-value">
                Container Code: ${esc(d.container_code || 'Not entered')}<br>
                Confidence: ${d.confidence != null ? (d.confidence * 100).toFixed(1) + '%' : 'N/A'}
            </div>
        </div>
    `).join('') || '<div class="detail-item"><div class="detail-value">No detections</div></div>';

    modalContent.innerHTML = `
        <div class="detail-section">
            <h4 class="section-title">Record Information</h4>
            <div class="detail-item">
                <div class="detail-label">Upload ID:</div>
                <div class="detail-value">${record.id}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">Filename:</div>
                <div class="detail-value">${esc(record.filename)}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">Image Size:</div>
                <div class="detail-value">${esc(record.image_size)}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">Created At:</div>
                <div class="detail-value">${esc(toIST(record.created_at))}</div>
            </div>
            <div class="detail-section" style="margin-top:12px;">
                <h4 class="section-title">Detections</h4>
                ${detectionsHtml}
            </div>
            ${record.annotated_image_url ? `
            <div class="detail-item" style="display:block; margin-top:10px;">
                <div class="detail-label">Annotated Image:</div>
                <img src="${record.annotated_image_url}" alt="Annotated" style="max-width:100%; margin-top:6px; border:1px solid #ddd; border-radius:4px;">
            </div>` : ''}
        </div>
    `;
}

function saveToDatabase(recordId) {
    const url = `${recordsConfig.apiBaseUrl}/records/${recordId}/save`;
    
    const csrfToken = document.querySelector('meta[name="csrf-token"]').content;
    fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrfToken,
        }
    })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showError(data.error);
                return;
            }
            
            showSuccess('Record saved to database successfully');
            loadRecords();
            closeModal();
        })
        .catch(error => {
            showError('Failed to save record to database');
            console.error('Error saving record:', error);
        });
}

function exportRecords() {
    if (recordsConfig.selectedRecords.length === 0) {
        showError('Please select records to export');
        return;
    }
    
    showSuccess('Exporting selected records...');
    
    const url = `${recordsConfig.apiBaseUrl}/export?ids=${recordsConfig.selectedRecords.join(',')}`;
    window.location.href = url;
    
    recordsConfig.selectedRecords = [];
    document.getElementById('selectAllCheckbox').checked = false;
}

function refreshRecords() {
    recordsConfig.currentPage = 1;
    loadRecords();
}

function filterRecords() {
    recordsConfig.currentPage = 1;
    applyFiltersAndPagination();
}

function sortTable(column) {
    if (recordsConfig.currentSort.column === column) {
        recordsConfig.currentSort.direction = recordsConfig.currentSort.direction === 'asc' ? 'desc' : 'asc';
    } else {
        recordsConfig.currentSort.column = column;
        recordsConfig.currentSort.direction = 'asc';
    }
    
    // Only filename and created_at have real sort options in #sortSelect.
    // The old code appended '_desc' onto an already-suffixed key, producing
    // 'filename_asc_desc' -- not a valid <option> value, so the select fell
    // back to '' and sortRecords()'s switch matched nothing. Net effect: the
    // second click on any column header silently disabled sorting entirely,
    // and the Size / Detections headers sorted by filename.
    const sortMap = {
        'filename': { asc: 'filename_asc', desc: 'filename_desc' },
        'created_at': { asc: 'created_at_asc', desc: 'created_at_desc' }
    };

    const option = sortMap[column];
    if (!option) {
        // No sort option exists for this column -- leave the current sort
        // alone rather than silently applying an unrelated one.
        return;
    }

    document.getElementById('sortSelect').value = option[recordsConfig.currentSort.direction];
    filterRecords();
}

function showModal() {
    document.getElementById('recordModal').style.display = 'block';
}

function closeModal() {
    document.getElementById('recordModal').style.display = 'none';
}

function showLoading(show) {
    document.getElementById('loadingSpinner').style.display = show ? 'block' : 'none';
}

function showSuccess(message) {
    alert(message);
}

function showError(message) {
    alert('Error: ' + message);
}

function formatFileSize(bytes) {
    if (!bytes) return '0 B';
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${sizes[i]}`;
}

function setupEventListeners() {
    document.getElementById('filterForm').addEventListener('input', debounce(filterRecords, 500));
    
    window.filterRecords = filterRecords;
    window.changePage = changePage;
    window.viewRecordDetails = viewRecordDetails;
    window.saveToDatabase = saveToDatabase;
    window.exportRecords = exportRecords;
    window.refreshRecords = refreshRecords;
    window.sortTable = sortTable;
    window.toggleSelectAll = toggleSelectAll;
    window.toggleRecordSelection = toggleRecordSelection;
    window.closeModal = closeModal;
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

document.addEventListener('DOMContentLoaded', initRecordsPage);

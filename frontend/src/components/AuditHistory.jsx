import React, { useState } from 'react';
import { 
  IconHistory, 
  IconShieldAlert, 
  IconShieldCheck, 
  IconTrash, 
  IconDownload, 
  IconSearch,
  IconArrowRight
} from './Icons';

export default function AuditHistory({ history = [], onClearHistory, onSelectScan }) {
  const [filter, setFilter] = useState('ALL');
  const [search, setSearch] = useState('');

  const filteredHistory = history.filter(item => {
    const matchesFilter = 
      filter === 'ALL' ? true :
      filter === 'SPAM' ? item.prediction === 'SPAM' :
      item.prediction === 'HAM';

    const matchesSearch = search.trim() === '' ||
      item.message.toLowerCase().includes(search.toLowerCase());

    return matchesFilter && matchesSearch;
  });

  const spamCount = history.filter(h => h.prediction === 'SPAM').length;
  const hamCount = history.filter(h => h.prediction === 'HAM').length;

  const exportCSV = () => {
    if (history.length === 0) return;
    const headers = ['Timestamp', 'Verdict', 'Confidence (%)', 'Spam Prob (%)', 'Ham Prob (%)', 'Latency (ms)', 'Message'];
    const rows = history.map(h => [
      h.timestamp,
      h.prediction,
      h.confidence,
      h.spamProbability || 0,
      h.hamProbability || 0,
      h.latency || 0,
      `"${h.message.replace(/"/g, '""')}"`
    ]);

    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map(e => e.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `xlstm_audit_log_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="history-container">
      <div className="section-header-row">
        <div>
          <div className="section-eyebrow">
            <IconHistory size={14} className="eyebrow-icon" />
            <span>SESSION AUDIT TRAIL</span>
          </div>
          <h2>Diagnostic Scan History</h2>
          <p>
            Continuous audit log of SMS sequences inspected during this operational session.
          </p>
        </div>

        {history.length > 0 && (
          <div className="history-top-actions">
            <button className="secondary-btn" onClick={exportCSV}>
              <IconDownload size={14} />
              <span>Export CSV</span>
            </button>
            <button className="secondary-btn text-danger" onClick={onClearHistory}>
              <IconTrash size={14} />
              <span>Clear History</span>
            </button>
          </div>
        )}
      </div>

      {history.length === 0 ? (
        <div className="history-empty-card">
          <IconHistory size={48} className="empty-icon" />
          <h3>No Scans Recorded Yet</h3>
          <p>
            Diagnose SMS messages using the Live Inspector or Batch Scanner to populate this audit trail.
          </p>
        </div>
      ) : (
        <div className="history-content-wrap">
          {/* Summary Pills */}
          <div className="history-stats-bar">
            <div className="history-stat-chip">
              <span>Total Scans:</span>
              <strong className="font-mono">{history.length}</strong>
            </div>
            <div className="history-stat-chip chip-spam">
              <span>Threats Flagged:</span>
              <strong className="font-mono spam-text">{spamCount}</strong>
            </div>
            <div className="history-stat-chip chip-ham">
              <span>Verified Clean:</span>
              <strong className="font-mono ham-text">{hamCount}</strong>
            </div>
          </div>

          {/* Filter and Search Bar */}
          <div className="table-controls-bar">
            <div className="filter-button-group">
              <button 
                className={`filter-btn ${filter === 'ALL' ? 'active' : ''}`}
                onClick={() => setFilter('ALL')}
              >
                All ({history.length})
              </button>
              <button 
                className={`filter-btn filter-spam ${filter === 'SPAM' ? 'active' : ''}`}
                onClick={() => setFilter('SPAM')}
              >
                Spam Threats ({spamCount})
              </button>
              <button 
                className={`filter-btn filter-ham ${filter === 'HAM' ? 'active' : ''}`}
                onClick={() => setFilter('HAM')}
              >
                Safe Ham ({hamCount})
              </button>
            </div>

            <div className="search-input-wrap">
              <IconSearch size={15} className="search-icon" />
              <input
                type="text"
                placeholder="Search audit entries..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
              />
            </div>
          </div>

          {/* History List */}
          <div className="history-list-card">
            {filteredHistory.map((item) => {
              const isSpam = item.prediction === 'SPAM';
              return (
                <div key={item.id} className={`history-row ${isSpam ? 'row-spam' : 'row-ham'}`}>
                  <div className="history-left">
                    <span className={`status-badge ${isSpam ? 'badge-spam' : 'badge-ham'}`}>
                      {isSpam ? <IconShieldAlert size={12} /> : <IconShieldCheck size={12} />}
                      <span>{item.prediction}</span>
                    </span>
                    <span className="history-timestamp font-mono">{item.timestamp}</span>
                  </div>

                  <div className="history-center">
                    <p className="history-message-text">{item.message}</p>
                  </div>

                  <div className="history-right">
                    <div className="history-metric-box">
                      <span className="history-metric-val font-mono">{item.confidence}%</span>
                      <span className="history-metric-label">Confidence</span>
                    </div>

                    {onSelectScan && (
                      <button 
                        className="history-reinspect-btn"
                        onClick={() => onSelectScan(item.message)}
                        title="Re-inspect in Live Inspector"
                      >
                        <span>Inspect</span>
                        <IconArrowRight size={13} />
                      </button>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

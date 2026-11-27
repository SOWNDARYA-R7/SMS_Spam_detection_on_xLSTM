import React, { useState } from 'react';
import { 
  IconZap, 
  IconShieldAlert, 
  IconShieldCheck, 
  IconDownload, 
  IconTrash, 
  IconSearch, 
  IconSparkles,
  IconAlertCircle
} from './Icons';

const SAMPLE_BATCH = [
  "URGENT: Your account has been compromised. Verify identity immediately at http://secure-bank.cc",
  "Hey mate, are we still meeting tomorrow for lunch at 1 PM?",
  "WINNER! You've been selected for a $5,000 Walmart gift card! Call 09061701461 to claim your reward.",
  "Your delivery order from Amazon has shipped and will arrive Thursday by 8pm. Track at amazon.com/orders",
  "FREE ENTRY to win a £500 cash prize! Text WIN to 80082. T&Cs apply.",
  "Mom: Can you pick up milk and bread on your way home?",
  "CRITICAL ALERT: IRS notice pending against your SSN. Call 800-441-2919 immediately to avoid arrest.",
  "Your Uber driver is arriving in a Silver Toyota Camry (License Plate: 7XYZ92)."
];

export default function BatchScanner({ onScanComplete }) {
  const [rawText, setRawText] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState([]);
  const [filter, setFilter] = useState('ALL');
  const [searchQuery, setSearchQuery] = useState('');
  const [error, setError] = useState('');

  const loadSampleBatch = () => {
    setRawText(SAMPLE_BATCH.join('\n\n'));
    setResults([]);
    setError('');
  };

  const clearAll = () => {
    setRawText('');
    setResults([]);
    setError('');
  };

  const runBatchScan = async () => {
    const lines = rawText
      .split('\n')
      .map(l => l.trim())
      .filter(l => l.length > 0);

    if (lines.length === 0) {
      setError('Please input at least one SMS message (one per line).');
      return;
    }

    setLoading(true);
    setError('');

    try {
      // First attempt fast /api/predict/batch endpoint
      const API_URL = import.meta.env.VITE_API_URL;

        const response = await fetch(`${API_URL}/api/predict/batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: lines })
      });

      if (response.ok) {
        const data = await response.json();
        setResults(data.results || []);
        
        // Notify audit history
        if (onScanComplete && data.results) {
          data.results.forEach((item, idx) => {
            onScanComplete({
              id: Date.now() + idx,
              timestamp: new Date().toLocaleTimeString(),
              message: item.message,
              prediction: item.prediction,
              confidence: item.confidence,
              spamProbability: item.spam_probability,
              hamProbability: item.ham_probability,
              latency: 12
            });
          });
        }
      } else {
        // Fallback to parallel individual calls
        const promises = lines.map(async (msg, index) => {
          const res = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: msg })
          });
          const d = await res.json();
          return {
            index,
            message: msg,
            prediction: d.prediction,
            confidence: d.confidence,
            ham_probability: d.ham_probability,
            spam_probability: d.spam_probability
          };
        });

        const batchRes = await Promise.all(promises);
        setResults(batchRes);
      }
    } catch (err) {
      setError(err.message || 'Error processing batch messages. Ensure FastAPI server is running.');
    } finally {
      setLoading(false);
    }
  };

  const exportCSV = () => {
    if (results.length === 0) return;
    const headers = ['Index', 'Classification', 'Confidence (%)', 'Spam Prob (%)', 'Ham Prob (%)', 'Message'];
    const rows = results.map(r => [
      r.index + 1,
      r.prediction,
      r.confidence,
      r.spam_probability,
      r.ham_probability,
      `"${r.message.replace(/"/g, '""')}"`
    ]);

    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map(e => e.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `xlstm_batch_scan_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Filtered Results
  const filteredResults = results.filter(item => {
    const matchesFilter = 
      filter === 'ALL' ? true : 
      filter === 'SPAM' ? item.prediction === 'SPAM' : 
      item.prediction === 'HAM';

    const matchesSearch = searchQuery.trim() === '' || 
      item.message.toLowerCase().includes(searchQuery.toLowerCase());

    return matchesFilter && matchesSearch;
  });

  const spamCount = results.filter(r => r.prediction === 'SPAM').length;
  const hamCount = results.filter(r => r.prediction === 'HAM').length;
  const threatRate = results.length > 0 ? Math.round((spamCount / results.length) * 100) : 0;

  return (
    <div className="batch-container">
      <div className="section-header-row">
        <div>
          <div className="section-eyebrow">
            <IconZap size={14} className="eyebrow-icon" />
            <span>ENTERPRISE BULK PROCESSING</span>
          </div>
          <h2>Multi-SMS Neural Batch Scanner</h2>
          <p>
            Screen high-volume text messages simultaneously. Evaluate threat levels, 
            detect campaigns, and export structured compliance audit logs.
          </p>
        </div>

        <div className="batch-actions">
          <button className="secondary-btn" onClick={loadSampleBatch}>
            <IconSparkles size={14} />
            <span>Load Sample Suite</span>
          </button>
          {rawText && (
            <button className="secondary-btn" onClick={clearAll}>
              <IconTrash size={14} />
              <span>Clear</span>
            </button>
          )}
        </div>
      </div>

      {/* Input Box */}
      <div className="batch-input-card">
        <label htmlFor="batch-textarea">Enter or paste SMS messages (separated by line breaks):</label>
        <textarea
          id="batch-textarea"
          className="batch-textarea"
          placeholder="Paste multiple SMS messages here, one per line..."
          value={rawText}
          onChange={(e) => setRawText(e.target.value)}
          rows={6}
        />

        <div className="batch-submit-row">
          <span className="batch-line-count">
            {rawText.split('\n').filter(l => l.trim().length > 0).length} message(s) queued
          </span>
          <button
            className="analyze-button"
            onClick={runBatchScan}
            disabled={loading || !rawText.trim()}
          >
            {loading ? (
              <>
                <span className="spinner" />
                <span>Processing Neural Batch...</span>
              </>
            ) : (
              <>
                <IconZap size={16} />
                <span>Run Batch Classification</span>
              </>
            )}
          </button>
        </div>

        {error && (
          <div className="error-banner">
            <IconAlertCircle size={18} />
            <p>{error}</p>
          </div>
        )}
      </div>

      {/* Results View */}
      {results.length > 0 && (
        <div className="batch-results-wrap">
          {/* Summary Metric Cards */}
          <div className="batch-stats-grid">
            <div className="kpi-card">
              <span className="kpi-label">Total Messages</span>
              <span className="kpi-value font-mono">{results.length}</span>
            </div>
            <div className="kpi-card kpi-spam">
              <span className="kpi-label">Spam Threats Detected</span>
              <span className="kpi-value font-mono spam-text">{spamCount}</span>
            </div>
            <div className="kpi-card kpi-ham">
              <span className="kpi-label">Legitimate Ham</span>
              <span className="kpi-value font-mono ham-text">{hamCount}</span>
            </div>
            <div className="kpi-card">
              <span className="kpi-label">Threat Incident Rate</span>
              <span className="kpi-value font-mono">{threatRate}%</span>
            </div>
          </div>

          {/* Table Controls */}
          <div className="table-controls-bar">
            <div className="filter-button-group">
              <button 
                className={`filter-btn ${filter === 'ALL' ? 'active' : ''}`}
                onClick={() => setFilter('ALL')}
              >
                All ({results.length})
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

            <div className="table-right-tools">
              <div className="search-input-wrap">
                <IconSearch size={15} className="search-icon" />
                <input
                  type="text"
                  placeholder="Filter messages..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
              </div>

              <button className="export-btn" onClick={exportCSV}>
                <IconDownload size={14} />
                <span>Export CSV</span>
              </button>
            </div>
          </div>

          {/* Results Table */}
          <div className="table-responsive-container">
            <table className="batch-table">
              <thead>
                <tr>
                  <th style={{ width: '60px' }}>#</th>
                  <th>SMS Content</th>
                  <th style={{ width: '130px' }}>Classification</th>
                  <th style={{ width: '130px' }}>Confidence</th>
                  <th style={{ width: '180px' }}>Probability Split</th>
                </tr>
              </thead>
              <tbody>
                {filteredResults.map((item, idx) => {
                  const isSpam = item.prediction === 'SPAM';
                  return (
                    <tr key={idx} className={isSpam ? 'row-spam' : 'row-ham'}>
                      <td className="font-mono text-muted">{idx + 1}</td>
                      <td className="message-cell">{item.message}</td>
                      <td>
                        <span className={`status-badge ${isSpam ? 'badge-spam' : 'badge-ham'}`}>
                          {isSpam ? <IconShieldAlert size={12} /> : <IconShieldCheck size={12} />}
                          <span>{item.prediction}</span>
                        </span>
                      </td>
                      <td className="font-mono font-medium">
                        {item.confidence}%
                      </td>
                      <td>
                        <div className="mini-prob-bar">
                          <div 
                            className={`mini-bar-fill ${isSpam ? 'fill-spam' : 'fill-ham'}`}
                            style={{ width: `${isSpam ? item.spam_probability : item.ham_probability}%` }}
                          />
                        </div>
                        <span className="mini-prob-subtext">
                          Spam: {item.spam_probability}% | Ham: {item.ham_probability}%
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

import React, { useState, useMemo } from 'react';
import { 
  IconShield, 
  IconShieldAlert, 
  IconShieldCheck, 
  IconSparkles, 
  IconCopy, 
  IconCheck, 
  IconTrash, 
  IconArrowRight, 
  IconAlertCircle,
  IconZap,
  IconActivity
} from './Icons';

const PRESETS = [
  {
    category: 'Spam / Phishing',
    label: 'Bank Fraud Alert',
    badge: 'Spam',
    text: 'URGENT: Your Chase bank account has been flagged for suspicious activity. Verify identity immediately at http://chase-security-verify.cc to prevent permanent suspension.'
  },
  {
    category: 'Spam / Phishing',
    label: 'Package On Hold',
    badge: 'Spam',
    text: 'USPS Notice: Your parcel package #US88219 could not be delivered due to missing house number. Please update info at bit.ly/usps-track-now within 12 hours.'
  },
  {
    category: 'Spam / Phishing',
    label: 'Lottery Prize Claim',
    badge: 'Spam',
    text: 'Congratulations! You have been selected as our lucky cash winner of $1,000,000! Text CLAIM to 88484 now to receive your transfer wire.'
  },
  {
    category: 'Legitimate / Ham',
    label: 'Friend Casual Chat',
    badge: 'Ham',
    text: 'Hey Sarah! Are we still meeting tomorrow for coffee at 10 AM at the cafe downtown? Let me know if that time works for you.'
  },
  {
    category: 'Legitimate / Ham',
    label: '2FA Security Code',
    badge: 'Ham',
    text: 'Your verification code for Google Account is 749210. Valid for 10 minutes. Do not share this code with anyone.'
  },
  {
    category: 'Legitimate / Ham',
    label: 'Doctor Appointment',
    badge: 'Ham',
    text: 'Reminder: Your dental appointment with Dr. Watson is scheduled for Thursday, Dec 4th at 2:30 PM. Reply 1 to confirm or 2 to reschedule.'
  }
];

const SUSPICIOUS_KEYWORDS = [
  'urgent', 'congratulations', 'winner', 'free', 'prize', 'claim', 'cash',
  'suspended', 'verify', 'account', 'password', 'wire', 'transfer', 'call now',
  'click here', 'limited time', 'selected', 'won', 'lottery', 'crypto', 'bonus'
];

export default function LiveInspector({ onScanComplete }) {
  const [message, setMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [copied, setCopied] = useState(false);
  const [scanLatency, setScanLatency] = useState(null);

  // Live Heuristic Evaluation
  const heuristics = useMemo(() => {
    if (!message.trim()) return null;

    const lower = message.toLowerCase();
    const matchedKeywords = SUSPICIOUS_KEYWORDS.filter(kw => lower.includes(kw));
    const urlPattern = /(https?:\/\/[^\s]+|www\.[^\s]+|[a-zA-Z0-9-]+\.(cc|xyz|tk|top|info|ru|live)[^\s]*)/gi;
    const urls = message.match(urlPattern) || [];
    const phonePattern = /(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\b\d{5,6}\b/g;
    const phoneMatches = message.match(phonePattern) || [];

    // Caps ratio
    const letters = message.replace(/[^a-zA-Z]/g, '');
    const caps = message.replace(/[^A-Z]/g, '');
    const capsRatio = letters.length > 0 ? Math.round((caps.length / letters.length) * 100) : 0;

    return {
      keywords: matchedKeywords,
      urls,
      phoneMatches,
      capsRatio,
      charCount: message.length,
      wordCount: message.trim().split(/\s+/).filter(Boolean).length
    };
  }, [message]);

  const runPrediction = async (textToScan = message) => {
    const target = textToScan.trim();
    if (!target) {
      setError('Please provide an SMS message to evaluate.');
      return;
    }

    setLoading(true);
    setError('');
    const startTime = performance.now();

    try {
      const API_URL = import.meta.env.VITE_API_URL;
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: target })
      });

      const data = await response.json();
      const elapsed = Math.round(performance.now() - startTime);
      setScanLatency(elapsed);

      if (!response.ok) {
        throw new Error(data.detail || 'Prediction failed');
      }

      setResult(data);

      if (onScanComplete) {
        onScanComplete({
          id: Date.now(),
          timestamp: new Date().toLocaleTimeString(),
          message: target,
          prediction: data.prediction,
          confidence: data.confidence,
          spamProbability: data.spam_probability,
          hamProbability: data.ham_probability,
          latency: elapsed
        });
      }
    } catch (err) {
      setError(err.message || 'Cannot reach API. Ensure FastAPI server is running on port 8000.');
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      runPrediction();
    }
  };

  const loadPreset = (preset) => {
    setMessage(preset.text);
    setResult(null);
    setError('');
    runPrediction(preset.text);
  };

  const handleCopyReport = () => {
    if (!result) return;
    const report = `xLSTM Sentinel Diagnosis Report:
Prediction: ${result.prediction} (${result.confidence}% confidence)
Spam Probability: ${result.spam_probability}%
Ham Probability: ${result.ham_probability}%
Message: "${message}"`;
    navigator.clipboard.writeText(report);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleClear = () => {
    setMessage('');
    setResult(null);
    setError('');
    setScanLatency(null);
  };

  return (
    <div className="inspector-container">
      {/* Top Banner / Hero */}
      <div className="inspector-hero">
        <div className="hero-text-block">
          <div className="section-eyebrow">
            <IconSparkles size={14} className="eyebrow-icon" />
            <span>DEEP SEQUENCE CLASSIFICATION</span>
          </div>
          <h2>Live SMS Neural Threat Inspector</h2>
          <p>
            Evaluate incoming text messages through our trained 4-block xLSTM model 
            with matrix memory (mLSTM) and scalar memory (sLSTM) sequence reasoning.
          </p>
        </div>

        {/* Quick attack sample chips */}
        <div className="preset-strip">
          <span className="preset-label">Test Case Library:</span>
          <div className="preset-buttons">
            {PRESETS.map((p, idx) => (
              <button
                key={idx}
                type="button"
                className={`preset-chip ${p.badge === 'Spam' ? 'chip-spam' : 'chip-ham'}`}
                onClick={() => loadPreset(p)}
                title={p.text}
              >
                <span className="preset-dot" />
                <span className="preset-name">{p.label}</span>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main Two-Column Layout */}
      <div className="inspector-grid">
        {/* Left Column: Input & Heuristics */}
        <div className="inspector-card input-card">
          <div className="card-header">
            <div className="header-title-wrap">
              <span className="step-number">01</span>
              <h3>Message Input & Heuristic Pre-scan</h3>
            </div>
            {message && (
              <button className="text-action-btn" onClick={handleClear}>
                <IconTrash size={14} /> Clear
              </button>
            )}
          </div>

          <div className="textarea-wrapper">
            <textarea
              className="message-textarea"
              placeholder="Paste or type SMS message sequence here... (e.g. 'URGENT: Your account has been compromised. Verify at http://...')"
              value={message}
              onChange={(e) => {
                setMessage(e.target.value);
                if (result) setResult(null);
                if (error) setError('');
              }}
              onKeyDown={handleKeyDown}
              rows={6}
            />

            <div className="textarea-footer">
              <div className="character-stats">
                <span>{heuristics?.charCount || 0} characters</span>
                <span className="separator">•</span>
                <span>{heuristics?.wordCount || 0} words</span>
                {heuristics && heuristics.capsRatio > 25 && (
                  <>
                    <span className="separator">•</span>
                    <span className="warn-badge">{heuristics.capsRatio}% CAPS</span>
                  </>
                )}
              </div>
              <span className="hotkey-hint">Press <kbd>Ctrl</kbd> + <kbd>Enter</kbd> to analyze</span>
            </div>
          </div>

          {/* Live Heuristics Indicators */}
          {heuristics && message.trim() && (
            <div className="heuristic-bar">
              <div className="heuristic-title">
                <IconZap size={14} /> Heuristic Signals:
              </div>
              <div className="heuristic-tags">
                {heuristics.keywords.length > 0 ? (
                  <span className="h-tag h-tag-warn" title={`Keywords: ${heuristics.keywords.join(', ')}`}>
                    🚨 {heuristics.keywords.length} Spam keywords
                  </span>
                ) : (
                  <span className="h-tag h-tag-safe">✓ Clean vocabulary</span>
                )}

                {heuristics.urls.length > 0 && (
                  <span className="h-tag h-tag-danger" title={`URLs: ${heuristics.urls.join(', ')}`}>
                    🔗 {heuristics.urls.length} Link(s) detected
                  </span>
                )}

                {heuristics.phoneMatches.length > 0 && (
                  <span className="h-tag h-tag-info">
                    📞 Shortcode/Phone detected
                  </span>
                )}

                {heuristics.capsRatio > 40 && (
                  <span className="h-tag h-tag-warn">
                    ⚠️ Excessive Capitalization
                  </span>
                )}
              </div>
            </div>
          )}

          <div className="action-row">
            <button
              className="analyze-button"
              onClick={() => runPrediction()}
              disabled={loading || !message.trim()}
            >
              {loading ? (
                <>
                  <span className="spinner" />
                  <span>Inferring xLSTM Layers...</span>
                </>
              ) : (
                <>
                  <span>Run xLSTM Neural Scan</span>
                  <IconArrowRight size={16} />
                </>
              )}
            </button>
          </div>

          {error && (
            <div className="error-banner">
              <IconAlertCircle size={18} />
              <div>
                <strong>Inference Error</strong>
                <p>{error}</p>
              </div>
            </div>
          )}
        </div>

        {/* Right Column: Threat Diagnosis Dossier */}
        <div className="inspector-card result-card">
          <div className="card-header">
            <div className="header-title-wrap">
              <span className="step-number">02</span>
              <h3>Neural Threat Diagnosis Dossier</h3>
            </div>
            {result && (
              <button 
                className="text-action-btn"
                onClick={handleCopyReport}
                title="Copy diagnosis report"
              >
                {copied ? <IconCheck size={14} /> : <IconCopy size={14} />}
                {copied ? 'Copied' : 'Copy Report'}
              </button>
            )}
          </div>

          {loading ? (
            <div className="result-loading-state">
              <div className="scanning-radar">
                <div className="radar-circle circle-1" />
                <div className="radar-circle circle-2" />
                <div className="radar-sweep" />
                <IconShield size={38} className="radar-icon" />
              </div>
              <h4>Propagating Sequence Through xLSTM</h4>
              <p>Evaluating mLSTM Covariance States & Attention Weights...</p>
            </div>
          ) : result ? (
            <div className="result-content-body">
              {/* Verdict Header Badge */}
              <div className={`verdict-banner ${result.prediction === 'SPAM' ? 'verdict-spam' : 'verdict-ham'}`}>
                <div className="verdict-icon-box">
                  {result.prediction === 'SPAM' ? (
                    <IconShieldAlert size={36} />
                  ) : (
                    <IconShieldCheck size={36} />
                  )}
                </div>
                <div className="verdict-text-group">
                  <span className="verdict-kicker">VERDICT CLASSIFICATION</span>
                  <div className="verdict-title-row">
                    <h2>{result.prediction === 'SPAM' ? 'MALICIOUS SPAM' : 'LEGITIMATE HAM'}</h2>
                    <span className="confidence-pill">
                      {result.confidence}% Confidence
                    </span>
                  </div>
                  <p>
                    {result.prediction === 'SPAM'
                      ? 'High probability of unsolicited advertising, phishing, or social engineering fraud.'
                      : 'High probability of standard personal, transactional, or notification text.'}
                  </p>
                </div>
              </div>

              {/* Probability Dual-Comparison Bars */}
              <div className="probability-breakdown-card">
                <div className="prob-header">
                  <span>Sequence Softmax Distribution</span>
                  <span className="prob-total">Total: 100%</span>
                </div>

                <div className="prob-meter-group">
                  {/* Spam Bar */}
                  <div className="prob-row">
                    <div className="prob-label-row">
                      <span className="prob-name spam-text">Spam Threat Likelihood</span>
                      <strong className="prob-val spam-text">{result.spam_probability}%</strong>
                    </div>
                    <div className="prob-track">
                      <div 
                        className="prob-fill fill-spam"
                        style={{ width: `${result.spam_probability}%` }}
                      />
                    </div>
                  </div>

                  {/* Ham Bar */}
                  <div className="prob-row">
                    <div className="prob-label-row">
                      <span className="prob-name ham-text">Legitimate Ham Likelihood</span>
                      <strong className="prob-val ham-text">{result.ham_probability}%</strong>
                    </div>
                    <div className="prob-track">
                      <div 
                        className="prob-fill fill-ham"
                        style={{ width: `${result.ham_probability}%` }}
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* Diagnostic Metadata Grid */}
              <div className="meta-stats-grid">
                <div className="meta-card">
                  <span className="meta-card-label">Threat Severity</span>
                  <span className={`meta-card-value ${result.prediction === 'SPAM' ? 'spam-text' : 'ham-text'}`}>
                    {result.prediction === 'SPAM' 
                      ? (result.confidence > 90 ? 'Critical' : 'Elevated') 
                      : 'Minimal (Clean)'}
                  </span>
                </div>

                <div className="meta-card">
                  <span className="meta-card-label">Inference Latency</span>
                  <span className="meta-card-value font-mono">
                    {scanLatency ? `${scanLatency} ms` : '~15 ms'}
                  </span>
                </div>

                <div className="meta-card">
                  <span className="meta-card-label">Active Layers</span>
                  <span className="meta-card-value">3x mLSTM + 1x sLSTM</span>
                </div>

                <div className="meta-card">
                  <span className="meta-card-label">Token Limit</span>
                  <span className="meta-card-value font-mono">50 Tokens Max</span>
                </div>
              </div>

              {/* Action Recommendation */}
              <div className={`recommendation-box ${result.prediction === 'SPAM' ? 'rec-spam' : 'rec-ham'}`}>
                <div className="rec-title">
                  <IconShield size={16} />
                  <span>Recommended Action:</span>
                </div>
                <p>
                  {result.prediction === 'SPAM'
                    ? 'Do NOT click any embedded hyperlinks or call attached contact numbers. Block the originating sender number and report to spam registry.'
                    : 'This message matches typical human-to-human or authenticated automated service patterns. Safe to interact with.'}
                </p>
              </div>
            </div>
          ) : (
            <div className="result-placeholder-state">
              <div className="placeholder-icon-wrap">
                <IconShield size={44} className="placeholder-shield" />
              </div>
              <h4>Awaiting SMS Sequence</h4>
              <p>
                Enter an SMS message on the left or select a sample from the test library above to execute real-time xLSTM neural classification.
              </p>
              <div className="feature-checklist">
                <div className="check-item">
                  <IconCheck size={14} className="check-icon" />
                  <span>Multi-head mLSTM Covariance Memory</span>
                </div>
                <div className="check-item">
                  <IconCheck size={14} className="check-icon" />
                  <span>Stabilized sLSTM Exponential Gating</span>
                </div>
                <div className="check-item">
                  <IconCheck size={14} className="check-icon" />
                  <span>Sub-millisecond Pre-tokenization</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

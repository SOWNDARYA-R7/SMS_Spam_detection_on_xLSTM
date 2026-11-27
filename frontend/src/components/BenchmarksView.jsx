import React, { useState } from 'react';
import { 
  IconBarChart, 
  IconCheck, 
  IconShield, 
  IconActivity,
  IconCpu
} from './Icons';

const BENCHMARK_MODELS = [
  {
    name: 'xLSTM (Proposed)',
    type: 'Extended Neural LSTM',
    accuracy: 94.70,
    precision: 74.19,
    recall: 92.62,
    f1: 82.39,
    highlight: true,
    badge: 'Proposed Model'
  },
  {
    name: 'TF-IDF + SVM',
    type: 'Classical Machine Learning',
    accuracy: 98.38,
    precision: 97.12,
    recall: 90.60,
    f1: 93.75,
    highlight: false,
    badge: 'Linear Baseline'
  },
  {
    name: 'LSTM',
    type: 'Standard Recurrent Network',
    accuracy: 97.85,
    precision: 97.71,
    recall: 85.91,
    f1: 91.43,
    highlight: false,
    badge: 'RNN Baseline'
  },
  {
    name: 'GRU',
    type: 'Gated Recurrent Unit',
    accuracy: 97.67,
    precision: 97.67,
    recall: 84.56,
    f1: 90.65,
    highlight: false,
    badge: 'Gated Baseline'
  },
  {
    name: 'Bi-LSTM',
    type: 'Bidirectional LSTM',
    accuracy: 97.31,
    precision: 95.42,
    recall: 83.89,
    f1: 89.29,
    highlight: false,
    badge: 'Bidirectional Baseline'
  },
  {
    name: 'XGBoost',
    type: 'Gradient Boosted Trees',
    accuracy: 96.23,
    precision: 100.00,
    recall: 71.81,
    f1: 83.59,
    highlight: false,
    badge: 'Tree Baseline'
  }
];

export default function BenchmarksView() {
  const [activeMetric, setActiveMetric] = useState('recall');

  const metricDescriptions = {
    recall: 'Spam Detection Recall: The percentage of actual spam attacks successfully captured (xLSTM achieves 92.62%, outperforming standard LSTM, GRU, Bi-LSTM and XGBoost).',
    accuracy: 'Overall Accuracy: The total ratio of correct classifications across both Ham and Spam test samples.',
    precision: 'Precision: The exactness of spam flags (ratio of true spam vs false positives).',
    f1: 'F1-Score: The harmonic mean balancing precision and recall.'
  };

  return (
    <div className="benchmarks-container">
      <div className="section-header-row">
        <div>
          <div className="section-eyebrow">
            <IconBarChart size={14} className="eyebrow-icon" />
            <span>EMPIRICAL RESEARCH EVALUATION</span>
          </div>
          <h2>Model Comparative Benchmarks</h2>
          <p>
            Rigorous performance evaluation conducted on 5,572 SMS messages with an 80:20 stratified split 
            (1,114 held-out test messages).
          </p>
        </div>
      </div>

      {/* Metric Selector Tabs */}
      <div className="metric-selector-bar">
        <span className="selector-label">Evaluate by Metric:</span>
        <div className="selector-buttons">
          <button 
            className={`metric-tab ${activeMetric === 'recall' ? 'active' : ''}`}
            onClick={() => setActiveMetric('recall')}
          >
            Spam Recall (Proposed Advantage)
          </button>
          <button 
            className={`metric-tab ${activeMetric === 'accuracy' ? 'active' : ''}`}
            onClick={() => setActiveMetric('accuracy')}
          >
            Overall Accuracy
          </button>
          <button 
            className={`metric-tab ${activeMetric === 'f1' ? 'active' : ''}`}
            onClick={() => setActiveMetric('f1')}
          >
            F1-Score
          </button>
          <button 
            className={`metric-tab ${activeMetric === 'precision' ? 'active' : ''}`}
            onClick={() => setActiveMetric('precision')}
          >
            Precision
          </button>
        </div>
      </div>

      <div className="metric-explainer-banner">
        <IconActivity size={16} />
        <span>{metricDescriptions[activeMetric]}</span>
      </div>

      {/* Visual Comparison Bar Chart */}
      <div className="chart-card">
        <h3>Model Comparison: {activeMetric.toUpperCase()} (%)</h3>
        <div className="bar-chart-visual">
          {BENCHMARK_MODELS.map((m, idx) => {
            const val = m[activeMetric];
            return (
              <div key={idx} className={`chart-bar-row ${m.highlight ? 'chart-bar-highlight' : ''}`}>
                <div className="bar-label-group">
                  <span className="bar-model-name">
                    {m.name}
                    {m.highlight && <span className="highlight-tag">PROPOSED</span>}
                  </span>
                  <span className="bar-model-type">{m.type}</span>
                </div>
                <div className="bar-track-wrap">
                  <div 
                    className={`bar-fill-element ${m.highlight ? 'fill-accent' : 'fill-baseline'}`}
                    style={{ width: `${Math.max(val, 10)}%` }}
                  >
                    <span className="bar-value-label">{val}%</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Two Column Section: Full Table + Confusion Matrix */}
      <div className="benchmark-details-grid">
        {/* Full Comparison Table */}
        <div className="bench-card table-bench-card">
          <div className="card-header">
            <h3>Full Architecture Comparison Table</h3>
            <span className="sample-badge">N = 1,114 Test Samples</span>
          </div>

          <div className="table-responsive-container">
            <table className="benchmark-table">
              <thead>
                <tr>
                  <th>Model Architecture</th>
                  <th>Accuracy</th>
                  <th>Precision</th>
                  <th>Recall (Spam)</th>
                  <th>F1-Score</th>
                </tr>
              </thead>
              <tbody>
                {BENCHMARK_MODELS.map((m, idx) => (
                  <tr key={idx} className={m.highlight ? 'row-highlighted' : ''}>
                    <td>
                      <div className="model-cell">
                        <strong>{m.name}</strong>
                        <span className="model-subtext">{m.type}</span>
                      </div>
                    </td>
                    <td className="font-mono">{m.accuracy}%</td>
                    <td className="font-mono">{m.precision}%</td>
                    <td className={`font-mono font-bold ${m.highlight ? 'recall-lead' : ''}`}>
                      {m.recall}%
                    </td>
                    <td className="font-mono">{m.f1}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Confusion Matrix Interactive Visualizer */}
        <div className="bench-card matrix-bench-card">
          <div className="card-header">
            <h3>xLSTM Confusion Matrix Breakdown</h3>
            <span className="sample-badge">Test Evaluation</span>
          </div>

          <div className="matrix-wrapper">
            <div className="matrix-labels-top">
              <span>Predicted Ham</span>
              <span>Predicted Spam</span>
            </div>

            <div className="matrix-grid-2x2">
              {/* True Negative */}
              <div className="matrix-cell cell-tn">
                <span className="cell-role">True Negative (TN)</span>
                <span className="cell-count font-mono">917</span>
                <span className="cell-desc">Legitimate Ham Correctly Allowed</span>
              </div>

              {/* False Positive */}
              <div className="matrix-cell cell-fp">
                <span className="cell-role">False Positive (FP)</span>
                <span className="cell-count font-mono">48</span>
                <span className="cell-desc">Ham Flagged as Spam</span>
              </div>

              {/* False Negative */}
              <div className="matrix-cell cell-fn">
                <span className="cell-role">False Negative (FN)</span>
                <span className="cell-count font-mono">11</span>
                <span className="cell-desc">Spam Missed (Low Rate!)</span>
              </div>

              {/* True Positive */}
              <div className="matrix-cell cell-tp">
                <span className="cell-role">True Positive (TP)</span>
                <span className="cell-count font-mono">138</span>
                <span className="cell-desc">Spam Threats Successfully Caught</span>
              </div>
            </div>

            <div className="matrix-legend">
              <div className="legend-stat">
                <span className="stat-num font-mono">92.62%</span>
                <span className="stat-caption">Spam Detection Recall</span>
              </div>
              <div className="legend-divider" />
              <div className="legend-stat">
                <span className="stat-num font-mono">11</span>
                <span className="stat-caption">Total Misses out of 149 Spams</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

import React from 'react';
import { 
  IconLayers, 
  IconCpu, 
  IconShield, 
  IconActivity,
  IconSparkles
} from './Icons';

export default function ArchitectureExplorer() {
  const blocks = [
    {
      index: 1,
      type: 'mLSTM',
      name: 'Matrix Memory Block 1',
      role: 'Associative Memory & Key-Value Retrieval',
      desc: 'Maintains a matrix memory state C_t updated via outer-product covariance (v_t ⊗ k_t^T), allowing high-capacity contextual storage.'
    },
    {
      index: 2,
      type: 'sLSTM',
      name: 'Scalar Memory Block 2',
      role: 'Exponential Gating & Normalization',
      desc: 'Features exponential input and forget gates with a running max tracker m_t for numerical stability, balancing short and long-range SMS patterns.'
    },
    {
      index: 3,
      type: 'mLSTM',
      name: 'Matrix Memory Block 3',
      role: 'Multi-Head Contextual Refinement',
      desc: 'Projects sequence representations across 4 parallel attention heads to identify non-linear phishing triggers and deceptive phrases.'
    },
    {
      index: 4,
      type: 'mLSTM',
      name: 'Matrix Memory Block 4',
      role: 'Sequence Summary & Syntactic Encoding',
      desc: 'Aggregates token-level representations into a unified semantic sequence ready for global pooling and threat decision.'
    }
  ];

  const hyperparams = [
    { label: 'Embedding Dimension', value: '128' },
    { label: 'Total xLSTM Blocks', value: '4 Blocks' },
    { label: 'Block Structure', value: 'mLSTM, sLSTM, mLSTM, mLSTM' },
    { label: 'Multi-Head Attention', value: '4 Heads' },
    { label: 'Max Sequence Length', value: '50 Tokens' },
    { label: 'Vocabulary Size', value: '7,700+ unique tokens' },
    { label: 'Pooling Mechanism', value: 'Mean Pooling across sequence' },
    { label: 'Classifier Layer', value: 'Linear(128 -> 2) + Softmax' },
    { label: 'Training Optimizer', value: 'AdamW with Weight Decay' },
    { label: 'Loss Objective', value: 'Cross Entropy Loss' }
  ];

  return (
    <div className="architecture-container">
      <div className="section-header-row">
        <div>
          <div className="section-eyebrow">
            <IconLayers size={14} className="eyebrow-icon" />
            <span>NEURAL NETWORK SPECIFICATIONS</span>
          </div>
          <h2>Extended LSTM (xLSTM) Architecture Deep Dive</h2>
          <p>
            An in-depth look at the recurrent sequence architecture that overcomes the storage 
            limitations of traditional LSTMs using matrix memory and exponential stabilization.
          </p>
        </div>
      </div>

      {/* Layer Pipeline Diagram */}
      <div className="pipeline-card">
        <h3>End-to-End Inference Pipeline</h3>
        <div className="pipeline-flow">
          <div className="pipeline-node node-input">
            <span className="node-type">Input</span>
            <strong>Raw SMS Text</strong>
            <span className="node-detail">Cleaned & Tokenized</span>
          </div>

          <div className="pipeline-connector">→</div>

          <div className="pipeline-node node-embed">
            <span className="node-type">Layer 1</span>
            <strong>Word Embedding</strong>
            <span className="node-detail">128-D Dense Vectors</span>
          </div>

          <div className="pipeline-connector">→</div>

          <div className="pipeline-node node-core">
            <span className="node-type">Backbone</span>
            <strong>4x xLSTM Blocks</strong>
            <span className="node-detail">3 mLSTM + 1 sLSTM</span>
          </div>

          <div className="pipeline-connector">→</div>

          <div className="pipeline-node node-pool">
            <span className="node-type">Aggregation</span>
            <strong>Mean Pooling</strong>
            <span className="node-detail">Sequence Reduction</span>
          </div>

          <div className="pipeline-connector">→</div>

          <div className="pipeline-node node-output">
            <span className="node-type">Head</span>
            <strong>Dense Classifier</strong>
            <span className="node-detail">Ham vs Spam Softmax</span>
          </div>
        </div>
      </div>

      {/* mLSTM vs sLSTM Comparison Cards */}
      <div className="mechanisms-grid">
        <div className="mechanism-card mlstm-card">
          <div className="mechanism-badge badge-mlstm">mLSTM (Matrix Memory)</div>
          <h4>Covariance Storage with Outer Products</h4>
          <p>
            Instead of storing memory in scalar cell states, mLSTM maintains a matrix state 
            that allows dynamic key-value storage and retrieval. This enables the model to recall 
            specific phishing keywords even when surrounded by deceptive filler text.
          </p>
          <div className="math-pill">
            <code>C_t = f_t · C_{'{t-1}'} + v_t ⊗ k_t^T</code>
          </div>
        </div>

        <div className="mechanism-card slstm-card">
          <div className="mechanism-badge badge-slstm">sLSTM (Scalar Memory)</div>
          <h4>Exponential Gating with Stabilizer State</h4>
          <p>
            sLSTM enhances classic LSTM cells by introducing exponential gating functions. 
            A numerical stabilizer state tracks running maxima, preventing vanishing or 
            exploding gradient issues across varied message lengths.
          </p>
          <div className="math-pill">
            <code>m_t = max(f_t + m_{'{t-1}'}, i_t)</code>
          </div>
        </div>
      </div>

      {/* Block Sequence Cards */}
      <div className="blocks-section">
        <h3>Composition of the 4 xLSTM Blocks</h3>
        <div className="blocks-grid">
          {blocks.map((b) => (
            <div key={b.index} className="block-card">
              <div className="block-card-header">
                <span className={`block-badge ${b.type === 'mLSTM' ? 'badge-mlstm' : 'badge-slstm'}`}>
                  {b.type}
                </span>
                <span className="block-index">Layer {b.index} of 4</span>
              </div>
              <h4>{b.name}</h4>
              <span className="block-role">{b.role}</span>
              <p>{b.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Hyperparameters Table */}
      <div className="hyperparams-card">
        <h3>Model Configuration & Hyperparameters</h3>
        <div className="hyperparams-grid">
          {hyperparams.map((hp, idx) => (
            <div key={idx} className="hyperparam-item">
              <span className="hyperparam-label">{hp.label}</span>
              <strong className="hyperparam-value font-mono">{hp.value}</strong>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import LiveInspector from './components/LiveInspector';
import BatchScanner from './components/BatchScanner';
import BenchmarksView from './components/BenchmarksView';
import ArchitectureExplorer from './components/ArchitectureExplorer';
import AuditHistory from './components/AuditHistory';
import './App.css';

export default function App() {
  const API_URL = (
  import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'
).replace(/\/+$/, '');
  const [activeTab, setActiveTab] = useState('inspector');
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('xlstm_theme') || 'dark';
  });
  const [apiStatus, setApiStatus] = useState('checking');
  const [latency, setLatency] = useState(null);
  const [history, setHistory] = useState(() => {
    try {
      const saved = localStorage.getItem('xlstm_audit_history');
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });

  // Apply theme attribute to document
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('xlstm_theme', theme);
  }, [theme]);

  // Ping backend health check
  useEffect(() => {
    let isMounted = true;
    const checkHealth = async () => {
      const start = performance.now();
      try {
        const res = await fetch(`${API_URL}/api/health`);
        if (res.ok) {
          const elapsed = Math.round(performance.now() - start);
          if (isMounted) {
            setApiStatus('connected');
            setLatency(elapsed);
          }
        } else {
          // Check fallback /
          const fallbackRes = await fetch(`${API_URL}`);
          if (fallbackRes.ok && isMounted) {
            setApiStatus('connected');
            setLatency(Math.round(performance.now() - start));
          } else if (isMounted) {
            setApiStatus('disconnected');
          }
        }
      } catch {
        if (isMounted) {
          setApiStatus('disconnected');
        }
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, [API_URL]);

  // Save audit history to local storage
  const handleScanComplete = (scanRecord) => {
    setHistory((prev) => {
      const updated = [scanRecord, ...prev.slice(0, 99)];
      try {
        localStorage.setItem('xlstm_audit_history', JSON.stringify(updated));
      } catch (err) {
        console.error('Failed to save history to storage:', err);
      }
      return updated;
    });
  };

  const handleClearHistory = () => {
    setHistory([]);
    localStorage.removeItem('xlstm_audit_history');
  };

  return (
    <div className="dashboard-app">
      <Header
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        theme={theme}
        setTheme={setTheme}
        apiStatus={apiStatus}
        latency={latency}
        historyCount={history.length}
      />

      <main className="dashboard-main-content">
        {activeTab === 'inspector' && (
          <LiveInspector onScanComplete={handleScanComplete} />
        )}

        {activeTab === 'batch' && (
          <BatchScanner onScanComplete={handleScanComplete} />
        )}

        {activeTab === 'benchmarks' && (
          <BenchmarksView />
        )}

        {activeTab === 'architecture' && (
          <ArchitectureExplorer />
        )}

        {activeTab === 'history' && (
          <AuditHistory 
            history={history} 
            onClearHistory={handleClearHistory}
            onSelectScan={() => setActiveTab('inspector')}
          />
        )}
      </main>

      <footer className="dashboard-footer">
        <div className="footer-content">
          <div className="footer-left">
            <span className="footer-brand">xLSTM Sentinel AI</span>
            <span className="footer-sep">•</span>
            <span>Extended Long Short-Term Memory Neural Threat Defense</span>
          </div>
          <div className="footer-right">
            <span>PyTorch 2.x</span>
            <span className="footer-sep">•</span>
            <span>FastAPI Backend</span>
            <span className="footer-sep">•</span>
            <span>React 19 Enterprise UI</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

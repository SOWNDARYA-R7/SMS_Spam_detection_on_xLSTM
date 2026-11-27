import React from 'react';
import { 
  IconShield, 
  IconActivity, 
  IconBarChart, 
  IconLayers, 
  IconHistory, 
  IconSun, 
  IconMoon, 
  IconZap,
  IconCpu
} from './Icons';

export default function Header({ 
  activeTab, 
  setActiveTab, 
  theme, 
  setTheme, 
  apiStatus, 
  latency, 
  historyCount = 0 
}) {
  const tabs = [
    { id: 'inspector', label: 'Live Inspector', icon: IconShield },
    { id: 'batch', label: 'Batch Scanner', icon: IconZap },
    { id: 'benchmarks', label: 'Model Benchmarks', icon: IconBarChart },
    { id: 'architecture', label: 'xLSTM Architecture', icon: IconLayers },
    { id: 'history', label: 'Audit Log', icon: IconHistory, count: historyCount }
  ];

  return (
    <header className="dashboard-header">
      <div className="header-top">
        <div className="brand-group">
          <div className="brand-icon-wrap">
            <IconShield size={24} className="brand-shield-icon" />
          </div>
          <div className="brand-text">
            <div className="brand-title-row">
              <h1>xLSTM Sentinel</h1>
            </div>
            <p>Enterprise SMS Threat Intelligence & Neural Sequence Defense</p>
          </div>
        </div>

        <div className="header-telemetry">
          <div className="telemetry-pill">
            <IconActivity size={14} className="telemetry-icon" />
            <span className="telemetry-label">Latency:</span>
            <span className="telemetry-val">{latency ? `${latency}ms` : '--'}</span>
          </div>

          <div className="telemetry-pill architecture-pill">
            <IconCpu size={14} className="telemetry-icon" />
            <span className="telemetry-label">Engine:</span>
            <span className="telemetry-val">4-Block xLSTM</span>
          </div>

          <button 
            className="theme-toggle-btn"
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
            aria-label="Toggle theme"
          >
            {theme === 'dark' ? <IconSun size={17} /> : <IconMoon size={17} />}
          </button>
        </div>
      </div>

      <nav className="dashboard-nav">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              className={`nav-tab-btn ${isActive ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              <Icon size={16} />
              <span>{tab.label}</span>
              {typeof tab.count === 'number' && tab.count > 0 && (
                <span className="tab-badge">{tab.count}</span>
              )}
            </button>
          );
        })}
      </nav>
    </header>
  );
}

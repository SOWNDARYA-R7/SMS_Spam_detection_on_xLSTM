
import { useState } from "react";
import "./App.css";

function App() {
  const [message, setMessage] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const predictMessage = async () => {
    if (!message.trim()) {
      setError("Please enter an SMS message.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ message })
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Prediction failed");
      }

      setResult(data);
    } catch (err) {
      setError(err.message || "Unable to connect to the backend.");
    } finally {
      setLoading(false);
    }
  };

  const clearMessage = () => {
    setMessage("");
    setResult(null);
    setError("");
  };

  return (
    <div className="app">
      <nav className="navbar">
        <h2>🛡️ SMS Shield</h2>
        <span>Powered by xLSTM AI</span>
      </nav>

      <main className="container">
        <div className="hero">
          <span className="tag">AI-POWERED CLASSIFICATION</span>
          <h1>Detect SMS Spam <span>Instantly.</span></h1>
          <p>
            Analyze your text messages using our trained
            xLSTM deep learning model.
          </p>
        </div>

        <section className="predict-card">
          <label>Enter your SMS message</label>

          <textarea
            placeholder="Paste your SMS message here..."
            value={message}
            onChange={(e) => {
              setMessage(e.target.value);
              setResult(null);
              setError("");
            }}
          />

          <div className="actions">
            <span>{message.length} characters</span>
            <button className="clear-btn" onClick={clearMessage}>
              Clear
            </button>
          </div>

          <button
            className="predict-btn"
            onClick={predictMessage}
            disabled={loading}
          >
            {loading ? "Analyzing..." : "Analyze Message →"}
          </button>

          {error && <p className="error">{error}</p>}

          {result && (
            <div className={`result ${result.prediction.toLowerCase()}`}>
              <div className="result-heading">
                <span>
                  {result.prediction === "HAM" ? "🛡️" : "⚠️"}
                </span>
                <div>
                  <small>CLASSIFICATION RESULT</small>
                  <h2>{result.prediction}</h2>
                  <p>
                    {result.prediction === "HAM"
                      ? "This message was classified as legitimate."
                      : "This message was classified as spam."}
                  </p>
                </div>
              </div>

              <div className="confidence">
                <div>
                  <span>Prediction confidence</span>
                  <strong>{result.confidence}%</strong>
                </div>
                <div className="progress">
                  <div
                    style={{ width: `${result.confidence}%` }}
                  />
                </div>
              </div>

              <div className="probabilities">
                <div>
                  <span>Ham probability</span>
                  <strong>{result.ham_probability}%</strong>
                </div>
                <div>
                  <span>Spam probability</span>
                  <strong>{result.spam_probability}%</strong>
                </div>
              </div>
            </div>
          )}
        </section>

        <section className="examples">
          <h3>Try example messages</h3>
          <button onClick={() =>
            setMessage("Hey! Are we meeting tomorrow at 10 AM?")
          }>
            ✓ Normal message
          </button>
          <button onClick={() =>
            setMessage("Congratulations! You won a free prize. Claim now!")
          }>
            ⚠ Spam example
          </button>
        </section>

        <footer>
          <p>SMS Spam Detection using xLSTM</p>
          <span>Built with React, FastAPI and PyTorch</span>
        </footer>
      </main>
    </div>
  );
}

export default App;

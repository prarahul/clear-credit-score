import { useState } from 'react'
import './App.css'

interface CreditRequest {
  checking_status: string;
  duration: number;
  credit_history: string;
  purpose: string;
  credit_amount: number;
  savings_status: string;
  employment: string;
  installment_commitment: number;
  personal_status: string;
  other_parties: string;
  residence_since: number;
  property_magnitude: string;
  age: number;
  other_payment_plans: string;
  housing: string;
  existing_credits: number;
  job: string;
  num_dependents: number;
  own_telephone: string;
  foreign_worker: string;
}

interface CreditResponse {
  prob_default: number;
  risk: string;
}

function App() {
  const [formData, setFormData] = useState<CreditRequest>({
    checking_status: '<0',
    duration: 12,
    credit_history: 'existing paid',
    purpose: 'car (new)',
    credit_amount: 2500,
    savings_status: '<100',
    employment: '>=7',
    installment_commitment: 2,
    personal_status: 'male single',
    other_parties: 'none',
    residence_since: 3,
    property_magnitude: 'real estate',
    age: 35,
    other_payment_plans: 'none',
    housing: 'own',
    existing_credits: 1,
    job: 'skilled',
    num_dependents: 1,
    own_telephone: 'yes',
    foreign_worker: 'yes'
  });

  const [result, setResult] = useState<CreditResponse | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    
    try {
      const response = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });
      
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'low': return '#22c55e';
      case 'medium': return '#f59e0b';
      case 'high': return '#ef4444';
      default: return '#6b7280';
    }
  };

  return (
    <div className="app">
      <h1>Credit Risk Assessment</h1>
      
      <form onSubmit={handleSubmit} className="credit-form">
        <div className="form-grid">
          <div>
            <label>Credit Amount:</label>
            <input
              type="number"
              value={formData.credit_amount}
              onChange={(e) => setFormData({...formData, credit_amount: Number(e.target.value)})}
            />
          </div>
          
          <div>
            <label>Duration (months):</label>
            <input
              type="number"
              value={formData.duration}
              onChange={(e) => setFormData({...formData, duration: Number(e.target.value)})}
            />
          </div>
          
          <div>
            <label>Age:</label>
            <input
              type="number"
              value={formData.age}
              onChange={(e) => setFormData({...formData, age: Number(e.target.value)})}
            />
          </div>
          
          <div>
            <label>Checking Status:</label>
            <select
              value={formData.checking_status}
              onChange={(e) => setFormData({...formData, checking_status: e.target.value})}
            >
              <option value="<0">&lt; 0 DM</option>
              <option value="0<=X<200">0-200 DM</option>
              <option value=">=200">&ge; 200 DM</option>
              <option value="no checking">No checking</option>
            </select>
          </div>
          
          <div>
            <label>Purpose:</label>
            <select
              value={formData.purpose}
              onChange={(e) => setFormData({...formData, purpose: e.target.value})}
            >
              <option value="car (new)">Car (new)</option>
              <option value="car (used)">Car (used)</option>
              <option value="furniture/equipment">Furniture/Equipment</option>
              <option value="education">Education</option>
              <option value="business">Business</option>
              <option value="other">Other</option>
            </select>
          </div>
          
          <div>
            <label>Employment:</label>
            <select
              value={formData.employment}
              onChange={(e) => setFormData({...formData, employment: e.target.value})}
            >
              <option value="<1">&lt; 1 year</option>
              <option value="1<=X<4">1-4 years</option>
              <option value="4<=X<7">4-7 years</option>
              <option value=">=7">&ge; 7 years</option>
            </select>
          </div>
        </div>
        
        <button type="submit" disabled={loading} className="submit-btn">
          {loading ? 'Assessing...' : 'Assess Credit Risk'}
        </button>
      </form>

      {result && (
        <div className="result-card">
          <h2>Risk Assessment Result</h2>
          <div className="risk-indicator" style={{ backgroundColor: getRiskColor(result.risk) }}>
            <span className="risk-label">{result.risk.toUpperCase()} RISK</span>
            <span className="risk-probability">
              {(result.prob_default * 100).toFixed(1)}% default probability
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;

import { useState } from "react";
import { LoanApplicationForm } from "@/components/LoanApplicationForm";
import { RiskDashboard } from "@/components/RiskDashboard";
import { ModelMetrics } from "@/components/ModelMetrics";
import { calculateRisk } from "@/utils/riskCalculator";
import { Shield, TrendingUp, BarChart3 } from "lucide-react";

interface LoanApplicationData {
  loanAmount: string;
  annualIncome: string;
  employmentLength: string;
  homeOwnership: string;
  creditScore: string;
  debtToIncome: string;
  purpose: string;
}

interface RiskAssessment {
  riskScore: number;
  riskLevel: "low" | "medium" | "high";
  defaultProbability: number;
  features: {
    creditScore: { value: number; importance: number; impact: "positive" | "negative" };
    debtToIncome: { value: number; importance: number; impact: "positive" | "negative" };
    annualIncome: { value: number; importance: number; impact: "positive" | "negative" };
    employmentLength: { value: string; importance: number; impact: "positive" | "negative" };
    loanAmount: { value: number; importance: number; impact: "positive" | "negative" };
  };
}

const Index = () => {
  const [riskAssessment, setRiskAssessment] = useState<RiskAssessment | null>(null);

  const handleApplicationSubmit = (data: LoanApplicationData) => {
    const assessment = calculateRisk(data);
    setRiskAssessment(assessment);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-background to-card-elevated">
      {/* Header */}
      <div className="border-b bg-card/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-primary rounded-lg">
                <Shield className="h-6 w-6 text-primary-foreground" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-foreground">Credit Risk Analytics</h1>
                <p className="text-sm text-muted-foreground">ML-Powered Loan Default Prediction</p>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <div className="text-right">
                <div className="text-sm font-medium text-foreground">XGBoost Model v2.1</div>
                <div className="text-xs text-muted-foreground">ROC AUC: 87.2%</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column - Application Form */}
          <div className="lg:col-span-1">
            <LoanApplicationForm onApplicationSubmit={handleApplicationSubmit} />
          </div>

          {/* Right Column - Dashboard */}
          <div className="lg:col-span-2 space-y-8">
            <RiskDashboard assessment={riskAssessment} />
            <ModelMetrics />
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="border-t bg-card/30 mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="text-center text-sm text-muted-foreground">
            <p>Explainable AI for Financial Risk Assessment • Regulatory Compliant • Real-time Predictions</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Index;

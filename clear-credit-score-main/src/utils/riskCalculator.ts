// Mock credit risk calculation engine
// In a real implementation, this would call your trained ML model

interface LoanApplication {
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

export const calculateRisk = (application: LoanApplication): RiskAssessment => {
  const creditScore = parseInt(application.creditScore);
  const debtToIncome = parseFloat(application.debtToIncome);
  const annualIncome = parseInt(application.annualIncome);
  const loanAmount = parseInt(application.loanAmount);

  // Simple risk scoring algorithm (mock)
  let riskScore = 50; // Base score

  // Credit Score Impact (30% weight)
  if (creditScore >= 750) riskScore += 25;
  else if (creditScore >= 700) riskScore += 15;
  else if (creditScore >= 650) riskScore += 5;
  else if (creditScore >= 600) riskScore -= 5;
  else riskScore -= 20;

  // Debt-to-Income Impact (25% weight)
  if (debtToIncome <= 20) riskScore += 20;
  else if (debtToIncome <= 30) riskScore += 10;
  else if (debtToIncome <= 40) riskScore -= 5;
  else riskScore -= 15;

  // Income Impact (20% weight)
  const loanToIncomeRatio = loanAmount / annualIncome;
  if (loanToIncomeRatio <= 0.3) riskScore += 15;
  else if (loanToIncomeRatio <= 0.5) riskScore += 5;
  else if (loanToIncomeRatio <= 0.8) riskScore -= 5;
  else riskScore -= 15;

  // Employment Length Impact (15% weight)
  const employmentYears = getEmploymentYears(application.employmentLength);
  if (employmentYears >= 5) riskScore += 12;
  else if (employmentYears >= 2) riskScore += 6;
  else if (employmentYears >= 1) riskScore += 2;
  else riskScore -= 8;

  // Home Ownership Impact (10% weight)
  if (application.homeOwnership === "OWN") riskScore += 8;
  else if (application.homeOwnership === "MORTGAGE") riskScore += 4;
  else riskScore -= 2;

  // Ensure score is within bounds
  riskScore = Math.max(0, Math.min(100, riskScore));

  // Calculate default probability (inverse relationship with risk score)
  const defaultProbability = Math.max(0.01, Math.min(0.99, (100 - riskScore) / 100 * 0.4));

  // Determine risk level
  let riskLevel: "low" | "medium" | "high";
  if (riskScore >= 70) riskLevel = "low";
  else if (riskScore >= 40) riskLevel = "medium";
  else riskLevel = "high";

  // Calculate feature importance (mock SHAP values)
  const features = {
    creditScore: {
      value: creditScore,
      importance: 0.30,
      impact: creditScore >= 650 ? "positive" as const : "negative" as const
    },
    debtToIncome: {
      value: debtToIncome,
      importance: 0.25,
      impact: debtToIncome <= 35 ? "positive" as const : "negative" as const
    },
    annualIncome: {
      value: annualIncome,
      importance: 0.20,
      impact: loanToIncomeRatio <= 0.5 ? "positive" as const : "negative" as const
    },
    employmentLength: {
      value: application.employmentLength,
      importance: 0.15,
      impact: employmentYears >= 2 ? "positive" as const : "negative" as const
    },
    loanAmount: {
      value: loanAmount,
      importance: 0.10,
      impact: loanToIncomeRatio <= 0.4 ? "positive" as const : "negative" as const
    }
  };

  return {
    riskScore,
    riskLevel,
    defaultProbability,
    features
  };
};

const getEmploymentYears = (employmentLength: string): number => {
  if (employmentLength.includes("< 1")) return 0.5;
  if (employmentLength.includes("10+")) return 10;
  
  const match = employmentLength.match(/(\d+)/);
  return match ? parseInt(match[1]) : 0;
};
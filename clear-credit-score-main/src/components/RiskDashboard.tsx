import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { 
  AlertTriangle, 
  Shield, 
  TrendingDown, 
  TrendingUp, 
  DollarSign,
  Percent,
  Clock,
  Home
} from "lucide-react";

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

interface RiskDashboardProps {
  assessment: RiskAssessment | null;
}

export const RiskDashboard = ({ assessment }: RiskDashboardProps) => {
  if (!assessment) {
    return (
      <Card className="h-fit">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Shield className="h-5 w-5 text-muted-foreground" />
            Risk Assessment
          </CardTitle>
          <CardDescription>
            Submit a loan application to see risk analysis
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-muted-foreground">
            <AlertTriangle className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p>No assessment data available</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  const getRiskColor = (level: string) => {
    switch (level) {
      case "low": return "text-risk-low";
      case "medium": return "text-risk-medium";
      case "high": return "text-risk-high";
      default: return "text-muted-foreground";
    }
  };

  const getRiskBadgeVariant = (level: string) => {
    switch (level) {
      case "low": return "default";
      case "medium": return "secondary";
      case "high": return "destructive";
      default: return "outline";
    }
  };

  const getFeatureIcon = (feature: string) => {
    switch (feature) {
      case "creditScore": return <TrendingUp className="h-4 w-4" />;
      case "debtToIncome": return <Percent className="h-4 w-4" />;
      case "annualIncome": return <DollarSign className="h-4 w-4" />;
      case "employmentLength": return <Clock className="h-4 w-4" />;
      case "loanAmount": return <DollarSign className="h-4 w-4" />;
      default: return <TrendingUp className="h-4 w-4" />;
    }
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Shield className="h-5 w-5 text-primary" />
            Risk Assessment Overview
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center space-y-2">
              <div className="text-3xl font-bold">
                {assessment.riskScore}/100
              </div>
              <div className="text-sm text-muted-foreground">Risk Score</div>
              <Progress 
                value={assessment.riskScore} 
                className="h-2"
              />
            </div>

            <div className="text-center space-y-2">
              <Badge 
                variant={getRiskBadgeVariant(assessment.riskLevel)}
                className="text-sm px-3 py-1"
              >
                {assessment.riskLevel.toUpperCase()} RISK
              </Badge>
              <div className="text-sm text-muted-foreground">Risk Level</div>
            </div>

            <div className="text-center space-y-2">
              <div className={`text-2xl font-bold ${getRiskColor(assessment.riskLevel)}`}>
                {(assessment.defaultProbability * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-muted-foreground">Default Probability</div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5 text-primary" />
            Feature Importance & SHAP Analysis
          </CardTitle>
          <CardDescription>
            Key factors influencing the risk prediction
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {Object.entries(assessment.features)
              .sort(([,a], [,b]) => b.importance - a.importance)
              .map(([feature, data]) => (
                <div key={feature} className="flex items-center justify-between p-3 rounded-lg bg-card-elevated">
                  <div className="flex items-center gap-3">
                    <div className="p-2 rounded-full bg-primary-light">
                      {getFeatureIcon(feature)}
                    </div>
                    <div>
                      <div className="font-medium capitalize">
                        {feature.replace(/([A-Z])/g, ' $1').trim()}
                      </div>
                      <div className="text-sm text-muted-foreground">
                        Value: {typeof data.value === 'number' ? 
                          (feature === 'debtToIncome' ? `${data.value}%` : 
                           feature.includes('Income') || feature.includes('Amount') ? `$${data.value.toLocaleString()}` :
                           data.value) : 
                          data.value}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="text-right">
                      <div className="text-sm font-medium">
                        {(data.importance * 100).toFixed(1)}%
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Importance
                      </div>
                    </div>
                    <div className={`p-1 rounded ${data.impact === 'positive' ? 'text-risk-low' : 'text-risk-high'}`}>
                      {data.impact === 'positive' ? 
                        <TrendingUp className="h-4 w-4" /> : 
                        <TrendingDown className="h-4 w-4" />
                      }
                    </div>
                  </div>
                </div>
              ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
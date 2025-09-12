import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { 
  BarChart3, 
  Target, 
  TrendingUp, 
  Activity,
  CheckCircle2,
  AlertCircle
} from "lucide-react";

interface ModelPerformance {
  rocAuc: number;
  precision: number;
  recall: number;
  f1Score: number;
  accuracy: number;
  calibration: number;
}

const mockModelMetrics: ModelPerformance = {
  rocAuc: 0.87,
  precision: 0.82,
  recall: 0.78,
  f1Score: 0.80,
  accuracy: 0.85,
  calibration: 0.91
};

export const ModelMetrics = () => {
  const formatMetric = (value: number) => (value * 100).toFixed(1);

  const getMetricStatus = (value: number, thresholds: { good: number; fair: number }) => {
    if (value >= thresholds.good) return { status: "excellent", color: "text-risk-low", icon: CheckCircle2 };
    if (value >= thresholds.fair) return { status: "good", color: "text-risk-medium", icon: AlertCircle };
    return { status: "needs improvement", color: "text-risk-high", icon: AlertCircle };
  };

  const metrics = [
    {
      name: "ROC AUC",
      value: mockModelMetrics.rocAuc,
      description: "Overall model performance",
      thresholds: { good: 0.8, fair: 0.7 },
      icon: Target
    },
    {
      name: "Precision",
      value: mockModelMetrics.precision,
      description: "Accuracy of positive predictions",
      thresholds: { good: 0.8, fair: 0.7 },
      icon: BarChart3
    },
    {
      name: "Recall",
      value: mockModelMetrics.recall,
      description: "Coverage of actual positives",
      thresholds: { good: 0.75, fair: 0.65 },
      icon: TrendingUp
    },
    {
      name: "F1 Score",
      value: mockModelMetrics.f1Score,
      description: "Harmonic mean of precision & recall",
      thresholds: { good: 0.75, fair: 0.65 },
      icon: Activity
    },
    {
      name: "Calibration",
      value: mockModelMetrics.calibration,
      description: "Reliability of probability estimates",
      thresholds: { good: 0.85, fair: 0.75 },
      icon: CheckCircle2
    }
  ];

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <BarChart3 className="h-5 w-5 text-primary" />
          Model Performance Metrics
        </CardTitle>
        <CardDescription>
          Real-time model validation and performance indicators
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {metrics.map((metric) => {
            const status = getMetricStatus(metric.value, metric.thresholds);
            const StatusIcon = status.icon;
            
            return (
              <div key={metric.name} className="p-4 rounded-lg bg-card-elevated space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <metric.icon className="h-4 w-4 text-primary" />
                    <span className="font-medium">{metric.name}</span>
                  </div>
                  <StatusIcon className={`h-4 w-4 ${status.color}`} />
                </div>
                
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-2xl font-bold">
                      {formatMetric(metric.value)}%
                    </span>
                    <Badge variant="outline" className="text-xs">
                      {status.status}
                    </Badge>
                  </div>
                  
                  <Progress 
                    value={metric.value * 100} 
                    className="h-2"
                  />
                  
                  <p className="text-xs text-muted-foreground">
                    {metric.description}
                  </p>
                </div>
              </div>
            );
          })}
        </div>

        <div className="mt-6 p-4 rounded-lg bg-primary-light">
          <div className="flex items-start gap-3">
            <CheckCircle2 className="h-5 w-5 text-primary mt-0.5" />
            <div>
              <h4 className="font-medium text-primary">Model Status: Production Ready</h4>
              <p className="text-sm text-primary/80 mt-1">
                All metrics are within acceptable ranges for regulatory compliance. 
                Model was last retrained on Dec 8, 2024 with 98.7% data quality score.
              </p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
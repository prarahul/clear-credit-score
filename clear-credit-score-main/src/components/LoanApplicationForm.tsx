import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Calculator, TrendingUp } from "lucide-react";

interface LoanApplicationData {
  loanAmount: string;
  annualIncome: string;
  employmentLength: string;
  homeOwnership: string;
  creditScore: string;
  debtToIncome: string;
  purpose: string;
}

interface LoanApplicationFormProps {
  onApplicationSubmit: (data: LoanApplicationData) => void;
}

export const LoanApplicationForm = ({ onApplicationSubmit }: LoanApplicationFormProps) => {
  const [formData, setFormData] = useState<LoanApplicationData>({
    loanAmount: "",
    annualIncome: "",
    employmentLength: "",
    homeOwnership: "",
    creditScore: "",
    debtToIncome: "",
    purpose: "",
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onApplicationSubmit(formData);
  };

  const handleInputChange = (field: keyof LoanApplicationData, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  return (
    <Card className="h-fit">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Calculator className="h-5 w-5 text-primary" />
          Loan Application
        </CardTitle>
        <CardDescription>
          Enter applicant details to assess credit risk
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="loanAmount">Loan Amount ($)</Label>
              <Input
                id="loanAmount"
                type="number"
                placeholder="25000"
                value={formData.loanAmount}
                onChange={(e) => handleInputChange("loanAmount", e.target.value)}
                required
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="annualIncome">Annual Income ($)</Label>
              <Input
                id="annualIncome"
                type="number"
                placeholder="75000"
                value={formData.annualIncome}
                onChange={(e) => handleInputChange("annualIncome", e.target.value)}
                required
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="creditScore">Credit Score</Label>
              <Input
                id="creditScore"
                type="number"
                placeholder="720"
                min="300"
                max="850"
                value={formData.creditScore}
                onChange={(e) => handleInputChange("creditScore", e.target.value)}
                required
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="debtToIncome">Debt-to-Income Ratio (%)</Label>
              <Input
                id="debtToIncome"
                type="number"
                placeholder="25"
                min="0"
                max="100"
                value={formData.debtToIncome}
                onChange={(e) => handleInputChange("debtToIncome", e.target.value)}
                required
              />
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="employmentLength">Employment Length</Label>
            <Select onValueChange={(value) => handleInputChange("employmentLength", value)}>
              <SelectTrigger>
                <SelectValue placeholder="Select employment length" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="< 1 year">Less than 1 year</SelectItem>
                <SelectItem value="1 year">1 year</SelectItem>
                <SelectItem value="2 years">2 years</SelectItem>
                <SelectItem value="3 years">3 years</SelectItem>
                <SelectItem value="4 years">4 years</SelectItem>
                <SelectItem value="5 years">5 years</SelectItem>
                <SelectItem value="6 years">6 years</SelectItem>
                <SelectItem value="7 years">7 years</SelectItem>
                <SelectItem value="8 years">8 years</SelectItem>
                <SelectItem value="9 years">9 years</SelectItem>
                <SelectItem value="10+ years">10+ years</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="homeOwnership">Home Ownership</Label>
            <Select onValueChange={(value) => handleInputChange("homeOwnership", value)}>
              <SelectTrigger>
                <SelectValue placeholder="Select home ownership status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="RENT">Rent</SelectItem>
                <SelectItem value="OWN">Own</SelectItem>
                <SelectItem value="MORTGAGE">Mortgage</SelectItem>
                <SelectItem value="OTHER">Other</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="purpose">Loan Purpose</Label>
            <Select onValueChange={(value) => handleInputChange("purpose", value)}>
              <SelectTrigger>
                <SelectValue placeholder="Select loan purpose" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="debt_consolidation">Debt Consolidation</SelectItem>
                <SelectItem value="credit_card">Credit Card</SelectItem>
                <SelectItem value="home_improvement">Home Improvement</SelectItem>
                <SelectItem value="major_purchase">Major Purchase</SelectItem>
                <SelectItem value="medical">Medical</SelectItem>
                <SelectItem value="car">Car</SelectItem>
                <SelectItem value="vacation">Vacation</SelectItem>
                <SelectItem value="moving">Moving</SelectItem>
                <SelectItem value="other">Other</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <Button type="submit" className="w-full">
            <TrendingUp className="mr-2 h-4 w-4" />
            Analyze Risk
          </Button>
        </form>
      </CardContent>
    </Card>
  );
};
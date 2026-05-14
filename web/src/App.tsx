import { useState } from "react";
import { BudgetForm } from "./components/BudgetForm";
import { BudgetMatrix } from "./components/BudgetMatrix";
import { EstimateForm } from "./components/EstimateForm";
import { EstimateResult } from "./components/EstimateResult";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./components/ui/tabs";
import type { BudgetResponse, EstimateResponse } from "./lib/api";

export default function App() {
  const [estimateResult, setEstimateResult] = useState<EstimateResponse | null>(
    null,
  );
  const [budgetResult, setBudgetResult] = useState<BudgetResponse | null>(null);

  return (
    <div className="mx-auto max-w-7xl px-4 py-8">
      <h1 className="mb-6 text-2xl font-bold">vLLM Memory Estimator</h1>

      <Tabs defaultValue="estimate" className="w-full">
        <TabsList className="mb-4">
          <TabsTrigger value="estimate">Estimate</TabsTrigger>
          <TabsTrigger value="budget">Token Budget</TabsTrigger>
        </TabsList>

        <TabsContent value="estimate">
          <div className="grid grid-cols-1 gap-8 lg:grid-cols-[380px_1fr]">
            <EstimateForm onResult={setEstimateResult} />
            <EstimateResult data={estimateResult} />
          </div>
        </TabsContent>

        <TabsContent value="budget">
          <div className="grid grid-cols-1 gap-8 lg:grid-cols-[380px_1fr]">
            <BudgetForm onResult={setBudgetResult} />
            <BudgetMatrix data={budgetResult} />
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}

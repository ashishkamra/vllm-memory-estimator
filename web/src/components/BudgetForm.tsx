import { Loader2 } from "lucide-react";
import { type FormEvent, useState } from "react";
import type { BudgetResponse } from "../lib/api";
import { fetchBudget } from "../lib/api";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import { Switch } from "./ui/switch";

interface Props {
  onResult: (r: BudgetResponse) => void;
}

export function BudgetForm({ onResult }: Props) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [modelId, setModelId] = useState("");
  const [gpuMemory, setGpuMemory] = useState("80");
  const [quantization, setQuantization] = useState("none");
  const [dtype, setDtype] = useState("auto");
  const [kvCacheDtype, setKvCacheDtype] = useState("auto");
  const [tp, setTp] = useState("1");
  const [pp, setPp] = useState("1");
  const [dp, setDp] = useState("1");
  const [expertParallel, setExpertParallel] = useState(false);
  const [enforceEager, setEnforceEager] = useState(false);
  const [seqLengths, setSeqLengths] = useState("");
  const [seqCounts, setSeqCounts] = useState("");

  function parseIntList(s: string): number[] | null {
    if (!s.trim()) return null;
    return s
      .split(",")
      .map((x) => parseInt(x.trim()))
      .filter((n) => !isNaN(n) && n > 0);
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!modelId.trim() || !gpuMemory) return;

    setLoading(true);
    setError(null);

    try {
      const res = await fetchBudget({
        model_id: modelId.trim(),
        gpu_memory_gib: parseFloat(gpuMemory),
        tensor_parallel_size: parseInt(tp) || 1,
        pipeline_parallel_size: parseInt(pp) || 1,
        data_parallel_size: parseInt(dp) || 1,
        enable_expert_parallel: expertParallel,
        quantization: quantization === "none" ? null : quantization,
        dtype: dtype === "auto" ? null : dtype,
        kv_cache_dtype: kvCacheDtype === "auto" ? null : kvCacheDtype,
        enforce_eager: enforceEager,
        seq_lengths: parseIntList(seqLengths),
        seq_counts: parseIntList(seqCounts),
      });

      if (!res.ok) {
        setError(res.error || "Unknown error");
      } else {
        onResult(res);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="space-y-4 rounded-lg border p-4"
    >
      <div>
        <Label htmlFor="budget-model-id">Model ID</Label>
        <Input
          id="budget-model-id"
          placeholder="meta-llama/Llama-3.1-8B"
          value={modelId}
          onChange={(e) => setModelId(e.target.value)}
          required
        />
      </div>

      <div>
        <Label htmlFor="gpu-memory">GPU Memory (GiB)</Label>
        <Input
          id="gpu-memory"
          type="number"
          step="0.1"
          placeholder="80"
          value={gpuMemory}
          onChange={(e) => setGpuMemory(e.target.value)}
          required
        />
      </div>

      <div className="grid grid-cols-3 gap-3">
        <div>
          <Label>Quantization</Label>
          <Select value={quantization} onValueChange={setQuantization}>
            <SelectTrigger><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="none">None</SelectItem>
              <SelectItem value="fp8">FP8</SelectItem>
              <SelectItem value="int8">INT8</SelectItem>
              <SelectItem value="int4">INT4</SelectItem>
              <SelectItem value="awq">AWQ</SelectItem>
              <SelectItem value="gptq">GPTQ</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div>
          <Label>Dtype</Label>
          <Select value={dtype} onValueChange={setDtype}>
            <SelectTrigger><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="auto">Auto</SelectItem>
              <SelectItem value="float16">float16</SelectItem>
              <SelectItem value="bfloat16">bfloat16</SelectItem>
              <SelectItem value="float32">float32</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div>
          <Label>KV Cache Dtype</Label>
          <Select value={kvCacheDtype} onValueChange={setKvCacheDtype}>
            <SelectTrigger><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="auto">Auto</SelectItem>
              <SelectItem value="fp8">FP8</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-3">
        <div>
          <Label htmlFor="budget-tp">Tensor Parallel</Label>
          <Input
            id="budget-tp"
            type="number"
            min={1}
            value={tp}
            onChange={(e) => setTp(e.target.value)}
          />
        </div>
        <div>
          <Label htmlFor="budget-pp">Pipeline Parallel</Label>
          <Input
            id="budget-pp"
            type="number"
            min={1}
            value={pp}
            onChange={(e) => setPp(e.target.value)}
          />
        </div>
        <div>
          <Label htmlFor="budget-dp">Data Parallel</Label>
          <Input
            id="budget-dp"
            type="number"
            min={1}
            value={dp}
            onChange={(e) => setDp(e.target.value)}
          />
        </div>
      </div>

      <div className="flex items-center gap-6">
        <div className="flex items-center gap-2">
          <Switch
            id="budget-expert-parallel"
            checked={expertParallel}
            onCheckedChange={setExpertParallel}
          />
          <Label htmlFor="budget-expert-parallel">Expert Parallel</Label>
        </div>
        <div className="flex items-center gap-2">
          <Switch
            id="budget-enforce-eager"
            checked={enforceEager}
            onCheckedChange={setEnforceEager}
          />
          <Label htmlFor="budget-enforce-eager">Enforce Eager</Label>
        </div>
      </div>

      <div>
        <Label htmlFor="seq-lengths">Sequence Lengths (comma-separated)</Label>
        <Input
          id="seq-lengths"
          placeholder="auto (powers of 2 up to model max)"
          value={seqLengths}
          onChange={(e) => setSeqLengths(e.target.value)}
        />
      </div>
      <div>
        <Label htmlFor="seq-counts">Sequence Counts (comma-separated)</Label>
        <Input
          id="seq-counts"
          placeholder="auto (1,4,8,16,32,64,128,256,512)"
          value={seqCounts}
          onChange={(e) => setSeqCounts(e.target.value)}
        />
      </div>

      {error && (
        <div className="rounded-md bg-red-50 p-3 text-sm text-red-700">
          {error}
        </div>
      )}

      <Button
        type="submit"
        disabled={loading || !modelId.trim() || !gpuMemory}
        className="w-full"
      >
        {loading ? (
          <>
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            Computing Budget...
          </>
        ) : (
          "Compute Token Budget"
        )}
      </Button>
    </form>
  );
}

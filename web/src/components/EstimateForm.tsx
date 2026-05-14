import { Loader2 } from "lucide-react";
import { type FormEvent, useState } from "react";
import type { EstimateResponse } from "../lib/api";
import { fetchEstimate } from "../lib/api";
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
  onResult: (r: EstimateResponse) => void;
}

export function EstimateForm({ onResult }: Props) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [modelId, setModelId] = useState("");
  const [maxSeqLen, setMaxSeqLen] = useState("");
  const [maxActiveSeqs, setMaxActiveSeqs] = useState("256");
  const [quantization, setQuantization] = useState("none");
  const [dtype, setDtype] = useState("auto");
  const [kvCacheDtype, setKvCacheDtype] = useState("auto");
  const [tp, setTp] = useState("1");
  const [pp, setPp] = useState("1");
  const [dp, setDp] = useState("1");
  const [expertParallel, setExpertParallel] = useState(false);
  const [enforceEager, setEnforceEager] = useState(false);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!modelId.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const res = await fetchEstimate({
        model_id: modelId.trim(),
        max_seq_len: maxSeqLen ? parseInt(maxSeqLen) : null,
        max_active_seqs: parseInt(maxActiveSeqs) || 256,
        quantization: quantization === "none" ? null : quantization,
        dtype: dtype === "auto" ? null : dtype,
        kv_cache_dtype: kvCacheDtype === "auto" ? null : kvCacheDtype,
        tensor_parallel_size: parseInt(tp) || 1,
        pipeline_parallel_size: parseInt(pp) || 1,
        data_parallel_size: parseInt(dp) || 1,
        enable_expert_parallel: expertParallel,
        enforce_eager: enforceEager,
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
        <Label htmlFor="model-id">Model ID</Label>
        <Input
          id="model-id"
          placeholder="meta-llama/Llama-3.1-8B"
          value={modelId}
          onChange={(e) => setModelId(e.target.value)}
          required
        />
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div>
          <Label htmlFor="max-seq-len">Max Sequence Length</Label>
          <Input
            id="max-seq-len"
            type="number"
            placeholder="auto"
            value={maxSeqLen}
            onChange={(e) => setMaxSeqLen(e.target.value)}
          />
        </div>
        <div>
          <Label htmlFor="max-active-seqs">Max Active Sequences</Label>
          <Input
            id="max-active-seqs"
            type="number"
            value={maxActiveSeqs}
            onChange={(e) => setMaxActiveSeqs(e.target.value)}
          />
        </div>
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
          <Label htmlFor="tp">Tensor Parallel</Label>
          <Input
            id="tp"
            type="number"
            min={1}
            value={tp}
            onChange={(e) => setTp(e.target.value)}
          />
        </div>
        <div>
          <Label htmlFor="pp">Pipeline Parallel</Label>
          <Input
            id="pp"
            type="number"
            min={1}
            value={pp}
            onChange={(e) => setPp(e.target.value)}
          />
        </div>
        <div>
          <Label htmlFor="dp">Data Parallel</Label>
          <Input
            id="dp"
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
            id="expert-parallel"
            checked={expertParallel}
            onCheckedChange={setExpertParallel}
          />
          <Label htmlFor="expert-parallel">Expert Parallel</Label>
        </div>
        <div className="flex items-center gap-2">
          <Switch
            id="enforce-eager"
            checked={enforceEager}
            onCheckedChange={setEnforceEager}
          />
          <Label htmlFor="enforce-eager">Enforce Eager</Label>
        </div>
      </div>

      {error && (
        <div className="rounded-md bg-red-50 p-3 text-sm text-red-700">
          {error}
        </div>
      )}

      <Button type="submit" disabled={loading || !modelId.trim()} className="w-full">
        {loading ? (
          <>
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            Estimating...
          </>
        ) : (
          "Estimate Memory"
        )}
      </Button>
    </form>
  );
}

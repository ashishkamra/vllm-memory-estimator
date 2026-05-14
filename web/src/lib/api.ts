export interface EstimateRequest {
  model_id: string;
  max_seq_len?: number | null;
  max_active_seqs?: number;
  dtype?: string | null;
  kv_cache_dtype?: string | null;
  quantization?: string | null;
  tensor_parallel_size?: number;
  pipeline_parallel_size?: number;
  data_parallel_size?: number;
  enable_expert_parallel?: boolean;
  enforce_eager?: boolean;
  block_size?: number | null;
  max_num_batched_tokens?: number | null;
}

export interface ComponentEstimate {
  nominal_gib: number;
  lower_gib: number;
  upper_gib: number;
}

export interface EstimateResponse {
  ok: boolean;
  error?: string;
  model_id?: string;
  architecture?: string;
  parameter_count?: number;
  max_active_seqs?: number;
  max_seq_len?: number;
  tensor_parallel_size?: number;
  pipeline_parallel_size?: number;
  data_parallel_size?: number;
  enable_expert_parallel?: boolean;
  total_gpus?: number;
  quantization?: {
    method: string;
    weight_dtype: { name: string; bits: number };
    activation_dtype: { name: string; bits: number };
    kv_cache_dtype: { name: string; bits: number };
  };
  estimate?: {
    parameters: ComponentEstimate;
    activations: ComponentEstimate;
    kv_cache: ComponentEstimate;
    kv_cache_spec_type: string;
    workspace: ComponentEstimate;
    total: ComponentEstimate;
    vllm_overhead: ComponentEstimate;
    total_with_vllm: ComponentEstimate;
    total_cluster?: ComponentEstimate;
  };
}

export interface BudgetRequest {
  model_id: string;
  gpu_memory_gib: number;
  tensor_parallel_size?: number;
  pipeline_parallel_size?: number;
  data_parallel_size?: number;
  enable_expert_parallel?: boolean;
  quantization?: string | null;
  dtype?: string | null;
  kv_cache_dtype?: string | null;
  enforce_eager?: boolean;
  block_size?: number | null;
  seq_lengths?: number[] | null;
  seq_counts?: number[] | null;
  max_num_batched_tokens?: number | null;
}

export interface BudgetCell {
  seq_len: number;
  seqs: number;
  total_gib: number;
  fits: boolean;
  remaining_gib: number;
  parameter_gib: number;
  activation_gib: number;
  kv_cache_gib: number;
  overhead_gib: number;
}

export interface BudgetResponse {
  ok: boolean;
  error?: string;
  model_id?: string;
  gpu_memory_gib?: number;
  architecture?: string;
  quantization?: string;
  kv_cache_spec_type?: string;
  tensor_parallel_size?: number;
  pipeline_parallel_size?: number;
  data_parallel_size?: number;
  enable_expert_parallel?: boolean;
  total_gpus?: number;
  model_max_seq_len?: number;
  seq_lengths?: number[];
  seq_counts?: number[];
  matrix?: BudgetCell[][];
  max_seqs_per_context?: { seq_len: number; max_seqs: number | null }[];
}

async function post<T>(url: string, body: unknown): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(`HTTP ${res.status}: ${res.statusText}`);
  }
  return res.json();
}

export function fetchEstimate(req: EstimateRequest): Promise<EstimateResponse> {
  return post("/api/estimate", req);
}

export function fetchBudget(req: BudgetRequest): Promise<BudgetResponse> {
  return post("/api/budget", req);
}

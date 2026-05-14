import type { BudgetCell, BudgetResponse } from "../lib/api";

interface Props {
  data: BudgetResponse | null;
}

function cellColor(cell: BudgetCell, gpuMem: number): string {
  if (!cell.fits) return "rgba(200,0,0,0.08)";
  const pct = cell.total_gib / gpuMem;
  const green = Math.max(55, Math.min(255, Math.round(200 * (1 - pct)) + 55));
  return `rgba(0,${green},0,0.12)`;
}

function Tooltip({ cell }: { cell: BudgetCell }) {
  return (
    <div className="text-xs leading-relaxed">
      <div>Parameters: {cell.parameter_gib.toFixed(2)} GiB</div>
      <div>KV Cache: {cell.kv_cache_gib.toFixed(2)} GiB</div>
      <div>Activations: {cell.activation_gib.toFixed(2)} GiB</div>
      <div>Overhead: {cell.overhead_gib.toFixed(2)} GiB</div>
      <div className="mt-1 border-t pt-1 font-semibold">
        Total: {cell.total_gib.toFixed(2)} GiB
      </div>
    </div>
  );
}

export function BudgetMatrix({ data }: Props) {
  if (!data || !data.ok || !data.matrix) {
    return (
      <div className="flex items-center justify-center rounded-lg border border-dashed p-12 text-gray-400">
        Enter model details and GPU memory to see the token budget matrix.
      </div>
    );
  }

  const gpuMem = data.gpu_memory_gib!;
  const seqLengths = data.seq_lengths!;
  const seqCounts = data.seq_counts!;
  const matrix = data.matrix!;
  const maxSeqs = data.max_seqs_per_context!;

  const dims: string[] = [];
  if (data.tensor_parallel_size && data.tensor_parallel_size > 1)
    dims.push(`TP=${data.tensor_parallel_size}`);
  if (data.pipeline_parallel_size && data.pipeline_parallel_size > 1)
    dims.push(`PP=${data.pipeline_parallel_size}`);
  if (data.data_parallel_size && data.data_parallel_size > 1)
    dims.push(`DP=${data.data_parallel_size}`);
  if (data.enable_expert_parallel) dims.push("EP");

  const totalGpus = data.total_gpus ?? 1;

  return (
    <div className="space-y-4">
      {/* Multi-GPU banner */}
      {totalGpus > 1 && (
        <div className="rounded-lg border border-blue-300 bg-blue-50 px-4 py-3 text-sm">
          <strong>{totalGpus} GPUs</strong> ({dims.join(", ")}) — all values
          below are <strong>per GPU</strong> ({gpuMem} GiB each)
        </div>
      )}

      {/* Summary cards */}
      <div className="flex flex-wrap gap-3">
        <Card label="Model" value={data.model_id ?? ""} />
        <Card label="GPU Memory" value={`${gpuMem} GiB / GPU`} />
        <Card label="Architecture" value={data.architecture ?? "unknown"} />
        <Card label="Quantization" value={data.quantization ?? "dense"} />
        <Card label="KV Cache" value={data.kv_cache_spec_type ?? "full"} />
        <Card
          label="Max Context"
          value={data.model_max_seq_len?.toLocaleString() ?? "-"}
        />
        {dims.length > 0 && (
          <Card label="Parallelism" value={dims.join(", ")} />
        )}
      </div>

      {/* Matrix table */}
      <div className="overflow-x-auto rounded-lg border">
        <table className="w-full border-collapse text-sm">
          <thead>
            <tr className="bg-gray-50">
              <th className="border-b px-3 py-2 text-right font-semibold">
                Context
              </th>
              {seqCounts.map((sc) => (
                <th
                  key={sc}
                  className="border-b px-3 py-2 text-right font-semibold"
                >
                  {sc} seq
                </th>
              ))}
              <th className="border-b bg-blue-50 px-3 py-2 text-right font-semibold">
                Max Seqs
              </th>
            </tr>
          </thead>
          <tbody>
            {seqLengths.map((sl, rowIdx) => (
              <tr key={sl} className="hover:bg-gray-50/50">
                <td className="border-b bg-gray-50 px-3 py-2 text-right font-semibold tabular-nums">
                  {sl.toLocaleString()}
                </td>
                {seqCounts.map((sc, colIdx) => {
                  const cell = matrix[rowIdx][colIdx];
                  return (
                    <td
                      key={sc}
                      className="group relative border-b px-3 py-2 text-right tabular-nums"
                      style={{ backgroundColor: cellColor(cell, gpuMem) }}
                    >
                      {cell.fits ? cell.total_gib.toFixed(1) : "---"}
                      <div className="pointer-events-none absolute right-0 bottom-full z-10 mb-1 hidden rounded-md border bg-white p-2 shadow-lg group-hover:block">
                        <Tooltip cell={cell} />
                      </div>
                    </td>
                  );
                })}
                <td className="border-b bg-blue-50 px-3 py-2 text-right font-semibold tabular-nums">
                  {formatMaxSeqs(maxSeqs[rowIdx]?.max_seqs)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="text-sm text-gray-500">
        Each cell: estimated <strong>per-GPU</strong> memory (GiB).{" "}
        <strong>---</strong> = exceeds {gpuMem} GiB limit. Hover over cells for
        component breakdown.
      </p>
    </div>
  );
}

function formatMaxSeqs(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || ms === 0) return "---";
  if (ms >= 4096) return "4096+";
  return ms.toLocaleString();
}

function Card({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border bg-gray-50 px-4 py-2">
      <div className="text-xs text-gray-500">{label}</div>
      <div className="text-sm font-semibold truncate max-w-[200px]" title={value}>
        {value}
      </div>
    </div>
  );
}

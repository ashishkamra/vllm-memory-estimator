import type { ComponentEstimate, EstimateResponse } from "../lib/api";

interface Props {
  data: EstimateResponse | null;
}

function Row({
  label,
  comp,
  maxNominal,
  color,
}: {
  label: string;
  comp: ComponentEstimate;
  maxNominal: number;
  color: string;
}) {
  const pct = maxNominal > 0 ? (comp.nominal_gib / maxNominal) * 100 : 0;
  return (
    <tr>
      <td className="py-2 pr-4 text-sm font-medium">{label}</td>
      <td className="px-2 py-2 text-right text-sm tabular-nums">
        {comp.nominal_gib.toFixed(2)}
      </td>
      <td className="px-2 py-2 text-right text-sm tabular-nums text-gray-500">
        {comp.lower_gib.toFixed(2)}
      </td>
      <td className="px-2 py-2 text-right text-sm tabular-nums text-gray-500">
        {comp.upper_gib.toFixed(2)}
      </td>
      <td className="w-40 py-2 pl-4">
        <div className="h-4 w-full rounded bg-gray-100">
          <div
            className="h-full rounded"
            style={{ width: `${Math.min(pct, 100)}%`, backgroundColor: color }}
          />
        </div>
      </td>
    </tr>
  );
}

export function EstimateResult({ data }: Props) {
  if (!data || !data.ok || !data.estimate) {
    return (
      <div className="flex items-center justify-center rounded-lg border border-dashed p-12 text-gray-400">
        Enter model details and click Estimate to see results.
      </div>
    );
  }

  const est = data.estimate;
  const maxNominal = est.total_with_vllm.nominal_gib;

  const colors = {
    parameters: "#3b82f6",
    activations: "#f59e0b",
    kv_cache: "#10b981",
    workspace: "#8b5cf6",
    vllm_overhead: "#6b7280",
  };

  const totalGpus = data.total_gpus ?? 1;

  return (
    <div className="space-y-6">
      {/* Summary cards */}
      <div className="flex flex-wrap gap-3">
        <Card label="Model" value={data.model_id ?? ""} />
        <Card label="Architecture" value={data.architecture ?? "unknown"} />
        <Card
          label="Parameters"
          value={
            data.parameter_count
              ? `${(data.parameter_count / 1e9).toFixed(1)}B`
              : "-"
          }
        />
        <Card
          label="Quantization"
          value={data.quantization?.method ?? "dense"}
        />
        <Card label="Max Seq Len" value={data.max_seq_len?.toLocaleString() ?? "-"} />
        <Card label="Max Seqs" value={data.max_active_seqs?.toString() ?? "-"} />
        {totalGpus > 1 && (
          <Card label="GPUs" value={totalGpus.toString()} />
        )}
      </div>

      {/* Memory breakdown table */}
      <div className="overflow-x-auto rounded-lg border">
        <table className="w-full">
          <thead>
            <tr className="border-b bg-gray-50">
              <th className="py-2 pr-4 pl-4 text-left text-sm font-semibold">
                Component
              </th>
              <th className="px-2 py-2 text-right text-sm font-semibold">
                Nominal (GiB)
              </th>
              <th className="px-2 py-2 text-right text-sm font-semibold">
                Lower
              </th>
              <th className="px-2 py-2 text-right text-sm font-semibold">
                Upper
              </th>
              <th className="py-2 pl-4 pr-4 text-left text-sm font-semibold">
                Relative
              </th>
            </tr>
          </thead>
          <tbody className="pl-4">
            <tr><td colSpan={5} className="pl-4"><div className="border-b" /></td></tr>
            <Row label="Parameters" comp={est.parameters} maxNominal={maxNominal} color={colors.parameters} />
            <Row label="Activations" comp={est.activations} maxNominal={maxNominal} color={colors.activations} />
            <Row
              label={est.kv_cache_spec_type === "full" ? "KV Cache" : `KV Cache (${est.kv_cache_spec_type})`}
              comp={est.kv_cache}
              maxNominal={maxNominal}
              color={colors.kv_cache}
            />
            <Row label="Workspace" comp={est.workspace} maxNominal={maxNominal} color={colors.workspace} />
            <tr><td colSpan={5} className="pl-4"><div className="border-b" /></td></tr>
            <Row label="vLLM Overhead" comp={est.vllm_overhead} maxNominal={maxNominal} color={colors.vllm_overhead} />
            <tr><td colSpan={5} className="pl-4"><div className="border-b border-black" /></td></tr>
            <tr className="font-semibold">
              <td className="py-2 pr-4 text-sm">
                {totalGpus > 1 ? "Total (per GPU)" : "Total (vLLM)"}
              </td>
              <td className="px-2 py-2 text-right text-sm tabular-nums">
                {est.total_with_vllm.nominal_gib.toFixed(2)}
              </td>
              <td className="px-2 py-2 text-right text-sm tabular-nums text-gray-500">
                {est.total_with_vllm.lower_gib.toFixed(2)}
              </td>
              <td className="px-2 py-2 text-right text-sm tabular-nums text-gray-500">
                {est.total_with_vllm.upper_gib.toFixed(2)}
              </td>
              <td />
            </tr>
            {totalGpus > 1 && est.total_cluster && (
              <tr className="font-semibold text-blue-700">
                <td className="py-2 pr-4 text-sm">
                  Total Cluster ({totalGpus} GPUs)
                </td>
                <td className="px-2 py-2 text-right text-sm tabular-nums">
                  {est.total_cluster.nominal_gib.toFixed(2)}
                </td>
                <td className="px-2 py-2 text-right text-sm tabular-nums opacity-70">
                  {est.total_cluster.lower_gib.toFixed(2)}
                </td>
                <td className="px-2 py-2 text-right text-sm tabular-nums opacity-70">
                  {est.total_cluster.upper_gib.toFixed(2)}
                </td>
                <td />
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
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

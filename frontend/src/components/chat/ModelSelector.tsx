import { useModels } from "../../api/hooks";
import { resolveModel, useModelSettings } from "../../lib/modelSettings";
import { ErrorBox, Notice, Spinner } from "../common";

function NumberSetting({
  label,
  value,
  onChange,
  min,
  max,
  step,
  hint,
}: {
  label: string;
  value: number | null;
  onChange: (value: number | null) => void;
  min: number;
  max: number;
  step: number;
  hint: string;
}) {
  return (
    <label className="setting">
      {label}
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        placeholder={hint}
        value={value ?? ""}
        onChange={(e) => {
          const raw = e.target.value;
          const parsed = Number(raw);
          onChange(raw === "" || Number.isNaN(parsed) ? null : parsed);
        }}
      />
    </label>
  );
}

export function ModelSelector() {
  const models = useModels();
  const { settings, update } = useModelSettings();

  if (models.isPending) return <Spinner label="Checking model providers…" />;
  if (models.isError) {
    return <ErrorBox error={models.error} title="Could not list models." onRetry={() => models.refetch()} />;
  }

  const { provider, model } = resolveModel(settings, models.data);
  if (!provider) return <Notice kind="warning">No LLM provider is configured on the backend.</Notice>;

  return (
    <div className="model-selector">
      <label className="setting">
        Provider
        <select
          value={provider.name}
          onChange={(e) => update({ provider: e.target.value, model: null })}
        >
          {models.data.providers.map((p) => (
            <option key={p.name} value={p.name}>
              {p.name} ({p.is_local ? "local" : "cloud"})
              {p.available ? "" : " – unreachable"}
            </option>
          ))}
        </select>
      </label>

      <label className="setting">
        Model
        <select
          value={model ?? ""}
          disabled={provider.models.length === 0}
          onChange={(e) => update({ provider: provider.name, model: e.target.value })}
        >
          {model && !provider.models.includes(model) && <option value={model}>{model}</option>}
          {provider.models.length === 0 && !model && <option value="">No models available</option>}
          {provider.models.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </select>
      </label>

      {!provider.available && (
        <Notice kind="warning">
          {provider.name} is unreachable{provider.error ? `: ${provider.error}` : "."}
        </Notice>
      )}
      {!provider.is_local && (
        <Notice kind="warning">
          <strong>Cloud processing.</strong> Your question and the retrieved passages from your
          documents are sent to {provider.name}.
        </Notice>
      )}
      {models.data.local_only && (
        <p className="muted small">Local-only mode is on: cloud providers are disabled on the backend.</p>
      )}

      <details className="advanced">
        <summary>Generation settings</summary>
        <NumberSetting
          label="Temperature"
          value={settings.temperature}
          onChange={(temperature) => update({ temperature })}
          min={0}
          max={2}
          step={0.1}
          hint="default"
        />
        <NumberSetting
          label="Max answer tokens"
          value={settings.maxTokens}
          onChange={(maxTokens) => update({ maxTokens })}
          min={1}
          max={32768}
          step={64}
          hint="default"
        />
        <NumberSetting
          label="Passages (top-k)"
          value={settings.topK}
          onChange={(topK) => update({ topK })}
          min={1}
          max={50}
          step={1}
          hint="default"
        />
        <button
          type="button"
          className="btn btn-small"
          onClick={() => update({ temperature: null, maxTokens: null, topK: null })}
        >
          Reset to defaults
        </button>
      </details>
    </div>
  );
}

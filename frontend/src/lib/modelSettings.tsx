import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";

import type { ModelsResponse, ProviderModels } from "../api/types";

/** The user's model choice. null fields mean "use the backend default". */
export interface ModelSettings {
  provider: string | null;
  model: string | null;
  temperature: number | null;
  maxTokens: number | null;
  topK: number | null;
}

export const DEFAULT_SETTINGS: ModelSettings = {
  provider: null,
  model: null,
  temperature: null,
  maxTokens: null,
  topK: null,
};

const STORAGE_KEY = "research-assistant.model-settings";

function load(): ModelSettings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return { ...DEFAULT_SETTINGS, ...(JSON.parse(raw) as Partial<ModelSettings>) };
  } catch {
    // Storage unavailable or corrupt: fall back to defaults.
  }
  return DEFAULT_SETTINGS;
}

function save(settings: ModelSettings): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  } catch {
    // Not fatal: the choice just will not survive a reload.
  }
}

export interface ResolvedModel {
  provider: ProviderModels | null;
  model: string | null;
}

/**
 * Resolve the saved choice against what the backend currently offers. A saved
 * provider or model that no longer exists falls back to the default, so the
 * interface never shows a model that would not actually be used.
 */
export function resolveModel(settings: ModelSettings, models: ModelsResponse | undefined): ResolvedModel {
  if (!models || models.providers.length === 0) return { provider: null, model: null };
  const provider =
    models.providers.find((p) => p.name === settings.provider) ??
    models.providers.find((p) => p.name === models.default_provider) ??
    models.providers[0]!;

  const saved = settings.provider === provider.name ? settings.model : null;
  // While a provider is unreachable its models cannot be checked; keep the saved
  // choice rather than silently switching to another model.
  if (saved && (provider.models.includes(saved) || !provider.available)) {
    return { provider, model: saved };
  }
  return { provider, model: provider.default_model ?? provider.models[0] ?? null };
}

interface ModelSettingsContextValue {
  settings: ModelSettings;
  update: (patch: Partial<ModelSettings>) => void;
}

const ModelSettingsContext = createContext<ModelSettingsContextValue | null>(null);

export function ModelSettingsProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<ModelSettings>(load);
  const update = useCallback((patch: Partial<ModelSettings>) => {
    setSettings((current) => {
      const next = { ...current, ...patch };
      save(next);
      return next;
    });
  }, []);
  const value = useMemo(() => ({ settings, update }), [settings, update]);
  return <ModelSettingsContext.Provider value={value}>{children}</ModelSettingsContext.Provider>;
}

export function useModelSettings(): ModelSettingsContextValue {
  const value = useContext(ModelSettingsContext);
  if (!value) throw new Error("useModelSettings must be used inside ModelSettingsProvider");
  return value;
}

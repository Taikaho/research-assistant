import { NavLink, Outlet } from "react-router";

import { useModels } from "../api/hooks";
import { resolveModel, useModelSettings } from "../lib/modelSettings";
import { LocalityBadge } from "./common";

/** Always-visible indicator of the model that will answer the next question. */
export function ActiveModelBadge() {
  const { settings } = useModelSettings();
  const models = useModels();

  if (models.isPending) return <span className="active-model muted">Loading models…</span>;
  if (models.isError) return <span className="active-model text-error">Models unavailable</span>;

  const { provider, model } = resolveModel(settings, models.data);
  if (!provider) return <span className="active-model text-error">No LLM provider configured</span>;
  return (
    <span className="active-model" data-testid="active-model">
      <LocalityBadge isLocal={provider.is_local} />
      <span>
        {provider.name} · {model ?? "no model available"}
      </span>
      {!provider.available && <span className="text-error"> (unreachable)</span>}
    </span>
  );
}

export function Layout() {
  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">Research Assistant</div>
        <nav className="main-nav">
          <NavLink to="/chat">Chat</NavLink>
          <NavLink to="/library">Library</NavLink>
          <NavLink to="/collections">Collections</NavLink>
        </nav>
        <ActiveModelBadge />
      </header>
      <main className="app-main">
        <Outlet />
      </main>
    </div>
  );
}

async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: options.body && !(options.body instanceof FormData) ? { "Content-Type": "application/json" } : undefined,
    ...options,
  });
  if (response.status === 204) return null;
  const text = await response.text();
  let payload = null;
  try {
    payload = text ? JSON.parse(text) : null;
  } catch {
    payload = { detail: text };
  }
  if (!response.ok) {
    const detail = payload && payload.detail;
    let message = typeof detail === "string" ? detail : `Request failed (${response.status})`;
    if (detail && detail.rejected) {
      message = detail.rejected.map((r) => `${r.name}: ${r.reason}`).join("; ");
    }
    const error = new Error(message);
    error.status = response.status;
    throw error;
  }
  return payload;
}

export const api = {
  health: () => request("/api/health"),
  collections: () => request("/api/collections"),
  createCollection: (name) => request("/api/collections", { method: "POST", body: JSON.stringify({ name }) }),
  deleteCollection: (id) => request(`/api/collections/${id}`, { method: "DELETE" }),
  collection: (id) => request(`/api/collections/${id}`),
  stats: (id) => request(`/api/collections/${id}/stats`),
  upload: (id, files) => {
    const form = new FormData();
    for (const file of files) form.append("files", file, file.name);
    return request(`/api/collections/${id}/documents`, { method: "POST", body: form });
  },
  deleteDocument: (id, doc) => request(`/api/collections/${id}/documents/${doc}`, { method: "DELETE" }),
  report: (id, doc) => request(`/api/collections/${id}/documents/${doc}/report`),
  pageUrl: (id, doc, page) => `/api/collections/${id}/documents/${doc}/page/${page}`,
  ask: (id, question, maxHops) => request(`/api/collections/${id}/ask`, { method: "POST", body: JSON.stringify({ question, max_hops: maxHops }) }),
  entities: (id, q, limit = 24) => request(`/api/collections/${id}/entities?q=${encodeURIComponent(q)}&limit=${limit}`),
  entity: (id, eid) => request(`/api/collections/${id}/entities/${eid}`),
  graph: (id, focus = []) => request(`/api/collections/${id}/graph?focus=${focus.join(",")}&limit=60`),
};

import { api } from "/static/api.js";
import { TrailMap } from "/static/map.js";

const $ = (sel) => document.querySelector(sel);
const state = { collections: [], cid: null, docs: [], jobs: [], poll: null, result: null, map: null, entityTimer: null, statsFor: null };

const KIND_LABEL = { pdf: "PDF", image: "IMG", text: "TXT", pending: "…" };
const MODE_LABEL = {
  lookup: "Lookup",
  choose: "Comparison",
  same: "Same-attribute check",
  both: "Shared-property check",
  verify: "Yes / no check",
  connect: "Connection",
};

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch]);
}

function escapeRegex(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function highlight(text, terms, answerTerms = []) {
  const all = [...new Set([...answerTerms, ...terms].filter((t) => t && t.length > 1))].sort((a, b) => b.length - a.length);
  if (!all.length) return escapeHtml(text);
  const pattern = new RegExp(`(?<![\\p{L}\\p{N}])(${all.map(escapeRegex).join("|")})`, "giu");
  const answers = new Set(answerTerms.map((t) => t.toLowerCase()));
  return text
    .split(pattern)
    .map((piece, i) => {
      if (i % 2 === 0) return escapeHtml(piece);
      const cls = answers.has(piece.toLowerCase()) ? ' class="is-answer"' : "";
      return `<mark${cls}>${escapeHtml(piece)}</mark>`;
    })
    .join("");
}

function plural(count, word) {
  return `${count.toLocaleString()} ${word}${count === 1 ? "" : "s"}`;
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function toast(message, isError = false) {
  const el = $("#toast");
  el.textContent = message;
  el.classList.toggle("is-error", isError);
  el.classList.add("is-visible");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => el.classList.remove("is-visible"), isError ? 6000 : 3200);
}

function applyTheme(theme) {
  if (theme) document.documentElement.dataset.theme = theme;
  else delete document.documentElement.dataset.theme;
  if (state.map) state.map.draw();
}

function toggleTheme() {
  const current = document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  const next = current === "dark" ? "light" : "dark";
  try { localStorage.setItem("waymark.theme", next); } catch {}
  applyTheme(next);
}

async function loadCollections(selectId) {
  state.collections = await api.collections();
  let stored = null;
  try { stored = localStorage.getItem("waymark.collection"); } catch {}
  const wanted = selectId || state.cid || stored;
  state.cid = state.collections.some((c) => c.id === wanted) ? wanted : state.collections[0]?.id;
  const select = $("#collection");
  select.innerHTML = state.collections
    .map((c) => `<option value="${c.id}" ${c.id === state.cid ? "selected" : ""}>${escapeHtml(c.name)} (${c.documents})</option>`)
    .join("");
  try { localStorage.setItem("waymark.collection", state.cid); } catch {}
  await refreshCollection();
}

async function refreshCollection() {
  if (!state.cid) return;
  const detail = await api.collection(state.cid);
  state.docs = detail.documents;
  state.jobs = detail.jobs;
  const option = [...$("#collection").options].find((o) => o.value === state.cid);
  if (option) option.textContent = `${detail.name} (${state.docs.length})`;
  renderDocs();
  const busy = state.jobs.some((j) => j.status === "queued" || j.status === "running");
  clearTimeout(state.poll);
  if (busy) {
    state.poll = setTimeout(() => refreshCollection().catch(() => {}), 1200);
  } else {
    const ready = state.docs.filter((d) => d.status === "ready").map((d) => d.id).join(",");
    if (state.statsFor !== `${state.cid}:${ready}`) {
      state.statsFor = `${state.cid}:${ready}`;
      refreshStats();
      searchEntities("");
    }
  }
}

function renderDocs() {
  const list = $("#docs");
  const jobsByDoc = new Map(state.jobs.map((j) => [j.document, j]));
  list.innerHTML = state.docs
    .map((d) => {
      const job = jobsByDoc.get(d.id);
      const running = d.status === "queued" || d.status === "running";
      let meta = "";
      if (running) {
        meta = `${escapeHtml(job ? job.stage : "queued")}`;
      } else if (d.status === "failed") {
        meta = `<span class="doc-meta is-error">${escapeHtml(d.error || "failed")}</span>`;
      } else {
        const extras = [];
        extras.push(d.kind === "image" ? "image" : plural(d.pages, "page"));
        extras.push(plural(d.passages, "passage"));
        if (d.meta && d.meta.ocr_pages) extras.push(`${d.meta.ocr_pages} read by OCR`);
        if (d.meta && d.meta.figures) extras.push(`${d.meta.figures} figures`);
        extras.push(formatBytes(d.bytes));
        meta = escapeHtml(extras.join(", "));
      }
      const progress = running ? `<div class="doc-progress"><i style="width:${Math.round((job ? job.progress : 0) * 100)}%"></i></div>` : "";
      const reportBtn = d.has_report ? `<button data-report="${d.id}" aria-label="Image analysis for ${escapeHtml(d.name)}" title="Image analysis"><svg viewBox="0 0 24 24" width="16" height="16"><rect x="3" y="5" width="18" height="14" rx="2" fill="none" stroke="currentColor" stroke-width="1.8"/><circle cx="9" cy="10" r="2" fill="currentColor"/><path d="M5 17l5-5 4 4 2-2 3 3" fill="none" stroke="currentColor" stroke-width="1.8"/></svg></button>` : "";
      const previewBtn = d.status === "ready" && (d.kind === "pdf" || d.kind === "image") && !d.has_report ? `<button data-preview="${d.id}" aria-label="Preview ${escapeHtml(d.name)}" title="Preview first page"><svg viewBox="0 0 24 24" width="16" height="16"><path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12z" fill="none" stroke="currentColor" stroke-width="1.8"/><circle cx="12" cy="12" r="3" fill="currentColor"/></svg></button>` : "";
      const deleteBtn = running ? "" : `<button data-delete="${d.id}" aria-label="Remove ${escapeHtml(d.name)}" title="Remove"><svg viewBox="0 0 24 24" width="16" height="16"><path d="M6 6l12 12M18 6L6 18" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg></button>`;
      return `<li class="doc">
        <span class="doc-kind" data-kind="${d.kind}">${KIND_LABEL[d.kind] || "DOC"}</span>
        <div><div class="doc-name" title="${escapeHtml(d.name)}">${escapeHtml(d.name)}</div><div class="doc-meta">${meta}</div>${progress}</div>
        <div class="doc-actions">${reportBtn}${previewBtn}${deleteBtn}</div>
      </li>`;
    })
    .join("");
  $("#docs-empty").hidden = state.docs.length > 0;
}

async function refreshStats() {
  const el = $("#stats");
  try {
    const stats = await api.stats(state.cid);
    if (stats.empty) {
      el.innerHTML = "";
      renderSuggestions([]);
      return;
    }
    el.innerHTML = `<b>${stats.passages.toLocaleString()}</b> passages, <b>${stats.entities.toLocaleString()}</b> entities, <b>${stats.edges.toLocaleString()}</b> links`;
    const top = await api.entities(state.cid, "", 12);
    renderSuggestions(top.filter((e) => e.kind === "named"));
  } catch {
    el.textContent = "";
  }
}

function renderSuggestions(entities) {
  const box = $("#suggestions");
  if (entities.length < 2) {
    box.innerHTML = "";
    return;
  }
  const [a, b, c] = entities;
  const prompts = [`How is ${a.name} connected to ${b.name}?`];
  if (c) prompts.push(`What links ${b.name} and ${c.name}?`);
  const person = entities.find((e) => e.class === "PERSON");
  if (person) prompts.push(`Where did ${person.name} work?`);
  box.innerHTML = prompts.map((p) => `<button type="button" data-suggest="${escapeHtml(p)}">${escapeHtml(p)}</button>`).join("");
}

async function upload(files) {
  if (!files.length || !state.cid) return;
  try {
    const result = await api.upload(state.cid, files);
    const accepted = result.jobs.length;
    const rejected = result.rejected.map((r) => `${r.name} (${r.reason})`);
    toast(rejected.length ? `Queued ${accepted}. Skipped ${rejected.join(", ")}` : `Queued ${accepted} ${accepted === 1 ? "file" : "files"} for indexing`, rejected.length > 0 && !accepted);
    await refreshCollection();
  } catch (error) {
    toast(error.message, true);
  }
}

async function ask(event) {
  event?.preventDefault();
  const question = $("#question").value.trim();
  if (question.length < 2) {
    $("#question").focus();
    return;
  }
  const button = $("#ask-btn");
  button.disabled = true;
  button.innerHTML = `<span class="spinner" aria-hidden="true"></span> Tracing`;
  try {
    const result = await api.ask(state.cid, question, Number($("#hops").value));
    state.result = result;
    renderResult(result);
  } catch (error) {
    toast(error.message, true);
  } finally {
    button.disabled = false;
    button.textContent = "Trace";
  }
}

function trailIds(chain) {
  if (!chain || !chain.hops.length) return [];
  const ids = [];
  if (chain.hops[0].from) ids.push(chain.hops[0].from.id);
  for (const hop of chain.hops) if (hop.to && !ids.includes(hop.to.id)) ids.push(hop.to.id);
  return ids;
}

function sourceButton(sentence) {
  const origin = sentence.origin && sentence.origin !== "text" ? `<span class="origin">${sentence.origin === "ocr" ? "OCR" : "figure"}</span>` : "";
  return `<button class="source" data-doc="${sentence.doc}" data-page="${sentence.page}">${escapeHtml(sentence.doc_name || sentence.doc)}, page ${sentence.page} ${origin}</button>`;
}

function renderResult(result) {
  $("#result").hidden = false;
  const answer = result.answer;
  const chain = result.chains[0];
  const mode = MODE_LABEL[result.mode] || result.mode;
  const hops = chain ? chain.hops.length : 0;
  const answerText = answer ? (answer.kind === "bool" ? answer.text.charAt(0).toUpperCase() + answer.text.slice(1) : answer.text) : "No trail reached an answer";
  const confidence = answer ? Math.round(answer.confidence * 100) : 0;
  $("#answer").innerHTML = `
    <div class="answer-text ${answer ? "" : "is-empty"} reveal">${escapeHtml(answerText)}</div>
    <div class="confidence">${answer ? `Confidence ${confidence}%` : ""}<div class="confidence-bar"><i style="width:${confidence}%"></i></div></div>
    <div class="answer-sub">${escapeHtml(mode)}${hops ? `, ${hops} ${hops === 1 ? "hop" : "hops"}` : ""}, ${result.elapsed_ms} ms${result.alternatives.length ? `. Also considered: ${result.alternatives.slice(0, 3).map((a) => escapeHtml(a.text)).join(", ")}` : ""}</div>`;

  $("#anchors").innerHTML = result.anchors.length
    ? `Started from ${result.anchors
        .map((a) => {
          const split = a.options.length > 1;
          const sense = a.entity ? `${escapeHtml(a.entity.name)}${split ? ` <small>${a.options.length} senses</small>` : ""}` : escapeHtml(a.text);
          return `<button class="anchor-chip ${split ? "is-split" : ""}" data-entity="${a.entity ? a.entity.id : ""}" title="${split ? "Context picked this sense out of several with the same name" : "Matched entity"}">${sense}</button>`;
        })
        .join("")}`
    : "No named starting point in the question, so the trail began from the strongest passages.";

  const trail = $("#trail");
  if (!chain) {
    trail.innerHTML = `<li class="waypoint"><span class="waypoint-node"></span><p>Nothing in this collection connects these terms closely enough. Try naming a person, place or term exactly as the documents spell it.</p></li>`;
  } else {
    const names = new Set();
    for (const hop of chain.hops) {
      if (hop.from) names.add(hop.from.name);
      if (hop.to) names.add(hop.to.name);
    }
    const answerTerms = answer && answer.kind !== "bool" && answer.kind !== "path" ? [answer.text] : [];
    const parts = [];
    const first = chain.hops[0].from;
    if (first) {
      parts.push(`<li class="waypoint is-start reveal"><span class="waypoint-node"></span><button class="waypoint-name" data-entity="${first.id}">${escapeHtml(first.name)} <small>${escapeHtml(first.type)}</small></button></li>`);
    }
    chain.hops.forEach((hop, i) => {
      const isLast = i === chain.hops.length - 1;
      const target = hop.to;
      const sentence = hop.sentence;
      const card = `<div class="evidence-card"><blockquote>${highlight(sentence.text, [...names], answerTerms)}</blockquote>${sourceButton(sentence)}</div>`;
      let head = "";
      if (target) {
        head = `<button class="waypoint-name" data-entity="${target.id}">${escapeHtml(isLast && answer && answer.kind === "entity" ? answer.text : target.name)} <small>${escapeHtml(target.type)}</small></button>`;
      } else if (isLast && answer) {
        head = `<span class="waypoint-name">${escapeHtml(answer.text)}</span>`;
      }
      parts.push(`<li class="waypoint ${isLast ? "is-answer" : ""} reveal" style="animation-delay:${(i + 1) * 70}ms"><span class="waypoint-node"></span>${card}${head}</li>`);
    });
    trail.innerHTML = parts.join("");
  }

  const alts = result.chains.slice(1);
  $("#alt-trails").hidden = alts.length === 0;
  $("#alt-list").innerHTML = alts
    .map((c) => {
      const names = [];
      if (c.hops[0]?.from) names.push(c.hops[0].from.name);
      c.hops.forEach((h) => names.push(h.to ? h.to.name : "value"));
      return `<div class="alt-trail">${escapeHtml(names.join(" › "))}<br>${c.hops.map((h) => `<small>${escapeHtml(h.sentence.text.slice(0, 160))}</small>`).join("<br>")}</div>`;
    })
    .join("");

  const trailTerms = chain ? [...new Set(chain.hops.flatMap((h) => [h.from?.name, h.to?.name]).filter(Boolean))] : [];
  const onTrail = new Set(chain ? chain.hops.map((h) => h.sentence.passage) : []);
  $("#panel-evidence").innerHTML = result.passages
    .map((p) => `<article class="passage ${onTrail.has(p.id) ? "is-on-trail" : ""}"><p>${highlight(p.text, [...trailTerms, ...result.cues.filter((c) => c.length > 3)])}</p>${sourceButton({ doc: p.doc, doc_name: p.doc_name, page: p.page })}</article>`)
    .join("");

  renderResultEntities(result, chain);
  selectTab("evidence");
  state.map = state.map || new TrailMap($("#map"), (id) => openEntity(id));
  state.map.load(result.graph, trailIds(chain));
  $("#result").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
}

function renderResultEntities(result, chain) {
  const seen = new Map();
  for (const a of result.anchors) {
    if (a.entity) seen.set(a.entity.id, { ...a.entity, note: a.options.length > 1 ? `chosen over ${a.options.length - 1} other ${a.options.length === 2 ? "sense" : "senses"}` : "named in the question" });
  }
  if (chain) {
    for (const h of chain.hops) {
      if (h.to && !seen.has(h.to.id)) seen.set(h.to.id, { ...h.to, note: "reached along the trail" });
    }
  }
  $("#panel-entities").innerHTML = seen.size
    ? `<ul class="entity-list">${[...seen.values()]
        .map((e) => `<li><button data-entity="${e.id}"><strong>${escapeHtml(e.name)}</strong><span><span class="tag" data-class="${e.class}">${escapeHtml(e.type)}</span> ${escapeHtml(e.note)}</span></button></li>`)
        .join("")}</ul>`
    : `<p class="map-hint">No entities on this trail.</p>`;
}

function selectTab(name) {
  document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
  document.querySelectorAll(".panel").forEach((p) => (p.hidden = p.dataset.panel !== name));
  if (name === "map" && state.map) {
    state.map.resize();
    state.map.heat = Math.max(state.map.heat, 0.5);
    state.map.start();
  }
}

async function searchEntities(query) {
  if (!state.cid) return;
  try {
    const items = await api.entities(state.cid, query, 30);
    $("#entity-list").innerHTML = items.length
      ? items
          .map((e) => `<li><button data-entity="${e.id}"><strong>${escapeHtml(e.name)}${e.qualifier ? ` (${escapeHtml(e.qualifier)})` : ""}</strong><span><span class="tag" data-class="${e.class}">${escapeHtml(e.label)}</span> ${plural(e.mentions, "mention")} in ${plural(e.passages, "passage")}</span></button></li>`)
          .join("")
      : `<li class="map-hint">${query ? "No entity matches that name." : "Entities appear here once a document is indexed."}</li>`;
  } catch {
    $("#entity-list").innerHTML = "";
  }
}

function openSheet(title, html) {
  $("#sheet-title").textContent = title;
  $("#sheet-body").innerHTML = html;
  const sheet = $("#sheet");
  if (!sheet.open) sheet.showModal();
  $("#sheet-body").scrollTop = 0;
}

async function openEntity(id) {
  if (id === "" || id === undefined || id === null) return;
  try {
    const e = await api.entity(state.cid, id);
    const senses = e.senses.length
      ? `<h4>Other entities sharing a name</h4><p class="map-hint">These were kept apart because their surrounding context disagreed.</p><div class="chips">${e.senses.map((s) => `<button data-entity="${s.id}">${escapeHtml(s.name)} <small>${escapeHtml(s.label)}, ${s.mentions}</small></button>`).join("")}</div>`
      : "";
    const html = `
      <p><span class="tag" data-class="${e.class}">${escapeHtml(e.label)}</span> ${plural(e.mentions, "mention")} across ${plural(e.passages, "passage")}${e.qualifier ? `, qualified as ${escapeHtml(e.qualifier)}` : ""}</p>
      <h4>Also written as</h4>
      <p>${e.surfaces.map((s) => `${escapeHtml(s.text)} <small>(${s.count})</small>`).join(", ")}</p>
      ${senses}
      <h4>Strongest links</h4>
      <div class="chips">${e.neighbours.map((n) => `<button data-entity="${n.id}">${escapeHtml(n.name)}</button>`).join("") || "<span class='map-hint'>No links yet.</span>"}</div>
      <h4>Where it appears</h4>
      ${e.samples.map((s) => `<div class="sample">${highlight(s.sentence, [s.surface])}<small>${escapeHtml(s.doc_name)}, page ${s.page}</small></div>`).join("")}
    `;
    openSheet(e.name, html);
  } catch (error) {
    toast(error.message, true);
  }
}

function openPage(doc, page) {
  const meta = state.docs.find((d) => d.id === doc);
  const name = meta ? meta.name : "Document";
  if (meta && meta.kind === "text") {
    toast("Plain-text files have no page preview");
    return;
  }
  openSheet(`${name}, page ${page}`, `<img class="page-image" src="${api.pageUrl(state.cid, doc, page)}" alt="Page ${page} of ${escapeHtml(name)}" loading="lazy">`);
}

async function openReport(doc) {
  try {
    const data = await api.report(state.cid, doc);
    const r = data.report;
    if (!r) {
      openPage(doc, 1);
      return;
    }
    const polygons = r.lines
      .map((l) => `<polygon points="${l.box.map(([x, y]) => `${(x * 1000).toFixed(1)},${(y * 1000).toFixed(1)}`).join(" ")}"><title>${escapeHtml(l.text)}</title></polygon>`)
      .join("");
    const swatches = r.palette.map((p) => `<span title="${p.hex} ${(p.share * 100).toFixed(0)}%" style="flex:${p.share};background:${p.hex}"></span>`).join("");
    const html = `
      <div class="image-stage"><img src="${api.pageUrl(state.cid, doc, 1)}" alt="${escapeHtml(data.name)}"><svg viewBox="0 0 1000 1000" preserveAspectRatio="none">${polygons}</svg></div>
      <h4>What this looks like</h4>
      <p><strong>${escapeHtml(r.kind)}</strong>: ${escapeHtml(r.kind_reasons.join("; "))}.</p>
      <div class="metrics">
        <div class="metric"><b>${r.width}×${r.height}</b><span>${escapeHtml(r.format)} ${escapeHtml(r.mode)}</span></div>
        <div class="metric"><b>${r.lines.length}</b><span>text lines read</span></div>
        <div class="metric"><b>${Math.round(r.text_coverage * 100)}%</b><span>of the frame is text</span></div>
        <div class="metric"><b>${Math.round(r.brightness * 100)}%</b><span>mean brightness</span></div>
        <div class="metric"><b>${r.colorfulness.toFixed(2)}</b><span>colourfulness</span></div>
        <div class="metric"><b>${Math.round(r.edge_density * 100)}%</b><span>edge density</span></div>
      </div>
      <h4>Palette</h4>
      <div class="swatches">${swatches}</div>
      ${r.numbers.length ? `<h4>Figures in the image</h4><p>${r.numbers.map(escapeHtml).join(", ")}</p>` : ""}
      ${Object.keys(r.exif).length ? `<h4>Camera data</h4><p>${Object.entries(r.exif).map(([k, v]) => `${escapeHtml(k)}: ${escapeHtml(v)}`).join("<br>")}</p>` : ""}
      <h4>Text read from the image</h4>
      <div class="ocr-text">${escapeHtml(r.text || "No legible text.")}</div>`;
    openSheet(data.name, html);
  } catch (error) {
    toast(error.message, true);
  }
}

function bind() {
  $("#ask-form").addEventListener("submit", ask);
  $("#question").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) ask(e);
  });
  $("#hops").addEventListener("input", (e) => ($("#hops-out").value = e.target.value));
  $("#theme").addEventListener("click", toggleTheme);
  $("#collection").addEventListener("change", (e) => {
    state.cid = e.target.value;
    state.statsFor = null;
    $("#result").hidden = true;
    try { localStorage.setItem("waymark.collection", state.cid); } catch {}
    refreshCollection();
  });
  $("#new-collection").addEventListener("click", async () => {
    const name = prompt("Name the new collection");
    if (!name || !name.trim()) return;
    try {
      const created = await api.createCollection(name.trim());
      state.statsFor = null;
      $("#result").hidden = true;
      await loadCollections(created.id);
    } catch (error) {
      toast(error.message, true);
    }
  });
  $("#delete-collection").addEventListener("click", async () => {
    const current = state.collections.find((c) => c.id === state.cid);
    if (!current || !confirm(`Delete “${current.name}” and every file in it?`)) return;
    try {
      await api.deleteCollection(state.cid);
      state.cid = null;
      state.statsFor = null;
      $("#result").hidden = true;
      await loadCollections();
      toast("Collection deleted");
    } catch (error) {
      toast(error.message, true);
    }
  });
  const drop = $("#drop");
  $("#file-input").addEventListener("change", (e) => {
    upload([...e.target.files]);
    e.target.value = "";
  });
  ["dragenter", "dragover"].forEach((type) =>
    drop.addEventListener(type, (e) => {
      e.preventDefault();
      drop.classList.add("is-over");
    }),
  );
  ["dragleave", "drop"].forEach((type) => drop.addEventListener(type, () => drop.classList.remove("is-over")));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    upload([...e.dataTransfer.files]);
  });
  drop.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      $("#file-input").click();
    }
  });
  $("#entity-search").addEventListener("input", (e) => {
    clearTimeout(state.entityTimer);
    state.entityTimer = setTimeout(() => searchEntities(e.target.value.trim()), 180);
  });
  $("#sheet-close").addEventListener("click", () => $("#sheet").close());
  $("#sheet").addEventListener("click", (e) => {
    if (e.target === $("#sheet")) $("#sheet").close();
  });
  $("#toggle-library").addEventListener("click", () => {
    const open = $("#library").classList.toggle("is-open");
    $("#toggle-library").setAttribute("aria-expanded", String(open));
  });
  document.addEventListener("click", (e) => {
    const target = e.target.closest("[data-entity],[data-doc],[data-delete],[data-report],[data-preview],[data-suggest],[data-tab]");
    if (!target) return;
    if (target.dataset.tab) return selectTab(target.dataset.tab);
    if (target.dataset.suggest) {
      $("#question").value = target.dataset.suggest;
      return ask();
    }
    if (target.dataset.entity !== undefined && target.dataset.entity !== "") return openEntity(Number(target.dataset.entity));
    if (target.dataset.doc) return openPage(target.dataset.doc, Number(target.dataset.page || 1));
    if (target.dataset.report) return openReport(target.dataset.report);
    if (target.dataset.preview) return openPage(target.dataset.preview, 1);
    if (target.dataset.delete) {
      const doc = state.docs.find((d) => d.id === target.dataset.delete);
      if (!doc || !confirm(`Remove “${doc.name}” from this collection?`)) return;
      api
        .deleteDocument(state.cid, doc.id)
        .then(() => {
          toast("Removed");
          state.statsFor = null;
          return refreshCollection();
        })
        .catch((error) => toast(error.message, true));
    }
  });
}

async function boot() {
  let theme = null;
  try { theme = localStorage.getItem("waymark.theme"); } catch {}
  applyTheme(theme);
  bind();
  try {
    await loadCollections();
  } catch (error) {
    toast(`Could not reach the server: ${error.message}`, true);
  }
}

boot();

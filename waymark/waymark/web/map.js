const CLASS_SHAPE = { PERSON: "circle", PLACE: "diamond", ORG: "square", WORK: "square", EVENT: "diamond", CONCEPT: "dot", NORP: "dot", MISC: "circle" };

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export class TrailMap {
  constructor(canvas, onPick) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.onPick = onPick;
    this.nodes = [];
    this.edges = [];
    this.trail = [];
    this.drag = null;
    this.hover = null;
    this.frame = null;
    this.heat = 0;
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas);
    canvas.addEventListener("pointerdown", (e) => this.down(e));
    canvas.addEventListener("pointermove", (e) => this.move(e));
    canvas.addEventListener("pointerup", (e) => this.up(e));
    canvas.addEventListener("pointerleave", () => { this.hover = null; this.draw(); });
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    if (!rect.width) return;
    const ratio = window.devicePixelRatio || 1;
    this.canvas.width = Math.round(rect.width * ratio);
    this.canvas.height = Math.round(rect.height * ratio);
    this.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    this.width = rect.width;
    this.height = rect.height;
    this.draw();
  }

  load(graph, trail = []) {
    const previous = new Map(this.nodes.map((n) => [n.id, n]));
    this.trail = trail;
    const trailIndex = new Map(trail.map((id, i) => [id, i]));
    const w = this.width || 600;
    const h = this.height || 400;
    this.nodes = graph.nodes.map((n, i) => {
      const old = previous.get(n.id);
      const order = trailIndex.get(n.id);
      const angle = (i / Math.max(graph.nodes.length, 1)) * Math.PI * 2;
      const x = old ? old.x : order !== undefined ? w * (0.15 + 0.7 * (order / Math.max(trail.length - 1, 1))) : w / 2 + Math.cos(angle) * w * 0.32;
      const y = old ? old.y : order !== undefined ? h / 2 : h / 2 + Math.sin(angle) * h * 0.32;
      return { ...n, x, y, vx: 0, vy: 0, onTrail: order !== undefined, order };
    });
    const byId = new Map(this.nodes.map((n) => [n.id, n]));
    this.edges = graph.edges
      .map((e) => ({ source: byId.get(e.source), target: byId.get(e.target), weight: e.weight }))
      .filter((e) => e.source && e.target);
    this.trailEdges = [];
    for (let i = 0; i + 1 < trail.length; i++) {
      const a = byId.get(trail[i]);
      const b = byId.get(trail[i + 1]);
      if (a && b) this.trailEdges.push([a, b]);
    }
    this.heat = 1;
    this.start();
  }

  start() {
    if (this.frame) return;
    const tick = () => {
      this.step();
      this.draw();
      this.heat *= 0.985;
      if (this.heat > 0.02 || this.drag) {
        this.frame = requestAnimationFrame(tick);
      } else {
        this.frame = null;
      }
    };
    this.frame = requestAnimationFrame(tick);
  }

  step() {
    const nodes = this.nodes;
    const w = this.width || 600;
    const h = this.height || 400;
    const k = Math.sqrt((w * h) / Math.max(nodes.length, 1)) * 0.55;
    for (let i = 0; i < nodes.length; i++) {
      const a = nodes[i];
      for (let j = i + 1; j < nodes.length; j++) {
        const b = nodes[j];
        let dx = a.x - b.x;
        let dy = a.y - b.y;
        let d2 = dx * dx + dy * dy;
        if (d2 < 0.01) { dx = Math.random() - 0.5; dy = Math.random() - 0.5; d2 = 0.5; }
        const force = (k * k) / d2;
        const fx = dx * force * 0.02;
        const fy = dy * force * 0.02;
        a.vx += fx; a.vy += fy;
        b.vx -= fx; b.vy -= fy;
      }
    }
    for (const e of this.edges) {
      const dx = e.target.x - e.source.x;
      const dy = e.target.y - e.source.y;
      const d = Math.sqrt(dx * dx + dy * dy) || 1;
      const pull = ((d - k) / d) * 0.04;
      e.source.vx += dx * pull; e.source.vy += dy * pull;
      e.target.vx -= dx * pull; e.target.vy -= dy * pull;
    }
    for (const n of nodes) {
      if (n.onTrail) {
        const targetX = w * (0.12 + 0.76 * (n.order / Math.max(this.trail.length - 1, 1)));
        n.vx += (targetX - n.x) * 0.03;
        n.vy += (h / 2 - n.y) * 0.03;
      } else {
        n.vx += (w / 2 - n.x) * 0.002;
        n.vy += (h / 2 - n.y) * 0.002;
      }
      if (this.drag && this.drag.node === n) continue;
      n.vx *= 0.82; n.vy *= 0.82;
      n.x = Math.max(16, Math.min(w - 16, n.x + n.vx * this.heat * 4));
      n.y = Math.max(16, Math.min(h - 16, n.y + n.vy * this.heat * 4));
    }
  }

  radius(n) {
    return Math.min(16, 4 + Math.log2(2 + n.mentions) * 1.8) + (n.onTrail ? 3 : 0);
  }

  draw() {
    const ctx = this.ctx;
    const w = this.width || 0;
    const h = this.height || 0;
    if (!w) return;
    const ink = cssVar("--ink");
    const muted = cssVar("--muted");
    const line = cssVar("--line");
    const route = cssVar("--route");
    const landmark = cssVar("--landmark");
    const panel = cssVar("--panel");
    ctx.clearRect(0, 0, w, h);
    ctx.lineWidth = 1;
    ctx.strokeStyle = line;
    for (const e of this.edges) {
      ctx.globalAlpha = Math.min(0.9, 0.25 + Math.log2(1 + e.weight) * 0.15);
      ctx.beginPath();
      ctx.moveTo(e.source.x, e.source.y);
      ctx.lineTo(e.target.x, e.target.y);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    ctx.setLineDash([6, 5]);
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = route;
    for (const [a, b] of this.trailEdges || []) {
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }
    ctx.setLineDash([]);
    const last = this.trail[this.trail.length - 1];
    for (const n of this.nodes) {
      const r = this.radius(n);
      const isAnswer = n.id === last && this.trail.length > 1;
      ctx.fillStyle = isAnswer ? landmark : n.onTrail ? route : panel;
      ctx.strokeStyle = isAnswer ? landmark : n.onTrail ? route : muted;
      ctx.lineWidth = n === this.hover ? 2.5 : 1.5;
      this.shape(n, r);
      ctx.fill();
      ctx.stroke();
    }
    ctx.font = "600 12px Public Sans, system-ui, sans-serif";
    ctx.textBaseline = "middle";
    for (const n of this.nodes) {
      const important = n.onTrail || n === this.hover || n.mentions > 6 || this.nodes.length <= 24;
      if (!important) continue;
      const label = n.name.length > 28 ? `${n.name.slice(0, 27)}…` : n.name;
      const x = n.x + this.radius(n) + 5;
      ctx.lineWidth = 4;
      ctx.strokeStyle = panel;
      ctx.strokeText(label, x, n.y);
      ctx.fillStyle = n.onTrail ? ink : muted;
      ctx.fillText(label, x, n.y);
    }
  }

  shape(n, r) {
    const ctx = this.ctx;
    const kind = CLASS_SHAPE[n.class] || "circle";
    ctx.beginPath();
    if (kind === "square") {
      ctx.rect(n.x - r * 0.85, n.y - r * 0.85, r * 1.7, r * 1.7);
    } else if (kind === "diamond") {
      ctx.moveTo(n.x, n.y - r);
      ctx.lineTo(n.x + r, n.y);
      ctx.lineTo(n.x, n.y + r);
      ctx.lineTo(n.x - r, n.y);
      ctx.closePath();
    } else {
      ctx.arc(n.x, n.y, kind === "dot" ? r * 0.75 : r, 0, Math.PI * 2);
    }
  }

  locate(e) {
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    let best = null;
    let bestD = Infinity;
    for (const n of this.nodes) {
      const d = Math.hypot(n.x - x, n.y - y);
      if (d < this.radius(n) + 6 && d < bestD) { best = n; bestD = d; }
    }
    return { x, y, node: best };
  }

  down(e) {
    const hit = this.locate(e);
    if (!hit.node) return;
    this.canvas.setPointerCapture(e.pointerId);
    this.drag = { node: hit.node, moved: false };
    this.heat = Math.max(this.heat, 0.4);
    this.start();
  }

  move(e) {
    const hit = this.locate(e);
    if (this.drag) {
      this.drag.node.x = hit.x;
      this.drag.node.y = hit.y;
      this.drag.moved = true;
      this.draw();
      return;
    }
    if (hit.node !== this.hover) {
      this.hover = hit.node;
      this.canvas.style.cursor = hit.node ? "pointer" : "grab";
      this.draw();
    }
  }

  up(e) {
    if (!this.drag) return;
    const { node, moved } = this.drag;
    this.drag = null;
    if (!moved && this.onPick) this.onPick(node.id);
  }
}

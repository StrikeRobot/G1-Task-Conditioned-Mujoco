// MapView: draw occupancy PNG + robot + path + goal; pan/zoom/right-click goal.
class MapView {
  constructor(canvas, onGoal) {
    this.cv = canvas; this.ctx = canvas.getContext("2d");
    this.onGoal = onGoal;
    this.img = null; this.meta = null;
    this.scale = 3.0; this.tx = 0; this.ty = 0;
    this.pose = [0, 0, 0]; this.path = []; this.goal = null;
    this._fitted = false;
    // Left-click (no drag) sets a goal; left-drag pans. Distinguish by movement.
    canvas.addEventListener("mousedown", e => {
      if (e.button !== 0) return;
      this._drag = [e.offsetX, e.offsetY];
      this._down = [e.offsetX, e.offsetY];
      this._moved = false;
    });
    canvas.addEventListener("mousemove", e => {
      if (!this._drag) return;
      if (Math.hypot(e.offsetX - this._down[0], e.offsetY - this._down[1]) > 4)
        this._moved = true;
      this.tx += e.offsetX - this._drag[0];
      this.ty += e.offsetY - this._drag[1];
      this._drag = [e.offsetX, e.offsetY];
    });
    window.addEventListener("mouseup", e => {
      if (this._drag && !this._moved) {        // a click, not a pan -> set goal
        const r = this.cv.getBoundingClientRect();
        const w = this.canvasToWorld(e.clientX - r.left, e.clientY - r.top);
        if (w) this.onGoal(w[0], w[1]);
      }
      this._drag = null;
    });
    canvas.addEventListener("wheel", e => {
      e.preventDefault();
      const f = e.deltaY < 0 ? 1.15 : 1 / 1.15;
      this.scale *= f;
      this.tx = e.offsetX - (e.offsetX - this.tx) * f;
      this.ty = e.offsetY - (e.offsetY - this.ty) * f;
    });
    canvas.addEventListener("dblclick", () => this.fit());
    canvas.addEventListener("contextmenu", e => {     // right-click also sets goal
      e.preventDefault();
      const w = this.canvasToWorld(e.offsetX, e.offsetY);
      if (w) this.onGoal(w[0], w[1]);
    });
  }

  setMap(blob, meta) {
    this.meta = meta;
    createImageBitmap(blob).then(b => {
      this.img = b;
      if (!this._fitted) { this.fit(); this._fitted = true; }
    });
  }

  fit() {
    if (!this.img) return;
    const s = Math.min(this.cv.width / this.img.width,
                       this.cv.height / this.img.height);
    this.scale = s;
    this.tx = (this.cv.width - this.img.width * s) / 2;
    this.ty = (this.cv.height - this.img.height * s) / 2;
  }

  // PNG is already flipud: pixel row 0 = largest world y
  worldToCanvas(x, y) {
    const m = this.meta; if (!m) return null;
    const px = (x - m.origin[0]) / m.res;
    const py = m.h - (y - m.origin[1]) / m.res;
    return [px * this.scale + this.tx, py * this.scale + this.ty];
  }

  canvasToWorld(cx, cy) {
    const m = this.meta; if (!m) return null;
    const px = (cx - this.tx) / this.scale;
    const py = (cy - this.ty) / this.scale;
    return [m.origin[0] + px * m.res, m.origin[1] + (m.h - py) * m.res];
  }

  draw() {
    const { ctx, cv } = this;
    cv.width = cv.clientWidth; cv.height = cv.clientHeight;
    ctx.fillStyle = "#b9b9bd"; ctx.fillRect(0, 0, cv.width, cv.height);
    if (this.img) {
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(this.img, this.tx, this.ty,
                    this.img.width * this.scale, this.img.height * this.scale);
    }
    if (this.path && this.path.length > 1) {
      ctx.strokeStyle = "#1e90ff"; ctx.lineWidth = 3;
      ctx.lineJoin = "round"; ctx.beginPath();
      this.path.forEach((p, i) => {
        const c = this.worldToCanvas(p[0], p[1]);
        i ? ctx.lineTo(c[0], c[1]) : ctx.moveTo(c[0], c[1]);
      });
      ctx.stroke();
      ctx.fillStyle = "#1e90ff";                 // small dots along the path
      this.path.forEach((p, i) => {
        if (i % 5) return;
        const c = this.worldToCanvas(p[0], p[1]);
        ctx.beginPath(); ctx.arc(c[0], c[1], 1.6, 0, 7); ctx.fill();
      });
    }
    if (this.goal) {
      const g = this.worldToCanvas(this.goal[0], this.goal[1]);
      if (g) {
        ctx.strokeStyle = "#ff7a00"; ctx.fillStyle = "#ff7a00"; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(g[0], g[1], 8, 0, 7); ctx.stroke();   // ring
        ctx.beginPath(); ctx.arc(g[0], g[1], 2.5, 0, 7); ctx.fill();   // center
        ctx.beginPath();                                               // crosshair
        ctx.moveTo(g[0] - 12, g[1]); ctx.lineTo(g[0] + 12, g[1]);
        ctx.moveTo(g[0], g[1] - 12); ctx.lineTo(g[0], g[1] + 12);
        ctx.stroke();
      }
    }
    const r = this.worldToCanvas(this.pose[0], this.pose[1]);
    if (r) {
      ctx.fillStyle = "#e22"; ctx.beginPath(); ctx.arc(r[0], r[1], 6, 0, 7); ctx.fill();
      ctx.strokeStyle = "#e22"; ctx.lineWidth = 2; ctx.beginPath();
      ctx.moveTo(r[0], r[1]);
      ctx.lineTo(r[0] + 14 * Math.cos(-this.pose[2]),
                 r[1] + 14 * Math.sin(-this.pose[2]));
      ctx.stroke();
    }
  }
}

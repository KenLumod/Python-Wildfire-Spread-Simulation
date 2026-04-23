"""
================================================================================
  WILDFIRE SPREAD SIMULATION
  CS422 – Computational Modelling
  Agent-Based Model with Stochastic Spread
================================================================================
"""

import math
import random
import time
import tkinter as tk
from tkinter import font as tkfont


# ================================================================================
# SECTION 1 — SIMULATION ENGINE
# Model Logic, Formulas, and Calculations
# ================================================================================

# ── SIMULATION CONSTANTS ────────────────────────────────────────────────────────
COLS            = 50
ROWS            = 40
CELL            = 12

# Cell state identifiers
TREE            = 0
BURNING         = 1
ASH             = 2
BREAK           = 3

# Base spread probability indexed by wind speed level 1–5
BASE_PROB       = [0.04, 0.07, 0.13, 0.22, 0.35]

# Probability a burning cell extinguishes each tick
BURN_OUT_CHANCE = 0.28

# 8-directional neighbor offsets (dr, dc)
NEIGHBORS       = [(-1, 0), (1, 0), (0, -1), (0, 1),
                   (-1, -1), (-1, 1), (1, -1), (1, 1)]

# Wind boost coefficient used in spread probability formula
WIND_BOOST_COEFF = 0.35

# Probability clamp bounds
PROB_MIN = 0.02
PROB_MAX = 0.98

# Cell colors used for rendering
COLORS = {
    TREE:    "#27a446",
    ASH:     "#404754",
    BREAK:   "#7a5230",
}
FIRE_COLORS = ["#e84c1e", "#f5870a", "#faa030"]


# ── SPREAD PROBABILITY FORMULA ──────────────────────────────────────────────────
#
#   P = min(0.98, max(0.02, p₀ + (n̂ · ŵ) × 0.35))
#
#   Where:
#     p₀      = base probability at current wind speed
#     n̂ · ŵ   = dot product of neighbor direction and wind direction
#     0.35    = wind boost coefficient
#
def spread_prob(neighbor_dc, neighbor_dr, wind_dx, wind_dy, wind_speed):
    """
    Compute fire spread probability from a burning cell to a neighbor.

    Parameters
    ----------
    neighbor_dc : int  Column delta toward the neighbor (-1, 0, or 1)
    neighbor_dr : int  Row delta toward the neighbor (-1, 0, or 1)
    wind_dx     : int  Wind direction x-component
    wind_dy     : int  Wind direction y-component
    wind_speed  : int  Wind speed level 1–5

    Returns
    -------
    float  Clamped probability in [0.02, 0.98]
    """
    base       = BASE_PROB[wind_speed - 1]
    dot        = neighbor_dc * wind_dx + neighbor_dr * wind_dy
    wind_boost = dot * WIND_BOOST_COEFF
    return min(PROB_MAX, max(PROB_MIN, base + wind_boost))


# ── SIMULATION INTERVAL FORMULA ─────────────────────────────────────────────────
#
#   Δt = max(60, 520 − v × 80)   [milliseconds]
#
#   Where v is wind speed level 1–5.
#   Higher wind speed → shorter interval → faster simulation.
#
def get_interval(wind_speed):
    """
    Compute tick interval in milliseconds based on wind speed.

    Parameters
    ----------
    wind_speed : int  Wind speed level 1–5

    Returns
    -------
    int  Interval in milliseconds
    """
    return max(60, 520 - wind_speed * 80)


# ── GRID INITIALISATION ─────────────────────────────────────────────────────────
def init_grid():
    """
    Create a fresh ROWS × COLS grid where every cell is TREE.

    Returns
    -------
    list[list[int]]  2D grid of TREE state integers
    """
    return [[TREE for _ in range(COLS)] for _ in range(ROWS)]


def copy_grid(src):
    """
    Shallow-copy a grid so tick transitions are computed from a stable snapshot.

    Parameters
    ----------
    src : list[list[int]]  Source grid

    Returns
    -------
    list[list[int]]  Independent copy
    """
    return [row[:] for row in src]


# ── ABM SIMULATION STEP ─────────────────────────────────────────────────────────
def simulation_step(grid, wind_dx, wind_dy, wind_speed):
    """
    Advance the simulation by one tick using Agent-Based Model rules.

    Rules applied per burning cell:
      1. Burn-out: 28% chance the cell transitions to ASH.
      2. Spread:   For each of 8 neighbors that is TREE,
                   ignite it with probability P (spread_prob formula).

    Parameters
    ----------
    grid       : list[list[int]]  Current grid state
    wind_dx    : int               Wind x-direction component
    wind_dy    : int               Wind y-direction component
    wind_speed : int               Wind speed level 1–5

    Returns
    -------
    list[list[int]]  Updated grid after one tick
    """
    next_grid = copy_grid(grid)

    for r in range(ROWS):
        for c in range(COLS):
            if grid[r][c] == BURNING:
                # Burn-out check
                if random.random() < BURN_OUT_CHANCE:
                    next_grid[r][c] = ASH

                # Spread to neighbors
                for (dr, dc) in NEIGHBORS:
                    nr, nc = r + dr, c + dc
                    if nr < 0 or nr >= ROWS or nc < 0 or nc >= COLS:
                        continue
                    if grid[nr][nc] != TREE:
                        continue
                    p = spread_prob(dc, dr, wind_dx, wind_dy, wind_speed)
                    if random.random() < p:
                        next_grid[nr][nc] = BURNING

    return next_grid


# ── METRICS CALCULATION ─────────────────────────────────────────────────────────
#
#   burned_pct = (ASH cells + BURNING cells) / total cells × 100
#   perimeter  = burning_count × 0.12   [km]
#
def compute_metrics(grid):
    """
    Compute simulation metrics from the current grid state.

    Parameters
    ----------
    grid : list[list[int]]  Current grid

    Returns
    -------
    dict with keys:
        burned_pct  (float)  Percentage of grid burned or burning
        perimeter   (float)  Estimated fire perimeter in km
        fire_fronts (int)    Count of currently burning cells
        has_fire    (bool)   Whether any burning cells remain
    """
    total   = ROWS * COLS
    burned  = 0
    burning = 0

    for r in range(ROWS):
        for c in range(COLS):
            if grid[r][c] == ASH:
                burned += 1
            elif grid[r][c] == BURNING:
                burning += 1

    burned_pct  = (burned + burning) / total * 100
    perimeter   = burning * 0.12
    has_fire    = burning > 0

    return {
        "burned_pct":  round(burned_pct, 1),
        "perimeter":   round(perimeter, 1),
        "fire_fronts": burning,
        "has_fire":    has_fire,
    }


# ── ERROR METRICS ────────────────────────────────────────────────────────────────
#
#   MSE   = (1/n) × Σ (actual_i − expected_i)²
#   RMSE  = √MSE
#   NRMSE = (RMSE / (max − min)) × 100%
#
#   Baseline: linear (uniform) spread from 0% to final burned %.
#
def compute_error_metrics(burn_history):
    """
    Compute MSE, RMSE, and NRMSE comparing actual burn progression
    against a uniform linear spread baseline.

    Parameters
    ----------
    burn_history : list[float]  Sequence of burned_pct values over time

    Returns
    -------
    dict  Keys: mse, rmse, nrmse  (or None values if insufficient data)
    """
    if len(burn_history) < 5:
        return {"mse": None, "rmse": None, "nrmse": None}

    n       = len(burn_history)
    max_pct = burn_history[-1]

    sum_sq_err = 0.0
    for i, actual in enumerate(burn_history):
        expected    = (i / (n - 1)) * max_pct
        sum_sq_err += (actual - expected) ** 2

    mse   = sum_sq_err / n
    rmse  = math.sqrt(mse)
    nrmse = (rmse / max_pct * 100) if max_pct > 0 else 0.0

    return {
        "mse":   round(mse,   2),
        "rmse":  round(rmse,  2),
        "nrmse": round(nrmse, 1),
    }


# ================================================================================
# SECTION 2 — UI AND FRONTEND
# Rendering, Layout, Controls, and Event Handling
# ================================================================================

# ── SPEED LABELS ────────────────────────────────────────────────────────────────
SPEED_LABELS = {1: "Very Low", 2: "Low", 3: "Medium", 4: "High", 5: "Very High"}

# ── COMPASS DIRECTION DEFINITIONS ───────────────────────────────────────────────
COMPASS_DIRS = [
    {"label": "↖", "dx": -1, "dy": -1, "row": 0, "col": 0},
    {"label": "↑", "dx":  0, "dy": -1, "row": 0, "col": 1},
    {"label": "↗", "dx":  1, "dy": -1, "row": 0, "col": 2},
    {"label": "←", "dx": -1, "dy":  0, "row": 1, "col": 0},
    {"label": "·", "dx":  0, "dy":  0, "row": 1, "col": 1, "center": True},
    {"label": "→", "dx":  1, "dy":  0, "row": 1, "col": 2},
    {"label": "↙", "dx": -1, "dy":  1, "row": 2, "col": 0},
    {"label": "↓", "dx":  0, "dy":  1, "row": 2, "col": 1},
    {"label": "↘", "dx":  1, "dy":  1, "row": 2, "col": 2},
]

# ── THEME COLORS ────────────────────────────────────────────────────────────────
BG          = "#0d1117"
SURFACE     = "#161b22"
SURFACE_2   = "#1c2230"
BORDER      = "#21293a"
BORDER_LIT  = "#2e3d55"
TEXT        = "#cdd9f5"
TEXT_MUTED  = "#6e7e9e"
TEXT_DIM    = "#4a566e"
ACCENT      = "#f5870a"
GREEN       = "#27a446"
BTN_START   = "#27a446"
BTN_PAUSE   = "#c8992a"
BTN_RESET   = "#c0302a"


class WildfireApp:
    """
    Main application class.
    Manages the Tkinter window, all widgets, the simulation loop,
    and delegates all model logic to the functions in Section 1.
    """

    # ── INIT ────────────────────────────────────────────────────────────────────
    def __init__(self, root):
        self.root = root
        self.root.title("Wildfire Spread Simulation — CS422")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)

        # ── Simulation state ────────────────────────────────────────────────────
        self.grid         = init_grid()
        self.running      = False
        self.tick         = 0
        self.after_id     = None
        self.wind_dx      = 1
        self.wind_dy      = 0
        self.wind_speed   = 3
        self.tool         = "ignite"   # "ignite" or "break"
        self.burn_history = []
        self.chart_data   = []         # list of (tick, pct) for chart drawing

        # ── Build UI ────────────────────────────────────────────────────────────
        self._build_header()
        self._build_shell()

        # ── Initial render ───────────────────────────────────────────────────────
        self._render_grid()
        self._update_metrics_display(compute_metrics(self.grid))
        self._update_error_display({"mse": None, "rmse": None, "nrmse": None})
        self._draw_chart()

    # ── HEADER ──────────────────────────────────────────────────────────────────
    def _build_header(self):
        hdr = tk.Frame(self.root, bg=BG)
        hdr.pack(pady=(14, 0))

        tk.Label(
            hdr, text="🔥 Wildfire Spread Simulation",
            bg=BG, fg=TEXT,
            font=("Helvetica", 18, "bold"),
        ).pack()

        tk.Label(
            hdr, text="AGENT-BASED MODEL  ·  CS422 COMPUTATIONAL MODELLING",
            bg=BG, fg=TEXT_MUTED,
            font=("Helvetica", 8),
        ).pack(pady=(2, 0))

    # ── MAIN SHELL ──────────────────────────────────────────────────────────────
    def _build_shell(self):
        shell = tk.Frame(self.root, bg=SURFACE, bd=0, highlightthickness=0)
        shell.pack(fill="both", expand=True, padx=16, pady=14)

        self._build_left_panel(shell)
        self._build_center_panel(shell)
        self._build_right_panel(shell)

    # ── LEFT CONTROL PANEL ──────────────────────────────────────────────────────
    def _build_left_panel(self, parent):
        panel = tk.Frame(parent, bg=SURFACE, width=210)
        panel.pack(side="left", fill="y", padx=(10, 6), pady=10)
        panel.pack_propagate(False)

        self._section_label(panel, "Simulation Controls")

        # Wind direction
        self._section_label(panel, "Wind Direction", small=True)
        self._build_compass(panel)

        # Wind speed
        self._section_label(panel, "Wind Speed", small=True)
        speed_row = tk.Frame(panel, bg=SURFACE)
        speed_row.pack(fill="x", pady=(0, 4))
        tk.Label(speed_row, text="Speed", bg=SURFACE, fg=TEXT_MUTED,
                 font=("Helvetica", 9)).pack(side="left")
        self.speed_val_label = tk.Label(
            speed_row, text=SPEED_LABELS[self.wind_speed],
            bg=SURFACE, fg=ACCENT, font=("Helvetica", 9, "bold"))
        self.speed_val_label.pack(side="right")

        self.speed_slider = tk.Scale(
            panel, from_=1, to=5, orient="horizontal",
            bg=SURFACE, fg=TEXT, troughcolor=BORDER,
            highlightthickness=0, bd=0,
            activebackground=ACCENT,
            command=self._on_speed_change,
        )
        self.speed_slider.set(self.wind_speed)
        self.speed_slider.pack(fill="x", pady=(0, 8))

        # Simulation buttons
        self._section_label(panel, "Controls", small=True)
        self._make_btn(panel, "▶  Start Simulation", BTN_START, self._start_sim)
        self._make_btn(panel, "⏸  Pause",            BTN_PAUSE, self._pause_sim)
        self._make_btn(panel, "↺  Reset",             BTN_RESET, self._reset_sim)

        # Status
        status_row = tk.Frame(panel, bg=SURFACE_2,
                               highlightbackground=BORDER, highlightthickness=1)
        status_row.pack(fill="x", pady=(8, 0), ipady=4, ipadx=8)
        self.status_dot_canvas = tk.Canvas(
            status_row, width=10, height=10, bg=SURFACE_2,
            highlightthickness=0)
        self.status_dot_canvas.pack(side="left", padx=(6, 4))
        self._draw_status_dot("idle")
        self.status_label = tk.Label(
            status_row, text="Idle",
            bg=SURFACE_2, fg=TEXT_MUTED, font=("Helvetica", 9))
        self.status_label.pack(side="left")

        # Tool selection
        self._section_label(panel, "Interaction Tools", small=True)
        tool_row = tk.Frame(panel, bg=SURFACE)
        tool_row.pack(fill="x", pady=(0, 6))

        self.ignite_btn = tk.Button(
            tool_row, text="Ignite",
            bg=SURFACE_2, fg=ACCENT,
            activebackground=BORDER_LIT, activeforeground=ACCENT,
            font=("Helvetica", 9, "bold"),
            bd=0, highlightthickness=1, highlightbackground=ACCENT,
            cursor="hand2", command=self._select_ignite,
        )
        self.ignite_btn.pack(side="left", expand=True, fill="x", padx=(0, 4))

        self.break_btn = tk.Button(
            tool_row, text="Firebreak",
            bg=SURFACE_2, fg=TEXT_MUTED,
            activebackground=BORDER_LIT, activeforeground=ACCENT,
            font=("Helvetica", 9, "bold"),
            bd=0, highlightthickness=1, highlightbackground=BORDER,
            cursor="hand2", command=self._select_break,
        )
        self.break_btn.pack(side="left", expand=True, fill="x")

    def _build_compass(self, parent):
        compass_frame = tk.Frame(parent, bg=SURFACE)
        compass_frame.pack(pady=(4, 8))

        self.compass_buttons = {}
        for d in COMPASS_DIRS:
            is_center = d.get("center", False)
            btn = tk.Button(
                compass_frame,
                text=d["label"],
                width=2, height=1,
                bg=BORDER if not is_center else SURFACE,
                fg=TEXT_MUTED,
                activebackground=ACCENT,
                activeforeground="#fff",
                font=("Helvetica", 10),
                bd=0, highlightthickness=0,
                cursor="arrow" if is_center else "hand2",
                relief="flat",
            )
            btn.grid(row=d["row"], column=d["col"], padx=1, pady=1)
            if not is_center:
                dx, dy = d["dx"], d["dy"]
                btn.config(command=lambda b=btn, x=dx, y=dy: self._set_wind_dir(b, x, y))
                self.compass_buttons[(dx, dy)] = btn

        # Set default active direction (East)
        self._highlight_compass_btn(1, 0)

    # ── CENTER CANVAS ───────────────────────────────────────────────────────────
    def _build_center_panel(self, parent):
        center = tk.Frame(parent, bg=SURFACE)
        center.pack(side="left", fill="both", expand=True, pady=10)

        # Header row
        hdr = tk.Frame(center, bg=SURFACE)
        hdr.pack(fill="x", pady=(0, 6))
        tk.Label(hdr, text="SIMULATION GRID — 50 × 40",
                 bg=SURFACE, fg=TEXT_MUTED,
                 font=("Helvetica", 8, "bold")).pack(side="left")
        self.grid_status_label = tk.Label(
            hdr, text="⬤  Idle",
            bg=SURFACE, fg=TEXT_DIM, font=("Helvetica", 8))
        self.grid_status_label.pack(side="right")

        # Canvas
        self.canvas = tk.Canvas(
            center,
            width=COLS * CELL,
            height=ROWS * CELL,
            bg="#1a2a1a",
            highlightthickness=1,
            highlightbackground=BORDER,
            cursor="crosshair",
        )
        self.canvas.pack()

        # Bind mouse events
        self.canvas.bind("<ButtonPress-1>",   self._on_canvas_press)
        self.canvas.bind("<B1-Motion>",        self._on_canvas_drag)

    # ── RIGHT PANEL ─────────────────────────────────────────────────────────────
    def _build_right_panel(self, parent):
        panel = tk.Frame(parent, bg=SURFACE, width=220)
        panel.pack(side="left", fill="y", padx=(6, 10), pady=10)
        panel.pack_propagate(False)

        # Legend
        self._section_label(panel, "Legend")
        legend_items = [
            ("Tree",      "#27a446"),
            ("Burning",   "#e84c1e"),
            ("Ash",       "#404754"),
            ("Firebreak", "#7a5230"),
        ]
        for name, color in legend_items:
            row = tk.Frame(panel, bg=SURFACE)
            row.pack(fill="x", pady=2)
            swatch = tk.Canvas(row, width=14, height=14, bg=color,
                               highlightthickness=0)
            swatch.pack(side="left", padx=(0, 8))
            tk.Label(row, text=name, bg=SURFACE, fg=TEXT,
                     font=("Helvetica", 9)).pack(side="left")

        # Metric cards
        self._section_label(panel, "Metrics")
        metrics_frame = tk.Frame(panel, bg=SURFACE)
        metrics_frame.pack(fill="x", pady=(0, 6))

        self.metric_burned = self._metric_card(metrics_frame, "Burned",      "0.0%",  0, 0)
        self.metric_perim  = self._metric_card(metrics_frame, "Perimeter",   "0.0 km", 0, 1)
        self.metric_fronts = self._metric_card(metrics_frame, "Fire Fronts", "0",      1, 0)
        self.metric_tick   = self._metric_card(metrics_frame, "Tick",        "0",      1, 1)

        # Chart
        self._section_label(panel, "Burned Area Over Time")
        chart_frame = tk.Frame(panel, bg=SURFACE_2,
                               highlightbackground=BORDER, highlightthickness=1)
        chart_frame.pack(fill="x", pady=(0, 6))
        self.chart_canvas = tk.Canvas(
            chart_frame, width=200, height=100,
            bg=SURFACE_2, highlightthickness=0)
        self.chart_canvas.pack(padx=4, pady=4)
        self.chart_axis_label = tk.Label(
            chart_frame, text="0%  —  —",
            bg=SURFACE_2, fg=TEXT_DIM, font=("Courier", 7))
        self.chart_axis_label.pack(pady=(0, 4))

        # Error metrics
        self._section_label(panel, "Error Metrics")
        err_frame = tk.Frame(panel, bg=SURFACE_2,
                             highlightbackground=BORDER, highlightthickness=1)
        err_frame.pack(fill="x", pady=(0, 4))

        self.err_mse   = self._error_row(err_frame, "MSE",   "—", 0)
        self.err_rmse  = self._error_row(err_frame, "RMSE",  "—", 1)
        self.err_nrmse = self._error_row(err_frame, "NRMSE", "—", 2)

        tk.Label(
            err_frame,
            text="Compares simulated vs. expected\nuniform spread. Needs ≥5 ticks.",
            bg=SURFACE_2, fg=TEXT_DIM,
            font=("Helvetica", 7),
            justify="left",
        ).grid(row=3, column=0, columnspan=2, sticky="w", padx=8, pady=(4, 6))

    # ── UI HELPERS ──────────────────────────────────────────────────────────────
    def _section_label(self, parent, text, small=False):
        size = 7 if small else 8
        tk.Label(
            parent, text=text.upper(),
            bg=SURFACE, fg=TEXT_MUTED,
            font=("Helvetica", size, "bold"),
        ).pack(fill="x", pady=(8, 2))
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", pady=(0, 4))

    def _make_btn(self, parent, text, color, cmd):
        btn = tk.Button(
            parent, text=text,
            bg=color, fg="#ffffff",
            activebackground=color, activeforeground="#ffffff",
            font=("Helvetica", 9, "bold"),
            bd=0, highlightthickness=0,
            cursor="hand2", pady=7,
            command=cmd,
        )
        btn.pack(fill="x", pady=2)
        return btn

    def _metric_card(self, parent, label, value, row, col):
        card = tk.Frame(parent, bg=SURFACE_2,
                        highlightbackground=BORDER, highlightthickness=1)
        card.grid(row=row, column=col, padx=3, pady=3, sticky="nsew")
        parent.grid_columnconfigure(col, weight=1)

        tk.Label(card, text=label.upper(), bg=SURFACE_2, fg=TEXT_DIM,
                 font=("Helvetica", 7, "bold")).pack(anchor="w", padx=6, pady=(5, 0))
        val_label = tk.Label(card, text=value, bg=SURFACE_2, fg=ACCENT,
                             font=("Courier", 13, "bold"))
        val_label.pack(anchor="w", padx=6, pady=(0, 5))
        return val_label

    def _error_row(self, parent, label, value, row):
        tk.Label(parent, text=label, bg=SURFACE_2, fg=TEXT_DIM,
                 font=("Helvetica", 8, "bold")).grid(
            row=row, column=0, sticky="w", padx=8, pady=3)
        val = tk.Label(parent, text=value, bg=SURFACE_2, fg=ACCENT,
                       font=("Courier", 10, "bold"))
        val.grid(row=row, column=1, sticky="e", padx=8, pady=3)
        return val

    def _draw_status_dot(self, state):
        self.status_dot_canvas.delete("all")
        color = {
            "idle":    TEXT_DIM,
            "running": GREEN,
            "paused":  BTN_PAUSE,
            "done":    TEXT_MUTED,
        }.get(state, TEXT_DIM)
        self.status_dot_canvas.create_oval(1, 1, 9, 9, fill=color, outline="")

    def _highlight_compass_btn(self, dx, dy):
        for (bdx, bdy), btn in self.compass_buttons.items():
            if bdx == dx and bdy == dy:
                btn.config(bg=ACCENT, fg="#ffffff")
            else:
                btn.config(bg=BORDER, fg=TEXT_MUTED)

    # ── GRID RENDERER ───────────────────────────────────────────────────────────
    def _render_grid(self):
        self.canvas.delete("all")
        for r in range(ROWS):
            for c in range(COLS):
                state = self.grid[r][c]
                if state == BURNING:
                    color = random.choice(FIRE_COLORS)
                else:
                    color = COLORS[state]
                x1 = c * CELL
                y1 = r * CELL
                x2 = x1 + CELL - 1
                y2 = y1 + CELL - 1
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

    # ── CHART RENDERER ──────────────────────────────────────────────────────────
    def _draw_chart(self):
        c = self.chart_canvas
        c.delete("all")
        w, h = 200, 100

        # Grid lines
        for i in range(1, 4):
            y = (h // 4) * i
            c.create_line(0, y, w, y, fill=BORDER_LIT, width=1)
        for i in range(1, 4):
            x = (w // 4) * i
            c.create_line(x, 0, x, h, fill=BORDER_LIT, width=1)

        if len(self.chart_data) < 2:
            c.create_text(w // 2, h // 2, text="Awaiting data...",
                          fill=TEXT_MUTED, font=("Helvetica", 9))
            return

        max_pct = max(d[1] for d in self.chart_data) or 1
        n       = len(self.chart_data)

        # Build polygon for fill
        pts = []
        for i, (_, pct) in enumerate(self.chart_data):
            x = int(i / (n - 1) * w)
            y = int(h - (pct / max_pct) * (h - 4) - 2)
            pts.append((x, y))

        fill_pts = pts + [(w, h), (0, h)]
        flat = [coord for pt in fill_pts for coord in pt]
        c.create_polygon(flat, fill="#3a1f06", outline="")

        # Line
        line_pts = [coord for pt in pts for coord in pt]
        c.create_line(line_pts, fill=ACCENT, width=2, smooth=True)

        # Live dot
        lx, ly = pts[-1]
        c.create_oval(lx - 3, ly - 3, lx + 3, ly + 3,
                      fill=ACCENT, outline="#ffffff", width=1)

        self.chart_axis_label.config(
            text=f"0%  ——  {max_pct:.0f}%")

    # ── METRICS DISPLAY UPDATE ──────────────────────────────────────────────────
    def _update_metrics_display(self, metrics):
        self.metric_burned.config(text=f"{metrics['burned_pct']}%")
        self.metric_perim.config( text=f"{metrics['perimeter']} km")
        self.metric_fronts.config(text=str(metrics["fire_fronts"]))
        self.metric_tick.config(  text=str(self.tick))

    def _update_error_display(self, errors):
        self.err_mse.config(  text="—" if errors["mse"]   is None else f"{errors['mse']}")
        self.err_rmse.config( text="—" if errors["rmse"]  is None else f"{errors['rmse']}")
        self.err_nrmse.config(text="—" if errors["nrmse"] is None else f"{errors['nrmse']}%")

    def _set_status(self, state):
        labels = {
            "idle":    "Idle",
            "running": "Running",
            "paused":  "Paused",
            "done":    "Fire Out",
        }
        self.status_label.config(text=labels.get(state, "Idle"))
        self.grid_status_label.config(
            text=f"⬤  {labels.get(state, 'Idle')}",
            fg=GREEN if state == "running" else BTN_PAUSE if state == "paused" else TEXT_DIM,
        )
        self._draw_status_dot(state)

    # ── SIMULATION LOOP ─────────────────────────────────────────────────────────
    def _tick(self):
        """One simulation tick — delegated entirely to Section 1 functions."""
        if not self.running:
            return

        # Run ABM step (Section 1)
        self.grid = simulation_step(
            self.grid, self.wind_dx, self.wind_dy, self.wind_speed)
        self.tick += 1

        # Compute metrics (Section 1)
        metrics = compute_metrics(self.grid)
        self.burn_history.append(metrics["burned_pct"])
        if len(self.burn_history) > 120:
            self.burn_history.pop(0)
        self.chart_data.append((self.tick, metrics["burned_pct"]))
        if len(self.chart_data) > 120:
            self.chart_data.pop(0)

        # Compute error metrics (Section 1)
        errors = compute_error_metrics(self.burn_history)

        # Update display
        self._render_grid()
        self._update_metrics_display(metrics)
        self._update_error_display(errors)
        self._draw_chart()

        # Check termination
        if not metrics["has_fire"]:
            self._pause_sim()
            self._set_status("done")
            return

        # Schedule next tick
        interval = get_interval(self.wind_speed)
        self.after_id = self.root.after(interval, self._tick)

    # ── SIMULATION CONTROLS ─────────────────────────────────────────────────────
    def _start_sim(self):
        if self.running:
            return
        self.running = True
        self._set_status("running")
        self._tick()

    def _pause_sim(self):
        self.running = False
        if self.after_id:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        if self.tick > 0:
            self._set_status("paused")

    def _reset_sim(self):
        self._pause_sim()
        self.grid         = init_grid()
        self.tick         = 0
        self.burn_history = []
        self.chart_data   = []
        self._render_grid()
        self._update_metrics_display(compute_metrics(self.grid))
        self._update_error_display({"mse": None, "rmse": None, "nrmse": None})
        self._draw_chart()
        self._set_status("idle")

    # ── USER INPUT EVENTS ───────────────────────────────────────────────────────
    def _on_speed_change(self, val):
        self.wind_speed = int(float(val))
        self.speed_val_label.config(text=SPEED_LABELS[self.wind_speed])
        if self.running:
            # Restart with new interval
            if self.after_id:
                self.root.after_cancel(self.after_id)
            interval = get_interval(self.wind_speed)
            self.after_id = self.root.after(interval, self._tick)

    def _set_wind_dir(self, btn, dx, dy):
        self.wind_dx = dx
        self.wind_dy = dy
        self._highlight_compass_btn(dx, dy)

    def _select_ignite(self):
        self.tool = "ignite"
        self.ignite_btn.config(fg=ACCENT, highlightbackground=ACCENT)
        self.break_btn.config( fg=TEXT_MUTED, highlightbackground=BORDER)

    def _select_break(self):
        self.tool = "break"
        self.break_btn.config( fg=ACCENT, highlightbackground=ACCENT)
        self.ignite_btn.config(fg=TEXT_MUTED, highlightbackground=BORDER)

    def _cell_from_event(self, event):
        c = max(0, min(COLS - 1, event.x // CELL))
        r = max(0, min(ROWS - 1, event.y // CELL))
        return r, c

    def _apply_tool(self, event):
        r, c = self._cell_from_event(event)
        if self.tool == "ignite" and self.grid[r][c] == TREE:
            self.grid[r][c] = BURNING
            self._render_grid()
            self._update_metrics_display(compute_metrics(self.grid))
        elif self.tool == "break":
            self.grid[r][c] = BREAK
            self._render_grid()
            self._update_metrics_display(compute_metrics(self.grid))

    def _on_canvas_press(self, event):
        self._apply_tool(event)

    def _on_canvas_drag(self, event):
        self._apply_tool(event)


# ── ENTRY POINT ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = WildfireApp(root)
    root.mainloop()

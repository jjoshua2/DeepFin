"""log_temp trajectory vs CUMULATIVE OPTIMIZER STEPS, with the frozen force curve overlaid.

⚑ Replaces the withdrawn linear extrapolation (ledger `849591754` correction 3).
A restoring force means RELAXATION TOWARD A FIXED POINT, so a line through the early
slope is the wrong model and must never be a falsifier. The right test is whether the
measured slope FALLS as `log_temp` approaches the frozen root -- i.e. whether the
trajectory has the SHAPE the frozen sweep predicts.

x is cumulative optimizer STEPS, not iterations: views-targeting makes steps/iter vary
(87..100 in this window), so an iteration axis silently rescales the x-axis by ~15%.

The overlay has ONE free parameter, a scalar effective rate `eta`, fitted by least
squares on the F-only points. The SHAPE is the prediction; `eta` is not.
"""
from __future__ import annotations

import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUN = ("runs/pbt2_small/tune/"
       "train_trial_dea5e_00000_0_lr=0.0000_2026-08-16_12-38-11/progress.csv")
FLIP = 301  # F-only flip: w_sf_own_regret 0.7 -> 0.0

sweep = json.load(open("scratchpad/logtemp_sweep.json"))
curve = sorted(sweep["curve"], key=lambda d: d["log_temp"])
cl = np.array([c["log_temp"] for c in curve])
cg = np.array([c["F_only"] for c in curve])
l_star = float(sweep["l_star_F_only"])
l_ckpt = float(sweep["l_ckpt"])

# cumulative optimizer steps, keyed by iteration
steps: dict[int, int] = {}
cum = 0
for r in csv.DictReader(open(RUN)):
    it = int(r["training_iteration"])
    cum += int(float(r["trainer_steps_done"]))
    steps[it] = cum
flip_cum = steps.get(FLIP, 0)

traj = [json.loads(x) for x in open("scratchpad/f_only_logtemp_trajectory.jsonl") if x.strip()]
traj.sort(key=lambda d: d["iter"])
pts = [(d["iter"], d["policy_own_log_temp"], d.get("arm", "?")) for d in traj if d["iter"] in steps]
xs = np.array([steps[i] - flip_cum for i, _, _ in pts], dtype=float)
ys = np.array([v for _, v, _ in pts])
arms = [a for _, _, a in pts]
post = np.array([a == "F_only" for a in arms])

# measured local slope between consecutive F-only points, vs log_temp at the midpoint
px, py = xs[post], ys[post]
o = np.argsort(px)
px, py = px[o], py[o]
d_step = np.diff(px)
keep = d_step > 0
slope = np.diff(py)[keep] / d_step[keep]
mid = ((py[:-1] + py[1:]) / 2.0)[keep]

# ⚑ SIGN CONVENTION, read off the generator and NOT inferred: `logtemp_sweep.py`
# computes `gce += -((p - t) * z).sum()`, i.e. MINUS dCE/d(log T) -- G is already the
# DESCENT direction. So the update is `dl/dstep = +eta * G(l)`, and a fit that comes
# back with eta < 0 is a bug in the overlay, not a sign disagreement in the data.
g_at = np.interp(mid, cl, cg)
eta = float((slope @ g_at) / (g_at @ g_at)) if g_at.any() else 0.0
pred = eta * g_at

# ZERO-FREE-PARAMETER check: take eta = lr straight from the run and predict the
# total displacement over the window. This is the test with teeth; the fitted eta
# above only reports what rate the data implies.
LR = 3e-5
g_traj = np.interp(py, cl, cg)
pred_disp = (float((0.5 * (g_traj[:-1] + g_traj[1:]) * np.diff(px)).sum()) * LR
              if px.size > 1 else float("nan"))
meas_disp = float(py[-1] - py[0])
ss_res = float(((slope - pred) ** 2).sum())
ss_tot = float(((slope - slope.mean()) ** 2).sum())
r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5.0))

a1.axhline(l_star, color="crimson", ls="--", lw=1.4,
           label=f"frozen F-only root  l* = {l_star:.4f}")
a1.axhline(l_ckpt, color="gray", ls=":", lw=1.2, label=f"sweep ckpt  {l_ckpt:.4f}")
a1.axvline(0.0, color="k", lw=0.8, alpha=0.5)
a1.plot(xs[~post], ys[~post], "o", color="steelblue", ms=6, label="pre-flip (A+F / pre-arms)")
a1.plot(px, py, "o-", color="darkorange", ms=5, lw=1.5, label="F-only")
a1.set_xlabel(f"cumulative optimizer steps since the flip (iter {FLIP})")
a1.set_ylabel("policy_own.log_temp")
a1.set_title("Trajectory on a STEP axis — no extrapolation drawn")
a1.legend(fontsize=8, loc="best")
a1.grid(alpha=0.25)

a2.plot(cl, eta * cg, color="crimson", lw=2.0,
        label=f"frozen force curve  η·G(log T),  fitted η = {eta:.3g}")
a2.plot(cl, LR * cg, color="seagreen", lw=1.6, ls="--",
        label=f"same curve at η = lr = {LR:g}  (no free parameter)")
a2.axhline(0.0, color="k", lw=0.8, alpha=0.5)
a2.axvline(l_star, color="crimson", ls="--", lw=1.2, label="root (force = 0)")
a2.plot(mid, slope, "o", color="darkorange", ms=7, label="measured Δlog T / Δstep")
a2.set_xlabel("policy_own.log_temp")
a2.set_ylabel("d(log_temp) / d(optimizer step)")
a2.set_title(f"Measured slope vs the FROZEN curve (shape is the prediction)\n"
             f"one free scalar η;  R² = {r2:.3f}")
a2.legend(fontsize=8, loc="best")
a2.grid(alpha=0.25)

fig.tight_layout()
fig.savefig("scratchpad/logtemp_force_overlay.png", dpi=130)

print(f"flip at iter {FLIP}, cumulative step {flip_cum}")
print(f"F-only points: {int(post.sum())}, spanning {px.min():.0f}..{px.max():.0f} steps")
print(f"log_temp {py[0]:.5f} -> {py[-1]:.5f}   (delta {py[-1]-py[0]:+.5f})")
print(f"distance to frozen root: {py[-1]-l_star:+.5f}  ({abs(py[-1]-l_star)/abs(py[0]-l_star)*100:.1f}% of the initial gap remaining)")
print(f"fitted eta = {eta:.6g}   R2(shape) = {r2:.4f}   n_slopes = {slope.size}")
print(f"lr from the run = {LR:g}  -> fitted/lr = {eta/LR:.2f}x")
print(f"ZERO-PARAM displacement over the window: predicted {pred_disp:+.5f}  "
      f"measured {meas_disp:+.5f}  ratio {meas_disp/pred_disp:.2f}x")

# relaxation time under the FROZEN model itself (an ODE, not a line)
l, n, STEP = float(py[-1]), 0, 200
while l > l_star + 0.001 and n < 4_000_000:
    l += LR * float(np.interp(l, cl, cg)) * STEP
    n += STEP
print(f"frozen-model relaxation to within 0.001 of the root: "
      f"{n:,} more optimizer steps (~{n/95:.0f} iterations at ~95 steps/iter)")
print("\n  log_temp    measured dl/dstep      frozen +eta*G")
for m, s, p_ in zip(mid, slope, pred):
    print(f"  {m:+.5f}   {s:+.3e}        {p_:+.3e}")

"""
Final results table for the report: every run folder given, grouped by trajectory type.

Per run (all over the phase where the reference wants contact, unless noted):
    pad RMS / max   in-plane pad position error vs the reference (mm)
    force RMS       MuJoCo normal force vs the reference force (N)
    contact         share of the contact phase the pad is touching the wall (%)
    touch delay     first touch minus the start of the reference's contact phase (s)
    release delay   time after the contact phase ends until the pad leaves the wall for good (s)
    solver          failed solves / numeric retries / suboptimal solves, from events.csv

The trajectory type (push, vertical, horizontal, circle) is read from the reference itself,
so the folder names don't matter.

Run:  python results_table.py MPC_runs/final_*
      python results_table.py MPC_runs/push_1234 MPC_runs/vert_7 ...
"""

import sys
from pathlib import Path

import numpy as np


CONTACT_FORCE = 0.05        #N, MuJoCo normal force above this counts as touching
TYPE_ORDER = ["push", "vertical", "horizontal", "circle"]


def quat_to_rotation(q):
    #[w, x, y, z] -> rotation matrix
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def wall_axes(n):
    #Wall's in-plane axes: along (horizontal) and up
    along = np.cross(n, [0.0, 0.0, 1.0])
    along = along / np.linalg.norm(along)
    return along, np.cross(along, n)


def trajectory_type(ref_pad, ref_contact, n):
    #How far the reference moves along and up the wall while it wants contact
    along, up = wall_axes(n)
    moving = ref_pad[ref_contact]
    span_along = np.ptp(moving @ along)
    span_up = np.ptp(moving @ up)
    if max(span_along, span_up) < 0.01:
        return "push"
    if span_up > 3 * span_along:
        return "vertical"
    if span_along > 3 * span_up:
        return "horizontal"
    return "circle"


def run_metrics(run_folder):
    data = np.load(run_folder / "run_data.npz")
    times, states = data["times"], data["states"]
    force = data["normal_force"]
    n = data["wall_normal_vector"]

    #Reference at every saved time (the saved reference is the coarse plan)
    ref_time = data["reference_time"]
    ref_force = np.interp(times, ref_time, data["reference_normal_force"])
    ref_contact = np.round(np.interp(times, ref_time, data["reference_contact_flag"])) > 0.5
    ref_pad = np.stack([np.interp(times, ref_time, data["reference_pad_position"][:, i]) for i in range(3)], 1)

    #Pad offset (COM -> pad, body frame) from the start: the reference starts at the starting pad position
    R0 = quat_to_rotation(data["initial_state"][6:10])
    pad_offset = R0.T @ (data["reference_pad_position"][0] - data["initial_state"][0:3])
    pad = np.array([s[0:3] + quat_to_rotation(s[6:10]) @ pad_offset for s in states])

    #In-plane pad error (the normal direction is set by the wall during contact)
    error = pad - ref_pad
    error = error - (error @ n)[:, None] * n
    pad_error = 1000 * np.linalg.norm(error, axis=1)

    touching = force > CONTACT_FORCE
    contact_start = times[ref_contact][0]
    contact_end = times[ref_contact][-1]

    #Touch delay: first touch relative to the start of the contact phase (negative = early)
    touch_delay = times[touching][0] - contact_start if touching.any() else np.nan

    #Release: pad off the wall from some point after the contact phase until the end of the run
    after = times > contact_end
    still_touching = np.where(touching & after)[0]
    if touching[-1]:
        release_delay = np.nan
    elif len(still_touching) == 0:
        release_delay = 0.0
    else:
        release_delay = times[still_touching[-1] + 1] - contact_end

    #Solver events
    events_path = run_folder / "events.csv"
    events = [line.split(",")[2] for line in events_path.read_text().splitlines()[1:]] if events_path.exists() else []

    return dict(
        name=run_folder.name,
        seed=int(data["seed"]) if "seed" in data else -1,
        type=trajectory_type(ref_pad, ref_contact, n),
        pad_rms=float(np.sqrt(np.mean(pad_error[ref_contact] ** 2))),
        pad_max=float(pad_error[ref_contact].max()),
        force_rms=float(np.sqrt(np.mean((force[ref_contact] - ref_force[ref_contact]) ** 2))),
        contact=100 * float(touching[ref_contact].mean()),
        touch_delay=float(touch_delay),
        release_delay=float(release_delay),
        failed=sum(e in ("solve_failed", "solver_error") for e in events),
        retries=sum(e == "numeric_retry" for e in events),
        suboptimal=sum(e == "solve_suboptimal" for e in events),
    )


def spread(values, fmt):
    #"a" if all runs agree, otherwise "min–max"
    lo, hi = np.nanmin(values), np.nanmax(values)
    return format(lo, fmt) if format(lo, fmt) == format(hi, fmt) else f"{lo:{fmt}}–{hi:{fmt}}"


folders = [Path(p) for p in sys.argv[1:] if (Path(p) / "run_data.npz").exists()]
if not folders:
    sys.exit("No run folders with run_data.npz given. Usage: python results_table.py MPC_runs/final_*")
rows = [run_metrics(f) for f in folders]
rows.sort(key=lambda r: (TYPE_ORDER.index(r["type"]), r["seed"]))

#Every run
print(f"{'run':>24s} {'type':>10s} {'seed':>5s} | {'pad RMS':>8s} {'pad max':>8s} {'F RMS':>6s} "
      f"{'contact':>8s} {'touch':>6s} {'release':>8s} | {'fail':>4s} {'retry':>5s} {'subopt':>6s}")
print(f"{'':>24s} {'':>10s} {'':>5s} | {'mm':>8s} {'mm':>8s} {'N':>6s} {'%':>8s} {'s':>6s} {'s':>8s} |")
for r in rows:
    release = "stuck" if np.isnan(r["release_delay"]) else f"{r['release_delay']:.2f}"
    print(f"{r['name']:>24s} {r['type']:>10s} {r['seed']:5d} | {r['pad_rms']:8.1f} {r['pad_max']:8.1f} "
          f"{r['force_rms']:6.2f} {r['contact']:8.0f} {r['touch_delay']:6.2f} {release:>8s} | "
          f"{r['failed']:4d} {r['retries']:5d} {r['suboptimal']:6d}")

#Per trajectory type, as ranges over the seeds
print("\nSummary (range over seeds):")
print(f"{'type':>10s} {'runs':>5s} | {'pad RMS mm':>11s} {'F RMS N':>10s} {'contact %':>10s} "
      f"{'touch s':>11s} {'released':>9s} {'fails':>6s} {'retries':>8s}")
typst_rows = []
for kind in TYPE_ORDER:
    group = [r for r in rows if r["type"] == kind]
    if not group:
        continue
    get = lambda key: np.array([r[key] for r in group])
    released = int(np.sum(~np.isnan(get("release_delay"))))
    cells = [spread(get("pad_rms"), ".1f"), spread(get("force_rms"), ".2f"),
             spread(get("contact"), ".0f"), spread(get("touch_delay"), ".2f")]
    print(f"{kind:>10s} {len(group):5d} | {cells[0]:>11s} {cells[1]:>10s} {cells[2]:>10s} {cells[3]:>11s} "
          f"{released:>4d}/{len(group):<4d} {int(get('failed').sum()):6d} {int(get('retries').sum()):8d}")
    typst_rows.append(f"    [{kind.capitalize()}], [{cells[0]}], [{cells[1]}], [{cells[2]}], "
                      f"[{released}/{len(group)}],")

#Paste-ready Typst table
print("\nTypst:")
print("""#figure(
  table(
    columns: 5,
    align: (left, center, center, center, center),
    stroke: none,
    table.hline(),
    [*Trajectory*], [*Pad RMS (mm)*], [*Force RMS (N)*], [*Contact (%)*], [*Released*],
    table.hline(stroke: 0.5pt),""")
print("\n".join(typst_rows))
seeds = ", ".join(str(s) for s in sorted({r["seed"] for r in rows}))
print(f"""    table.hline(),
  ),
  caption: [Closed-loop results over randomized seeds {seeds} for each trajectory, given as the range over seeds. Errors are over the phase where the reference wants contact; pad error is in the wall plane, force error is MuJoCo vs. reference.],
) <results_table>""")

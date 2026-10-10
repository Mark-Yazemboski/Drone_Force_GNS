"""Where does an epoch's time go? GPU compute, or Python/CPU overhead?

Run on the same node you train on, with your run file's settings:
    python profile_training_speed.py --run-file Test_1.py
    python profile_training_speed.py --run-file Test_1.py --n-traj 60 --batch-sizes 256,1024,4096

It loads --n-traj training trajectories (enough to fill batches; the epoch
estimate is scaled to your full N_train), fits the normalization stats the same
way training does, then times, per batch and per stage:
    data     building the batch on the CPU (the per-sample Python loop in
             iterate_drone_chains), copying it to the GPU, rotation augmentation
    forward  unroll_loss (K model steps)
    backward loss.backward() + optimizer step
It does this at several batch sizes. The key test is how the time per batch
scales with batch size:
    ~flat (16x the samples for < ~3x the time)  -> OVERHEAD-BOUND: the GPU sits idle
        waiting for Python to launch thousands of tiny kernels. A faster GPU will
        NOT help; fewer launches (vectorized code, bigger batches, CUDA graphs) will.
    ~proportional to batch size                 -> COMPUTE-BOUND: the GPU is the limit.
It also times one validation pass and one drag-coefficient refit, which run
every val_interval / drag_refit_interval epochs, and prints a torch.profiler
summary of one training step (CPU time vs GPU time, number of kernel launches).
Nothing is trained or saved.
"""

import argparse
import dataclasses
import importlib.util
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from drone_data import build_drone_dataset, annotate_pad_distance
from train_drone_gns import (DroneForceModel, _make_phys, stage_windows, fit_aero_stats, fit_contact_stats,
                             _batches, _stage_noise, unroll_loss, _set_trainable, validation_loss,
                             kstep_validation, force_validation, fit_drag_coefficients)


def load_run_file(path):
    spec = importlib.util.spec_from_file_location("run_settings", os.path.abspath(path))
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    spec.loader.exec_module(mod)                 # top level only; the run file's __main__ block does not run
    return mod


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def stage_setup(model, phys, stage, s):
    _set_trainable(model, False)
    if stage == "aero":
        _set_trainable(model.aero, True)
        params = list(model.aero.parameters()) + model.params.thrust_parameters()
    else:
        _set_trainable(model.contact, True)
        params = list(model.contact.parameters()) + list(phys.parameters())
    model.train()
    return torch.optim.Adam(params, lr=1e-12)    # tiny lr: the timing is real, the weights barely move


def time_batches(model, phys, data, idx, s, device, stage, n_iter, warmup=3):
    opt = stage_setup(model, phys, stage, s)
    noise = _stage_noise(model, s, stage)
    t_data = t_fwd = t_bwd = 0.0
    n = 0
    it = _batches(data, idx, s, device, s.multistep, train=True, noise=noise)
    while n < warmup + n_iter:
        sync(device)
        t0 = time.perf_counter()
        try:
            b = next(it)
        except StopIteration:
            it = _batches(data, idx, s, device, s.multistep, train=True, noise=noise)
            continue
        sync(device)
        t1 = time.perf_counter()
        loss, p, raws = unroll_loss(model, phys, b, s, stage)    # includes its own float() syncs
        sync(device)
        t2 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        float(loss.detach())
        sync(device)
        t3 = time.perf_counter()
        if n >= warmup:
            t_data += t1 - t0
            t_fwd += t2 - t1
            t_bwd += t3 - t2
        n += 1
    return {k: 1e3 * v / n_iter for k, v in (("data", t_data), ("forward", t_fwd), ("backward", t_bwd))}


def profile_one_step(model, phys, data, idx, s, device, stage):
    from torch.profiler import profile, ProfilerActivity
    opt = stage_setup(model, phys, stage, s)
    b = next(_batches(data, idx, s, device, s.multistep, train=True, noise=_stage_noise(model, s, stage)))
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.type == "cuda" else [])
    for _ in range(2):                                           # warm up
        loss = unroll_loss(model, phys, b, s, stage)[0]
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    sync(device)
    with profile(activities=acts) as prof:
        loss = unroll_loss(model, phys, b, s, stage)[0]
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sync(device)
    ev = prof.key_averages()
    cpu_ms = sum(e.self_cpu_time_total for e in ev) / 1e3
    gpu_ms = n_kernels = 0
    if device.type == "cuda":
        dev_time = lambda e: getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0))
        gpu_ms = sum(dev_time(e) for e in ev) / 1e3
        n_kernels = sum(e.count for e in ev if dev_time(e) > 0 and not e.key.startswith("aten::")
                        and not e.key.startswith("cuda"))
    n_ops = sum(e.count for e in ev if e.key.startswith("aten::"))
    return cpu_ms, gpu_ms, n_kernels, n_ops


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-file", default=os.path.join(HERE, "run_drone.py"))
    ap.add_argument("--n-traj", type=int, default=40, help="trajectories to load for timing")
    ap.add_argument("--batch-sizes", default="256,1024,4096")
    ap.add_argument("--iters", type=int, default=15, help="timed batches per (stage, batch size)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-profiler", action="store_true")
    a = ap.parse_args()

    run = load_run_file(a.run_file)
    s0, cfg = run.settings, run.drone
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    n_full = len(run.train_range)

    print("=" * 72)
    print(f"torch {torch.__version__} | device {device}"
          + (f" ({torch.cuda.get_device_name(device)}, "
             f"{torch.cuda.get_device_properties(device).total_memory / 2**30:.0f} GB)" if device.type == "cuda" else "")
          + f" | CPU threads {torch.get_num_threads()} | CPUs visible {len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()}")
    print("=" * 72)

    ids = list(run.train_range)[:a.n_traj]
    data, _ = build_drone_dataset(ids, run.trajectory_folder, cfg, run.rotor_speed_in_rpm, run.rotor_speed_hold)
    val_ids = list(run.val_range)[:max(5, a.n_traj // 4)]
    val, _ = build_drone_dataset(val_ids, run.trajectory_folder, cfg, run.rotor_speed_in_rpm, run.rotor_speed_hold)
    print(f"loaded {len(data)} train / {len(val)} val trajectories "
          f"(avg {sum(d['T'] for d in data) / len(data):.0f} frames); epoch estimates scaled to N_train = {n_full}")

    model = DroneForceModel(cfg, s0).to(device)
    phys = _make_phys(cfg, s0).to(device)
    annotate_pad_distance(data, model.pad_rest.cpu())
    annotate_pad_distance(val, model.pad_rest.cpu())
    w = stage_windows(data, val, s0)
    fit_aero_stats(model, data, w["tr_aero"], s0, device)
    fit_contact_stats(model, data, w["tr_contact"], w["tr_touch"], s0, device)
    scale = n_full / len(data)

    sizes = [int(x) for x in a.batch_sizes.split(",")]
    summary = {}
    for stage, key in (("aero", "tr_aero"), ("contact", "tr_contact")):
        idx = w[key]
        n_win = len(idx) * scale
        print(f"\n--- stage {stage}: ~{n_win:,.0f} training windows per epoch at N_train = {n_full} ---")
        print(f"{'batch':>6} | {'data ms':>8} {'fwd ms':>8} {'bwd ms':>8} {'total ms':>9} | "
              f"{'samples/s':>10} | {'batches/ep':>10} {'est. epoch s':>12}")
        rows = []
        for bs in sizes:
            s = dataclasses.replace(s0, batch_size=bs)
            t = time_batches(model, phys, data, idx, s, device, stage, a.iters)
            tot = sum(t.values())
            nb = max(1, int(-(-n_win // bs)))
            rows.append((bs, t, tot))
            print(f"{bs:>6} | {t['data']:8.1f} {t['forward']:8.1f} {t['backward']:8.1f} {tot:9.1f} | "
                  f"{1e3 * bs / tot:10,.0f} | {nb:10d} {nb * tot / 1e3:12.1f}")
        summary[stage] = rows

        s = dataclasses.replace(s0, batch_size=sizes[0])
        sync(device); t0 = time.perf_counter()
        vl = w["vl_aero"] if stage == "aero" else w["vl_contact"]
        va = w["va_aero"] if stage == "aero" else w["va_all"]
        validation_loss(model, phys, val, vl, s, device, stage)
        kstep_validation(model, val, va, s, device, stage == "contact")
        force_validation(model, val, va, s, device)
        sync(device)
        t_val = (time.perf_counter() - t0) * (len(run.val_range) / len(val))
        print(f"one validation pass (scaled to {len(run.val_range)} val trajectories): {t_val:.1f} s, "
              f"every {s0.val_interval} epochs -> +{t_val / s0.val_interval:.1f} s/epoch on average")
        if stage == "aero" and s0.learn_drag_coeffs and s0.drag_coeff_fit == "lstsq":
            sync(device); t0 = time.perf_counter()
            fit_drag_coefficients(model, data, idx, s, device, "network")
            sync(device)
            t_fit = time.perf_counter() - t0             # capped at stats_batches, so not scaled
            print(f"one drag refit: {t_fit:.1f} s, every {s0.drag_refit_interval} epochs "
                  f"-> +{t_fit / s0.drag_refit_interval:.1f} s/epoch on average")

        if not a.no_profiler:
            cpu_ms, gpu_ms, n_k, n_ops = profile_one_step(model, phys, data, idx, s, device, stage)
            line = f"profiler, one step at batch {sizes[0]}: {n_ops} aten ops, CPU {cpu_ms:.0f} ms"
            if device.type == "cuda":
                line += f", GPU busy {gpu_ms:.0f} ms over ~{n_k} kernel launches"
            print(line)

    # ---------------- verdict ----------------
    print("\n" + "=" * 72 + "\nVERDICT\n" + "=" * 72)
    for stage, rows in summary.items():
        (b0, t0, tot0), (b1, t1, tot1) = rows[0], rows[-1]
        ratio = tot1 / tot0
        data_frac = t0["data"] / tot0
        print(f"[{stage}] batch {b0} -> {b1} ({b1 / b0:.0f}x samples) costs {ratio:.1f}x the time per batch; "
              f"data is {100 * data_frac:.0f}% of a batch at {b0}.")
        dev = "GPU" if device.type == "cuda" else "CPU"
        if ratio < 0.25 * (b1 / b0):
            print(f"    OVERHEAD-BOUND at batch {b0}: the {dev} is mostly waiting on Python to launch kernels. "
                  f"A faster {dev} won't help much; bigger batches or fewer ops will.")
        else:
            print(f"    Close to COMPUTE-BOUND: time grows with batch size, so the {dev} itself is the limit.")
        if data_frac > 0.3:
            print("    The batch builder (per-sample Python loop) is a big share: vectorizing it is a free speedup.")
    print("Note: a bigger batch means fewer optimizer steps per epoch, so it is not free either; "
          "epochs at different batch sizes are not equivalent.")


if __name__ == "__main__":
    main()

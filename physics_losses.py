"""Define the optional physics penalties that guide the force GNS during training.
The PhysicsLosses module manages fixed or learnable friction and drag
coefficients and evaluates constraints on friction direction and magnitude,
the Coulomb friction cone, analytic fluid drag, and temporal fluid smoothness.
It scales these residuals so their magnitudes can be interpreted alongside
the trajectory loss. The file also records a bounded history of slip/contact
diagnostics and provides reset and summary functions so the experiment runner
can report how the physical constraints behaved during training.

======================================================================
NORMALIZATION
======================================================================
Every force-like residual is divided by phi_g = g*dt^2, the specific weight
of the cube per step^2 - so a raw value of 1.0 means "a violation the size of
gravity". Torque-like residuals are divided by the empirical angular
acceleration std. This makes the printed raw magnitudes interpretable and the
weights transferable across datasets. Calibration rule: after one epoch, read
the printed raws and set each weight so (weight * raw) is 1-10% of the
position loss.

Slip velocities and gates are DETACHED: the physics terms constrain the
predicted forces given the observed motion; they must not create an incentive
to change the motion to relax the constraint.
"""

from collections import deque

import numpy as np
import torch
import torch.nn as nn



# ======================================================================
# DIAGNOSTIC HISTORY
# ----------------------------------------------------------------------
# A process-local, bounded history of diagnostic snapshots produced by
# PhysicsLosses.slip_gate_report(). The trainer records one snapshot from
# the first batch of each reported epoch; this is separate from the loss
# history and is never used to compute gradients or update the model.
#
# Each snapshot is a dictionary containing:
#   - slip-gate coverage and counterfactual coverage at lower thresholds
#   - contact-node slip-speed percentiles and the configured gate parameters
#   - friction coefficient implied by the predicted forces and learned mu
#   - friction/slip alignment, misalignment angle, and force cancellation
#     for sliding and static contact
#   - contact weight/count and the optional timestep used for unit conversion
#
# The deque keeps only the newest 200 snapshots. summarize_diagnostics()
# normally averages the newest 20 snapshots for the run report, while
# reset_diagnostics() clears the history before a new training run.
# ======================================================================
PHYSICS_DIAGNOSTIC_HISTORY = deque(maxlen=200)


#Clears the physics diagnostic history buffer. 
def reset_diagnostics():
    PHYSICS_DIAGNOSTIC_HISTORY.clear()


#Summarizes the physics diagnostic history buffer.
#Returns a flat dict of floats averaged over the last `last_n` recorded epochs.
#Tracks data like mean alignment, implied friction coefficient, gate fraction,
#force cancellation for sliding and static contact, and the number of recorded epochs.
def summarize_diagnostics(last_n=20):

    out = {}
    if PHYSICS_DIAGNOSTIC_HISTORY:

        # Extract the last `last_n` snapshots from the diagnostic history.
        tail = list(PHYSICS_DIAGNOSTIC_HISTORY)[-last_n:]
        al = np.array([r["mean_align"] for r in tail], dtype=float)
        mi = np.array([r["mu_implied"] for r in tail], dtype=float)
        gf = np.array([r["gate_frac"] for r in tail], dtype=float)

        # Remove any non-finite values to avoid NaNs in the summary.
        al, mi, gf = al[np.isfinite(al)], mi[np.isfinite(mi)], gf[np.isfinite(gf)]


        if al.size:
            # Compute the mean alignment and its standard deviation.
            m = float(al.mean())
            out["diag_align"] = m
            out["diag_align_std"] = float(al.std(ddof=1)) if al.size > 1 else 0.0
            out["diag_misalign_deg"] = float(
                np.degrees(np.arccos(min(1.0, max(-1.0, m)))))
        if mi.size:
            # Compute the mean implied friction coefficient and its standard deviation.
            out["diag_mu_implied"] = float(mi.mean())
            out["diag_mu_implied_std"] = float(mi.std(ddof=1)) if mi.size > 1 else 0.0
        if gf.size:
            # Compute the mean gate fraction.
            out["diag_gate_frac"] = float(gf.mean())
        for key in ("cancel_slide", "cancel_static"):
            # Compute the mean force cancellation for sliding and static contact.
            v = np.array([r.get(key, np.nan) for r in tail], dtype=float)
            v = v[np.isfinite(v)]

            # Remove any non-finite values to avoid NaNs in the summary.
            if v.size:
                out[f"diag_{key}"] = float(v.mean())

        # Record the number of epochs included in the summary.
        out["diag_n_epochs"] = len(tail)
    return out


# PhysicsLosses module: encapsulates the physics-loss state and exposes methods for each violation term.
class PhysicsLosses(nn.Module):

    # Initialize the physics-loss module with the given parameters.
    def __init__(self, phi_g, ang_scale_vec, mu_init, learn_mu, k_init, learn_k,
                 slip_v0, slip_tau, eps=1e-9):
        super().__init__()
        self.register_buffer("phi_g", torch.as_tensor(float(phi_g)))
        self.register_buffer("ang_scale_vec",
                             torch.as_tensor(ang_scale_vec, dtype=torch.float32))

        # Convert the initial friction coefficient to log-space for stability and potential learning.
        log_mu = torch.log(torch.tensor(float(mu_init)))
        if learn_mu:
            self.log_mu = nn.Parameter(log_mu)
        else:
            self.register_buffer("log_mu", log_mu)

        
        # Convert the initial drag coefficient to log-space for stability and potential learning.
        # to better learn k/m, baseline should be turned off.
        log_k = torch.log(torch.tensor(float(k_init)))
        if learn_k:
            self.log_k = nn.Parameter(log_k)
        else:
            self.register_buffer("log_k", log_k)

        
        # Tangential-speed gate: low speeds are treated as static contact and
        # high speeds as sliding contact. slip_v0 is the 50% transition speed;
        # slip_tau controls how sharply the sigmoid changes between regimes.
        self.slip_v0 = slip_v0        # transition speed (m/step)
        self.slip_tau = slip_tau      # transition softness (m/step)
        self.eps = eps

    @property
    # Returns the friction coefficient mu
    def mu(self):
        return torch.exp(self.log_mu)

    @property
    # Returns the quadratic drag coefficient k/m
    def k_over_m(self):
        return torch.exp(self.log_k)

    @torch.no_grad()
    def slip_gate_report(self, phi_contact, c_w, v_node, wall_n, dt=None):
        """Are the sliding-friction terms awake? Mirrors the gating of
        h_friction_direction / h_friction_magnitude exactly.

        Those terms weight every node by (c_w * slip_gate). If the slip_gate
        factor is ~0 across the batch, they contribute nothing at ANY weight
        and mu - whose only gradient path is h_friction_magnitude - is being
        fit from whatever sliver of frames does get through.

        Returns a dict; see fmt_slip_gate_report() for the one-line print.
        phi_contact/c_w/v_node/wall_n: the SAME tensors you pass to
        compute_step_terms. dt (s) is optional and only converts the
        m/step figures to m/s for readability.
        """

        # Detach the velocity to avoid gradients flowing through it
        v = v_node.detach()

        # Tangential velocity relative to the wall normal
        v_t = v - (v * wall_n).sum(-1, keepdim=True) * wall_n
        speed = v_t.norm(dim=-1, keepdim=True)                 # (B,N,1) m/step

        # Unit vector in the direction of tangential velocity
        v_hat_d = v_t / (speed + self.eps)

        # Slip gate: sigmoid that transitions from static to sliding based on tangential speed
        gate = torch.sigmoid((speed - self.slip_v0) / self.slip_tau)

        # Detach the contact weight to avoid gradients flowing through it
        w_c = c_w.detach()

        # Total soft contact weight across the batch. This is the denominator
        # for contact-weighted sliding fractions; it is not physical mass.
        total_contact_weight = w_c.sum().clamp_min(self.eps)

        # Fraction of total contact weight classified as effectively sliding
        gate_frac = float((w_c * gate).sum() / total_contact_weight)

        #Sees only the nodes that are actually in contact. This filters out nodes
        #with negligible contact weight, focusing the slip-speed statistics on
        #the relevant population.
        in_contact = (w_c > 0.5).squeeze(-1)

        # Extract the slip speeds of nodes that are actually in contact.
        s = speed.squeeze(-1)[in_contact]

        # Compute the slip-speed percentiles for the in-contact nodes.
        # If there are no in-contact nodes, return NaN for all percentiles.
        # Basically, we are computing the 10th, 50th, 90th, and 99th percentiles
        # of the slip speed for the nodes that are actually in contact.
        if s.numel() == 0:
            pct = {q: float('nan') for q in (10, 50, 90, 99)}
        else:
            qs = torch.tensor([0.10, 0.50, 0.90, 0.99], device=s.device,
                              dtype=s.dtype)
            vals = torch.quantile(s, qs)
            pct = {q: float(x) for q, x in zip((10, 50, 90, 99), vals)}

        # This will record the counterfactual contact fractions at lower slip thresholds.
        # to basically see how the contact fraction would change if we used
        # a lower slip threshold.
        cf = {}
        for div in (3.0, 10.0, 30.0):
            g2 = torch.sigmoid((speed - self.slip_v0 / div) / self.slip_tau)
            cf[div] = float((w_c * g2).sum() / total_contact_weight)



        # mu implied by the model's OWN predicted forces on sliding nodes.
        # Coulomb says ||phi_t|| = mu * phi_n while sliding, so this is the
        # mu the force decomposition is currently consistent with -
        # independent of the learnable mu parameter.

        # Compute the normal and tangential components of the contact force.
        phi_n = (phi_contact.detach() * wall_n).sum(-1, keepdim=True)
        phi_t = phi_contact.detach() - phi_n * wall_n

        # Weight the tangential force by the contact gate for later calculations.
        wg = w_c * gate

        # Compute the magnitude of the tangential force for each contact node.
        mag = phi_t.norm(dim=-1, keepdim=True)

        # Compute the implied coefficient of friction based on the weighted tangential and normal forces.
        num = (wg * mag).sum()
        den = (wg * phi_n.clamp_min(0.0)).sum()
        mu_implied = float(num / den) if float(den) > self.eps else float('nan')



        # DIRECTIONAL alignment, measured directly instead of inferred from
        # the mu_param/mu_implied ratio. +1 = friction exactly opposes slip
        # (perfect Coulomb), 0 = perpendicular, -1 = friction DRIVES the
        # slip. Magnitude-weighted, so a negligible misaimed force does not
        # drag the average down. This is the crossing-arrow defect as a
        # scalar: mu_param = mu_implied * mean_align.
        # CANCELLATION FRACTION, per branch. 1 - ||sum phi_t|| / sum||phi_t||
        # over the nodes of each cube: 0 = all friction pulling together,
        # 1 = perfect cancellation (large opposing forces summing to
        # nothing). The prediction loss sees only the net, so a cancelling
        # field is free; and in the STATIC branch every other term is gated
        # off, so nothing else measures this at all.

        # section for computing cancellation fractions per branch.
        # basic idea: cancellation fraction measures how much the 
        # tangential forces cancel each other out within each branch.
        def _cancel(weight):

            # Compute the numerator and denominator for the cancellation fraction.
            # numerator: ||sum(weighted tangential forces)||
            # denominator: sum(||weighted tangential forces||)
            num = (weight * phi_t).sum(dim=1).norm(dim=-1)      
            den = (weight * mag).sum(dim=1).squeeze(-1)        

            # Avoid division by zero by checking if the denominator is greater than a small epsilon.
            ok = den > self.eps
            if not bool(ok.any()):
                return float('nan')

            # Compute the cancellation fraction as 1 - ||sum|| / sum||, averaged over valid branches.
            return float((1.0 - num[ok] / den[ok]).mean())

        # Compute cancellation fractions for sliding and static branches.
        cancel_slide = _cancel(w_c * gate)
        cancel_static = _cancel(w_c * (1.0 - gate))

        # section for computing friction alignment.
        align = -(phi_t * v_hat_d).sum(-1, keepdim=True) / (mag + self.eps)

        # Compute the weighted mean alignment of the tangential forces with the sliding direction.
        wgm = wg * mag
        mean_align = (float((wgm * align).sum() / wgm.sum())
                      if float(wgm.sum()) > self.eps else float('nan'))

        # Prepare the report dictionary with all relevant metrics.
        report = dict(gate_frac=gate_frac,
                      slip_v0=float(self.slip_v0),
                      slip_tau=float(self.slip_tau),
                      pct=pct, counterfactual=cf,
                      mu_implied=mu_implied, mu_param=float(self.mu),
                      mean_align=mean_align,
                      cancel_slide=cancel_slide, cancel_static=cancel_static,
                      misalign_deg=float(np.degrees(np.arccos(
                          min(1.0, max(-1.0, mean_align)))))
                      if mean_align == mean_align else float('nan'),
                      n_contact_nodes=float(total_contact_weight), dt=dt)

        # Append the report to the global diagnostic history and return it.
        PHYSICS_DIAGNOSTIC_HISTORY.append(report)

        return report

    @staticmethod
    # Format the slip-gate report for logging.
    def fmt_slip_gate_report(r):

        # Extract the time step and prepare conversion functions for velocity units.
        dt = r["dt"]

        # Determine the conversion factor for velocities based on the time step.
        to_ms = (lambda x: x / dt) if dt else (lambda x: float('nan'))
        u = "m/s" if dt else "m/step"
        conv = to_ms if dt else (lambda x: x)

        # Extract the percentile and counterfactual data from the report.
        p = r["pct"]
        cf = r["counterfactual"]

        # Begin constructing the formatted string for the slip-gate report.
        return (
            f"  Slip gate | OPEN {r['gate_frac']:6.1%} of contact weight  "
            f"| v0={conv(r['slip_v0']):.3f} {u}  "
            f"| contact slip p10/p50/p90/p99 = "
            f"{conv(p[10]):.3f}/{conv(p[50]):.3f}/{conv(p[90]):.3f}/{conv(p[99]):.3f} {u}\n"
            f"            | if v0 were /3: {cf[3.0]:5.1%}   /10: {cf[10.0]:5.1%}   "
            f"/30: {cf[30.0]:5.1%}   "
            f"| mu implied by predicted forces = {r['mu_implied']:.3f} "
            f"(mu param = {r['mu_param']:.3f})\n"
            f"            | friction alignment = {r['mean_align']:+.3f} "
            f"({r['misalign_deg']:.0f} deg off anti-parallel; 1.000 = perfect Coulomb)\n"
            f"            | cancellation  sliding {r['cancel_slide']:.3f}  "
            f"static {r['cancel_static']:.3f}   (0 = forces pull together, "
            f"1 = they cancel to nothing)"
        )
    
    #====================================================================================================================================
    #PHYSICS LOSSES 

    # ==================================================================
    # DIRECTION HALF OF COULOMB FRICTION
    # ------------------------------------------------------------------
    # This function computes the directional component of the Coulomb
    # friction loss, which penalizes misalignment between the tangential
    # friction force and the slip direction of the node.
    # ==================================================================
    def h_friction_direction(self, phi_contact, c_w, v_node, wall_n):
        """

        The per-node cost is  ||phi_t|| * (1 + phi_hat_t . vhat), which is 0
        when friction exactly opposes slip and 2||phi_t|| when it drives it.
        Weighting by magnitude means a large misaimed force is expensive and a
        negligible one is nearly free. LINEAR, not squared: the constant
        gradient keeps pushing all the way to alignment.
        """

        # Extract the tangential component of the node's velocity relative to the wall.
        v = v_node.detach()
        v_t = v - (v * wall_n).sum(-1, keepdim=True) * wall_n

        # Compute the speed and direction of the tangential velocity.
        speed = v_t.norm(dim=-1, keepdim=True)
        v_hat = v_t / (speed + self.eps)

        # Compute the slip gate, which smoothly transitions from 0 to 1 as the tangential speed 
        # exceeds the slip threshold.
        slip_gate = torch.sigmoid((speed - self.slip_v0) / self.slip_tau)

        # Decompose the contact force into normal and tangential components.
        phi_n = (phi_contact * wall_n).sum(-1, keepdim=True)
        phi_t = phi_contact - phi_n * wall_n

        # Compute the magnitude of the tangential force and its misalignment with the slip direction.
        mag = phi_t.norm(dim=-1, keepdim=True)
        misalign = 1.0 + (phi_t * v_hat).sum(-1, keepdim=True) / (mag + self.eps)

        # Compute the weighted misalignment loss.
        # Basically only counting the nodes that are slipping, as indicated by the slip gate.
        w = (c_w * slip_gate).detach()

        # Return the final weighted misalignment loss, normalized by the total contact weight.
        return (w * (mag / self.phi_g) * misalign).sum() / (c_w.detach().sum() + self.eps)


    # ==================================================================
    # MAGNITUDE HALF OF COULOMB FRICTION
    # ------------------------------------------------------------------
    # This function enforces the magnitude half of the Coulomb friction law.
    # It will penalize deviations of the tangential force magnitude from the 
    # Coulomb friction limit when sliding.
    # ==================================================================
    def h_friction_magnitude(self, phi_contact, c_w, v_node, wall_n):
        """
        MAGNITUDE half of Coulomb: ||phi_t|| = mu * phi_n while sliding.
        """

        # Compute the tangential velocity and the slip gate.
        v = v_node.detach()
        v_t = v - (v * wall_n).sum(-1, keepdim=True) * wall_n
        speed = v_t.norm(dim=-1, keepdim=True)
        slip_gate = torch.sigmoid((speed - self.slip_v0) / self.slip_tau)

        # Compute the normal and tangential components of the contact force.
        phi_n = (phi_contact * wall_n).sum(-1, keepdim=True)
        phi_t = phi_contact - phi_n * wall_n


        # phi_n is DETACHED: Coulomb says what friction may be GIVEN the
        # normal force. Left attached, this residual carries a normal-direction
        # gradient of size mu*|dL/dphi_t|, so the cheapest fix for
        # ||phi_t|| != mu*phi_n is to move phi_n - a well-determined,
        # position-loss-observable quantity - instead of the friction.
        # mu keeps its gradient - this is its only path.

        # Compute the residual for the magnitude half of Coulomb friction.
        resid = (phi_t.norm(dim=-1, keepdim=True)
                 - self.mu * phi_n.detach()) / self.phi_g

        # Only consider nodes that are in contact and sliding., and normalize by the contact weight.
        w = (c_w * slip_gate).detach()
        return (w * resid.pow(2)).sum() / (c_w.detach().sum() + self.eps)


    
    # ==================================================================
    # COULOMB FRICTION CONE
    # ------------------------------------------------------------------
    # This function enforces the inequality ||phi_t|| <= mu * phi_n.
    # Unlike the sliding losses, the cone does not require friction to be
    # moving or aligned with slip, so it is useful for static contact.
    # ==================================================================
    def h_friction_cone(self, phi_contact, c_w, v_node, wall_n):
        """
        Enforces the Coulomb friction-cone limit on static contact.

        The loss is zero when the tangential force is inside the cone. It
        penalizes only the amount by which the force exceeds mu * phi_n.
        """

        # Keep mu fixed for this loss so the model cannot reduce the penalty
        # by increasing the learned friction coefficient.
        mu = self.mu.detach()

        # Decompose the predicted contact force into normal and tangential parts.
        phi_n = (phi_contact * wall_n).sum(-1, keepdim=True)
        phi_t = phi_contact - phi_n * wall_n

        # Positive values violate the cone; values at or below zero are valid.
        excess = phi_t.norm(dim=-1, keepdim=True) - mu * phi_n.detach().clamp_min(0.0)

        # Apply the cone to the static side of the slip gate. Sliding nodes are
        # handled by the direction and magnitude losses instead.
        v = v_node.detach()
        v_t = v - (v * wall_n).sum(-1, keepdim=True) * wall_n
        speed = v_t.norm(dim=-1, keepdim=True)
        slip_gate = torch.sigmoid((speed - self.slip_v0) / self.slip_tau)
        w = c_w.detach() * (1.0 - slip_gate)

        # only cares about static nodes; sliding nodes are handled elsewhere.
        # normalized by the total contact weight to avoid scale issues.
        return ((w * (excess.clamp_min(0.0) / self.phi_g).pow(2)).sum()
                / (c_w.detach().sum() + self.eps))


    # ==================================================================
    # FLUID ANCHOR
    # ------------------------------------------------------------------
    # This function penalizes deviations of the predicted fluid acceleration from the analytic drag target.
    # helps guide the learned fluid acceleration toward physically plausible values.
    # ==================================================================
    def h_fluid_anchor(self, a_fluid_total, drag_target):
        # Equation: h_anchor = ||(a_fluid_total[b] - drag_target[b]) / phi_g||^2.
        return ((a_fluid_total - drag_target) / self.phi_g).pow(2).sum(-1).mean()

    # ==================================================================
    # FLUID TEMPORAL SMOOTHNESS
    # ------------------------------------------------------------------
    # This function penalizes abrupt changes in the predicted fluid wrench over time.
    # helps guide the learned fluid wrench toward physically plausible temporal behavior.
    # ==================================================================
    def h_fluid_temporal_smooth(self, fluid_series, torque_series=None):
        # Penalizes abrupt changes in the predicted fluid wrench over an
        # unroll. Fluid loads should vary smoothly, while contact impulses may
        # change rapidly.
        # Equation: h_smooth = mean_k mean_b
        # ||(fluid_series[k+1][b] - fluid_series[k][b]) / phi_g||^2.
        # If torque_series is provided, its matching normalized temporal
        # difference is added using ang_scale_vec:
        # ||(torque_series[k+1][b] - torque_series[k][b]) / ang_scale_vec||^2.
        # This constrains temporal variation, not the absolute fluid magnitude,
        # and returns zero when there is only one unroll step.

        # Return zero if there are not enough time steps to compute differences.
        if len(fluid_series) < 2:
            return fluid_series[0].new_zeros(())

        # Compute the squared normalized differences between consecutive fluid wrench predictions.
        diffs = [((b - a) / self.phi_g).pow(2).sum(-1).mean()
                 for a, b in zip(fluid_series[:-1], fluid_series[1:])]

        # Average the squared differences to get the temporal smoothness loss.
        total = torch.stack(diffs).mean()

        # Store the initial total before adding torque smoothness, if any.
        if torque_series is not None and len(torque_series) >= 2:
            tdiffs = [((b - a) / self.ang_scale_vec).pow(2).sum(-1).mean()
                      for a, b in zip(torque_series[:-1], torque_series[1:])]
            total = total + torch.stack(tdiffs).mean()
        return total

    # This will compute all per-step loss terms for a single time step.
    # Will check to see if the user assigned non-zero weights to each term before computing it.
    # Saves the raw (unweighted) loss terms in a dictionary and returns it.
    def compute_step_terms(self, phi_contact, c_w, v_node, wall_n,
                           a_fluid_total, drag_target, weights):
        
        raws = {}
        if weights.get("w_fric_dir", 0) > 0:
            raws["fric_dir"] = self.h_friction_direction(
                phi_contact, c_w, v_node, wall_n)
        if weights.get("w_fric_mag", 0) > 0:
            raws["fric_mag"] = self.h_friction_magnitude(
                phi_contact, c_w, v_node, wall_n)
        if weights.get("w_fric_cone", 0) > 0:
            raws["fric_cone"] = self.h_friction_cone(
                phi_contact, c_w, v_node, wall_n)
        if weights.get("w_fluid_anchor", 0) > 0:
            raws["fluid_anchor"] = self.h_fluid_anchor(a_fluid_total, drag_target)
        return raws

    @staticmethod
    # Computes the weighted total of the raw loss terms based on the provided weights.
    def weighted_total(raws, weights):
        total = 0.0
        for name, val in raws.items():
            total = total + weights.get("w_" + name, 0.0) * val
        return total

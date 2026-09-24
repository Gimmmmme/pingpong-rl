"""Strict physical table-tennis rally scorer.

The scorer is an evaluator only. It may consume simulator ground truth while
training or auditing, but its output is never part of the policy observation.
Paddle contact is accepted only on the thin blade plane and inside the
160 x 170 mm blade footprint. A handle, wrist, robot body, or inactive arm
cannot be mistaken for a legal paddle return.
"""
from __future__ import annotations

import torch


class RallyScorer:
    """Count legal returns from physical ball motion.

    ``state`` normally contains selected paddle fields (``paddle_pos`` and
    ``paddle_normal``), as in v1. v2 may additionally pass all four paddles:

    ``all_paddle_pos`` / ``all_paddle_normal``: ``[N,2,2,3]`` where the first
    index is table side and the second is left/right arm;
    ``all_paddle_tangent_y``: optional width axis of the blade;
    ``active_arms``: ``[N,2]``, 0=left and 1=right.

    If tangent axes are unavailable, the footprint test uses the conservative
    inscribed circle of radius 80 mm. This is intentionally stricter than a
    disk approximation and cannot include the handle outside the blade.
    """

    def __init__(self, n, device, *, table_top=.72, table_length=1.4,
                 table_width=.7, radius=.02, paddle_radius=.08,
                 paddle_half_width=.08, paddle_half_height=.085,
                 plane_tolerance=.012, net_height=.075):
        self.n = int(n)
        self.device = device
        self.table_top = float(table_top)
        self.half_length = float(table_length) / 2
        self.half_width = float(table_width) / 2
        self.radius = float(radius)
        # Keep the old keyword as an alias, but use physical v2 dimensions.
        self.paddle_half_width = float(paddle_half_width or paddle_radius)
        self.paddle_half_height = float(paddle_half_height)
        self.plane_tolerance = float(plane_tolerance)
        self.net_height = float(net_height)

        z = torch.zeros
        self.previous_pos = z((self.n, 3), device=device)
        self.previous_vel = z((self.n, 3), device=device)
        self.last_hitter = torch.full((self.n,), -1, device=device,
                                      dtype=torch.long)
        self.leg_bounces = torch.zeros(self.n, device=device, dtype=torch.long)
        self.hit_count = torch.zeros(self.n, device=device, dtype=torch.long)
        self.legal_returns = torch.zeros(self.n, device=device, dtype=torch.long)
        self.legal_hits = torch.zeros(self.n, device=device, dtype=torch.long)
        self.last_hit_legal = torch.zeros(self.n, device=device, dtype=torch.bool)
        self.invalid = torch.zeros(self.n, device=device, dtype=torch.bool)
        self.inactive_faults = torch.zeros(self.n, device=device, dtype=torch.long)
        self.last_hit_step = torch.full((self.n,), -1000, device=device,
                                        dtype=torch.long)
        self.last_bounce_step = torch.full((self.n,), -1000, device=device,
                                           dtype=torch.long)
        self.crossed_net = torch.zeros(self.n, device=device, dtype=torch.bool)
        self.step_index = 0

    def reset(self, state, ids=None):
        if ids is None:
            ids = torch.arange(self.n, device=self.device)
        self.previous_pos[ids] = state['ball_pos'][ids]
        self.previous_vel[ids] = state['ball_vel'][ids]
        self.last_hitter[ids] = -1
        self.leg_bounces[ids] = 0
        self.hit_count[ids] = 0
        self.legal_returns[ids] = 0
        self.legal_hits[ids] = 0
        self.last_hit_legal[ids] = False
        self.invalid[ids] = False
        self.inactive_faults[ids] = 0
        self.last_hit_step[ids] = self.step_index - 1000
        self.last_bounce_step[ids] = self.step_index - 1000
        self.crossed_net[ids] = False

    def _paddles(self, state):
        """Return positions, normals, width axes and active arm ids."""
        pp = state.get('all_paddle_pos')
        nn = state.get('all_paddle_normal')
        if pp is None or nn is None:
            # Legacy selected-paddle API. No inactive-arm fault can be
            # asserted because the evaluator was not given inactive poses.
            pp = state['paddle_pos'][:, :, None, :]
            nn = state['paddle_normal'][:, :, None, :]
            yy = None
            active = None
        else:
            if pp.ndim != 4 or tuple(pp.shape[1:3]) != (2, 2):
                raise ValueError('all_paddle_pos must have shape [N,2,2,3]')
            yy = state.get('all_paddle_tangent_y')
            active = state.get('active_arms')
            if active is not None:
                active = torch.as_tensor(active, device=self.device,
                                         dtype=torch.long)
        return pp, nn, yy, active

    @torch.no_grad()
    def update(self, state, dt):
        p = state['ball_pos']
        v = state['ball_vel']
        pp, normal, tangent_y, active = self._paddles(state)
        oldp = self.previous_pos
        oldv = self.previous_vel
        self.step_index += 1

        crossed = ((oldp[:, 0] * p[:, 0] < 0) &
                   (p[:, 2] > self.table_top + self.net_height + self.radius - .005))
        self.crossed_net |= crossed

        bounce = ((oldv[:, 2] < -.08) & (v[:, 2] > .08) &
                  (p[:, 2] < self.table_top + self.radius + .045))
        bounce &= (p[:, 0].abs() < self.half_length + self.radius)
        bounce &= (p[:, 1].abs() < self.half_width + self.radius)
        bounce &= ((self.step_index - self.last_bounce_step) * dt > .05)
        self.last_bounce_step[bounce] = self.step_index

        receiving = torch.where(p[:, 0] >= 0, 1, 0)
        wrong_bounce = bounce & (self.last_hitter >= 0) & (
            (receiving == self.last_hitter) | (~self.crossed_net))
        self.leg_bounces[bounce] += 1
        self.invalid |= wrong_bounce | (bounce & (self.leg_bounces > 1))
        legal_return = (bounce & (self.last_hitter >= 0) & self.last_hit_legal &
                        (~self.invalid) & (self.leg_bounces == 1))
        self.legal_returns[legal_return] += 1

        # Signed distance to the blade plane. Normal is normalized defensively
        # because an audit may provide noisy FK values.
        normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-7)
        rel = p[:, None, None, :] - pp
        plane_signed = (rel * normal).sum(-1)
        plane = plane_signed.abs()
        tangent_component = rel - plane_signed[..., None] * normal

        if tangent_y is not None:
            tangent_y = torch.as_tensor(tangent_y, device=p.device,
                                        dtype=p.dtype)
            tangent_y = tangent_y / tangent_y.norm(dim=-1, keepdim=True).clamp_min(1e-7)
            # Width axis is supplied by the link frame; the second in-plane
            # axis is normal x width (local Z for the v2 paddle).
            tangent_z = torch.linalg.cross(normal, tangent_y, dim=-1)
            tangent_z = tangent_z / tangent_z.norm(dim=-1, keepdim=True).clamp_min(1e-7)
            u = (tangent_component * tangent_y).sum(-1)
            w = (tangent_component * tangent_z).sum(-1)
            footprint = ((u / self.paddle_half_width) ** 2 +
                         (w / self.paddle_half_height) ** 2 <= 1.0)
        else:
            # Inscribed circle is a safe fallback for old selected-paddle
            # records and never reaches the 85 mm long-axis endcaps.
            footprint = tangent_component.square().sum(-1) <= self.paddle_half_width ** 2

        near = (plane <= self.radius + self.plane_tolerance) & footprint
        incoming = torch.stack([oldv[:, 0] < -.12, oldv[:, 0] > .12], 1)
        outgoing = torch.stack([v[:, 0] > .12, v[:, 0] < -.12], 1)
        direction_flip = (incoming & outgoing)[:, :, None]
        hit_candidates = near & direction_flip

        physical_contact = torch.ones(self.n, device=p.device, dtype=torch.bool)
        if 'contact_force' in state:
            # Contact force is evaluator-only. It gates a velocity reversal so
            # a nearby no-contact trajectory cannot score as a hit.
            force = torch.as_tensor(state['contact_force'], device=p.device)
            force = force.reshape(self.n, 3)
            physical_contact = force.norm(dim=-1) > 1e-4
            hit_candidates &= physical_contact[:, None, None]

        # If all four paddles are supplied, mark a reversal on an inactive arm
        # as a fault. Select active candidates when both shapes overlap.
        inactive_fault = torch.zeros(self.n, device=p.device, dtype=torch.bool)
        if active is not None and hit_candidates.shape[2] == 2:
            ar = active.clamp(0, 1)[:, :, None]
            arm_ids = torch.arange(2, device=p.device)[None, None, :]
            # A glancing inactive-arm contact is a fault even if it does not
            # reverse world X. A measured impulse plus a velocity discontinuity
            # is required, so proximity alone cannot trigger a fault.
            discontinuity = (v - oldv).norm(dim=-1) > .08
            inactive_candidates = near & (arm_ids != ar)
            inactive_candidates &= (physical_contact & discontinuity)[:, None, None]
            inactive_fault = inactive_candidates.any(dim=(1, 2))
            inactive_fault &= ((self.step_index - self.last_hit_step) * dt > .05)
            self.inactive_faults += inactive_fault.to(torch.long)
            selected_mask = hit_candidates & (arm_ids == ar)
            any_candidates = hit_candidates.any(dim=2)
            hit_candidates = torch.where(
                any_candidates[:, :, None],
                torch.where(selected_mask.any(dim=2, keepdim=True),
                            selected_mask, hit_candidates),
                hit_candidates)

        hit_by_side = hit_candidates.any(dim=2)
        hit = hit_by_side.any(dim=1)
        hit &= ((self.step_index - self.last_hit_step) * dt > .05)
        side = hit_by_side.to(torch.int).argmax(1)
        arm = hit_candidates.to(torch.int).argmax(2).gather(1, side[:, None]).squeeze(1)

        legal_hit = hit & (~inactive_fault) & (self.leg_bounces == 1) & (~self.invalid)
        legal_hit &= (self.last_hitter < 0) | (side != self.last_hitter)
        self.hit_count[hit] += 1
        self.legal_hits[legal_hit] += 1
        self.last_hit_legal[hit] = legal_hit[hit]
        self.invalid |= (hit & (~legal_hit)) | inactive_fault
        self.last_hitter[hit] = side[hit]
        self.leg_bounces[hit] = 0
        self.crossed_net[hit] = False
        self.last_hit_step[hit] = self.step_index

        out = ((p[:, 2] < self.table_top - .20) | (p[:, 0].abs() > 1.3) |
               (p[:, 1].abs() > .65))
        self.previous_pos.copy_(p)
        self.previous_vel.copy_(v)
        return {
            'hit': hit,
            'hit_side': side,
            'paddle_arm': arm,
            'legal_hit': legal_hit,
            'bounce': bounce,
            'legal_return': legal_return,
            'legal_hits': self.legal_hits.clone(),
            'inactive_fault': inactive_fault,
            'inactive_fault_count': self.inactive_faults.clone(),
            'wrong_bounce': wrong_bounce,
            'invalid': self.invalid.clone(),
            'out': out,
            'hit_count': self.hit_count.clone(),
            'legal_returns': self.legal_returns.clone(),
            'paddle_candidates': hit_candidates.clone(),
        }

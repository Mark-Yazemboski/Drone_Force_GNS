import torch
import numpy as np


# Logarithm map for SO(3): rotation matrix -> rotation vector
def so3_log(R):
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = ((tr - 1.0) * 0.5).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos_theta).unsqueeze(-1)                  # (..., 1)
    vee = torch.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1],
    ], dim=-1) * 0.5
    small = theta < 1e-4
    sin_safe = torch.sin(theta).clamp_min(1e-12)
    factor = torch.where(small, 1.0 + theta ** 2 / 6.0, theta / sin_safe)
    return factor * vee


# Exponential map for SO(3): rotation vector -> rotation matrix (Rodrigues' formula)
def so3_exp(w):
    theta = w.norm(dim=-1, keepdim=True).unsqueeze(-1)          # (..., 1, 1)
    K = torch.zeros(w.shape[:-1] + (3, 3), dtype=w.dtype, device=w.device)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -w[..., 2], w[..., 1], -w[..., 0]
    K[..., 1, 0], K[..., 2, 0], K[..., 2, 1] = w[..., 2], -w[..., 1], w[..., 0]
    small = theta < 1e-4
    theta_safe = theta.clamp_min(1e-12)
    A = torch.where(small, 1.0 - theta ** 2 / 6.0, torch.sin(theta_safe) / theta_safe)
    B = torch.where(small, 0.5 - theta ** 2 / 24.0, (1.0 - torch.cos(theta_safe)) / theta_safe ** 2)
    eye = torch.eye(3, device=w.device, dtype=w.dtype).expand(K.shape)
    return eye + A * K + B * (K @ K)

def quat_to_rot_matrix(q):
    # q: (..., 4) quaternion, scalar first [w, x, y, z] (MuJoCo's convention). Normalized here.
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z),     2 * (x * y - w * z),     2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z),     2 * (y * z - w * x),
            2 * (x * z - w * y),     2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(q.shape[:-1] + (3, 3))


def rot_matrix_to_quat(R):
    # R: (..., 3, 3) -> (..., 4) [w, x, y, z]. Branch-free Shepperd-style: uses the largest
    # of the four candidate denominators, so it's stable at every attitude.
    m00, m11, m22 = R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]
    t = torch.stack([1 + m00 + m11 + m22, 1 + m00 - m11 - m22,
                     1 - m00 + m11 - m22, 1 - m00 - m11 + m22], -1)
    i = t.argmax(-1, keepdim=True)
    s = 2 * torch.sqrt(t.gather(-1, i).clamp_min(1e-12)).squeeze(-1)
    q = torch.stack([
        torch.stack([s / 4, (R[..., 2, 1] - R[..., 1, 2]) / s, (R[..., 0, 2] - R[..., 2, 0]) / s, (R[..., 1, 0] - R[..., 0, 1]) / s], -1),
        torch.stack([(R[..., 2, 1] - R[..., 1, 2]) / s, s / 4, (R[..., 0, 1] + R[..., 1, 0]) / s, (R[..., 0, 2] + R[..., 2, 0]) / s], -1),
        torch.stack([(R[..., 0, 2] - R[..., 2, 0]) / s, (R[..., 0, 1] + R[..., 1, 0]) / s, s / 4, (R[..., 1, 2] + R[..., 2, 1]) / s], -1),
        torch.stack([(R[..., 1, 0] - R[..., 0, 1]) / s, (R[..., 0, 2] + R[..., 2, 0]) / s, (R[..., 1, 2] + R[..., 2, 1]) / s, s / 4], -1),
    ], -2)
    q = q.gather(-2, i.unsqueeze(-1).expand(*i.shape[:-1], 1, 4)).squeeze(-2)
    return q / q.norm(dim=-1, keepdim=True)


def orthonormalize(R):
    U, _, Vt = np.linalg.svd(R)
    return U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt
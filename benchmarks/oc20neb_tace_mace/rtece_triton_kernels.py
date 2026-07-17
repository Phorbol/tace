from __future__ import annotations

import torch


def _load_triton():
    try:
        import triton
        import triton.language as tl
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError("Triton is required for the fused rTECE force path") from exc
    return triton, tl


triton, tl = _load_triton()


@triton.jit
def _pair_force_kernel(
    pos,
    src_index,
    dst_index,
    density_grad,
    forces,
    num_edges: tl.constexpr,
    cutoff: tl.constexpr,
    num_radial: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < num_edges
    src = tl.load(src_index + offsets, mask=mask, other=0)
    dst = tl.load(dst_index + offsets, mask=mask, other=0)

    sx = tl.load(pos + src * 3 + 0, mask=mask, other=0.0)
    sy = tl.load(pos + src * 3 + 1, mask=mask, other=0.0)
    sz = tl.load(pos + src * 3 + 2, mask=mask, other=0.0)
    dx = tl.load(pos + dst * 3 + 0, mask=mask, other=0.0)
    dy = tl.load(pos + dst * 3 + 1, mask=mask, other=0.0)
    dz = tl.load(pos + dst * 3 + 2, mask=mask, other=0.0)

    vx = dx - sx
    vy = dy - sy
    vz = dz - sz
    dist = tl.sqrt(vx * vx + vy * vy + vz * vz)
    dist = tl.maximum(dist, 1.0e-12)
    inv_dist = 1.0 / dist
    ux = vx * inv_dist
    uy = vy * inv_dist
    uz = vz * inv_dist

    x = tl.minimum(tl.maximum(dist / cutoff, 0.0), 1.0)
    x2 = x * x
    x3 = x2 * x
    x4 = x3 * x
    x5 = x4 * x
    inside = dist < cutoff
    envelope = tl.where(inside, 1.0 - 10.0 * x3 + 15.0 * x4 - 6.0 * x5, 0.0)
    envelope_derivative = tl.where(inside, (-30.0 * x2 + 60.0 * x3 - 30.0 * x4) / cutoff, 0.0)

    width = cutoff / (num_radial - 1)
    inv_width2 = 1.0 / (width * width)
    edge_scale = tl.zeros((block_size,), dtype=tl.float32)
    for k in range(0, num_radial):
        center = width * k
        delta = dist - center
        gaussian = tl.exp(-0.5 * delta * delta * inv_width2)
        gaussian_derivative = gaussian * (-delta * inv_width2)
        radial_derivative = gaussian_derivative * envelope + gaussian * envelope_derivative
        grad = tl.load(density_grad + dst * num_radial + k, mask=mask, other=0.0)
        edge_scale += grad * radial_derivative

    gx = edge_scale * ux
    gy = edge_scale * uy
    gz = edge_scale * uz
    tl.atomic_add(forces + dst * 3 + 0, -gx, sem="relaxed", mask=mask)
    tl.atomic_add(forces + dst * 3 + 1, -gy, sem="relaxed", mask=mask)
    tl.atomic_add(forces + dst * 3 + 2, -gz, sem="relaxed", mask=mask)
    tl.atomic_add(forces + src * 3 + 0, gx, sem="relaxed", mask=mask)
    tl.atomic_add(forces + src * 3 + 1, gy, sem="relaxed", mask=mask)
    tl.atomic_add(forces + src * 3 + 2, gz, sem="relaxed", mask=mask)


def pair_forces_triton(
    *,
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    density_grad: torch.Tensor,
    cutoff: float,
    num_radial: int,
    block_size: int = 128,
) -> torch.Tensor:
    if pos.device.type != "cuda":
        raise RuntimeError("Triton pair force path requires a CUDA graph")
    if pos.dtype != torch.float32 or density_grad.dtype != torch.float32:
        raise RuntimeError("Triton pair force path currently supports float32 tensors only")
    if not pos.is_contiguous():
        pos = pos.contiguous()
    src = edge_index[0].contiguous()
    dst = edge_index[1].contiguous()
    density_grad = density_grad.contiguous()
    forces = torch.zeros_like(pos)
    num_edges = int(src.numel())
    grid = (triton.cdiv(num_edges, block_size),)
    _pair_force_kernel[grid](
        pos,
        src,
        dst,
        density_grad,
        forces,
        num_edges,
        float(cutoff),
        int(num_radial),
        block_size,
        num_warps=4,
    )
    return forces


@triton.jit
def _element_density_force_kernel(
    pos,
    src_index,
    dst_index,
    node_z,
    density_grad,
    element_density_grad,
    forces,
    num_edges: tl.constexpr,
    cutoff: tl.constexpr,
    num_radial: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < num_edges
    src = tl.load(src_index + offsets, mask=mask, other=0)
    dst = tl.load(dst_index + offsets, mask=mask, other=0)
    src_z = tl.load(node_z + src, mask=mask, other=0.0)

    sx = tl.load(pos + src * 3 + 0, mask=mask, other=0.0)
    sy = tl.load(pos + src * 3 + 1, mask=mask, other=0.0)
    sz = tl.load(pos + src * 3 + 2, mask=mask, other=0.0)
    dx = tl.load(pos + dst * 3 + 0, mask=mask, other=0.0)
    dy = tl.load(pos + dst * 3 + 1, mask=mask, other=0.0)
    dz = tl.load(pos + dst * 3 + 2, mask=mask, other=0.0)

    vx = dx - sx
    vy = dy - sy
    vz = dz - sz
    dist = tl.sqrt(vx * vx + vy * vy + vz * vz)
    dist = tl.maximum(dist, 1.0e-12)
    inv_dist = 1.0 / dist
    ux = vx * inv_dist
    uy = vy * inv_dist
    uz = vz * inv_dist

    x = tl.minimum(tl.maximum(dist / cutoff, 0.0), 1.0)
    x2 = x * x
    x3 = x2 * x
    x4 = x3 * x
    x5 = x4 * x
    inside = dist < cutoff
    envelope = tl.where(inside, 1.0 - 10.0 * x3 + 15.0 * x4 - 6.0 * x5, 0.0)
    envelope_derivative = tl.where(inside, (-30.0 * x2 + 60.0 * x3 - 30.0 * x4) / cutoff, 0.0)

    width = cutoff / (num_radial - 1)
    inv_width2 = 1.0 / (width * width)
    edge_scale = tl.zeros((block_size,), dtype=tl.float32)
    for k in range(0, num_radial):
        center = width * k
        delta = dist - center
        gaussian = tl.exp(-0.5 * delta * delta * inv_width2)
        gaussian_derivative = gaussian * (-delta * inv_width2)
        radial_derivative = gaussian_derivative * envelope + gaussian * envelope_derivative
        density_part = tl.load(density_grad + dst * num_radial + k, mask=mask, other=0.0)
        element_part = tl.load(element_density_grad + dst * num_radial + k, mask=mask, other=0.0)
        edge_scale += (density_part + element_part * src_z) * radial_derivative

    gx = edge_scale * ux
    gy = edge_scale * uy
    gz = edge_scale * uz
    tl.atomic_add(forces + dst * 3 + 0, -gx, sem="relaxed", mask=mask)
    tl.atomic_add(forces + dst * 3 + 1, -gy, sem="relaxed", mask=mask)
    tl.atomic_add(forces + dst * 3 + 2, -gz, sem="relaxed", mask=mask)
    tl.atomic_add(forces + src * 3 + 0, gx, sem="relaxed", mask=mask)
    tl.atomic_add(forces + src * 3 + 1, gy, sem="relaxed", mask=mask)
    tl.atomic_add(forces + src * 3 + 2, gz, sem="relaxed", mask=mask)


def element_density_forces_triton(
    *,
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_z: torch.Tensor,
    density_grad: torch.Tensor,
    element_density_grad: torch.Tensor,
    cutoff: float,
    num_radial: int,
    block_size: int = 128,
) -> torch.Tensor:
    if pos.device.type != "cuda":
        raise RuntimeError("Triton element-density force path requires a CUDA graph")
    if pos.dtype != torch.float32 or density_grad.dtype != torch.float32 or element_density_grad.dtype != torch.float32:
        raise RuntimeError("Triton element-density force path currently supports float32 tensors only")
    if node_z.dtype != torch.float32:
        raise RuntimeError("Triton element-density force path currently supports float32 atomic numbers only")
    if not pos.is_contiguous():
        pos = pos.contiguous()
    src = edge_index[0].contiguous()
    dst = edge_index[1].contiguous()
    node_z = node_z.contiguous()
    density_grad = density_grad.contiguous()
    element_density_grad = element_density_grad.contiguous()
    forces = torch.zeros_like(pos)
    num_edges = int(src.numel())
    grid = (triton.cdiv(num_edges, block_size),)
    _element_density_force_kernel[grid](
        pos,
        src,
        dst,
        node_z,
        density_grad,
        element_density_grad,
        forces,
        num_edges,
        float(cutoff),
        int(num_radial),
        block_size,
        num_warps=4,
    )
    return forces

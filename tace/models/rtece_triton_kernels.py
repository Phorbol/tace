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
def _direct_radius_count_edges_kernel(
    pos,
    counts,
    starts,
    edge_count,
    num_graphs: tl.constexpr,
    max_count: tl.constexpr,
    total_slots: tl.constexpr,
    cutoff_sq: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total_slots
    slots_per_graph = max_count * max_count
    graph = offsets // slots_per_graph
    local_pair = offsets - graph * slots_per_graph
    src_local = local_pair // max_count
    dst_local = local_pair - src_local * max_count
    count = tl.load(counts + graph, mask=mask & (graph < num_graphs), other=0)
    start = tl.load(starts + graph, mask=mask & (graph < num_graphs), other=0)
    valid = mask & (graph < num_graphs) & (src_local < count) & (dst_local < count) & (src_local != dst_local)
    src = start + src_local
    dst = start + dst_local

    sx = tl.load(pos + src * 3 + 0, mask=valid, other=0.0)
    sy = tl.load(pos + src * 3 + 1, mask=valid, other=0.0)
    sz = tl.load(pos + src * 3 + 2, mask=valid, other=0.0)
    dx = tl.load(pos + dst * 3 + 0, mask=valid, other=0.0)
    dy = tl.load(pos + dst * 3 + 1, mask=valid, other=0.0)
    dz = tl.load(pos + dst * 3 + 2, mask=valid, other=0.0)
    vx = sx - dx
    vy = sy - dy
    vz = sz - dz
    inside = valid & ((vx * vx + vy * vy + vz * vz) < cutoff_sq)
    hits = tl.sum(tl.where(inside, 1, 0), axis=0)
    tl.atomic_add(edge_count, hits, sem="relaxed")


@triton.jit
def _direct_radius_padded_edges_kernel(
    pos,
    counts,
    starts,
    edge_index,
    edge_count,
    num_graphs: tl.constexpr,
    max_count: tl.constexpr,
    total_slots: tl.constexpr,
    edge_stride: tl.constexpr,
    cutoff_sq: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total_slots
    slots_per_graph = max_count * max_count
    graph = offsets // slots_per_graph
    local_pair = offsets - graph * slots_per_graph
    src_local = local_pair // max_count
    dst_local = local_pair - src_local * max_count
    count = tl.load(counts + graph, mask=mask & (graph < num_graphs), other=0)
    start = tl.load(starts + graph, mask=mask & (graph < num_graphs), other=0)
    valid = mask & (graph < num_graphs) & (src_local < count) & (dst_local < count) & (src_local != dst_local)
    src = start + src_local
    dst = start + dst_local

    sx = tl.load(pos + src * 3 + 0, mask=valid, other=0.0)
    sy = tl.load(pos + src * 3 + 1, mask=valid, other=0.0)
    sz = tl.load(pos + src * 3 + 2, mask=valid, other=0.0)
    dx = tl.load(pos + dst * 3 + 0, mask=valid, other=0.0)
    dy = tl.load(pos + dst * 3 + 1, mask=valid, other=0.0)
    dz = tl.load(pos + dst * 3 + 2, mask=valid, other=0.0)
    vx = sx - dx
    vy = sy - dy
    vz = sz - dz
    inside = valid & ((vx * vx + vy * vy + vz * vz) < cutoff_sq)
    counter_ptrs = edge_count + tl.zeros((block_size,), dtype=tl.int64)
    out = tl.atomic_add(counter_ptrs, 1, sem="relaxed", mask=inside)
    tl.store(edge_index + out, src, mask=inside)
    tl.store(edge_index + edge_stride + out, dst, mask=inside)


def direct_radius_padded_edges_triton(
    *,
    pos: torch.Tensor,
    counts: torch.Tensor,
    starts: torch.Tensor,
    cutoff: float,
    block_size: int = 256,
) -> torch.Tensor:
    if pos.device.type != "cuda":
        raise RuntimeError("Triton direct-radius edge provider requires CUDA positions")
    if pos.dtype != torch.float32:
        raise RuntimeError("Triton direct-radius edge provider currently supports float32 positions only")
    if counts.device.type != "cuda" or starts.device.type != "cuda":
        raise RuntimeError("Triton direct-radius edge provider requires CUDA counts and starts")
    if counts.dtype != torch.long or starts.dtype != torch.long:
        raise RuntimeError("Triton direct-radius edge provider requires int64 counts and starts")
    if not pos.is_contiguous():
        pos = pos.contiguous()
    counts = counts.contiguous()
    starts = starts.contiguous()
    num_graphs = int(counts.numel())
    if num_graphs == 0:
        return torch.empty((2, 0), device=pos.device, dtype=torch.long)
    max_count = int(counts.max().detach().cpu())
    if max_count <= 1:
        return torch.empty((2, 0), device=pos.device, dtype=torch.long)
    total_slots = int(num_graphs * max_count * max_count)
    edge_index = torch.empty((2, total_slots), device=pos.device, dtype=torch.long)
    edge_count = torch.zeros((), device=pos.device, dtype=torch.long)
    grid = (triton.cdiv(total_slots, block_size),)
    _direct_radius_padded_edges_kernel[grid](
        pos,
        counts,
        starts,
        edge_index,
        edge_count,
        int(num_graphs),
        int(max_count),
        int(total_slots),
        int(total_slots),
        float(cutoff) * float(cutoff),
        int(block_size),
        num_warps=4,
    )
    count = int(edge_count.detach().cpu())
    return edge_index[:, :count]


def direct_radius_counted_edges_triton(
    *,
    pos: torch.Tensor,
    counts: torch.Tensor,
    starts: torch.Tensor,
    cutoff: float,
    block_size: int = 256,
) -> torch.Tensor:
    if pos.device.type != "cuda":
        raise RuntimeError("Triton counted direct-radius edge provider requires CUDA positions")
    if pos.dtype != torch.float32:
        raise RuntimeError("Triton counted direct-radius edge provider currently supports float32 positions only")
    if counts.device.type != "cuda" or starts.device.type != "cuda":
        raise RuntimeError("Triton counted direct-radius edge provider requires CUDA counts and starts")
    if counts.dtype != torch.long or starts.dtype != torch.long:
        raise RuntimeError("Triton counted direct-radius edge provider requires int64 counts and starts")
    if not pos.is_contiguous():
        pos = pos.contiguous()
    counts = counts.contiguous()
    starts = starts.contiguous()
    num_graphs = int(counts.numel())
    if num_graphs == 0:
        return torch.empty((2, 0), device=pos.device, dtype=torch.long)
    max_count = int(counts.max().detach().cpu())
    if max_count <= 1:
        return torch.empty((2, 0), device=pos.device, dtype=torch.long)
    total_slots = int(num_graphs * max_count * max_count)
    edge_count = torch.zeros((), device=pos.device, dtype=torch.long)
    grid = (triton.cdiv(total_slots, block_size),)
    _direct_radius_count_edges_kernel[grid](
        pos,
        counts,
        starts,
        edge_count,
        int(num_graphs),
        int(max_count),
        int(total_slots),
        float(cutoff) * float(cutoff),
        int(block_size),
        num_warps=4,
    )
    count = int(edge_count.detach().cpu())
    if count == 0:
        return torch.empty((2, 0), device=pos.device, dtype=torch.long)

    edge_index = torch.empty((2, count), device=pos.device, dtype=torch.long)
    edge_count.zero_()
    _direct_radius_padded_edges_kernel[grid](
        pos,
        counts,
        starts,
        edge_index,
        edge_count,
        int(num_graphs),
        int(max_count),
        int(total_slots),
        int(count),
        float(cutoff) * float(cutoff),
        int(block_size),
        num_warps=4,
    )
    return edge_index


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
def _element_density_descriptor_kernel(
    pos,
    src_index,
    dst_index,
    node_z,
    descriptors,
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

    x = tl.minimum(tl.maximum(dist / cutoff, 0.0), 1.0)
    x2 = x * x
    x3 = x2 * x
    x4 = x3 * x
    x5 = x4 * x
    inside = dist < cutoff
    envelope = tl.where(inside, 1.0 - 10.0 * x3 + 15.0 * x4 - 6.0 * x5, 0.0)

    width = cutoff / (num_radial - 1)
    inv_width2 = 1.0 / (width * width)
    descriptor_stride = num_radial * 2
    for k in range(0, num_radial):
        center = width * k
        delta = dist - center
        gaussian = tl.exp(-0.5 * delta * delta * inv_width2)
        radial = gaussian * envelope
        base = descriptors + dst * descriptor_stride + k
        tl.atomic_add(base, radial, sem="relaxed", mask=mask)
        tl.atomic_add(base + num_radial, radial * src_z, sem="relaxed", mask=mask)


def element_density_descriptors_triton(
    *,
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_z: torch.Tensor,
    cutoff: float,
    num_radial: int,
    block_size: int = 128,
) -> torch.Tensor:
    if pos.device.type != "cuda":
        raise RuntimeError("Triton element-density descriptor path requires a CUDA graph")
    if pos.dtype != torch.float32:
        raise RuntimeError("Triton element-density descriptor path currently supports float32 positions only")
    if node_z.dtype != torch.float32:
        raise RuntimeError("Triton element-density descriptor path currently supports float32 atomic numbers only")
    if not pos.is_contiguous():
        pos = pos.contiguous()
    src = edge_index[0].contiguous()
    dst = edge_index[1].contiguous()
    node_z = node_z.contiguous()
    descriptors = torch.zeros((int(node_z.numel()), int(num_radial) * 2), device=pos.device, dtype=pos.dtype)
    num_edges = int(src.numel())
    grid = (triton.cdiv(num_edges, block_size),)
    _element_density_descriptor_kernel[grid](
        pos,
        src,
        dst,
        node_z,
        descriptors,
        num_edges,
        float(cutoff),
        int(num_radial),
        block_size,
        num_warps=4,
    )
    return descriptors


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


@triton.jit
def _element_density_direct_padded_descriptor_kernel(
    pos,
    counts,
    starts,
    node_z,
    descriptors,
    num_graphs: tl.constexpr,
    max_count: tl.constexpr,
    total_slots: tl.constexpr,
    cutoff: tl.constexpr,
    num_radial: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total_slots
    slots_per_graph = max_count * max_count
    graph = offsets // slots_per_graph
    local_pair = offsets - graph * slots_per_graph
    src_local = local_pair // max_count
    dst_local = local_pair - src_local * max_count
    count = tl.load(counts + graph, mask=mask & (graph < num_graphs), other=0)
    start = tl.load(starts + graph, mask=mask & (graph < num_graphs), other=0)
    valid = mask & (graph < num_graphs) & (src_local < count) & (dst_local < count) & (src_local != dst_local)
    src = start + src_local
    dst = start + dst_local
    src_z = tl.load(node_z + src, mask=valid, other=0.0)

    sx = tl.load(pos + src * 3 + 0, mask=valid, other=0.0)
    sy = tl.load(pos + src * 3 + 1, mask=valid, other=0.0)
    sz = tl.load(pos + src * 3 + 2, mask=valid, other=0.0)
    dx = tl.load(pos + dst * 3 + 0, mask=valid, other=0.0)
    dy = tl.load(pos + dst * 3 + 1, mask=valid, other=0.0)
    dz = tl.load(pos + dst * 3 + 2, mask=valid, other=0.0)

    vx = dx - sx
    vy = dy - sy
    vz = dz - sz
    dist = tl.sqrt(vx * vx + vy * vy + vz * vz)
    dist = tl.maximum(dist, 1.0e-12)
    inside = valid & (dist < cutoff)

    x = tl.minimum(tl.maximum(dist / cutoff, 0.0), 1.0)
    x2 = x * x
    x3 = x2 * x
    x4 = x3 * x
    x5 = x4 * x
    envelope = tl.where(inside, 1.0 - 10.0 * x3 + 15.0 * x4 - 6.0 * x5, 0.0)

    width = cutoff / (num_radial - 1)
    inv_width2 = 1.0 / (width * width)
    descriptor_stride = num_radial * 2
    for k in range(0, num_radial):
        center = width * k
        delta = dist - center
        gaussian = tl.exp(-0.5 * delta * delta * inv_width2)
        radial = gaussian * envelope
        base = descriptors + dst * descriptor_stride + k
        tl.atomic_add(base, radial, sem="relaxed", mask=inside)
        tl.atomic_add(base + num_radial, radial * src_z, sem="relaxed", mask=inside)


@triton.jit
def _element_density_direct_padded_force_kernel(
    pos,
    counts,
    starts,
    node_z,
    density_grad,
    element_density_grad,
    forces,
    num_graphs: tl.constexpr,
    max_count: tl.constexpr,
    total_slots: tl.constexpr,
    cutoff: tl.constexpr,
    num_radial: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total_slots
    slots_per_graph = max_count * max_count
    graph = offsets // slots_per_graph
    local_pair = offsets - graph * slots_per_graph
    src_local = local_pair // max_count
    dst_local = local_pair - src_local * max_count
    count = tl.load(counts + graph, mask=mask & (graph < num_graphs), other=0)
    start = tl.load(starts + graph, mask=mask & (graph < num_graphs), other=0)
    valid = mask & (graph < num_graphs) & (src_local < count) & (dst_local < count) & (src_local != dst_local)
    src = start + src_local
    dst = start + dst_local
    src_z = tl.load(node_z + src, mask=valid, other=0.0)

    sx = tl.load(pos + src * 3 + 0, mask=valid, other=0.0)
    sy = tl.load(pos + src * 3 + 1, mask=valid, other=0.0)
    sz = tl.load(pos + src * 3 + 2, mask=valid, other=0.0)
    dx = tl.load(pos + dst * 3 + 0, mask=valid, other=0.0)
    dy = tl.load(pos + dst * 3 + 1, mask=valid, other=0.0)
    dz = tl.load(pos + dst * 3 + 2, mask=valid, other=0.0)

    vx = dx - sx
    vy = dy - sy
    vz = dz - sz
    dist = tl.sqrt(vx * vx + vy * vy + vz * vz)
    dist = tl.maximum(dist, 1.0e-12)
    inside = valid & (dist < cutoff)
    inv_dist = 1.0 / dist
    ux = vx * inv_dist
    uy = vy * inv_dist
    uz = vz * inv_dist

    x = tl.minimum(tl.maximum(dist / cutoff, 0.0), 1.0)
    x2 = x * x
    x3 = x2 * x
    x4 = x3 * x
    x5 = x4 * x
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
        density_part = tl.load(density_grad + dst * num_radial + k, mask=inside, other=0.0)
        element_part = tl.load(element_density_grad + dst * num_radial + k, mask=inside, other=0.0)
        edge_scale += (density_part + element_part * src_z) * radial_derivative

    gx = edge_scale * ux
    gy = edge_scale * uy
    gz = edge_scale * uz
    tl.atomic_add(forces + dst * 3 + 0, -gx, sem="relaxed", mask=inside)
    tl.atomic_add(forces + dst * 3 + 1, -gy, sem="relaxed", mask=inside)
    tl.atomic_add(forces + dst * 3 + 2, -gz, sem="relaxed", mask=inside)
    tl.atomic_add(forces + src * 3 + 0, gx, sem="relaxed", mask=inside)
    tl.atomic_add(forces + src * 3 + 1, gy, sem="relaxed", mask=inside)
    tl.atomic_add(forces + src * 3 + 2, gz, sem="relaxed", mask=inside)


def _validate_direct_padded_inputs(
    *,
    pos: torch.Tensor,
    counts: torch.Tensor,
    starts: torch.Tensor,
    node_z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
    if pos.device.type != "cuda":
        raise RuntimeError("Triton direct-padded element-density path requires CUDA positions")
    if pos.dtype != torch.float32:
        raise RuntimeError("Triton direct-padded element-density path currently supports float32 positions only")
    if counts.device.type != "cuda" or starts.device.type != "cuda" or node_z.device.type != "cuda":
        raise RuntimeError("Triton direct-padded element-density path requires CUDA counts, starts, and atomic numbers")
    if counts.dtype != torch.long or starts.dtype != torch.long:
        raise RuntimeError("Triton direct-padded element-density path requires int64 counts and starts")
    if node_z.dtype != torch.float32:
        raise RuntimeError("Triton direct-padded element-density path currently supports float32 atomic numbers only")
    if not pos.is_contiguous():
        pos = pos.contiguous()
    counts = counts.contiguous()
    starts = starts.contiguous()
    node_z = node_z.contiguous()
    num_graphs = int(counts.numel())
    max_count = int(counts.max().detach().cpu()) if num_graphs else 0
    total_slots = int(num_graphs * max_count * max_count)
    return pos, counts, starts, node_z, num_graphs, max_count, total_slots


def element_density_direct_padded_descriptors_triton(
    *,
    pos: torch.Tensor,
    counts: torch.Tensor,
    starts: torch.Tensor,
    node_z: torch.Tensor,
    cutoff: float,
    num_radial: int,
    block_size: int = 128,
) -> torch.Tensor:
    pos, counts, starts, node_z, num_graphs, max_count, total_slots = _validate_direct_padded_inputs(
        pos=pos,
        counts=counts,
        starts=starts,
        node_z=node_z,
    )
    descriptors = torch.zeros((int(node_z.numel()), int(num_radial) * 2), device=pos.device, dtype=pos.dtype)
    if num_graphs == 0 or max_count <= 1:
        return descriptors
    grid = (triton.cdiv(total_slots, block_size),)
    _element_density_direct_padded_descriptor_kernel[grid](
        pos,
        counts,
        starts,
        node_z,
        descriptors,
        int(num_graphs),
        int(max_count),
        int(total_slots),
        float(cutoff),
        int(num_radial),
        int(block_size),
        num_warps=4,
    )
    return descriptors


def element_density_direct_padded_forces_triton(
    *,
    pos: torch.Tensor,
    counts: torch.Tensor,
    starts: torch.Tensor,
    node_z: torch.Tensor,
    density_grad: torch.Tensor,
    element_density_grad: torch.Tensor,
    cutoff: float,
    num_radial: int,
    block_size: int = 128,
) -> torch.Tensor:
    pos, counts, starts, node_z, num_graphs, max_count, total_slots = _validate_direct_padded_inputs(
        pos=pos,
        counts=counts,
        starts=starts,
        node_z=node_z,
    )
    if density_grad.dtype != torch.float32 or element_density_grad.dtype != torch.float32:
        raise RuntimeError("Triton direct-padded element-density force path currently supports float32 gradients only")
    density_grad = density_grad.contiguous()
    element_density_grad = element_density_grad.contiguous()
    forces = torch.zeros_like(pos)
    if num_graphs == 0 or max_count <= 1:
        return forces
    grid = (triton.cdiv(total_slots, block_size),)
    _element_density_direct_padded_force_kernel[grid](
        pos,
        counts,
        starts,
        node_z,
        density_grad,
        element_density_grad,
        forces,
        int(num_graphs),
        int(max_count),
        int(total_slots),
        float(cutoff),
        int(num_radial),
        int(block_size),
        num_warps=4,
    )
    return forces

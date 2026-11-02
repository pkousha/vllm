# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real NCCL-EP regressions; run in the pinned four-GPU CUDA 13 environment."""

import gc
import threading
import weakref
from dataclasses import replace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.distributed.device_communicators.all2all import NcclEPAll2AllManager


def _bootstrap_worker(
    rank: int,
    init_method: str,
    noncontiguous: bool,
    stateless: bool,
    adapter: bool,
    kernel_backend: str | None = None,
):
    torch.set_num_threads(1)
    torch.accelerator.set_device_index(rank)
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=4,
            rank=rank,
            local_rank=rank,
            distributed_init_method=init_method,
            backend="gloo",
        )
        initialize_model_parallel(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2,
            backend="gloo",
        )
        memberships = [[0, 2], [1, 3]] if noncontiguous else [[0, 1], [2, 3]]
        groups = [dist.new_group(ranks, backend="gloo") for ranks in memberships]
        index = next(i for i, ranks in enumerate(memberships) if rank in ranks)
        metadata_group = None
        if stateless:
            from vllm.distributed.utils import StatelessProcessGroup

            store = dist.FileStore(init_method.removeprefix("file://") + ".store", 4)
            metadata_group = StatelessProcessGroup(
                rank=dist.get_rank(groups[index]),
                world_size=2,
                store=dist.PrefixStore(str(index), store),
            )
        manager = NcclEPAll2AllManager(groups[index], metadata_group)
        assert manager.world_size == 2
        assert manager.dp_world_size == 1
        value = torch.tensor([rank + 1.0], device="cuda")
        import nccl.core as core

        result = torch.empty_like(value)
        manager.nccl_comm.allreduce(value, result, op=core.SUM)
        torch.accelerator.synchronize()
        assert result.item() == sum(r + 1 for r in memberships[index])
        if adapter:
            _check_dispatch(manager)
        if kernel_backend is not None and kernel_backend.startswith("dbo_"):
            _check_dbo_lifetime(manager, kernel_backend.removeprefix("dbo_"))
        elif kernel_backend is not None and kernel_backend.startswith("lifetime_"):
            _check_lifetime(manager, kernel_backend.removeprefix("lifetime_"))
        elif kernel_backend == "batched":
            _check_batched(manager)
        elif kernel_backend is not None:
            _check_experts(manager, kernel_backend)
        manager.destroy()
        dist.barrier()
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.parametrize("noncontiguous", [False, True])
@pytest.mark.parametrize("stateless", [False, True])
def test_bootstrap_uses_the_ep_subgroup_root(tmp_path, noncontiguous, stateless):
    """Communicators on subgroups excluding global zero exchange real data."""
    assert torch.accelerator.device_count() >= 4
    mp.spawn(
        _bootstrap_worker,
        args=(f"file://{tmp_path}/bootstrap", noncontiguous, stateless, False),
        nprocs=4,
        join=True,
    )


@pytest.mark.parametrize("activation_format", [None, "batched"])
def test_parallel_config_preserves_existing_constructors(activation_format):
    from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig

    kwargs = (
        {}
        if activation_format is None
        else {"nccl_ep_activation_format": activation_format}
    )
    config = FusedMoEParallelConfig(
        tp_size=2,
        pcp_size=1,
        dp_size=1,
        ep_size=2,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=0,
        sp_size=2,
        use_ep=True,
        all2all_backend="nccl_ep",
        enable_eplb=False,
        **kwargs,
    )
    assert config.nccl_ep_activation_format == (activation_format or "standard")
    assert config.use_batched_activation_format == (activation_format == "batched")
    legacy = FusedMoEParallelConfig(
        2, 1, 1, 2, 0, 0, 0, 0, 2, True, "mori", False, True
    )
    assert legacy.mori_ep_v2
    assert legacy.nccl_ep_activation_format == "standard"


def _check_dispatch(manager):
    from vllm.model_executor.layers.fused_moe.config import (
        FUSED_MOE_UNQUANTIZED_CONFIG,
    )
    from vllm.model_executor.layers.fused_moe.prepare_finalize.nccl_ep import (
        NcclEPStandardPrepareAndFinalize,
    )
    from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
        TopKWeightAndReduceContiguous,
    )

    group = manager.get_handle(
        dict(
            algorithm="standard",
            num_experts=4,
            max_dispatch_tokens_per_rank=16,
            hidden=1024,
            dtype_bytes=2,
        )
    )
    kwargs = dict(
        ep_group=group,
        num_dispatchers=2,
        dp_size=1,
        max_tokens_per_rank=16,
        rank_expert_offset=manager.rank * 2,
        num_experts=4,
        num_topk=2,
    )
    with pytest.raises(ValueError, match="FP8 transport"):
        NcclEPStandardPrepareAndFinalize(**kwargs, use_fp8_dispatch=True)
    prepare = _make_prepare(manager, "standard")
    assert isinstance(prepare, NcclEPStandardPrepareAndFinalize)
    cases = [
        ((1, 0), [[0, 1]]),
        ((2, 3), [[2, 3]]),
        ((2, 1), [[0, 2]]),
        ((3, 1), [[0, 2], [1, -1], [2, 3]]),
        ((1, 7), [[0, 1], [0, 2], [2, 3]]),
        ((16, 16), [[0, 1]]),
        ((0, 3), [[0, 1]]),
        ((0, 0), [[0, 1]]),
    ]
    for sizes, routes in cases:
        n = sizes[manager.rank]
        all_ids = [
            torch.tensor(
                [routes[i % len(routes)] for i in range(size)], dtype=torch.int64
            ).reshape(size, 2)
            for size in sizes
        ]
        ids = all_ids[manager.rank].to("cuda")
        x = torch.arange(n, device="cuda", dtype=torch.bfloat16)[:, None]
        x = (x + 1 + manager.rank * 8).expand(n, 1024).contiguous()
        weights = torch.tensor([0.25, 0.75], device="cuda").expand(n, 2).contiguous()
        recv_x, scales, meta, recv_ids, recv_weights = prepare.prepare(
            x,
            weights,
            ids,
            4,
            None,
            False,
            FUSED_MOE_UNQUANTIZED_CONFIG,
        )
        assert scales is None
        assert recv_ids is not None and recv_weights is not None and meta is not None
        offset = manager.rank * 2
        expected_counts = torch.zeros(2, dtype=torch.int32)
        expected_rows = 0
        for source_ids in all_ids:
            local = (source_ids >= offset) & (source_ids < offset + 2)
            expected_rows += int(local.any(dim=1).sum())
            for expert in range(2):
                expected_counts[expert] += int((source_ids == offset + expert).sum())
        assert recv_x.shape == (expected_rows, 1024)
        assert recv_ids.shape == recv_weights.shape == (expected_rows, 2)
        assert meta.expert_num_tokens is not None
        torch.testing.assert_close(meta.expert_num_tokens.cpu(), expected_counts)
        assert ((recv_ids >= 0) & (recv_ids < 4)).all()
        local = (recv_ids >= offset) & (recv_ids < offset + 2)
        assert (recv_weights[~local] == 0).all()
        fused = recv_x[:, None, :].expand(-1, 2, -1).clone()
        output = torch.empty_like(x)
        prepare.finalize(
            output,
            fused,
            recv_weights,
            recv_ids,
            False,
            TopKWeightAndReduceContiguous(),
        )
        torch.accelerator.synchronize()
        expected = x * (weights * (ids >= 0)).sum(dim=1, keepdim=True).to(x.dtype)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

    prepare.destroy()


def test_standard_dispatch_uses_ep_capacity_and_valid_topk_metadata(tmp_path):
    """EP=2/DP=1, skew, offset experts, invalid slots, and empty send/recv."""
    assert torch.accelerator.device_count() >= 4
    mp.spawn(
        _bootstrap_worker,
        args=(f"file://{tmp_path}/dispatch", True, False, True),
        nprocs=4,
        join=True,
    )


def _check_experts(manager, backend):
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FUSED_MOE_UNQUANTIZED_CONFIG,
        fp8_w8a8_moe_quant_config,
    )
    from vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe import (
        DeepGemmExperts,
    )
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
    from vllm.model_executor.layers.fused_moe.prepare_finalize.nccl_ep import (
        NcclEPStandardPrepareAndFinalize,
    )
    from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
        make_moe_prepare_and_finalize_no_dp_ep,
    )
    from vllm.utils.deep_gemm import get_mk_alignment_for_contiguous_layout
    from vllm.v1.worker.workspace import init_workspace_manager

    from .utils import make_dummy_moe_config, make_test_weights

    torch.manual_seed(42)
    init_workspace_manager(
        torch.device("cuda", torch.accelerator.current_device_index())
    )
    block_shape = list(get_mk_alignment_for_contiguous_layout())
    quant_dtype = torch.float8_e4m3fn if backend == "deep_gemm" else None
    if quant_dtype is not None:
        from .test_deepgemm import make_block_quant_fp8_weights

        w1, w2, s1, s2 = make_block_quant_fp8_weights(4, 512, 1024, block_shape)
    else:
        (_, w1, s1, _), (_, w2, s2, _) = make_test_weights(
            4,
            512,
            1024,
            torch.bfloat16,
        )
    offset = manager.rank * 2
    if quant_dtype is not None:
        assert s1 is not None and s2 is not None
        full_quant = fp8_w8a8_moe_quant_config(s1, s2, block_shape=block_shape)
        quant = fp8_w8a8_moe_quant_config(
            s1[offset : offset + 2],
            s2[offset : offset + 2],
            block_shape=block_shape,
        )
    else:
        full_quant = quant = FUSED_MOE_UNQUANTIZED_CONFIG
    group = manager.get_handle(
        dict(
            algorithm="standard",
            num_experts=4,
            max_dispatch_tokens_per_rank=16,
            hidden=1024,
            dtype_bytes=2,
        )
    )
    prepare = NcclEPStandardPrepareAndFinalize(
        group,
        num_dispatchers=2,
        dp_size=1,
        max_tokens_per_rank=16,
        rank_expert_offset=offset,
        num_experts=4,
        num_topk=2,
    )
    config = make_dummy_moe_config(
        num_experts=4,
        num_local_experts=2,
        experts_per_token=2,
        hidden_dim=1024,
        intermediate_size=512,
    )
    expert_type = DeepGemmExperts if backend == "deep_gemm" else TritonExperts
    kernel = FusedMoEKernel(prepare, expert_type(config, quant))
    reference = FusedMoEKernel(
        make_moe_prepare_and_finalize_no_dp_ep(use_monolithic=False),
        expert_type(replace(config, num_local_experts=4), full_quant),
    )
    expert_map = torch.full((4,), -1, dtype=torch.int32, device="cuda")
    expert_map[offset : offset + 2] = torch.arange(2, device="cuda", dtype=torch.int32)
    torch.manual_seed(43 + manager.rank)
    x = torch.randn(3, 1024, device="cuda", dtype=torch.bfloat16) / 10
    weights = torch.tensor([[0.25, 0.75]] * 3, device="cuda")
    for ids in ([[0, 1], [0, 2], [2, 3]], [[2, 3], [1, 3], [0, 1]]):
        ids = torch.tensor(ids, device="cuda", dtype=torch.int64)
        with set_forward_context(None, VllmConfig()):
            expected = reference.apply(
                x, w1, w2, weights, ids, MoEActivation.SILU, 4, None, False
            )
            actual = kernel.apply(
                hidden_states=x,
                w1=w1[offset : offset + 2],
                w2=w2[offset : offset + 2],
                topk_weights=weights,
                topk_ids=ids,
                activation=MoEActivation.SILU,
                global_num_experts=4,
                expert_map=expert_map,
                apply_router_weight_on_input=False,
            )
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-2)
        torch.accelerator.synchronize()

    prepare.destroy()


@pytest.mark.parametrize("backend", ["triton", "deep_gemm"])
def test_standard_expert_kernels_consume_dispatch_metadata(tmp_path, backend):
    """Real expert computation must agree with a replicated reference."""
    assert torch.accelerator.device_count() >= 4
    mp.spawn(
        _bootstrap_worker,
        args=(f"file://{tmp_path}/experts", True, False, False, backend),
        nprocs=4,
        join=True,
    )


def _check_batched(manager):
    from vllm.model_executor.layers.fused_moe.config import (
        FUSED_MOE_UNQUANTIZED_CONFIG,
    )
    from vllm.model_executor.layers.fused_moe.prepare_finalize.nccl_ep_batched import (
        NcclEPBatchedPrepareAndFinalize,
    )
    from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
        TopKWeightAndReduceDelegate,
    )

    group = manager.get_handle(
        dict(
            algorithm="batched",
            num_experts=4,
            max_dispatch_tokens_per_rank=16,
            hidden=1024,
            dtype_bytes=2,
        )
    )
    kwargs = dict(
        ep_group=group,
        max_tokens_per_rank=16,
        num_dispatchers=2,
        num_experts=4,
        num_local_experts=2,
    )
    permutation = torch.tensor([2, 0, 3, 1], device="cuda", dtype=torch.int64)
    remapping = NcclEPBatchedPrepareAndFinalize(
        **kwargs, global_to_physical=permutation
    )
    mapped = remapping._map_global_to_physical_ids(
        torch.tensor(
            [[-1, 0], [1, 4]],
            device="cuda",
            dtype=torch.int64,
        )
    )
    assert mapped.cpu().tolist() == [[-1, 2], [0, -1]]
    prepare = _make_prepare(manager, "batched")
    assert isinstance(prepare, NcclEPBatchedPrepareAndFinalize)
    x = torch.full(
        (manager.rank + 1, 1024), manager.rank + 1, device="cuda", dtype=torch.bfloat16
    )
    ids = torch.tensor([[2, 3]] * x.shape[0], device="cuda", dtype=torch.int64)
    weights = torch.tensor([[0.25, 0.75]] * x.shape[0], device="cuda")
    hook, receiver = prepare.prepare_async(
        x,
        weights,
        ids,
        4,
        None,
        False,
        FUSED_MOE_UNQUANTIZED_CONFIG,
    )
    hook()
    received, scales, meta, _, _ = receiver()
    assert scales is None and meta is not None and meta.expert_num_tokens is not None
    expected_count = 3 if manager.rank == 1 else 0
    assert meta.expert_num_tokens.cpu().tolist() == [expected_count, expected_count]
    assert received.shape == (2, 32, 1024)
    assert (received[:, expected_count:] == 0).all()
    output = torch.empty_like(x)
    hook, receiver = prepare.finalize_async(
        output,
        received,
        weights,
        ids,
        False,
        TopKWeightAndReduceDelegate(),
    )
    hook()
    receiver()
    torch.accelerator.synchronize()
    torch.testing.assert_close(output, x, rtol=0, atol=0)
    prepare.destroy()


def test_batched_counts_padding_and_invalid_remapping(tmp_path):
    assert torch.accelerator.device_count() >= 4
    mp.spawn(
        _bootstrap_worker,
        args=(f"file://{tmp_path}/batched", True, False, False, "batched"),
        nprocs=4,
        join=True,
    )


def _make_prepare(manager, activation_format, routing_tables=None):
    from vllm.model_executor.layers.fused_moe.all2all_utils import (
        maybe_make_prepare_finalize,
    )
    from vllm.model_executor.layers.fused_moe.config import (
        FUSED_MOE_UNQUANTIZED_CONFIG,
        FusedMoEParallelConfig,
    )

    from .utils import make_dummy_moe_config

    parallel = FusedMoEParallelConfig(
        tp_size=2,
        pcp_size=1,
        dp_size=1,
        ep_size=2,
        tp_rank=manager.rank,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=manager.rank,
        sp_size=2,
        use_ep=True,
        all2all_backend="nccl_ep",
        enable_eplb=False,
        nccl_ep_activation_format=activation_format,
    )
    config = replace(
        make_dummy_moe_config(
            num_experts=4,
            num_local_experts=2,
            experts_per_token=2,
            hidden_dim=1024,
            intermediate_size=512,
            max_num_tokens=16,
        ),
        moe_parallel_config=parallel,
    )
    return maybe_make_prepare_finalize(
        config,
        FUSED_MOE_UNQUANTIZED_CONFIG,
        all2all_manager=manager,
        routing_tables=routing_tables,
    )


def _check_lifetime(manager, activation_format):
    from vllm.model_executor.layers.fused_moe.config import (
        FUSED_MOE_UNQUANTIZED_CONFIG,
    )
    from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
        TopKWeightAndReduceContiguous,
        TopKWeightAndReduceDelegate,
    )

    routing_tables = None
    if activation_format == "batched":
        permutation = torch.tensor([2, 0, 3, 1], device="cuda", dtype=torch.int64)
        inverse = permutation.argsort()
        routing_tables = (
            permutation,
            inverse,
            inverse[manager.rank * 2 : manager.rank * 2 + 2],
        )
    prepare = _make_prepare(manager, activation_format, routing_tables)
    pointer = None
    free_after_warmup = None
    for iteration in range(20):
        n = 1 + iteration % 3
        x = torch.full((n, 1024), iteration + 1, device="cuda", dtype=torch.bfloat16)
        ids = torch.tensor([[0, 2]] * n, device="cuda", dtype=torch.int64)
        weights = torch.full((n, 2), 0.5, device="cuda")
        output = torch.empty_like(x)
        routing_ref = weakref.ref(ids)
        result = prepare.prepare_async(
            x, weights, ids, 4, None, False, FUSED_MOE_UNQUANTIZED_CONFIG
        )
        del ids
        gc.collect()
        assert routing_ref() is not None, "native routing storage was released"
        if iteration == 0:
            with pytest.raises(RuntimeError, match="unfinished invocation"):
                prepare.prepare_async(
                    x,
                    weights,
                    routing_ref(),
                    4,
                    None,
                    False,
                    FUSED_MOE_UNQUANTIZED_CONFIG,
                )
        pressure = [
            torch.full((n, 2), -1, device="cuda", dtype=torch.int64) for _ in range(32)
        ]
        del pressure
        if isinstance(result, tuple):
            hook, receiver = result
            hook()
        else:
            receiver = result
        received, _, _, recv_ids, recv_weights = receiver()
        current_pointer = prepare.handles[0].ptr
        if pointer is None:
            pointer = current_pointer
        assert current_pointer == pointer, (
            "native handle was replaced instead of reused"
        )
        if activation_format == "standard":
            fused = received[:, None, :].expand(-1, 2, -1).clone()
            reducer = TopKWeightAndReduceContiguous()
        else:
            fused = received
            recv_ids, recv_weights = routing_ref(), weights
            reducer = TopKWeightAndReduceDelegate()
        if iteration % 2 == 0:
            prepare.finalize(output, fused, recv_weights, recv_ids, False, reducer)
        else:
            receiver = prepare.finalize_async(
                output, fused, recv_weights, recv_ids, False, reducer
            )
            if isinstance(receiver, tuple):
                hook, receiver = receiver
                hook()
            receiver()
        torch.accelerator.synchronize()
        torch.testing.assert_close(output, x, rtol=0, atol=0)
        del result, receiver, received, fused, output, recv_ids, recv_weights
        gc.collect()
        torch.accelerator.empty_cache()
        if iteration == 4:
            free_after_warmup = torch.accelerator.get_memory_info()[0]
    assert torch.accelerator.get_memory_info()[0] >= free_after_warmup, (
        "native allocations grew"
    )
    # Teardown must also finish a staged dispatch whose hook was never consumed.
    ids = torch.tensor([[0, 2]] * x.shape[0], device="cuda", dtype=torch.int64)
    prepare.prepare_async(x, weights, ids, 4, None, False, FUSED_MOE_UNQUANTIZED_CONFIG)
    handle = prepare.handles[0]
    group = prepare.ep_group
    manager.destroy()
    with pytest.raises(Exception, match="destroyed"):
        _ = handle.ptr
    with pytest.raises(Exception, match="destroyed"):
        _ = group.ptr
    with pytest.raises(RuntimeError, match="destroyed"):
        prepare.prepare(x, weights, ids, 4, None, False, FUSED_MOE_UNQUANTIZED_CONFIG)
    manager.destroy()


@pytest.mark.parametrize("activation_format", ["standard", "batched"])
def test_native_handle_reuse_retains_routing_and_stabilizes_memory(
    tmp_path, activation_format
):
    """Native allocations stay bounded and teardown invalidates handles first."""
    assert torch.accelerator.device_count() >= 4
    mp.spawn(
        _bootstrap_worker,
        args=(
            f"file://{tmp_path}/lifetime",
            True,
            False,
            False,
            "lifetime_" + activation_format,
        ),
        nprocs=4,
        join=True,
    )


def _check_dbo_lifetime(manager, activation_format):
    from vllm.forward_context import create_forward_context
    from vllm.model_executor.layers.fused_moe.config import (
        FUSED_MOE_UNQUANTIZED_CONFIG,
    )
    from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
        TopKWeightAndReduceContiguous,
        TopKWeightAndReduceDelegate,
    )
    from vllm.v1.worker.ubatching import (
        dbo_maybe_run_recv_hook,
        dbo_register_recv_hook,
        dbo_yield,
        make_ubatch_contexts,
    )

    prepare = _make_prepare(manager, activation_format)
    device = torch.accelerator.current_device_index()
    barrier = threading.Barrier(3)
    contexts = make_ubatch_contexts(
        2,
        torch.cuda.current_stream(),
        torch.cuda.Stream(),
        [create_forward_context(None, VllmConfig()) for _ in range(2)],
        barrier,
    )
    pointers: list[set[int]] = [set(), set()]
    errors: list[BaseException] = []

    def run(context):
        try:
            torch.accelerator.set_device_index(device)
            with context:
                for iteration in range(8):
                    dbo_maybe_run_recv_hook()
                    x = torch.full(
                        (context.id + 1, 1024),
                        1 + manager.rank + context.id * 4 + iteration * 8,
                        device="cuda",
                        dtype=torch.bfloat16,
                    )
                    ids = torch.tensor(
                        [[0, 2]] * x.shape[0], device="cuda", dtype=torch.int64
                    )
                    weights = torch.full((x.shape[0], 2), 0.5, device="cuda")
                    result = prepare.prepare_async(
                        x, weights, ids, 4, None, False, FUSED_MOE_UNQUANTIZED_CONFIG
                    )
                    if isinstance(result, tuple):
                        hook, receiver = result
                        dbo_register_recv_hook(hook)
                        dbo_yield()
                    else:
                        receiver = result
                    received, _, _, recv_ids, recv_weights = receiver()
                    pointers[context.id].add(prepare.handles[context.id].ptr)
                    if activation_format == "standard":
                        fused = received[:, None, :].expand(-1, 2, -1).clone()
                        reducer = TopKWeightAndReduceContiguous()
                    else:
                        fused = received
                        recv_ids, recv_weights = ids, weights
                        reducer = TopKWeightAndReduceDelegate()
                    output = torch.empty_like(x)
                    result = prepare.finalize_async(
                        output, fused, recv_weights, recv_ids, False, reducer
                    )
                    if isinstance(result, tuple):
                        hook, receiver = result
                        dbo_register_recv_hook(hook)
                        dbo_yield()
                    else:
                        receiver = result
                    receiver()
                    torch.testing.assert_close(output, x, rtol=0, atol=0)
                    dbo_maybe_run_recv_hook()
                    dbo_yield()
        except BaseException as error:
            errors.append(error)
            for context in contexts:
                context.cpu_wait_event.set()

    threads = [threading.Thread(target=run, args=(c,), daemon=True) for c in contexts]
    for thread in threads:
        thread.start()
    barrier.wait(timeout=30)
    contexts[0].cpu_wait_event.set()
    for thread in threads:
        thread.join(timeout=120)
        assert not thread.is_alive(), "microbatch scheduling did not finish"
    if errors:
        raise errors[0]
    assert all(len(slot) == 1 for slot in pointers)
    assert pointers[0].isdisjoint(pointers[1])
    handles = list(prepare.handles)
    manager.destroy()
    for handle in handles:
        with pytest.raises(Exception, match="destroyed"):
            _ = handle.ptr


@pytest.mark.parametrize("activation_format", ["standard", "batched"])
def test_native_handle_lifetime_isolated_between_microbatch_slots(
    tmp_path, activation_format
):
    assert torch.accelerator.device_count() >= 4
    mp.spawn(
        _bootstrap_worker,
        args=(
            f"file://{tmp_path}/dbo",
            True,
            False,
            False,
            "dbo_" + activation_format,
        ),
        nprocs=4,
        join=True,
    )

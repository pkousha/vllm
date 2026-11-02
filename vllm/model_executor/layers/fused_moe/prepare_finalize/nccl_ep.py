# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import nccl.core as nccl_core  # type: ignore[import-not-found]
import nccl.ep as nccl_ep  # type: ignore[import-not-found]
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed.device_communicators.all2all import nccl_ep_max_recv_tokens
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceContiguous,
    TopKWeightAndReduceDelegate,
)
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from vllm.v1.worker.ubatching import (
    dbo_current_ubatch_id,
    dbo_enabled,
    dbo_switch_to_comm,
    dbo_switch_to_compute,
    dbo_switch_to_compute_sync,
    dbo_yield_and_switch_from_comm_to_compute,
    dbo_yield_and_switch_from_compute_to_comm,
)

logger = init_logger(__name__)

_TORCH_TO_NCCL_DTYPE = {
    torch.bfloat16: nccl_core.BFLOAT16,
    torch.float16: nccl_core.FLOAT16,
    torch.float32: nccl_core.FLOAT32,
    torch.int64: nccl_core.INT64,
    torch.int32: nccl_core.INT32,
    torch.float8_e4m3fn: nccl_core.FLOAT8E4M3,
}


def _to_nccl_tensor(t: torch.Tensor) -> nccl_ep.Tensor:
    dtype = _TORCH_TO_NCCL_DTYPE.get(t.dtype)
    assert dtype is not None, f"Unsupported dtype {t.dtype} for NCCL EP"
    return nccl_ep.Tensor(t.data_ptr(), dtype=int(dtype), shape=tuple(t.shape))


def _get_cuda_stream():
    return torch.cuda.current_stream().cuda_stream


class NcclEPPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """Own eager-only native handles and their pointer-backed tensors."""

    def __init__(self, ep_group: nccl_ep.Group):
        super().__init__()
        self.ep_group = ep_group
        self.handles: list[nccl_ep.Handle | None] = [None, None]
        self._tensors: list[list[torch.Tensor]] = [[], []]
        self._descriptors: list[list[object]] = [[], []]
        self._done: list[torch.cuda.Event | None] = [None, None]
        self._pending: list[torch.cuda.Stream | None] = [None, None]
        self._active = [False, False]
        self._num_topk: int | None = None
        self._destroyed = False

    def _check_not_destroyed(self):
        if self._destroyed:
            raise RuntimeError("NCCL EP prepare/finalize has been destroyed.")

    def _get_handle(self, index, layout, routing, tensors, layout_info=None):
        self._check_not_destroyed()
        if self._active[index]:
            raise RuntimeError("NCCL EP microbatch still has an unfinished invocation.")
        if self._num_topk is not None and routing.shape[1] != self._num_topk:
            raise ValueError("NCCL EP handle reuse requires a constant top-k.")
        done = self._done[index]
        if done is not None:
            # shortcut: host wait before reuse; defer stream-aware retention to tuning.
            done.synchronize()
        self._tensors[index] = [routing, *tensors]
        self._active[index] = True
        stream = _get_cuda_stream()
        handle = self.handles[index]
        if handle is None:
            handle = self.ep_group.create_handle(
                layout,
                _to_nccl_tensor(routing),
                layout_info=layout_info,
                config=nccl_ep.HandleConfig(),
                stream=stream,
            )
            self.handles[index] = handle
        else:
            handle.update(
                _to_nccl_tensor(routing), layout_info=layout_info, stream=stream
            )
        self._num_topk = routing.shape[1]
        return handle

    def _complete(self, index):
        self._check_not_destroyed()
        pending = self._pending[index]
        if pending is not None:
            handle = self.handles[index]
            assert handle is not None
            # The pinned LL continuation launches on the original send stream.
            with torch.cuda.stream(pending):
                handle.complete(stream=pending.cuda_stream)
                done = torch.cuda.Event()
                done.record(pending)
            torch.cuda.current_stream(pending.device).wait_event(done)
            self._done[index] = done
            self._pending[index] = None

    def _finish(self, index):
        self._check_not_destroyed()
        assert self._pending[index] is None
        stream = torch.cuda.current_stream()
        previous = self._done[index]
        if previous is not None:
            stream.wait_event(previous)
        for tensor in self._tensors[index]:
            tensor.record_stream(stream)
        # Only routing remains cached by the idle native handle.
        self._tensors[index] = self._tensors[index][:1]
        self._descriptors[index].clear()
        done = torch.cuda.Event()
        done.record(stream)
        self._done[index] = done
        self._active[index] = False

    def destroy(self):
        if self._destroyed:
            return
        for index in range(len(self.handles)):
            self._complete(index)
        devices = {tensor.device for tensors in self._tensors for tensor in tensors}
        for device in devices:
            torch.accelerator.synchronize(device)
        for index, handle in enumerate(self.handles):
            if handle is not None:
                handle.destroy()
                self.handles[index] = None
            self._tensors[index].clear()
            self._descriptors[index].clear()
            self._done[index] = None
            self._active[index] = False
        self._destroyed = True


class NcclEPStandardPrepareAndFinalize(NcclEPPrepareAndFinalize):
    """Prepare/Finalize using NCCL EP with FLAT layout
    (Standard activation format).
    """

    def __init__(
        self,
        ep_group: nccl_ep.Group,
        num_dispatchers: int,
        dp_size: int,
        max_tokens_per_rank: int,
        rank_expert_offset: int,
        num_experts: int,
        num_topk: int,
        use_fp8_dispatch: bool = False,
    ):
        super().__init__(ep_group)
        self.num_dispatchers_ = num_dispatchers
        self.dp_size = dp_size
        self.max_tokens_per_rank = max_tokens_per_rank
        self.rank_expert_offset = rank_expert_offset
        self.num_experts = num_experts
        self.num_topk = num_topk
        if use_fp8_dispatch:
            raise ValueError("NCCL EP FP8 transport is not supported yet; use BF16.")

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def output_is_reduced(self) -> bool:
        return True

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return self.max_tokens_per_rank

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int64

    def _do_dispatch(
        self,
        tokens: torch.Tensor,
        rank_topk_ids: torch.Tensor,
        rank_topk_weights: torch.Tensor,
        a1_scale: torch.Tensor | None,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool,
    ) -> Callable:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("NCCL EP CUDA graph capture is not supported yet.")
        num_tokens = tokens.shape[0]
        assert num_tokens <= self.max_tokens_per_rank
        hidden = tokens.shape[1]
        topk = rank_topk_ids.shape[1]
        dbo_yield_and_switch_from_compute_to_comm()
        stream = _get_cuda_stream()

        recv_total = torch.empty(1, dtype=torch.int32, device=tokens.device)
        expert_counts = torch.empty(
            self.num_experts // self.num_dispatchers_,
            dtype=torch.int32,
            device=tokens.device,
        )
        layout_info = nccl_ep.LayoutInfo(
            expert_counters=_to_nccl_tensor(expert_counts),
            recv_total_counter=_to_nccl_tensor(recv_total),
            recv_topk_idx_kind=nccl_ep.ExpertIdKind.LOCAL,
        )
        a2a_idx = dbo_current_ubatch_id()
        ep_handle = self._get_handle(
            a2a_idx,
            nccl_ep.Layout.FLAT,
            rank_topk_ids,
            (tokens, rank_topk_weights, recv_total, expert_counts),
            layout_info=layout_info,
        )

        # FLAT sends each token at most once to each EP rank, irrespective of DP.
        max_recv = nccl_ep_max_recv_tokens(
            self.max_tokens_per_rank, self.num_dispatchers_
        )
        recv_tokens = torch.empty(
            (max_recv, hidden),
            dtype=tokens.dtype,
            device=tokens.device,
        )
        recv_topk_weights = torch.empty(
            (max_recv, topk),
            dtype=torch.float32,
            device=tokens.device,
        )
        recv_topk_idx = torch.empty(
            (max_recv, topk),
            dtype=torch.int64,
            device=tokens.device,
        )
        self._tensors[a2a_idx].extend((recv_tokens, recv_topk_weights, recv_topk_idx))
        dispatch_inputs = nccl_ep.DispatchInputs(
            tokens=_to_nccl_tensor(tokens),
            topk_weights=_to_nccl_tensor(rank_topk_weights),
        )
        dispatch_outputs = nccl_ep.DispatchOutputs(
            tokens=_to_nccl_tensor(recv_tokens),
            topk_weights=_to_nccl_tensor(recv_topk_weights),
            topk_idx=_to_nccl_tensor(recv_topk_idx),
        )

        ep_handle.dispatch(
            dispatch_inputs,
            dispatch_outputs,
            config=nccl_ep.DispatchConfig(send_only=0),
            stream=stream,
        )
        ep_handle.complete(stream=stream)

        dbo_switch_to_compute_sync()

        return lambda: self._receiver(
            recv_tokens,
            recv_topk_idx,
            recv_topk_weights,
            recv_total,
            expert_counts,
            a1_scale,
            quant_config,
            defer_input_quant=defer_input_quant,
        )

    def _receiver(
        self,
        recv_x: torch.Tensor,
        recv_topk_idx: torch.Tensor | None,
        recv_topk_weights: torch.Tensor | None,
        recv_total: torch.Tensor,
        expert_counts: torch.Tensor,
        a1_scale: torch.Tensor | None,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool,
    ) -> mk.PrepareResultType:
        num_recv = int(recv_total.item())
        if not 0 <= num_recv <= recv_x.shape[0]:
            raise RuntimeError(
                f"NCCL EP received {num_recv} rows for capacity {recv_x.shape[0]}"
            )
        expert_x = recv_x[:num_recv]
        expert_x_scale = None
        if recv_topk_idx is not None:
            recv_topk_idx = recv_topk_idx[:num_recv]
        if recv_topk_weights is not None:
            recv_topk_weights = recv_topk_weights[:num_recv]

        if recv_topk_idx is not None:
            valid = (recv_topk_idx >= 0) & (recv_topk_idx < expert_counts.numel())
            # Match DeepEP HT: expert_map filters a valid nonlocal global ID.
            invalid_id = -1
            if self.num_dispatchers_ > 1:
                invalid_id = self.num_experts - 1 if self.rank_expert_offset == 0 else 0
            recv_topk_idx = torch.where(
                valid, recv_topk_idx + self.rank_expert_offset, invalid_id
            )
            if recv_topk_weights is not None:
                recv_topk_weights = torch.where(valid, recv_topk_weights, 0.0)

        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=expert_counts,
            expert_num_tokens_cpu=expert_counts.cpu(),
        )

        if not defer_input_quant and expert_x.numel() != 0:
            expert_x, expert_x_scale = moe_kernel_quantize_input(
                expert_x,
                a1_scale,
                quant_dtype=quant_config.quant_dtype,
                per_act_token_quant=quant_config.per_act_token_quant,
                block_shape=quant_config.block_shape,
            )

        return (
            expert_x,
            expert_x_scale,
            expert_tokens_meta,
            recv_topk_idx,
            recv_topk_weights,
        )

    def supports_async(self) -> bool:
        return True

    def prepare_async(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.ReceiverType:
        self._check_not_destroyed()
        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1 = a1 * topk_weights.to(a1.dtype)

        if a1.dtype != torch.bfloat16:
            raise ValueError("NCCL EP currently requires BF16 transport inputs.")
        a1_post_scale = (
            quant_config.a1_gscale
            if quant_config.quant_dtype == "nvfp4"
            else quant_config.a1_scale
        )

        return self._do_dispatch(
            tokens=a1,
            rank_topk_ids=topk_ids,
            rank_topk_weights=topk_weights,
            a1_scale=a1_post_scale,
            quant_config=quant_config,
            defer_input_quant=defer_input_quant,
        )

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType:
        receiver = self.prepare_async(
            a1,
            topk_weights,
            topk_ids,
            num_experts,
            expert_map,
            apply_router_weight_on_input,
            quant_config,
            defer_input_quant,
        )
        return receiver()

    def _finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
        do_async: bool,
    ) -> Callable | None:
        self._check_not_destroyed()
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("NCCL EP CUDA graph capture is not supported yet.")
        a2a_idx = dbo_current_ubatch_id()
        ep_handle = self.handles[a2a_idx]
        assert ep_handle is not None

        if fused_expert_output.numel() != 0:
            if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
                weight_and_reduce_impl = TopKWeightAndReduceContiguous()
            fused_expert_output = weight_and_reduce_impl.apply(
                output=None,
                fused_expert_output=fused_expert_output,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )

        else:
            fused_expert_output = fused_expert_output.reshape(0, output.shape[-1])

        dbo_yield_and_switch_from_compute_to_comm()
        stream = _get_cuda_stream()

        assert fused_expert_output.dtype == torch.bfloat16, (
            f"NCCL EP combine requires bfloat16, got {fused_expert_output.dtype}"
        )

        combined_output = torch.empty_like(output)
        self._tensors[a2a_idx].extend((fused_expert_output, combined_output, output))

        combine_inputs = nccl_ep.CombineInputs(
            tokens=_to_nccl_tensor(fused_expert_output),
        )
        combine_outputs = nccl_ep.CombineOutputs(
            tokens=_to_nccl_tensor(combined_output),
        )

        ep_handle.combine(
            combine_inputs,
            combine_outputs,
            config=nccl_ep.CombineConfig(send_only=0),
            stream=stream,
        )
        ep_handle.complete(stream=stream)

        dbo_switch_to_compute()

        if do_async:

            def _receiver():
                dbo_switch_to_comm()
                output.copy_(combined_output, non_blocking=True)
                dbo_yield_and_switch_from_comm_to_compute()
                self._finish(a2a_idx)

            return _receiver
        else:
            assert not dbo_enabled()
            output.copy_(combined_output, non_blocking=True)
            self._finish(a2a_idx)
            return None

    def finalize_async(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> Callable:
        receiver = self._finalize(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
            True,
        )
        assert receiver is not None
        return receiver

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        self._finalize(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
            False,
        )

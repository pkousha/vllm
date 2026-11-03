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
from vllm.utils.torch_utils import aux_stream
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


def _quantize_dispatch_input(tokens, quant_config):
    if (
        quant_config.quant_dtype != torch.float8_e4m3fn
        or not quant_config.is_block_quantized
    ):
        raise ValueError("NCCL EP FP8 transport requires block-FP8 expert inputs.")
    assert quant_config.block_shape is not None
    if tokens.shape[1] % quant_config.block_shape[1] or tokens.shape[1] % 16:
        raise ValueError(
            "NCCL EP FP8 rows must align to the quantization block and 16B."
        )
    num_scales = tokens.shape[1] // quant_config.block_shape[1]
    if tokens.shape[0] == 0:
        tokens = torch.empty_like(tokens, dtype=torch.float8_e4m3fn)
        scales = torch.empty((0, num_scales), dtype=torch.float32, device=tokens.device)
    else:
        tokens, scales = moe_kernel_quantize_input(
            tokens,
            quant_config.a1_scale,
            quant_config.quant_dtype,
            quant_config.per_act_token_quant,
            quant_config.block_shape,
        )
        assert scales is not None
    # FWD requires a 16-byte physical scale row, including for small hidden sizes.
    if num_scales % 4:
        scales = torch.nn.functional.pad(scales, (0, -num_scales % 4))
    return tokens, scales.contiguous()


class NcclEPPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """Own native handles and their pointer-backed tensors."""

    def __init__(self, ep_group: nccl_ep.Group):
        super().__init__()
        self.ep_group = ep_group
        self.handles: list[nccl_ep.Handle | None] = [None, None]
        self._tensors: list[list[torch.Tensor]] = [[], []]
        self._descriptors: list[list[object]] = [[], []]
        self._done: list[tuple[torch.cuda.Event, torch.cuda.Stream] | None] = [
            None,
            None,
        ]
        self._pending: list[torch.cuda.Stream | None] = [None, None]
        self._active = [False, False]
        self._num_topk: int | None = None
        self._devices: set[torch.device] = set()
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
        handle = self.handles[index]
        capturing = torch.cuda.is_current_stream_capturing()
        if handle is None and capturing:
            raise RuntimeError("Warm up NCCL EP before CUDA graph capture.")
        done = self._done[index]
        self._done[index] = None
        # Capture/replay must be ordered by the caller after eager work.
        if not capturing and done is not None:
            event, producer = done
            consumer = torch.cuda.current_stream(routing.device)
            if consumer != producer:
                # PERF_REVIEW(pkousha): DBO may already order this reuse;
                # other cross-stream callers still need the dependency.
                consumer.wait_event(event)
        self._tensors[index] = [routing, *tensors]
        self._devices.add(routing.device)
        self._active[index] = True
        stream = _get_cuda_stream()
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
            consumer = torch.cuda.current_stream(pending.device)
            # The pinned LL continuation launches on the original send stream.
            with torch.cuda.stream(pending):
                handle.complete(stream=pending.cuda_stream)
                if consumer != pending:
                    done = torch.cuda.Event()
                    done.record(pending)
                    consumer.wait_event(done)
            self._pending[index] = None

    def _finish(self, index):
        self._check_not_destroyed()
        assert self._pending[index] is None
        stream = torch.cuda.current_stream()
        # PERF_REVIEW(pkousha): Check which allocator stream records are necessary;
        # prove ownership under allocator pressure before removing them.
        for tensor in self._tensors[index]:
            tensor.record_stream(stream)
        self._descriptors[index].clear()
        if torch.cuda.is_current_stream_capturing():
            # Graph pools own temporaries; callers own static inputs across replay.
            self._tensors[index].clear()
            self._done[index] = None
        else:
            # Only routing remains cached by the idle native handle.
            self._tensors[index] = self._tensors[index][:1]
            done = torch.cuda.Event()
            done.record(stream)
            self._done[index] = (done, stream)
        self._active[index] = False

    def destroy(self):
        if self._destroyed:
            return
        for index in range(len(self.handles)):
            self._complete(index)
        for device in self._devices:
            torch.accelerator.synchronize(device)
        for index, handle in enumerate(self.handles):
            if handle is not None:
                handle.destroy()
                self.handles[index] = None
            self._tensors[index].clear()
            self._descriptors[index].clear()
            self._done[index] = None
            self._active[index] = False
        self._devices.clear()
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
        self.use_fp8_dispatch = use_fp8_dispatch
        self._combine_stream = aux_stream()

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
        token_scales: torch.Tensor | None,
        rank_topk_ids: torch.Tensor,
        rank_topk_weights: torch.Tensor,
        a1_scale: torch.Tensor | None,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool,
    ) -> Callable:
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
        if token_scales is not None:
            self._tensors[a2a_idx].append(token_scales)

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
        recv_scales = None
        if token_scales is not None:
            recv_scales = torch.empty(
                (max_recv, token_scales.shape[1]),
                dtype=token_scales.dtype,
                device=tokens.device,
            )
            self._tensors[a2a_idx].append(recv_scales)
        self._tensors[a2a_idx].extend((recv_tokens, recv_topk_weights, recv_topk_idx))
        dispatch_inputs = nccl_ep.DispatchInputs(
            tokens=_to_nccl_tensor(tokens),
            topk_weights=_to_nccl_tensor(rank_topk_weights),
            scales=_to_nccl_tensor(token_scales) if token_scales is not None else None,
        )
        dispatch_outputs = nccl_ep.DispatchOutputs(
            tokens=_to_nccl_tensor(recv_tokens),
            topk_weights=_to_nccl_tensor(recv_topk_weights),
            topk_idx=_to_nccl_tensor(recv_topk_idx),
            scales=_to_nccl_tensor(recv_scales) if recv_scales is not None else None,
        )

        # PERF_REVIEW(pkousha): Pinned HT eager dispatch reads a count on CPU;
        # captured fixed-buffer dispatch skips that native synchronization.
        ep_handle.dispatch(
            dispatch_inputs,
            dispatch_outputs,
            config=nccl_ep.DispatchConfig(
                quantization_recipe=(
                    nccl_ep.DispatchQuantizationRecipe.FWD
                    if token_scales is not None
                    else nccl_ep.DispatchQuantizationRecipe.NONE
                ),
            ),
            stream=stream,
        )
        ep_handle.complete(stream=stream)

        dbo_switch_to_compute_sync()

        return lambda: self._receiver(
            recv_tokens,
            recv_scales,
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
        recv_scales: torch.Tensor | None,
        recv_topk_idx: torch.Tensor | None,
        recv_topk_weights: torch.Tensor | None,
        recv_total: torch.Tensor,
        expert_counts: torch.Tensor,
        a1_scale: torch.Tensor | None,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool,
    ) -> mk.PrepareResultType:
        torch._assert_async(
            (recv_total >= 0) & (recv_total <= recv_x.shape[0]),
            "NCCL EP receive count exceeds the buffer capacity",
        )
        valid_rows = (torch.arange(recv_x.shape[0], device=recv_x.device) < recv_total)[
            :, None
        ]
        # PERF_REVIEW(pkousha): Masking and local quantization visit full capacity;
        # investigate smaller per-graph bounds without dropping the padding mask.
        # Byte views also support FP8, whose masked_fill is not implemented.
        expert_x = (
            recv_x.view(torch.uint8).masked_fill(~valid_rows, 0).view(recv_x.dtype)
        )
        expert_x_scale = None
        if recv_scales is not None:
            assert quant_config.block_shape is not None
            num_scales = recv_x.shape[1] // quant_config.block_shape[1]
            expert_x_scale = recv_scales[:, :num_scales].masked_fill(~valid_rows, 1.0)

        if recv_topk_idx is not None:
            valid = (
                valid_rows
                & (recv_topk_idx >= 0)
                & (recv_topk_idx < expert_counts.numel())
            )
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
            # PERF_REVIEW(pkousha): GPU-only counts permit capture; measure
            # DeepGEMM's larger workspace bound against exact CPU sizing.
            expert_num_tokens_cpu=None,
        )

        if expert_x_scale is None and not defer_input_quant and expert_x.numel() != 0:
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
            raise ValueError(
                "NCCL EP expects BF16 inputs before transport quantization."
            )
        a1_post_scale = (
            quant_config.a1_gscale
            if quant_config.quant_dtype == "nvfp4"
            else quant_config.a1_scale
        )
        token_scales = None
        if self.use_fp8_dispatch and not defer_input_quant:
            a1, token_scales = _quantize_dispatch_input(a1, quant_config)

        return self._do_dispatch(
            tokens=a1,
            token_scales=token_scales,
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

        assert fused_expert_output.dtype == torch.bfloat16, (
            f"NCCL EP combine requires bfloat16, got {fused_expert_output.dtype}"
        )

        combined_output = output
        if (
            not output.is_contiguous()
            or output.data_ptr() % 16 != 0
            or output.untyped_storage().data_ptr()
            == fused_expert_output.untyped_storage().data_ptr()
        ):
            # PERF_REVIEW(pkousha): Native HT stores need aligned, contiguous,
            # disjoint output; keep the temporary/copy for incompatible views.
            combined_output = torch.empty_like(
                output, memory_format=torch.contiguous_format
            )
        self._tensors[a2a_idx].extend((fused_expert_output, output))
        if combined_output is not output:
            self._tensors[a2a_idx].append(combined_output)

        combine_inputs = nccl_ep.CombineInputs(
            tokens=_to_nccl_tensor(fused_expert_output),
        )
        combine_outputs = nccl_ep.CombineOutputs(
            tokens=_to_nccl_tensor(combined_output),
        )

        combine_stream = torch.cuda.current_stream()
        if do_async and not dbo_enabled():
            assert self._combine_stream is not None
            combine_stream = self._combine_stream
            if combine_stream != torch.cuda.current_stream():
                # PERF_REVIEW(pkousha): Fork/join permits shared-expert overlap;
                # measure the two GPU handoffs against the overlap they enable.
                combine_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(combine_stream):
            ep_handle.combine(
                combine_inputs,
                combine_outputs,
                config=nccl_ep.CombineConfig(send_only=0),
                stream=combine_stream.cuda_stream,
            )
            ep_handle.complete(stream=combine_stream.cuda_stream)

        dbo_switch_to_compute()

        if do_async:

            def _receiver():
                dbo_switch_to_comm()
                consumer = torch.cuda.current_stream()
                if consumer != combine_stream:
                    consumer.wait_stream(combine_stream)
                if combined_output is not output:
                    output.copy_(combined_output, non_blocking=True)
                dbo_yield_and_switch_from_comm_to_compute()
                self._finish(a2a_idx)

            return _receiver
        else:
            assert not dbo_enabled()
            if combined_output is not output:
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

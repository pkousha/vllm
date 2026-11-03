# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import nccl.ep as nccl_ep  # type: ignore[import-not-found]
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed.device_communicators.all2all import nccl_ep_max_recv_tokens
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.prepare_finalize.nccl_ep import (
    NcclEPPrepareAndFinalize,
    _get_cuda_stream,
    _quantize_dispatch_input,
    _to_nccl_tensor,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
)
from vllm.model_executor.layers.fused_moe.utils import (
    moe_kernel_quantize_input,
    normalize_batched_scales_shape,
)
from vllm.v1.worker.ubatching import (
    dbo_current_ubatch_id,
    dbo_maybe_run_recv_hook,
)

logger = init_logger(__name__)


class NcclEPBatchedPrepareAndFinalize(NcclEPPrepareAndFinalize):
    """Prepare/Finalize using NCCL EP with EXPERT_MAJOR layout
    (BatchedExperts activation format).
    """

    def __init__(
        self,
        ep_group: nccl_ep.Group,
        max_tokens_per_rank: int,
        num_dispatchers: int,
        num_experts: int,
        num_local_experts: int,
        use_fp8_dispatch: bool = False,
        global_to_physical: torch.Tensor | None = None,
        physical_to_global: torch.Tensor | None = None,
        local_expert_global_ids: torch.Tensor | None = None,
    ):
        super().__init__(ep_group)
        self.max_tokens_per_rank = max_tokens_per_rank
        self.use_fp8_dispatch = use_fp8_dispatch
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.num_dispatchers_ = num_dispatchers

        topk_indices_dtype = self.topk_indices_dtype()

        def _maybe_cast(tensor: torch.Tensor | None) -> torch.Tensor | None:
            if tensor is None or topk_indices_dtype is None:
                return tensor
            return tensor.to(dtype=topk_indices_dtype)

        self.global_to_physical = _maybe_cast(global_to_physical)
        self.physical_to_global = _maybe_cast(physical_to_global)
        self.local_expert_global_ids = _maybe_cast(local_expert_global_ids)

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def output_is_reduced(self) -> bool:
        return True

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.BatchedExperts

    def max_num_tokens_per_rank(self) -> int | None:
        return self.max_tokens_per_rank

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int64

    def _map_global_to_physical_ids(self, topk_ids: torch.Tensor) -> torch.Tensor:
        if self.global_to_physical is None:
            return topk_ids
        valid = (topk_ids >= 0) & (topk_ids < self.global_to_physical.numel())
        physical_ids = self.global_to_physical[
            topk_ids.clamp(0, self.global_to_physical.numel() - 1)
        ]
        return torch.where(valid, physical_ids, -1)

    def _do_quant(
        self,
        x: torch.Tensor,
        a1_dtype: torch.dtype,
        quant_config: FusedMoEQuantConfig,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        num_experts, max_tokens, hidden_dim = x.size()
        q_dtype = quant_config.quant_dtype

        if q_dtype == "nvfp4":
            q_dtype = None
            logger.debug_once("Using NCCL EP bfloat16 dispatch for NVFP4 MoE.")

        x = x.view((-1, hidden_dim))
        x, x_scales = moe_kernel_quantize_input(
            x,
            quant_config.a1_scale,
            q_dtype,
            quant_config.per_act_token_quant,
            quant_config.block_shape,
        )
        x = x.view((num_experts, -1, hidden_dim))

        if q_dtype is not None:
            assert x_scales is not None
            x_scales = normalize_batched_scales_shape(x_scales, num_experts)

        return x, x_scales

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
    ) -> tuple[Callable, mk.ReceiverType]:
        self._check_not_destroyed()
        if a1.dtype != torch.bfloat16:
            raise ValueError(
                "NCCL EP expects BF16 inputs before transport quantization."
            )
        if a1.shape[1] % 512:
            raise ValueError(
                "NCCL EP LL BF16 combine requires hidden size divisible by 512."
            )
        if defer_input_quant:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support defer_input_quant=True."
            )

        a2a_idx = dbo_current_ubatch_id()

        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1 = a1 * topk_weights.to(a1.dtype)

        assert a1.shape[0] <= self.max_tokens_per_rank
        hidden = a1.shape[1]
        stream = _get_cuda_stream()

        dispatch_topk_ids = self._map_global_to_physical_ids(topk_ids)
        tokens, token_scales = a1, None
        if self.use_fp8_dispatch:
            tokens, token_scales = _quantize_dispatch_input(a1, quant_config)

        ep_handle = self._get_handle(
            a2a_idx,
            nccl_ep.Layout.EXPERT_MAJOR,
            dispatch_topk_ids,
            (a1, topk_ids, topk_weights, tokens),
        )
        if token_scales is not None:
            self._tensors[a2a_idx].append(token_scales)

        max_recv = nccl_ep_max_recv_tokens(
            self.max_tokens_per_rank, self.num_dispatchers_
        )
        # PERF_REVIEW(pkousha): Full-capacity token/scale initialization costs
        # bandwidth each replay; preserve safe padding when reducing these writes.
        recv_tokens = torch.zeros(
            (self.num_local_experts, max_recv, hidden),
            dtype=tokens.dtype,
            device=a1.device,
        )
        recv_scales = None
        if token_scales is not None:
            recv_scales = torch.ones(
                (self.num_local_experts, max_recv, token_scales.shape[1]),
                dtype=token_scales.dtype,
                device=a1.device,
            )
            self._tensors[a2a_idx].append(recv_scales)
        recv_count = torch.empty(
            self.num_local_experts,
            dtype=torch.int32,
            device=a1.device,
        )

        self._tensors[a2a_idx].extend((recv_tokens, recv_count))
        dispatch_inputs = nccl_ep.DispatchInputs(
            tokens=_to_nccl_tensor(tokens),
            scales=_to_nccl_tensor(token_scales) if token_scales is not None else None,
        )
        dispatch_outputs = nccl_ep.DispatchOutputs(
            tokens=_to_nccl_tensor(recv_tokens),
            scales=_to_nccl_tensor(recv_scales) if recv_scales is not None else None,
        )
        dispatch_layout = nccl_ep.LayoutInfo(
            expert_counters=_to_nccl_tensor(recv_count),
        )

        self._descriptors[a2a_idx].extend(
            (dispatch_inputs, dispatch_outputs, dispatch_layout)
        )
        ep_handle.dispatch(
            dispatch_inputs,
            dispatch_outputs,
            layout_info=dispatch_layout,
            config=nccl_ep.DispatchConfig(
                send_only=1,
                quantization_recipe=(
                    nccl_ep.DispatchQuantizationRecipe.FWD
                    if token_scales is not None
                    else nccl_ep.DispatchQuantizationRecipe.NONE
                ),
            ),
            stream=stream,
        )

        self._pending[a2a_idx] = torch.cuda.current_stream()

        def hook():
            self._complete(a2a_idx)

        return (
            hook,
            lambda: self._receiver(
                recv_tokens,
                recv_scales,
                recv_count,
                a1.dtype,
                quant_config,
            ),
        )

    def _receiver(
        self,
        expert_x: torch.Tensor,
        expert_x_scale: torch.Tensor | None,
        expert_num_tokens: torch.Tensor,
        a1_dtype: torch.dtype,
        quant_config: FusedMoEQuantConfig,
    ) -> mk.PrepareResultType:
        if expert_x_scale is None:
            expert_x, expert_x_scale = self._do_quant(expert_x, a1_dtype, quant_config)
        else:
            assert quant_config.block_shape is not None
            num_scales = expert_x.shape[-1] // quant_config.block_shape[1]
            expert_x_scale = expert_x_scale[:, :, :num_scales].contiguous()

        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=expert_num_tokens,
            expert_num_tokens_cpu=None,
        )

        return expert_x, expert_x_scale, expert_tokens_meta, None, None

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
        hook, receiver = self.prepare_async(
            a1,
            topk_weights,
            topk_ids,
            num_experts,
            expert_map,
            apply_router_weight_on_input,
            quant_config,
            defer_input_quant=defer_input_quant,
        )
        hook()
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
    ) -> tuple[Callable, Callable]:
        self._check_not_destroyed()
        assert isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate), (
            "Weight application and reduction happens in the combine kernel."
        )

        a2a_idx = dbo_current_ubatch_id()
        ep_handle = self.handles[a2a_idx]
        assert ep_handle is not None

        combine_topk_weights = topk_weights
        if apply_router_weight_on_input:
            combine_topk_weights = torch.ones_like(topk_weights)

        stream = _get_cuda_stream()

        dbo_maybe_run_recv_hook()

        self._tensors[a2a_idx].extend(
            (fused_expert_output, output, combine_topk_weights)
        )
        combine_inputs = nccl_ep.CombineInputs(
            tokens=_to_nccl_tensor(fused_expert_output),
        )
        combine_outputs = nccl_ep.CombineOutputs(
            tokens=_to_nccl_tensor(output),
            topk_weights=_to_nccl_tensor(combine_topk_weights),
        )

        self._descriptors[a2a_idx].extend((combine_inputs, combine_outputs))
        ep_handle.combine(
            combine_inputs,
            combine_outputs,
            config=nccl_ep.CombineConfig(send_only=1),
            stream=stream,
        )

        self._pending[a2a_idx] = torch.cuda.current_stream()

        def recv_hook():
            self._complete(a2a_idx)

        return recv_hook, lambda: self._finish(a2a_idx)

    def finalize_async(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> tuple[Callable, Callable]:
        return self._finalize(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
            do_async=True,
        )

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        hook, receiver = self._finalize(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
            do_async=False,
        )
        hook()
        receiver()

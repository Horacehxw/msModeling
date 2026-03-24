from collections import deque
import logging

import torch
import torch.fx as fx

from ... import ops  # noqa: F401
from ..pass_base import TensorCastGraphModulePass
from ..topo_sort import stable_topo_sort

logger = logging.getLogger(__name__)


class DispatchFFNCombinePass(TensorCastGraphModulePass):
    _GROUPED_MATMUL_OPS = {
        torch.ops.tensor_cast.grouped_matmul.default,
        torch.ops.tensor_cast.grouped_matmul_quant.default,
        torch.ops.tensor_cast.grouped_matmul_quant_int4.default,
        torch.ops.tensor_cast.grouped_matmul_fp8.default,
        torch.ops.tensor_cast.grouped_matmul_mxfp4.default,
    }

    _GROUPED_MATMUL_SWIGLU_OPS = {
        torch.ops.tensor_cast.grouped_matmul_swiglu.default,
        torch.ops.tensor_cast.grouped_matmul_quant_swiglu.default,
        torch.ops.tensor_cast.grouped_matmul_quant_int4_swiglu.default,
        torch.ops.tensor_cast.grouped_matmul_fp8_swiglu.default,
        torch.ops.tensor_cast.grouped_matmul_mxfp4_swiglu.default,
    }

    _LINEAR_FFN_OPS = {
        torch.ops.tensor_cast.static_quant_linear.default,
        torch.ops.tensor_cast.static_quant_linear_int4.default,
        torch.ops.tensor_cast.fp8_linear.default,
        torch.ops.tensor_cast.mxfp4_linear.default,
    }

    _SWIGLU_OPS = {
        torch.ops.tensor_cast.swiglu.default,
        torch.ops.tensor_cast.grouped_matmul_swiglu.default,
        torch.ops.tensor_cast.grouped_matmul_quant_swiglu.default,
        torch.ops.tensor_cast.grouped_matmul_quant_int4_swiglu.default,
        torch.ops.tensor_cast.grouped_matmul_fp8_swiglu.default,
        torch.ops.tensor_cast.grouped_matmul_mxfp4_swiglu.default,
    }

    def __call__(self, gm: fx.GraphModule) -> fx.GraphModule:
        graph = gm.graph
        modified = False

        # Pre-scan: find all permute_tokens (start) and unpermute_tokens (end) nodes
        all_permute_starts = []
        all_unpermute_ends = []

        for node in graph.nodes:
            if self._is_permute_token(node):
                all_permute_starts.append(node)
            if self._is_unpermute_token(node):
                all_unpermute_ends.append(node)

        # Traverse from each permute_tokens to find corresponding unpermute_tokens
        processed_nodes = set()
        max_traverse_depth = 600

        for start_node in all_permute_starts:
            if start_node in processed_nodes:
                continue

            # Collect region nodes with forward BFS
            region_nodes, end_node = self._collect_region_nodes_forward(
                start_node, processed_nodes, max_traverse_depth
            )

            if not region_nodes or end_node is None:
                continue

            # Check required operators in region
            has_required_ops, reason = self._check_region_features(region_nodes)
            if not has_required_ops:
                logger.debug(
                    "DispatchFFNCombinePass skip region start=%s end=%s reason=%s",
                    start_node.name,
                    end_node.name,
                    reason,
                )
                continue

            # Replace region with fused operator
            with graph.inserting_before(end_node):
                fused_node = graph.create_node(
                    "call_function",
                    torch.ops.tensor_cast.dispatch_ffn_combine.default,
                    args=(start_node.args[0], start_node.args[1]),
                    kwargs={},
                    name="dispatch_ffn_combine_fused",
                )

            # Redirect all uses of unpermute_tokens to fused node
            end_node.replace_all_uses_with(fused_node)
            processed_nodes.update(region_nodes)
            modified = True

        # Clean up graph and recompile
        if modified:
            stable_topo_sort(gm)
            gm.graph.eliminate_dead_code()
            gm.graph.lint()
            gm.recompile()

        return gm

    # Forward BFS to collect region nodes with depth limit
    def _collect_region_nodes_forward(
        self, start_node: fx.Node, processed: set, max_depth: int
    ) -> tuple[set, fx.Node]:
        region = set()
        q = deque([(start_node, 0)])
        end_node = None

        while q and end_node is None:
            n, depth = q.popleft()

            if depth > max_depth or n in region or n in processed:
                continue

            region.add(n)

            # Stop traversal if unpermute_tokens is found
            if self._is_unpermute_token(n):
                end_node = n
                continue

            # Skip placeholders/constants
            if n.op in ["placeholder", "get_attr"]:
                continue

            # Traverse to next nodes
            for user in n.users:
                if isinstance(user, fx.Node) and user not in region:
                    q.append((user, depth + 1))

        return (region, end_node) if end_node else (set(), None)

    # Node type check helpers
    def _is_permute_token(self, node: fx.Node) -> bool:
        return (node.op == "call_function"
                and node.target == torch.ops.tensor_cast.permute_tokens.default)

    def _is_unpermute_token(self, node: fx.Node) -> bool:
        return (node.op == "call_function"
                and node.target == torch.ops.tensor_cast.unpermute_tokens.default)

    def _is_grouped_matmul(self, node: fx.Node) -> bool:
        return (node.op == "call_function"
                and node.target in self._GROUPED_MATMUL_OPS)

    def _is_grouped_matmul_swiglu(self, node: fx.Node) -> bool:
        return (node.op == "call_function"
                and node.target in self._GROUPED_MATMUL_SWIGLU_OPS)

    def _is_linear_ffn(self, node: fx.Node) -> bool:
        return node.op == "call_function" and node.target in self._LINEAR_FFN_OPS

    def _is_swiglu(self, node: fx.Node) -> bool:
        return node.op == "call_function" and node.target in self._SWIGLU_OPS

    # Check if region contains required MoE FFN operators
    def _check_region_features(self, region: set) -> tuple[bool, str]:
        has_ffn_compute = False
        has_permute = False
        has_unpermute = False
        has_swiglu = False

        for node in region:
            if self._is_permute_token(node):
                has_permute = True
            if self._is_unpermute_token(node):
                has_unpermute = True
            if (
                self._is_grouped_matmul(node)
                or self._is_grouped_matmul_swiglu(node)
                or self._is_linear_ffn(node)
            ):
                has_ffn_compute = True
            if self._is_swiglu(node):
                has_swiglu = True

        # Validate required operator counts
        if not has_permute:
            return False, "missing_permute_tokens"
        if not has_unpermute:
            return False, "missing_unpermute_tokens"
        if not has_ffn_compute:
            return False, "missing_ffn_compute_ops"
        if not has_swiglu:
            return False, "missing_swiglu"
        return True, "matched"

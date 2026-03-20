"""FlashCommV1 pass for three decoder-region rewrite patterns.

Pattern 1:
    all_reduce -> _internal_mark_region_begin -> rms_norm/add_rms_norm
    => reduce_scatter -> _internal_mark_region_begin -> norm(local) -> all_gather

Pattern 2:
    _internal_mark_region_begin + all_reduce -> add_rms_norm2
    => _internal_mark_region_begin + reduce_scatter -> add_rms_norm2
    Tuple outputs are gathered unless they continue into pattern 3.

Pattern 3:
    add_rms_norm2 residual + all_reduce[/view]
        -> add -> _internal_mark_region_end -> rms_norm
    => local residual + reduce_scatter[/view]
        -> add -> _internal_mark_region_end -> rms_norm(local) -> all_gather
"""

import logging
import operator
from typing import List, Optional, Set, Tuple

import torch
from torch.fx import GraphModule, Node

from ... import config
from ..pass_base import TensorCastGraphModulePass

logger = logging.getLogger(__name__)

# Single-output norm ops that can benefit from FlashCommV1 pattern 1/3
_SINGLE_OUTPUT_NORM_OPS: Set[torch._ops.OpOverloadPacket] = {
    torch.ops.tensor_cast.rms_norm.default,
    torch.ops.tensor_cast.add_rms_norm.default,
}

# Communication ops that precede norm in the pattern
_COMM_OPS: Set[torch._ops.OpOverloadPacket] = {
    torch.ops.tensor_cast.all_reduce.default,
}

# Internal marker node to skip
_INTERNAL_MARKER_OPS: Set[torch._ops.OpOverloadPacket] = {
    torch.ops.tensor_cast._internal_mark_region_begin.default,
}

_INTERNAL_MARKER_END_OPS: Set[torch._ops.OpOverloadPacket] = {
    torch.ops.tensor_cast._internal_mark_region_end.default,
}

_INTERNAL_COPY_OPS: Set[torch._ops.OpOverloadPacket] = {
    torch.ops.tensor_cast._internal_copy_region.default,
}

_ADD_OPS: Set[torch._ops.OpOverloadPacket] = {
    torch.ops.aten.add.Tensor,
}

_VIEW_OPS: Set[torch._ops.OpOverloadPacket] = {
    torch.ops.aten.view.default,
    torch.ops.aten.reshape.default,
}

# Sequence dimension (dim=1) for transformer models
_SEQ_DIM = 1

Pattern1Match = Tuple[Node, Node, Node]
Pattern2Match = Tuple[Node, Node, Node]
Pattern3Match = Tuple[Node, Node, Node, Node, Node]


class FlashCommV1Pass(TensorCastGraphModulePass):
    """FX pass for the three FlashCommV1 decoder-region rewrite patterns."""

    def __call__(self, gm: GraphModule) -> GraphModule:
        """Apply FlashCommV1 transformation to the graph."""
        if not config.compilation.passes.enable_flashcomm_v1:
            return gm

        graph = gm.graph

        # Infer world_size from the first all_reduce node's rank_group
        world_size = self._get_world_size_from_graph(graph)
        if world_size <= 1:
            logger.debug("No TP detected (world_size <= 1), skipping FlashCommV1 pass")
            return gm

        pattern1_matches = self._find_pattern1_matches(graph)
        pattern2_matches = self._find_pattern2_matches(graph)

        for comm_node, marker_node, norm_node in pattern1_matches:
            self._rewrite_pattern1(
                graph, comm_node, marker_node, norm_node, world_size
            )

        for marker_node, comm_node, norm_node in pattern2_matches:
            self._rewrite_pattern2(
                graph, marker_node, comm_node, norm_node, world_size
            )

        pattern3_matches = self._find_pattern3_matches(graph)
        for (
            comm_node,
            comm_output_node,
            add_node,
            region_end_node,
            norm_node,
        ) in pattern3_matches:
            self._rewrite_pattern3(
                graph,
                comm_node,
                comm_output_node,
                add_node,
                region_end_node,
                norm_node,
                world_size,
            )

        if (
            not pattern1_matches
            and not pattern2_matches
            and not pattern3_matches
        ):
            logger.debug("No FlashCommV1 patterns found for transformation")
            return gm

        logger.debug(
            "Transformed %d pattern1 matches, %d pattern2 matches, and %d "
            "pattern3 matches for FlashCommV1 (world_size=%d)",
            len(pattern1_matches),
            len(pattern2_matches),
            len(pattern3_matches),
            world_size,
        )

        comm_nodes_to_cleanup = {comm_node for comm_node, _, _ in pattern1_matches}
        comm_nodes_to_cleanup.update(
            comm_node for _, comm_node, _ in pattern2_matches
        )
        comm_nodes_to_cleanup.update(
            comm_node for comm_node, _, _, _, _ in pattern3_matches
        )
        for comm_node in comm_nodes_to_cleanup:
            if comm_node in graph.nodes and len(comm_node.users) == 0:
                graph.erase_node(comm_node)

        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()
        return gm

    def _get_world_size_from_graph(self, graph) -> int:
        """Infer world_size from all_reduce node's rank_group.

        Args:
            graph: The FX graph

        Returns:
            world_size inferred from rank_group length, or 1 if not found
        """
        for node in graph.nodes:
            if self._is_comm_op(node) and len(node.args) >= 3:
                rank_group = node.args[2]
                if isinstance(rank_group, list):
                    return len(rank_group)
        return 1

    def _find_pattern1_matches(self, graph) -> List[Pattern1Match]:
        """Match pattern 1: all_reduce -> region_begin -> norm."""
        # all_reduce -> region_begin -> rms_norm/add_rms_norm
        patterns = []

        for node in graph.nodes:
            if not self._is_comm_op(node):
                continue

            # Find all_reduce → _internal_mark_region_begin → rms_norm
            for user in node.users.keys():
                if not self._is_internal_marker(user):
                    continue

                # Found marker, continue to find its norm user
                for marker_user in user.users.keys():
                    if self._is_pattern13_norm_op(marker_user):
                        patterns.append((node, user, marker_user))
                        logger.debug(
                            "Found FlashCommV1 pattern 1: "
                            f"{node.target} → {user.target} → {marker_user.target}"
                        )

        return patterns

    def _find_pattern2_matches(self, graph) -> List[Pattern2Match]:
        """Match pattern 2: region_begin local input + all_reduce -> add_rms_norm2."""
        # region_begin + all_reduce -> add_rms_norm2
        patterns = []

        for node in graph.nodes:
            if not self._is_add_rms_norm2(node):
                continue

            marker_node = None
            comm_node = None
            for arg in node.args[:2]:
                if isinstance(arg, Node) and self._is_internal_marker(arg):
                    marker_node = arg
                elif isinstance(arg, Node) and self._is_comm_op(arg):
                    comm_node = arg

            if marker_node is None or comm_node is None:
                continue

            patterns.append((marker_node, comm_node, node))
            logger.debug(
                "Found FlashCommV1 pattern 2: "
                f"{marker_node.target} + {comm_node.target} → {node.target}"
            )

        return patterns

    def _is_comm_op(self, node: Node) -> bool:
        """Check if node is a communication op that can be transformed."""
        if node.op != "call_function":
            return False
        return node.target in _COMM_OPS

    def _is_pattern13_norm_op(self, node: Node) -> bool:
        """Check if node is a single-output norm op that benefits from FlashCommV1."""
        if node.op != "call_function":
            return False
        return node.target in _SINGLE_OUTPUT_NORM_OPS

    def _is_internal_marker(self, node: Node) -> bool:
        """Check if node is _internal_mark_region_begin."""
        if node.op != "call_function":
            return False
        return node.target in _INTERNAL_MARKER_OPS

    def _is_internal_marker_end(self, node: Node) -> bool:
        """Check if node is _internal_mark_region_end."""
        if node.op != "call_function":
            return False
        return node.target in _INTERNAL_MARKER_END_OPS

    def _is_internal_copy_region(self, node: Node) -> bool:
        """Check if node is _internal_copy_region."""
        if node.op != "call_function":
            return False
        return node.target in _INTERNAL_COPY_OPS

    def _is_add_op(self, node: Node) -> bool:
        """Check if node is aten.add.Tensor."""
        return node.op == "call_function" and node.target in _ADD_OPS

    def _is_view_op(self, node: Node) -> bool:
        """Check if node is a reshaping op between comm and add."""
        return node.op == "call_function" and node.target in _VIEW_OPS

    def _is_reduce_scatter(self, node: Node) -> bool:
        """Check if node is a reduce_scatter op."""
        if node.op != "call_function":
            return False
        return node.target == torch.ops.tensor_cast.reduce_scatter.default

    def _has_all_gather_user(self, node: Node) -> bool:
        """Check if the node is already gathered by a direct all_gather user."""
        return any(
            user.op == "call_function"
            and user.target == torch.ops.tensor_cast.all_gather.default
            for user in node.users.keys()
        )

    def _is_add_rms_norm2(self, node: Node) -> bool:
        """Check if node is add_rms_norm2."""
        return (
            node.op == "call_function"
            and node.target == torch.ops.tensor_cast.add_rms_norm2.default
        )

    def _is_pattern3_residual_getitem(self, node: Node) -> bool:
        """Pattern 3 only accepts the residual output from add_rms_norm2."""
        if node.op != "call_function" or node.target != operator.getitem:
            return False

        if len(node.args) != 2 or not isinstance(node.args[1], int):
            return False

        source = node.args[0]
        if not isinstance(source, Node) or not self._is_add_rms_norm2(source):
            return False

        return node.args[1] == 1

    def _is_getitem_node(self, node: Node, source: Node) -> bool:
        """Check if node extracts a tuple item from the given source."""
        return (
            node.op == "call_function"
            and node.target == operator.getitem
            and len(node.args) == 2
            and node.args[0] is source
            and isinstance(node.args[1], int)
        )

    def _get_or_insert_pattern1_reduce_scatter(
        self,
        graph,
        comm_node: Node,
        marker_node: Node,
        rank,
        rank_group,
    ) -> Node:
        """Reuse an existing reduce_scatter feeding marker or insert a new one."""
        marker_input = marker_node.args[0]
        if isinstance(marker_input, Node) and self._is_reduce_scatter(marker_input):
            return marker_input

        shard_dim = self._get_comm_reduce_scatter_dim(comm_node)
        with graph.inserting_after(comm_node):
            reduce_scatter = graph.call_function(
                torch.ops.tensor_cast.reduce_scatter.default,
                (comm_node.args[0], shard_dim, rank, rank_group),
            )

        marker_node.replace_input_with(comm_node, reduce_scatter)
        return reduce_scatter

    def _get_comm_reduce_scatter_dim(self, comm_node: Node) -> int:
        """Infer the sequence-shard dimension from the pre-comm input rank."""
        if not comm_node.args or not isinstance(comm_node.args[0], Node):
            return _SEQ_DIM

        meta_val = comm_node.args[0].meta.get("val")
        return self._get_shard_dim_from_meta_val(meta_val)

    def _get_shard_dim_from_meta_val(self, meta_val) -> int:
        """Infer the sequence-shard dimension from a meta tensor value."""
        if hasattr(meta_val, "dim"):
            rank = meta_val.dim()
            if rank == 2:
                return 0
            if rank >= 3:
                return _SEQ_DIM
        return _SEQ_DIM

    def _get_node_shard_dim(self, node: Node) -> int:
        """Infer the sequence-shard dimension from a node's meta value."""
        if not isinstance(node, Node):
            return _SEQ_DIM
        return self._get_shard_dim_from_meta_val(node.meta.get("val"))

    def _insert_comm_reduce_scatter(
        self,
        graph,
        comm_node: Node,
        rank,
        rank_group,
    ) -> Node:
        """Insert a reduce_scatter that matches the comm input layout."""
        with graph.inserting_after(comm_node):
            return graph.call_function(
                torch.ops.tensor_cast.reduce_scatter.default,
                (
                    comm_node.args[0],
                    self._get_comm_reduce_scatter_dim(comm_node),
                    rank,
                    rank_group,
                ),
            )

    def _get_comm_group_info(self, comm_node: Node, world_size: int):
        """Read rank and rank_group from the original all_reduce node."""
        rank = comm_node.args[1] if len(comm_node.args) > 1 else 0
        rank_group = (
            comm_node.args[2]
            if len(comm_node.args) > 2
            else list(range(world_size))
        )
        return rank, rank_group

    def _get_single_user_if(self, node: Node, predicate) -> Optional[Node]:
        """Return the only user if it matches the predicate."""
        users = list(node.users.keys())
        if len(users) != 1:
            return None
        user = users[0]
        if predicate(user):
            return user
        return None

    def _find_pattern3_norm_after_region_end(
        self, region_end_node: Node
    ) -> Optional[Node]:
        """Follow optional _internal_copy_region nodes to the final tail norm."""
        current_node = region_end_node
        visited = {region_end_node}

        while True:
            norm_node = self._get_single_user_if(
                current_node, self._is_pattern13_norm_op
            )
            if norm_node is not None:
                return norm_node

            next_copy = self._get_single_user_if(
                current_node, self._is_internal_copy_region
            )
            if next_copy is None or next_copy in visited:
                return None

            visited.add(next_copy)
            current_node = next_copy

    def _get_other_add_input(self, add_node: Node, input_node: Node):
        """Return the other add operand if input_node is one of the add operands."""
        if len(add_node.args) < 2:
            return None
        if add_node.args[0] is input_node:
            return add_node.args[1]
        if add_node.args[1] is input_node:
            return add_node.args[0]
        return None

    def _match_comm_to_add_input(
        self, node: Node
    ) -> Optional[Tuple[Node, Node]]:
        """Match a direct comm or comm->view path that feeds add."""
        if self._is_comm_op(node):
            return node, node

        if self._is_view_op(node) and len(node.args) > 0:
            source = node.args[0]
            if isinstance(source, Node) and self._is_comm_op(source):
                return source, node

        return None

    def _match_pattern3_tail_from_local_input(
        self, local_node: Node
    ) -> Optional[Pattern3Match]:
        """Match pattern 3 from the local residual branch."""
        # add_rms_norm2 residual + all_reduce[/view]
        # -> add -> region_end -> [copy_region]* -> rms_norm
        if not self._is_pattern3_residual_getitem(local_node):
            return None

        add_node = self._get_single_user_if(local_node, self._is_add_op)
        if add_node is None:
            return None

        other_input = self._get_other_add_input(add_node, local_node)
        if not isinstance(other_input, Node):
            return None

        comm_match = self._match_comm_to_add_input(other_input)
        if comm_match is None:
            return None
        comm_node, comm_output_node = comm_match

        region_end_node = self._get_single_user_if(
            add_node, self._is_internal_marker_end
        )
        if region_end_node is None:
            return None

        norm_node = self._find_pattern3_norm_after_region_end(region_end_node)
        if norm_node is None:
            return None

        return comm_node, comm_output_node, add_node, region_end_node, norm_node

    def _insert_pattern13_output_all_gather(
        self,
        graph,
        norm_node: Node,
        rank,
        rank_group,
    ) -> None:
        """Gather the single tensor output of a norm op."""
        if self._has_all_gather_user(norm_node):
            return

        shard_dim = self._get_node_shard_dim(norm_node)
        with graph.inserting_after(norm_node):
            all_gather = graph.call_function(
                torch.ops.tensor_cast.all_gather.default,
                (norm_node, shard_dim, rank, rank_group),
            )

        for user in list(norm_node.users.keys()):
            if user != all_gather:
                user.replace_input_with(norm_node, all_gather)

    def _insert_pattern2_output_all_gathers(
        self,
        graph,
        norm_node: Node,
        rank,
        rank_group,
    ) -> None:
        """Gather each tensor output produced by a tuple-valued norm op."""
        for user in list(norm_node.users.keys()):
            if not self._is_getitem_node(user, norm_node):
                continue

            if self._match_pattern3_tail_from_local_input(user) is not None:
                continue

            getitem_users = list(user.users.keys())
            if self._has_all_gather_user(user):
                continue

            with graph.inserting_after(user):
                all_gather = graph.call_function(
                    torch.ops.tensor_cast.all_gather.default,
                    (user, _SEQ_DIM, rank, rank_group),
                )

            for getitem_user in getitem_users:
                if getitem_user != all_gather:
                    getitem_user.replace_input_with(user, all_gather)

    def _rewrite_pattern2_comm_input_to_reduce_scatter(
        self,
        graph,
        comm_node: Node,
        norm_node: Node,
        rank,
        rank_group,
    ) -> Node:
        """Replace the all_reduce input of add_rms_norm2 with reduce_scatter."""
        for arg in norm_node.args[:2]:
            if isinstance(arg, Node) and self._is_reduce_scatter(arg):
                return arg

        reduce_scatter = self._insert_comm_reduce_scatter(
            graph, comm_node, rank, rank_group
        )

        norm_node.replace_input_with(comm_node, reduce_scatter)
        return reduce_scatter

    def _rewrite_pattern2(
        self,
        graph,
        marker_node: Node,
        comm_node: Node,
        norm_node: Node,
        world_size: int,
    ) -> None:
        """Rewrite pattern 2 into local add_rms_norm2 with selective output gathers."""
        # region_begin + all_reduce -> add_rms_norm2
        # => region_begin + reduce_scatter -> add_rms_norm2, gather non-tail outputs
        if len(comm_node.args) == 0:
            logger.debug(
                "Skip FlashCommV1 pattern 2 because comm_node has already been erased"
            )
            return

        rank, rank_group = self._get_comm_group_info(comm_node, world_size)

        self._rewrite_pattern2_comm_input_to_reduce_scatter(
            graph, comm_node, norm_node, rank, rank_group
        )
        self._insert_pattern2_output_all_gathers(graph, norm_node, rank, rank_group)

        logger.debug(
            "Transformed FlashCommV1 pattern 2: "
            f"{marker_node.target} + {comm_node.target} → {norm_node.target} "
            f"with world_size={world_size}"
        )

    def _rewrite_pattern1(
        self,
        graph,
        comm_node: Node,
        marker_node: Node,
        norm_node: Node,
        world_size: int,
    ) -> None:
        """Rewrite pattern 1 into reduce_scatter -> region_begin -> norm -> all_gather."""
        # all_reduce -> region_begin -> rms_norm/add_rms_norm
        # => reduce_scatter -> region_begin -> norm(local) -> all_gather
        if len(comm_node.args) == 0:
            logger.debug(
                "Skip FlashCommV1 pattern 1 because comm_node has already been erased"
            )
            return

        rank, rank_group = self._get_comm_group_info(comm_node, world_size)

        self._get_or_insert_pattern1_reduce_scatter(
            graph, comm_node, marker_node, rank, rank_group
        )

        self._insert_pattern13_output_all_gather(
            graph, norm_node, rank, rank_group
        )

        logger.debug(
            "Transformed FlashCommV1 pattern 1: "
            f"{comm_node.target} → {marker_node.target} → {norm_node.target} "
            f"with world_size={world_size}"
        )

    def _find_pattern3_matches(
        self,
        graph,
    ) -> List[Pattern3Match]:
        """Match pattern 3: pattern2 residual + all_reduce[/view] -> add -> end -> rms_norm."""
        # add_rms_norm2 residual + all_reduce[/view] -> add -> region_end -> rms_norm
        patterns = []
        seen = set()

        for node in graph.nodes:
            if not self._is_pattern3_residual_getitem(node):
                continue

            match = self._match_pattern3_tail_from_local_input(node)
            if match is None:
                continue

            key = tuple(id(n) for n in match)
            if key in seen:
                continue
            seen.add(key)
            patterns.append(match)
            logger.debug(
                "Found FlashCommV1 pattern 3: "
                f"{match[0].target} → {match[2].target} → "
                f"{match[3].target} → {match[4].target}"
            )

        return patterns

    def _rewrite_pattern3(
        self,
        graph,
        comm_node: Node,
        comm_output_node: Node,
        add_node: Node,
        region_end_node: Node,
        norm_node: Node,
        world_size: int,
    ) -> None:
        """Rewrite pattern 3 into local tail add -> region_end -> norm -> all_gather."""
        # add_rms_norm2 residual + all_reduce[/view] -> add -> region_end -> rms_norm
        # => local residual + reduce_scatter[/view] -> add -> region_end -> rms_norm(local) -> all_gather
        if len(comm_node.args) == 0:
            logger.debug(
                "Skip FlashCommV1 pattern 3 because comm_node has already been erased"
            )
            return

        rank, rank_group = self._get_comm_group_info(comm_node, world_size)

        if comm_output_node is comm_node:
            if comm_node not in add_node.all_input_nodes:
                return
        elif comm_node not in comm_output_node.all_input_nodes:
            return

        reduce_scatter = self._insert_comm_reduce_scatter(
            graph, comm_node, rank, rank_group
        )

        if comm_output_node is comm_node:
            add_node.replace_input_with(comm_node, reduce_scatter)
        else:
            comm_output_node.replace_input_with(comm_node, reduce_scatter)

        self._insert_pattern13_output_all_gather(
            graph, norm_node, rank, rank_group
        )
        logger.debug(
            "Transformed FlashCommV1 pattern 3: "
            f"{comm_node.target} → {add_node.target} → "
            f"{region_end_node.target} → {norm_node.target} "
            f"with world_size={world_size}"
        )

"""Decoupled TensorCast adapter for profiling comparison.

This module provides a clean interface between the profiling comparison
tool and TensorCast simulation, avoiding tight coupling.
"""

from dataclasses import dataclass
from typing import Dict, List

from ..config.schema import TensorCastConfig

from .base import BaseParser, OperationData, ProfilingResult


@dataclass
class TensorCastSimulationResult:
    """Result from TensorCast simulation.

    Attributes:
        operations: List of operation data
        total_time_s: Total execution time in seconds
        config: Simulation configuration used
    """

    operations: List[OperationData]
    total_time_s: float
    config: Dict


class TensorCastAdapter(BaseParser):
    """Adapter for running TensorCast simulation and extracting profiling data.

    This adapter decouples the profiling comparison tool from TensorCast
    implementation details.
    """

    def __init__(self, config: TensorCastConfig):
        """Initialize adapter with TensorCast configuration.

        Args:
            config: TensorCast simulation configuration
        """
        self.config = config

    def parse(self) -> ProfilingResult:
        """Run TensorCast simulation and return profiling result.

        Returns:
            ProfilingResult with operation data from simulation
        """
        # Import TensorCast components here to avoid circular imports
        from tensor_cast.core.model_runner import ModelRunner
        from tensor_cast.core.quantization.datatypes import QuantizeLinearAction
        from tensor_cast.core.user_config import UserInputConfig

        print("\n=== Running TensorCast Simulation ===")
        print(f"Model: {self.config.model_id}")
        print(f"Device: {self.config.device}")
        print(
            f"Parallelism: TP={self.config.tp_size}, DP={self.config.dp_size}, EP={self.config.ep}"
        )
        print(f"Quantization: {self.config.quantize_linear_action}")
        print(f"Batch size: {self.config.num_queries}")
        print(f"Query length: {self.config.query_length}")
        print(f"Context length: {self.config.context_length}")

        # Convert quantize_linear_action from string to enum
        quant_action = self.config.quantize_linear_action
        if isinstance(quant_action, str):
            quant_action = QuantizeLinearAction[quant_action]

        # Build user configuration
        user_config = UserInputConfig(
            model_id=self.config.model_id,
            device=self.config.device,
            world_size=self.config.world_size,
            tp_size=self.config.tp_size,
            dp_size=self.config.dp_size,
            ep=self.config.ep,
            quantize_linear_action=quant_action,
            num_queries=self.config.num_queries,
            query_len=self.config.query_length,
            context_length=self.config.context_length,
            word_embedding_tp=self.config.word_embedding_tp,
            lmhead_tp_size=self.config.lmhead_tp_size,
            enable_external_shared_experts=self.config.enable_external_shared_experts,
        )

        # Create model runner
        runner = ModelRunner(user_config)

        # Run simulation with profiling
        operations, total_time_s = self._run_with_profiling(runner)

        print(
            f"TensorCast simulation complete: {len(operations)} unique operations, "
            f"{total_time_s * 1e6:.2f} us total time"
        )

        return ProfilingResult(
            operations=operations,
            total_time_us=total_time_s * 1e6,
            source="tensorcast",
            metadata=self.config.to_dict(),
        )

    def _run_with_profiling(self, runner) -> tuple:
        """Run inference and capture profiling data.

        Args:
            runner: ModelRunner instance

        Returns:
            Tuple of (operations list, total time in seconds)
        """
        import torch

        from tensor_cast.core.input_generator import generate_inputs_varlen
        from tensor_cast.performance_model.memory_tracker import MemoryTracker

        from tensor_cast.runtime import Runtime

        # Prepare inputs
        requests = [runner.user_input.get_request_info()]
        input_kwargs = generate_inputs_varlen(
            runner.model,
            requests,
            block_size=runner.user_input.block_size,
        )

        # Run inference with runtime capture
        print("Running simulated inference...")

        with (
            Runtime(
                runner.perf_model,
                runner.device_profile,
                memory_tracker=MemoryTracker(runner.device_profile),
            ) as runtime,
            torch.no_grad(),
        ):
            _ = runner.model.forward(**input_kwargs)

        # Extract operations from runtime
        operations = self._extract_operations(runtime)

        # Get total time
        perf_model_name = (
            runtime.perf_models[0].name if runtime.perf_models else "unknown"
        )
        execution_time_s = runtime.total_execution_time_s()[perf_model_name]

        # Print table for reference
        table_result = runtime.table_averages(
            group_by_input_shapes=runner.user_input.dump_input_shapes
        )
        print(table_result)

        return operations, execution_time_s

    def _extract_operations(self, runtime) -> List[OperationData]:
        """Extract operation statistics from Runtime event_list.

        Args:
            runtime: Runtime instance with event data

        Returns:
            List of OperationData objects
        """
        from collections import defaultdict

        import torch

        # Aggregate by operation name
        op_stats = defaultdict(
            lambda: {
                "count": 0,
                "total_time": 0.0,
                "input_shapes": [],
                "output_shapes": [],
                "bound_classifications": [],
            }
        )

        # Get performance model name
        perf_model_name = (
            runtime.perf_models[0].name if runtime.perf_models else "unknown"
        )

        for event in runtime.event_list:
            op_info = event.op_invoke_info
            op_name = str(op_info.func)

            # Get execution time from performance results
            perf_result = event.perf_results.get(perf_model_name)
            duration_s = perf_result.execution_time_s if perf_result else 0.0

            # Get input shapes
            input_shapes = []
            for arg in op_info.args:
                if isinstance(arg, torch.Tensor):
                    input_shapes.append(str(list(arg.shape)))

            # Get output shapes
            output_shapes = []
            if isinstance(op_info.out, torch.Tensor):
                output_shapes.append(str(list(op_info.out.shape)))
            elif isinstance(op_info.out, (list, tuple)):
                for out_item in op_info.out:
                    if isinstance(out_item, torch.Tensor):
                        output_shapes.append(str(list(out_item.shape)))

            # Aggregate
            stats = op_stats[op_name]
            stats["count"] += 1
            stats["total_time"] += duration_s
            if input_shapes and not stats["input_shapes"]:
                stats["input_shapes"] = input_shapes
            if output_shapes and not stats["output_shapes"]:
                stats["output_shapes"] = output_shapes

            # Track bound classification
            if perf_result and hasattr(perf_result, "bound_classification"):
                stats["bound_classifications"].append(perf_result.bound_classification)

        # Convert to OperationData objects
        operations = []
        for op_name, stats in op_stats.items():
            # Determine most common bound classification
            if stats["bound_classifications"]:
                bound_classification = max(
                    set(stats["bound_classifications"]),
                    key=stats["bound_classifications"].count,
                )
            else:
                bound_classification = "unknown"

            count = stats["count"]
            total_time_us = stats["total_time"] * 1e6

            op = OperationData(
                op_type=op_name,
                count=count,
                total_time_us=total_time_us,
                avg_time_us=total_time_us / count if count > 0 else 0.0,
                input_shapes=stats["input_shapes"],
                output_shapes=stats["output_shapes"],
                bound_classification=bound_classification,
            )
            operations.append(op)

        return operations

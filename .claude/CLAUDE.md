User request:
在进行项目内部分析的时候可以用 `/home/horacehxw/Projects/aiconfigurator`, 最终的文档报告里面请引用项目 Github 仓 （https://github.com/ai-dynamo/aiconfigurator）

## ProfilingPerformanceModel 设计原则

ProfilingPerformanceModel 的设计应尽量贴合实测的 Profiling 算子，需要能和实际的 NPU Kernel 对齐算子和 Shape，不应为了迁就 TensorCast 当前的算子抽象而妥协。未来最好直接从 VLLM 实跑抓取算子图（而非依赖 TensorCast 的 dispatch trace）。
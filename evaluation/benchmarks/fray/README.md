# Fray Concurrency Bug Evaluation

This evaluation tests OpenHands agents' ability to fix concurrency bugs in Java programs using the Fray systematic concurrency testing tool.

## About Fray

[Fray](https://github.com/cmu-pasta/fray) is a tool for detecting concurrency bugs through systematic exploration of thread interleavings. It uses controlled scheduling to expose race conditions and other concurrency issues.

## Setup

### Prerequisites

1. Follow the [general setup instructions](../../README.md#setup)
2. Build the custom Docker image:

```bash
cd evaluation/benchmarks/fray
docker build -t openhands/fray-eval:v1.0 .
```

This image contains:
- Fray tool (version 0.6.9)
- Pre-cloned fray-benchmark repository from https://github.com/cmu-pasta/fray-benchmark
- Java 21 and Gradle setup

## Running the Evaluation

Basic command:

```bash
./evaluation/benchmarks/fray/scripts/run_infer.sh [model_config] [git-version] [eval_limit] [num_workers]
```

### Parameters

- `model_config` (required): LLM configuration from `config.toml` (e.g., `eval_gpt4_1106_preview`)
- `git-version` (optional): OpenHands version/commit (default: `HEAD`)
- `eval_limit` (optional): Number of instances to evaluate (default: all)
- `num_workers` (optional): Parallel workers (default: 1)

### Examples

```bash
# Run with GPT-4 on all instances
./evaluation/benchmarks/fray/scripts/run_infer.sh eval_gpt4_1106_preview

# Run on single instance with specific version
./evaluation/benchmarks/fray/scripts/run_infer.sh eval_gpt4_1106_preview HEAD 1 1

# Run with parallel workers
./evaluation/benchmarks/fray/scripts/run_infer.sh eval_gpt4_1106_preview HEAD 5 4
```

## Direct Python Execution

```bash
export PYTHONPATH=$(pwd)
poetry run python ./evaluation/benchmarks/fray/run_infer.py \
    --llm-config eval_gpt4_1106_preview \
    --max-iterations 30 \
    --eval-num-workers 1 \
    --eval-n-limit 1
```

## Success Criteria

An instance is marked successful if ALL of the following are true:

1. **Assertion Present**: The test assertion is not commented out or removed
2. **Code Compiles**: The fixed code compiles without errors using `./gradlew compileJava`
3. **Fray Passes**: Fray tests pass with 1000 iterations

## Current Dataset

The dataset is stored in `dataset.jsonl`. Each line is a JSON object with fields:
- `instance_id`: Unique identifier (e.g., "Reorder3Bad")
- `file_path`: Path to the Java file relative to fray-benchmark root
- `description`: Description of the concurrency bug
- `benchmark_category`: Benchmark suite (e.g., "sctbench")
- `subcategory`: Sub-category within the benchmark

### Current Instances

- **Reorder3Bad.java**: Memory ordering bug with concurrent reads and writes to volatile variables

### Adding More Instances

To add more instances, simply append new lines to `dataset.jsonl`:

```bash
echo '{"instance_id": "WronglockBad", "file_path": "src/main/java/cmu/pasta/fray/benchmark/sctbench/cs/origin/WronglockBad.java", "description": "Wrong lock protecting shared data", "benchmark_category": "sctbench", "subcategory": "cs/origin"}' >> evaluation/benchmarks/fray/dataset.jsonl
```

## Evaluation Output

Results are saved to `evaluation/evaluation_outputs/outputs/fray/*/output.jsonl`

Each result includes:
```json
{
  "instance_id": "Reorder3Bad",
  "test_result": {
    "assertion_present": true,
    "code_compiles": true,
    "fray_passes": true,
    "fray_exit_code": 0,
    "fray_output": "...",
    "error": null,
    "success": true
  },
  "metrics": {...},
  "history": [...]
}
```

## Troubleshooting

### Docker Image Build Fails

Ensure you can pull the base image:
```bash
docker pull ghcr.io/cmu-pasta/fray:0.6.9
```

### Gradle Commands Timeout

The default timeout is 600 seconds (10 minutes). If you need more time, edit `run_infer.py` and increase `sandbox_config.timeout`.

### Fray Tests Always Fail

1. Check if the bug is actually fixed
2. Verify the iteration count is sufficient (default: 1000)
3. Run manually in the Docker container to debug:
```bash
docker run -it --rm openhands/fray-eval:v1.0 bash
cd /workspace/fray-benchmark
./gradlew test --tests cmu.pasta.fray.benchmark.sctbench.cs.origin.Reorder3Bad -Dfray.maxIterations=1000 --no-daemon
```

## References

- [Fray GitHub](https://github.com/cmu-pasta/fray)
- [Fray Benchmark GitHub](https://github.com/cmu-pasta/fray-benchmark)
- [SCTBench Paper](https://conf.researchr.org/details/issta-2024/issta-2024-papers/60/SCTBench-Benchmarking-Bounded-Model-Checkers-on-Concurrent-Software)

"""Implements evaluation of agents on Fray concurrency bug fixing.

Fray is a tool for detecting concurrency bugs in Java programs through
systematic concurrency testing. This evaluation measures an agent's ability
to fix concurrency bugs such that Fray tests pass.
"""

import asyncio
import os
import re
from typing import Any

import pandas as pd

from evaluation.benchmarks.fray.dataset import get_fray_dataset
from evaluation.utils.shared import (
    EvalMetadata,
    EvalOutput,
    codeact_user_response,
    compatibility_for_eval_history_pairs,
    get_default_sandbox_config_for_eval,
    get_metrics,
    get_openhands_config_for_eval,
    make_metadata,
    prepare_dataset,
    reset_logger_for_multiprocessing,
    run_evaluation,
)
from openhands.controller.state.state import State
from openhands.core.config import (
    OpenHandsConfig,
    get_evaluation_parser,
    get_llm_config_arg,
)
from openhands.core.logger import openhands_logger as logger
from openhands.core.main import create_runtime, run_controller
from openhands.events.action import CmdRunAction, MessageAction
from openhands.events.observation import CmdOutputObservation
from openhands.runtime.base import Runtime
from openhands.utils.async_utils import call_async_from_sync

# Constants
FRAY_ITERATION_COUNT = 1000  # High iteration threshold for thorough testing
FRAY_BENCHMARK_ROOT = '/workspace/fray-benchmark'

AGENT_CLS_TO_FAKE_USER_RESPONSE_FN = {
    'CodeActAgent': codeact_user_response,
}

AGENT_CLS_TO_INST_SUFFIX = {
    'CodeActAgent': 'When you think you have fixed the concurrency bug, please finish the interaction using the "finish" tool.\n'
}


def get_config(
    metadata: EvalMetadata,
) -> OpenHandsConfig:
    """Get OpenHands configuration with custom Fray Docker image."""
    sandbox_config = get_default_sandbox_config_for_eval()
    # Use runtime_container_image instead of base_container_image
    # This tells OpenHands to use the image directly without building on top of it
    sandbox_config.runtime_container_image = 'openhands/fray-eval:v1.0'
    # Increase timeout for Gradle builds and Fray testing
    sandbox_config.timeout = 600  # 10 minutes

    config = get_openhands_config_for_eval(
        metadata=metadata,
        runtime='docker',
        sandbox_config=sandbox_config,
    )
    config.set_llm_config(metadata.llm_config)
    agent_config = config.get_agent_config(metadata.agent_class)
    agent_config.enable_prompt_extensions = False
    return config


def initialize_runtime(
    runtime: Runtime,
    instance: pd.Series,
):
    """Initialize the runtime for the agent.

    Verifies the fray-benchmark repository is present and working.
    """
    logger.info(f'{"-" * 50} BEGIN Runtime Initialization Fn {"-" * 50}')
    obs: CmdOutputObservation

    # Verify fray-benchmark exists
    action = CmdRunAction(command=f'ls {FRAY_BENCHMARK_ROOT}')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    assert obs.exit_code == 0, f'fray-benchmark directory not found: {obs.content}'

    # Change to benchmark directory
    action = CmdRunAction(command=f'cd {FRAY_BENCHMARK_ROOT}')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    assert obs.exit_code == 0

    # Verify the target file exists
    file_path = os.path.join(FRAY_BENCHMARK_ROOT, instance['file_path'])
    action = CmdRunAction(command=f'ls {file_path}')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    assert obs.exit_code == 0, f'Target file not found: {file_path}'

    # Read the file to verify it has assertions
    action = CmdRunAction(command=f'cat {file_path}')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    if obs.exit_code != 0:
        logger.warning(f'Could not read file {file_path}')
    elif 'assert' not in obs.content.lower():
        logger.warning(f'No assertions found in {file_path}')

    logger.info(f'{"-" * 50} END Runtime Initialization Fn {"-" * 50}')


def complete_runtime(
    runtime: Runtime,
    instance: pd.Series,
) -> dict[str, Any]:
    """Complete the runtime and run verification checks.

    This runs after the agent finishes and checks:
    1. Assertion is still present (not commented out)
    2. Code compiles
    3. Fray tests pass with high iteration count
    """
    logger.info(f'{"-" * 50} BEGIN Runtime Completion Fn {"-" * 50}')

    test_result = {
        'assertion_present': False,
        'code_compiles': False,
        'fray_passes': False,
        'fray_exit_code': None,
        'fray_output': '',
        'error': None,
    }

    file_path = os.path.join(FRAY_BENCHMARK_ROOT, instance['file_path'])

    # Check 1: Verify assertion is still present (not commented out)
    action = CmdRunAction(command=f'cat {file_path}')
    obs = runtime.run_action(action)

    if obs.exit_code == 0:
        file_content = obs.content
        # Look for assertion patterns that are NOT commented out
        # Check for common Java assertion patterns
        assertion_patterns = [
            r'^\s*assert\s+',  # assert statement
            r'^\s*assertTrue\(',  # JUnit assertTrue
            r'^\s*assertEquals\(',  # JUnit assertEquals
            r'^\s*assertNotNull\(',  # JUnit assertNotNull
            r'^\s*assertFalse\(',  # JUnit assertFalse
        ]

        has_assertion = False
        for line in file_content.split('\n'):
            # Skip commented lines
            stripped = line.strip()
            if stripped.startswith('//'):
                continue
            if stripped.startswith('/*') or stripped.startswith('*'):
                continue

            # Check for assertion patterns
            for pattern in assertion_patterns:
                if re.search(pattern, line):
                    has_assertion = True
                    logger.info(f'Found assertion: {line.strip()}')
                    break
            if has_assertion:
                break

        test_result['assertion_present'] = has_assertion
        logger.info(f'Assertion present: {has_assertion}')
    else:
        test_result['error'] = f'Failed to read file: {obs.content}'
        logger.error(test_result['error'])
        logger.info(f'{"-" * 50} END Runtime Completion Fn {"-" * 50}')
        return test_result

    # Check 2: Verify code compiles
    # Extract benchmark directory from file path (e.g., bms/SCTBench)
    # file_path is like: bms/SCTBench/src/main/java/...
    benchmark_dir = '/'.join(instance['file_path'].split('/')[:2])
    benchmark_path = os.path.join(FRAY_BENCHMARK_ROOT, benchmark_dir)

    action = CmdRunAction(
        command=f'cd {benchmark_path} && ./gradlew compileJava --no-daemon'
    )
    logger.info(f'Running compilation: {action.command}')
    obs = runtime.run_action(action)
    test_result['code_compiles'] = (obs.exit_code == 0)
    logger.info(f'Code compiles: {test_result["code_compiles"]}')

    if not test_result['code_compiles']:
        test_result['error'] = f'Compilation failed: {obs.content[-1000:]}'  # Last 1000 chars
        logger.error(test_result['error'])
        logger.info(f'{"-" * 50} END Runtime Completion Fn {"-" * 50}')
        return test_result

    # Check 3: Run Fray tests with high iteration count
    # Extract class path from file path
    # e.g., bms/SCTBench/src/main/java/cmu/pasta/fray/benchmark/sctbench/cs/origin/Reorder3Bad.java
    # -> cmu.pasta.fray.benchmark.sctbench.cs.origin.Reorder3Bad
    relative_path = instance['file_path'].split('src/main/java/')[-1].replace('.java', '').replace('/', '.')

    # Run Fray test with high iteration count
    fray_command = (
        f'cd {benchmark_path} && '
        f'./gradlew test --tests {relative_path} '
        f'-Dfray.maxIterations={FRAY_ITERATION_COUNT} '
        f'--no-daemon'
    )

    logger.info(f'Running Fray test: {fray_command}')
    action = CmdRunAction(command=fray_command)
    obs = runtime.run_action(action)

    test_result['fray_exit_code'] = obs.exit_code
    test_result['fray_output'] = obs.content[-2000:]  # Last 2000 chars for debugging
    test_result['fray_passes'] = (obs.exit_code == 0)

    logger.info(f'Fray test exit code: {obs.exit_code}')
    logger.info(f'Fray test passes: {test_result["fray_passes"]}')

    # Overall success requires all three checks
    test_result['success'] = (
        test_result['assertion_present']
        and test_result['code_compiles']
        and test_result['fray_passes']
    )

    logger.info(f'Overall success: {test_result["success"]}')
    logger.info(f'{"-" * 50} END Runtime Completion Fn {"-" * 50}')
    return test_result


def process_instance(
    instance: pd.Series,
    metadata: EvalMetadata,
    reset_logger: bool = True,
) -> EvalOutput:
    """Process a single evaluation instance."""
    config = get_config(metadata)

    if reset_logger:
        log_dir = os.path.join(metadata.eval_output_dir, 'infer_logs')
        reset_logger_for_multiprocessing(logger, instance['instance_id'], log_dir)
    else:
        logger.info(f'Starting evaluation for instance {instance["instance_id"]}.')

    file_path = os.path.join(FRAY_BENCHMARK_ROOT, instance['file_path'])
    instruction = (
        f'This Java program has a concurrency bug. Fix it so Fray tests pass.\n\n'
        f'The buggy file is located at: {file_path}\n\n'
        f'To test your fix, run: cd {FRAY_BENCHMARK_ROOT} && ./gradlew frayTest\n\n'
        f'IMPORTANT: You should NOT comment out or remove any assertions in the code.\n'
        f'IMPORTANT: You should ONLY interact with the environment provided to you AND NEVER ASK FOR HUMAN HELP.\n'
    )

    # Add agent-specific suffix
    instruction += AGENT_CLS_TO_INST_SUFFIX[metadata.agent_class]

    # Create and initialize runtime
    runtime = create_runtime(config)
    call_async_from_sync(runtime.connect)
    initialize_runtime(runtime, instance)

    state: State | None = asyncio.run(
        run_controller(
            config=config,
            initial_user_action=MessageAction(content=instruction),
            runtime=runtime,
            fake_user_response_fn=AGENT_CLS_TO_FAKE_USER_RESPONSE_FN.get(
                metadata.agent_class
            ),
        )
    )

    if state is None:
        raise ValueError('State should not be None.')

    # Get metrics
    metrics = get_metrics(state)

    # Run verification
    test_result = complete_runtime(runtime, instance)

    # Convert history to compatible format
    histories = compatibility_for_eval_history_pairs(state.history)

    # Save the output
    output = EvalOutput(
        instance_id=instance['instance_id'],
        instruction=instruction,
        metadata=metadata,
        history=histories,
        metrics=metrics,
        error=state.last_error if state and state.last_error else None,
        test_result=test_result,
    )

    runtime.close()
    return output


if __name__ == '__main__':
    parser = get_evaluation_parser()
    args, _ = parser.parse_known_args()

    # Load dataset
    dataset = get_fray_dataset()

    # Get LLM config
    llm_config = None
    if args.llm_config:
        llm_config = get_llm_config_arg(args.llm_config)
        llm_config.modify_params = False
    if llm_config is None:
        raise ValueError(f'Could not find LLM config: --llm_config {args.llm_config}')

    # Create metadata
    metadata = make_metadata(
        llm_config,
        'fray',
        args.agent_cls,
        args.max_iterations,
        args.eval_note,
        args.eval_output_dir,
    )

    # Prepare output
    output_file = os.path.join(metadata.eval_output_dir, 'output.jsonl')
    instances = prepare_dataset(dataset, output_file, args.eval_n_limit)

    # Run evaluation
    run_evaluation(
        instances,
        metadata,
        output_file,
        args.eval_num_workers,
        process_instance,
    )

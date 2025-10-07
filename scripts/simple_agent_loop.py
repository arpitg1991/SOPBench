"""Simple RL-style agent loop that follows SOP constraints.

This module provides a minimal environment and agent implementation that can
consume the Standard Operating Procedure (SOP) instructions shipped with
SOPBench tasks.  The goal is to demonstrate how an RL pipeline can interact
with the benchmark without requiring the full multi-agent orchestration
provided by ``swarm``.

The environment loads a task, exposes the SOP constraints as an ordered action
list, and executes those actions against the strict domain simulator.  The
agent can either follow the SOP deterministically or delegate action selection
to an LLM via :class:`~swarm.llm_handler.OpenAIHandler`.  After an episode
finishes the environment verifies whether the trajectory both respected the
SOP and achieved the expected outcome.

The script can be executed directly to run a single episode for inspection::

    python scripts/simple_agent_loop.py --domain bank --task-index 0

The resulting printed summary can be plugged into a larger RL training loop or
used as a sanity check for integration tests.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from env.task import task_default_dep_full, task_initializer
from env.variables import domain_keys


def _load_domain_tasks(domain: str, data_dir: Path) -> List[Dict[str, Any]]:
    """Load and flatten all tasks for ``domain`` from ``data_dir``.

    The JSON files in :mod:`data` map each user goal to a list of task
    instances.  This helper flattens the structure into a single list so that a
    caller can index tasks deterministically.
    """

    domain_file = data_dir / f"{domain}_tasks.json"
    if not domain_file.exists():
        raise FileNotFoundError(f"No task file found for domain '{domain}' at {domain_file}")

    with domain_file.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    tasks: List[Dict[str, Any]] = []
    for goal_tasks in data.values():
        tasks.extend(goal_tasks)
    if not tasks:
        raise ValueError(f"Task file '{domain_file}' does not contain any tasks")
    return tasks


def _lookup_nested(container: Any, key: str) -> Optional[Any]:
    """Recursively search ``container`` for ``key`` and return the first value found."""

    if isinstance(container, dict):
        if key in container:
            return container[key]
        for value in container.values():
            found = _lookup_nested(value, key)
            if found is not None:
                return found
    elif isinstance(container, (list, tuple, set)):
        for item in container:
            found = _lookup_nested(item, key)
            if found is not None:
                return found
    return None


def _resolve_argument(value: Any, context: Dict[str, Dict[str, Any]]) -> Any:
    """Resolve a placeholder ``value`` using the task ``context``.

    The environment exposes three distinct dictionaries: information known to
    the user, constraint parameters, and the initial database state.  Whenever
    the SOP uses a symbolic placeholder (for example ``"username"``) we map it
    to the concrete value stored in those dictionaries.  If no mapping is
    discovered we conservatively treat the placeholder as a literal string so
    that domain calls still succeed when the value was already concrete.
    """

    if not isinstance(value, str):
        return value

    for source in ("user_known", "constraint_parameters"):
        if value in context.get(source, {}):
            return context[source][value]

    database = context.get("initial_database", {})
    found = _lookup_nested(database, value)
    if found is not None:
        return found

    return value


@dataclass
class ActionSpec:
    """Specification for a single action derived from SOP constraints."""

    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


class _LLMPlanner:
    """Thin wrapper around :class:`OpenAIHandler` for single-agent rollouts."""

    def __init__(
        self,
        model_name: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        tool_call_mode: str,
    ) -> None:
        from swarm.llm_handler import OpenAIHandler

        self.handler = OpenAIHandler(
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            tool_calling=True,
        )
        self.tool_call_mode = tool_call_mode
        self.messages: List[Dict[str, Any]] = []
        self.assistant_tools: List[Dict[str, Any]] = []
        self.execution_log: List[Dict[str, Any]] = []
        self.state_snapshot: Dict[str, Any] = {}

    def reset(self, initial_state: Dict[str, Any]) -> None:
        """Seed the conversation with the SOP instructions and user prompt."""

        self.assistant_tools = initial_state.get("assistant_tools", [])
        self.execution_log = []
        self.state_snapshot = {
            "user_known": initial_state.get("user_known", {}),
            "constraint_parameters": initial_state.get("constraint_parameters", {}),
            "initial_database": initial_state.get("initial_database", {}),
        }

        system_prompt = initial_state.get("instructions", "")
        extra_guidance = (
            "\nYou must follow the SOP actions before attempting the goal. "
            "Always respond by calling exactly one tool; do not return free-form text."
        )
        self.messages = [{"role": "system", "content": system_prompt + extra_guidance}]

        user_prompt = initial_state.get("user_prompt", "")
        if user_prompt:
            self.messages.append({"role": "user", "content": user_prompt})

    def select_action(self, state: Dict[str, Any]) -> Tuple[str, Dict[str, Any], Optional[str]]:
        """Query the LLM for the next tool call based on ``state``."""

        summary = self._format_state(state)
        self.messages.append({"role": "user", "content": summary})

        request = {
            "messages": copy.deepcopy(self.messages),
            "tools": self.assistant_tools,
            "temperature": self.handler.temperature,
            "top_p": self.handler.top_p,
            "max_tokens": self.handler.max_tokens,
        }
        completion = self.handler.chat_completion(request, include_debugging_log=False, tool_call_mode=self.tool_call_mode)[
            "completion"
        ]

        assistant_message, tool_call = self._extract_tool_call(completion)
        self.messages.append(assistant_message)

        if tool_call is None:
            raise RuntimeError("Model response did not include a tool call")

        arguments_raw = tool_call["function"].get("arguments") or "{}"
        try:
            arguments = json.loads(arguments_raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse tool arguments: {arguments_raw}") from exc

        return tool_call["function"]["name"], arguments, tool_call.get("id")

    def observe(self, action_name: str, info: Dict[str, Any], tool_call_id: Optional[str]) -> None:
        """Append the tool execution result and update local history."""

        payload = {
            "success": info.get("success"),
            "payload": info.get("payload"),
        }
        tool_message = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": action_name,
            "content": json.dumps(payload, ensure_ascii=False, default=str),
        }
        self.messages.append(tool_message)

        self.execution_log.append(
            {
                "action": action_name,
                "success": info.get("success"),
                "payload": info.get("payload"),
            }
        )

    def _format_state(self, state: Dict[str, Any]) -> str:
        """Create a textual description of the environment state."""

        pending_specs = [
            {"name": getattr(spec, "name", str(spec)), "arguments": getattr(spec, "arguments", {})}
            for spec in state.get("pending_actions", [])
        ]

        context = {
            "goal": state.get("goal"),
            "goal_arguments": state.get("goal_arguments", {}),
            "pending_actions": pending_specs,
            "executed_actions": self.execution_log,
            "known_facts": self.state_snapshot,
        }

        return (
            "Follow the SOP and decide the next tool call. "
            "Here is the current structured state:\n" + json.dumps(context, indent=2, ensure_ascii=False, default=str)
        )

    def _extract_tool_call(self, completion: Any) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """Normalise the assistant message and associated tool call."""

        if hasattr(completion, "choices"):
            choice = completion.choices[0]
            message_obj = choice.message
            tool_calls_raw = getattr(message_obj, "tool_calls", None) or []
            assistant_message = {
                "role": message_obj.role,
                "content": getattr(message_obj, "content", None) or "",
            }
            if tool_calls_raw:
                assistant_message["tool_calls"] = [
                    {
                        "id": tool_call.id,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    for tool_call in tool_calls_raw
                ]
            tool_call = assistant_message.get("tool_calls", [None])[0]
        else:
            choice = completion["choices"][0]
            message_obj = choice["message"]
            assistant_message = {
                "role": message_obj.get("role", "assistant"),
                "content": message_obj.get("content", ""),
            }
            if message_obj.get("tool_calls"):
                assistant_message["tool_calls"] = message_obj["tool_calls"]
            tool_call = (assistant_message.get("tool_calls") or [None])[0]

        return assistant_message, tool_call

def _extract_constraint_actions(
    dependency: Iterable[Any],
    context: Dict[str, Dict[str, Any]],
) -> List[ActionSpec]:
    """Flatten a dependency tree into an ordered list of :class:`ActionSpec`.

    Each ``single`` node corresponds to a concrete action that the SOP mandates
    before attempting the target user goal.  Logical operators (``and``, ``or``,
    ``chain``, ``gate``) are unfolded depth-first.  Negated constraints (``not``)
    are recorded by flipping the expected boolean value.
    """

    actions: List[ActionSpec] = []

    if not dependency:
        return actions

    kind = dependency[0]
    if kind == "single":
        raw_name = dependency[1]
        arguments_template = dependency[2] if len(dependency) > 2 and dependency[2] else {}
        negated = raw_name.startswith("not ")
        name = raw_name[4:] if negated else raw_name
        arguments = {
            key: _resolve_argument(value, context) for key, value in arguments_template.items()
        }
        actions.append(ActionSpec(name=name, arguments=arguments))
        return actions

    if kind in {"and", "chain", "gate", "or"}:
        for child in dependency[1]:
            actions.extend(_extract_constraint_actions(child, context))
        return actions

    raise ValueError(f"Unsupported dependency kind: {kind}")


def _unpack_response(response: Any) -> Tuple[bool, Any]:
    """Normalize tool responses to ``(success, payload)`` tuples."""

    if isinstance(response, tuple):
        if not response:
            return False, None
        first = response[0]
        if isinstance(first, bool):
            success = first
            payload = response[1] if len(response) > 1 else None
        else:
            success = bool(first)
            payload = response
        return success, payload

    return bool(response), response


class SimpleSOPEnvironment:
    """Environment that executes SOP actions against the strict simulator."""

    def __init__(
        self,
        domain: str,
        tasks: List[Dict[str, Any]],
        default_constraint_option: str = "full",
        constraint_descr_format: str = "structured",
    ) -> None:
        self.domain = domain
        self.tasks = tasks
        self.default_constraint_option = default_constraint_option
        self.constraint_descr_format = constraint_descr_format
        self.task: Optional[Dict[str, Any]] = None
        self.instructions: Optional[str] = None
        self.goal_arguments: Dict[str, Any] = {}
        self.required_actions: List[ActionSpec] = []
        self.pending_actions: List[ActionSpec] = []
        self.history: List[Dict[str, Any]] = []
        self.system = None
        self.assistant_tools: List[Dict[str, Any]] = []
        self.task_prompt: str = ""
        self.user_instructions: str = ""

    def reset(self, task_index: int) -> Dict[str, Any]:
        """Reset the environment to ``task_index`` and return the initial state."""

        if not (0 <= task_index < len(self.tasks)):
            raise IndexError(f"Task index {task_index} is out of range for {len(self.tasks)} tasks")

        self.task = copy.deepcopy(self.tasks[task_index])

        dep_innate_full, default_dep_full, default_dep_full_descr = task_default_dep_full(
            self.domain,
            self.default_constraint_option,
            self.constraint_descr_format,
        )

        # Gather natural-language instructions from the prompt-based initializer.
        _, user_info_prompt, assistant_info_prompt, _ = task_initializer(
            self.domain,
            self.task,
            dep_innate_full,
            default_dep_full,
            default_dep_full_descr,
            included_functions=None,
            mode="prompt",
            shuffle_func=False,
            constraint_descr_format=self.constraint_descr_format,
        )
        self.instructions = assistant_info_prompt["instructions"]
        self.assistant_tools = assistant_info_prompt.get("tools", [])
        self.task_prompt = self.task.get("user_prompt", "")
        self.user_instructions = user_info_prompt.get("instructions", "")

        # Instantiate the strict simulator so that constraint violations surface
        # as failed actions.
        dep_full = copy.deepcopy(default_dep_full)
        dep_full[self.task["user_goal"]] = self.task["constraints"]
        self.system = domain_keys[f"{self.domain}_strict"](
            copy.deepcopy(self.task["initial_database"]),
            dep_innate_full,
            dep_full,
            self.task["constraint_parameters"],
        )

        context = {
            "user_known": self.task.get("user_known", {}),
            "constraint_parameters": self.task.get("constraint_parameters", {}),
            "initial_database": self.task.get("initial_database", {}),
        }

        self.required_actions = _extract_constraint_actions(self.task.get("constraints"), context)
        self.pending_actions = copy.deepcopy(self.required_actions)

        # Determine the arguments for the user goal using the directed action graph.
        self.goal_arguments = {}
        dag = self.task.get("directed_action_graph", {})
        for node in dag.get("nodes", []):
            if isinstance(node, list) and node and node[0] == self.task["user_goal"]:
                arguments_template = node[1] if len(node) > 1 and isinstance(node[1], dict) else {}
                self.goal_arguments = {
                    key: _resolve_argument(value, context) for key, value in arguments_template.items()
                }
                break

        self.history = []

        return {
            "instructions": self.instructions,
            "pending_actions": copy.deepcopy(self.pending_actions),
            "goal": self.task["user_goal"],
            "goal_arguments": copy.deepcopy(self.goal_arguments),
            "assistant_tools": copy.deepcopy(self.assistant_tools),
            "user_prompt": self.task_prompt,
            "user_instructions": self.user_instructions,
            "user_known": copy.deepcopy(self.task.get("user_known", {})),
            "constraint_parameters": copy.deepcopy(self.task.get("constraint_parameters", {})),
            "initial_database": copy.deepcopy(self.task.get("initial_database", {})),
        }

    def step(self, action_name: str, action_arguments: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute ``action_name`` with ``action_arguments`` against the simulator."""

        if self.task is None:
            raise RuntimeError("Environment must be reset before stepping")

        if action_arguments is None:
            action_arguments = {}

        if not hasattr(self.system, action_name):
            raise AttributeError(f"Domain system does not expose action '{action_name}'")

        raw_response = getattr(self.system, action_name)(**action_arguments)
        success, payload = _unpack_response(raw_response)

        if self.pending_actions and self.pending_actions[0].name == action_name:
            self.pending_actions.pop(0)

        self.history.append(
            {
                "action": action_name,
                "arguments": action_arguments,
                "success": success,
                "payload": payload,
            }
        )

        done = action_name == self.task["user_goal"]
        reward = 1.0 if done and success else 0.0

        state = {
            "instructions": self.instructions,
            "pending_actions": copy.deepcopy(self.pending_actions),
            "goal": self.task["user_goal"],
            "goal_arguments": copy.deepcopy(self.goal_arguments),
            "assistant_tools": copy.deepcopy(self.assistant_tools),
            "user_prompt": self.task_prompt,
            "user_instructions": self.user_instructions,
            "user_known": copy.deepcopy(self.task.get("user_known", {})),
            "constraint_parameters": copy.deepcopy(self.task.get("constraint_parameters", {})),
            "initial_database": copy.deepcopy(self.task.get("initial_database", {})),
        }

        info = {"success": success, "payload": payload, "raw_response": raw_response}
        return state, reward, done, info

    def verify(self) -> Dict[str, Any]:
        """Verify whether the recorded trajectory respects the SOP and outcome."""

        if self.task is None or not self.history:
            raise RuntimeError("No trajectory available for verification")

        executed = {entry["action"]: entry for entry in self.history}
        sop_compliance = True

        for spec in self.required_actions:
            entry = executed.get(spec.name)
            if entry is None or not entry["success"]:
                sop_compliance = False
                break

        final_entry = self.history[-1]
        actual_success = bool(final_entry["success"])
        expected_success = bool(self.task.get("action_should_succeed", True))

        return {
            "sop_compliance": sop_compliance,
            "expected_success": expected_success,
            "actual_success": actual_success,
            "matches_expectation": actual_success == expected_success,
            "num_steps": len(self.history),
        }


class SimpleSOPAgent:
    """Agent capable of running either a deterministic or LLM-driven policy."""

    def __init__(
        self,
        environment: SimpleSOPEnvironment,
        planner: str = "deterministic",
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        top_p: float = 0.01,
        max_tokens: int = 256,
        tool_call_mode: str = "fc",
    ) -> None:
        self.env = environment
        self.planner_mode = planner

        if planner not in {"deterministic", "llm"}:
            raise ValueError("planner must be either 'deterministic' or 'llm'")

        if self.planner_mode == "llm":
            if not model_name:
                raise ValueError("An assistant model must be provided when using the LLM planner")
            self.planner = _LLMPlanner(
                model_name=model_name,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                tool_call_mode=tool_call_mode,
            )
        else:
            self.planner = None

    def run_episode(self, task_index: int) -> Dict[str, Any]:
        """Execute a full episode on ``task_index`` and return the verification report."""

        state = self.env.reset(task_index)

        if self.planner_mode == "llm":
            assert self.planner is not None
            self.planner.reset(state)

        while True:
            if self.planner_mode == "deterministic":
                if state["pending_actions"]:
                    next_action = state["pending_actions"][0]
                    action_name = next_action.name
                    action_arguments = next_action.arguments
                else:
                    action_name = state["goal"]
                    action_arguments = state["goal_arguments"]
                tool_call_id = None
            else:
                assert self.planner is not None
                action_name, action_arguments, tool_call_id = self.planner.select_action(state)

            state, _, done, info = self.env.step(action_name, action_arguments)

            if self.planner_mode == "llm" and tool_call_id is not None:
                assert self.planner is not None
                self.planner.observe(action_name, info, tool_call_id)

            if done:
                break

        return self.env.verify()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a simplified SOP-compliant agent loop")
    parser.add_argument("--domain", type=str, default="bank", help="Domain name to evaluate")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing task files")
    parser.add_argument("--task-index", type=int, default=0, help="Index of the task to run")
    parser.add_argument(
        "--default-constraint-option",
        type=str,
        default="full",
        choices=["full", "required"],
        help="Default dependency configuration for unused actions",
    )
    parser.add_argument(
        "--constraint-descr-format",
        type=str,
        default="structured",
        choices=["old", "structured"],
        help="Formatting used when verbalising constraints",
    )
    parser.add_argument(
        "--planner",
        type=str,
        default="deterministic",
        choices=["deterministic", "llm"],
        help="Execution policy to use for choosing actions",
    )
    parser.add_argument(
        "--assistant-model",
        type=str,
        default=None,
        help="Model name to query when planner=llm",
    )
    parser.add_argument(
        "--assistant-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the assistant model",
    )
    parser.add_argument(
        "--assistant-top-p",
        type=float,
        default=0.01,
        help="Top-p nucleus sampling parameter for the assistant model",
    )
    parser.add_argument(
        "--assistant-max-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to request from the assistant model",
    )
    parser.add_argument(
        "--tool-call-mode",
        type=str,
        default="fc",
        choices=["fc", "react", "act-only", "react-v"],
        help="Strategy used by the handler when calling tools",
    )
    args = parser.parse_args()

    tasks = _load_domain_tasks(args.domain, args.data_dir)
    env = SimpleSOPEnvironment(
        domain=args.domain,
        tasks=tasks,
        default_constraint_option=args.default_constraint_option,
        constraint_descr_format=args.constraint_descr_format,
    )
    agent = SimpleSOPAgent(
        env,
        planner=args.planner,
        model_name=args.assistant_model,
        temperature=args.assistant_temperature,
        top_p=args.assistant_top_p,
        max_tokens=args.assistant_max_tokens,
        tool_call_mode=args.tool_call_mode,
    )

    report = agent.run_episode(args.task_index)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()


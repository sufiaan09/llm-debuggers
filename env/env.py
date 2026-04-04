"""
env.py — LLM Self-Debugging OpenEnv Environment
Implements the full OpenEnv spec: step() / reset() / state()
"""

import ast
import random
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from tasks.tasks import TASKS
from graders.grader import compute_reward


@dataclass
class Action:
    fix_code: str
    explanation: str
    bug_type: str

    VALID_BUG_TYPES = {
        "syntax_error", "shape_mismatch",
        "gradient_issue", "logic_error", "runtime_error"
    }

    def validate(self) -> Tuple[bool, str]:
        if not self.fix_code.strip():
            return False, "fix_code cannot be empty"
        if not self.explanation.strip():
            return False, "explanation cannot be empty"
        if self.bug_type not in self.VALID_BUG_TYPES:
            return False, f"bug_type must be one of {self.VALID_BUG_TYPES}"
        return True, "ok"


@dataclass
class Observation:
    buggy_code: str
    error_message: str
    task_description: str
    test_cases: List[Dict]
    attempt_number: int
    previous_attempts: List[Dict] = field(default_factory=list)
    task_id: str = ""
    task_level: str = ""

    def to_dict(self) -> Dict:
        return {
            "buggy_code": self.buggy_code,
            "error_message": self.error_message,
            "task_description": self.task_description,
            "test_cases": self.test_cases,
            "attempt_number": self.attempt_number,
            "previous_attempts": self.previous_attempts,
            "task_id": self.task_id,
            "task_level": self.task_level,
        }


@dataclass
class StepResult:
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]


class LLMDebuggingEnv:
    MAX_ATTEMPTS = 3
    ATTEMPT_PENALTY = 0.05
    MAX_SCORES = {"easy": 0.4, "medium": 0.7, "hard": 1.0}

    def __init__(self, task_level: str = "easy", seed: Optional[int] = 42):
        assert task_level in ("easy", "medium", "hard")
        self.task_level = task_level
        self.seed = seed
        self._rng = random.Random(seed)
        self._current_task = None
        self._attempt_number = 0
        self._previous_attempts = []
        self._done = False
        self._episode_reward = 0.0

    def reset(self, task_index: Optional[int] = None) -> Dict:
        tasks = TASKS[self.task_level]
        if task_index is not None:
            self._current_task = tasks[task_index % len(tasks)]
        else:
            self._current_task = self._rng.choice(tasks)
        self._attempt_number = 0
        self._previous_attempts = []
        self._done = False
        self._episode_reward = 0.0
        return self.state()

    def step(self, action: Action) -> StepResult:
        assert not self._done, "Episode done. Call reset() first."
        assert self._current_task is not None, "No task loaded. Call reset() first."
        valid, msg = action.validate()
        if not valid:
            raise ValueError(f"Invalid action: {msg}")
        self._attempt_number += 1
        reward_breakdown, total_reward = compute_reward(
            action=action,
            task=self._current_task,
            task_level=self.task_level,
            attempt_number=self._attempt_number,
        )
        max_score = self.MAX_SCORES[self.task_level]
        total_reward = min(total_reward, max_score)
        self._episode_reward = max(self._episode_reward, total_reward)
        attempt_record = {
            "attempt": self._attempt_number,
            "reward": round(total_reward, 4),
            "breakdown": reward_breakdown,
            "bug_type_submitted": action.bug_type,
            "fix_code_preview": action.fix_code[:120],
        }
        self._previous_attempts.append(attempt_record)
        self._done = (
            self._attempt_number >= self.MAX_ATTEMPTS
            or total_reward >= max_score * 0.95
        )
        obs = Observation(
            buggy_code=self._current_task["buggy_code"],
            error_message=self._current_task["error_message"],
            task_description=self._current_task["task_description"],
            test_cases=self._current_task["test_cases"],
            attempt_number=self._attempt_number,
            previous_attempts=self._previous_attempts,
            task_id=self._current_task["id"],
            task_level=self.task_level,
        )
        info = {
            "reward_breakdown": reward_breakdown,
            "episode_best_reward": self._episode_reward,
            "attempts_remaining": self.MAX_ATTEMPTS - self._attempt_number,
            "task_id": self._current_task["id"],
            "task_title": self._current_task["title"],
            "correct_bug_type": self._current_task["bug_type"],
        }
        return StepResult(observation=obs, reward=round(total_reward, 4), done=self._done, info=info)

    def state(self) -> Dict:
        if self._current_task is None:
            return {"status": "not_started", "task_level": self.task_level}
        return Observation(
            buggy_code=self._current_task["buggy_code"],
            error_message=self._current_task["error_message"],
            task_description=self._current_task["task_description"],
            test_cases=self._current_task["test_cases"],
            attempt_number=self._attempt_number,
            previous_attempts=self._previous_attempts,
            task_id=self._current_task["id"],
            task_level=self.task_level,
        ).to_dict()

    @property
    def action_space(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "fix_code": {"type": "string"},
                "explanation": {"type": "string"},
                "bug_type": {"type": "string", "enum": list(Action.VALID_BUG_TYPES)}
            }
        }

    @property
    def observation_space(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "buggy_code": {"type": "string"},
                "error_message": {"type": "string"},
                "task_description": {"type": "string"},
                "test_cases": {"type": "array"},
                "attempt_number": {"type": "integer"},
                "previous_attempts": {"type": "array"},
            }
        }
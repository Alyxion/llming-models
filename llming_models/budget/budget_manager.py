from __future__ import annotations

import logging
import os
import time as _time

from .budget_limit import BudgetLimit
from .budget_types import InsufficientBudgetError

logger = logging.getLogger(__name__)


class LLMBudgetManager:
    """Manages multiple budget limits in Euros for LLM operations.

    Handles budget reservation and tracking for LLM operations,
    ensuring that operations don't exceed allocated monetary limits.
    Budget is tracked in Euros and automatically scales with model-specific costs.

    ``reserve_output_ratio`` controls how conservatively output tokens are
    reserved.  The default (0.25) reserves 25% of ``max_output_tokens`` at
    full price -- enough for ~95% of real responses while avoiding the
    problem where a $2 remaining budget blocks all Opus calls because the
    worst-case reservation ($3.45) exceeds it.  Set to 1.0 to restore the
    old full-worst-case behaviour.

    If actual output exceeds the reservation, the overage is charged
    post-hoc by ``return_unused_budget_async``.
    """
    limits: dict[str, BudgetLimit]
    reserve_output_ratio: float

    def __init__(self, limits: list[BudgetLimit], *,
                 reserve_output_ratio: float = 0.25) -> None:
        self.limits = {limit.name: limit for limit in limits}
        self.reserve_output_ratio = max(0.01, min(1.0, reserve_output_ratio))
        logger.debug(
            "Initialized budget manager (reserve_ratio=%.2f) with limits: %s",
            self.reserve_output_ratio,
            [f"{name} ({limit.period.value}/{limit.scope.value}): {limit.amount}€"
             for name, limit in self.limits.items()],
        )

    async def available_budget_async(self, user_id: str | None = None) -> float:
        """Get the minimum available budget across all limits.

        For limits with PER_USER scope, ``user_id`` is required.
        The most restrictive limit wins.
        """
        budgets: dict[str, float] = {}
        for name, limit in self.limits.items():
            budgets[name] = await limit.get_available_budget_async(user_id=user_id)
        min_budget = min(budgets.values())
        logger.debug("Available budgets (user=%s): %s → min=%.4f€", user_id, budgets, min_budget)
        return min_budget

    async def reserve_budget_async(
        self,
        *,
        input_tokens: int,
        max_output_tokens: int,
        input_token_price: float,
        output_token_price: float,
        user_id: str | None = None,
    ) -> int:
        """Reserve budget for an LLM operation.

        Returns the effective number of output tokens that were reserved
        (after applying ``reserve_output_ratio``).  Pass this value to
        ``return_unused_budget_async`` so the refund is calculated
        correctly.

        Raises:
            InsufficientBudgetError: If there isn't enough budget available
        """
        if input_token_price < 0 or output_token_price < 0:
            raise ValueError("Token prices must be non-negative")
        effective_reserved_output = max(1, int(max_output_tokens * self.reserve_output_ratio))
        input_cost = input_tokens * (input_token_price / 1_000_000)
        output_cost = effective_reserved_output * (output_token_price / 1_000_000)
        required_budget = input_cost + output_cost
        if required_budget <= 0:
            return effective_reserved_output
        _pid = os.getpid()
        logger.info(
            "[OP] budget_reserve — starting (%.6f€, in=%d, reserved_out=%d/%d, ratio=%.2f, user=%s, pid=%d)",
            required_budget, input_tokens, effective_reserved_output, max_output_tokens,
            self.reserve_output_ratio, user_id, _pid,
        )
        _t0 = _time.monotonic()

        reserved_limits: list[str] = []
        try:
            for name, limit in self.limits.items():
                success = await limit.reserve_budget_async(required_budget, user_id=user_id)
                if not success:
                    available = await limit.get_available_budget_async(user_id=user_id)
                    msg = (f"Failed to reserve budget ({required_budget:.4f}€) from limit "
                           f"'{name}' (available: {available:.4f}€)")
                    for reserved_name in reserved_limits:
                        await self.limits[reserved_name].return_budget_async(required_budget, user_id=user_id)
                    raise InsufficientBudgetError(msg, limit_name=name)
                reserved_limits.append(name)
            logger.info(
                "[OP] budget_reserve — done (%.6f€, %.0fms, user=%s, pid=%d)",
                required_budget, (_time.monotonic() - _t0) * 1000, user_id, _pid,
            )
        except InsufficientBudgetError:
            raise
        except Exception as e:
            logger.warning(
                "[OP] budget_reserve — failed (%.6f€, %.0fms, user=%s, pid=%d): %s",
                required_budget, (_time.monotonic() - _t0) * 1000, user_id, _pid, e,
            )
            for name in reserved_limits:
                await self.limits[name].return_budget_async(required_budget, user_id=user_id)
            msg = f"Failed to reserve budget ({required_budget:.4f}€) due to error: {e}"
            raise InsufficientBudgetError(msg, limit_name=next(iter(self.limits.keys())))

        return effective_reserved_output

    async def return_unused_budget_async(
        self,
        *,
        reserved_output_tokens: int,
        actual_output_tokens: int,
        output_token_price: float,
        user_id: str | None = None,
    ) -> None:
        """Return unused budget after an operation completes.

        If the actual output exceeded the reservation (rare with a ratio),
        the overage is charged post-hoc.
        """
        if output_token_price < 0:
            raise ValueError("Token prices must be non-negative")
        _pid = os.getpid()
        _t0 = _time.monotonic()
        if actual_output_tokens > reserved_output_tokens:
            additional_tokens = actual_output_tokens - reserved_output_tokens
            additional_budget = additional_tokens * (output_token_price / 1_000_000)
            logger.info(
                "[OP] budget_return — overuse, reserving additional %.6f€ "
                "(reserved=%d, actual=%d, user=%s, pid=%d)",
                additional_budget, reserved_output_tokens, actual_output_tokens, user_id, _pid,
            )
            for limit in self.limits.values():
                await limit.reserve_budget_async(additional_budget, user_id=user_id)
            return

        unused_tokens = reserved_output_tokens - actual_output_tokens
        unused_budget = unused_tokens * (output_token_price / 1_000_000)

        for limit in self.limits.values():
            await limit.return_budget_async(unused_budget, user_id=user_id)
        logger.info(
            "[OP] budget_return — done (returned=%.6f€, reserved=%d, actual=%d, "
            "%.0fms, user=%s, pid=%d)",
            unused_budget, reserved_output_tokens, actual_output_tokens,
            (_time.monotonic() - _t0) * 1000, user_id, _pid,
        )

    async def reset_async(self) -> None:
        """Reset all budget limits."""
        for limit in self.limits.values():
            await limit.reset_async()

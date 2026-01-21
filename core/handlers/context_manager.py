"""
Context Window Manager

Manages context window limits and validates requests fit within model limits.
Critical for agentic workflows with large contexts.
"""

from __future__ import annotations
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .token_counter import TokenCounter, get_token_counter
from .capabilities import get_capabilities, ModelCapabilities


logger = logging.getLogger("shin-gateway")


# =============================================================================
# Context Limits
# =============================================================================

@dataclass
class ContextLimits:
    """Context window limits for a model"""
    model: str
    max_input_tokens: int
    max_output_tokens: int
    total_context: int

    @property
    def safe_input_limit(self) -> int:
        """Safe input limit leaving room for output"""
        return self.total_context - self.max_output_tokens


# =============================================================================
# Context Manager
# =============================================================================

class ContextManager:
    """
    Manage context window for agentic workflows.

    Provides validation, estimation, and advice for context usage.
    """

    def __init__(self, token_counter: Optional[TokenCounter] = None):
        self.token_counter = token_counter or get_token_counter()

    def get_limits(self, model: str) -> ContextLimits:
        """
        Get context limits for a model.

        Args:
            model: Model name or alias

        Returns:
            ContextLimits for the model
        """
        caps = get_capabilities(model)
        return ContextLimits(
            model=model,
            max_input_tokens=caps.max_context_tokens,
            max_output_tokens=caps.max_output_tokens,
            total_context=caps.max_context_tokens,
        )

    def estimate_input_tokens(self, request: Dict[str, Any]) -> int:
        """
        Estimate input tokens for a request.

        Args:
            request: Request object

        Returns:
            Estimated input token count
        """
        result = self.token_counter.count_request(request)
        return result.input_tokens

    def validate_request(
        self,
        request: Dict[str, Any],
        model: str
    ) -> tuple[bool, Optional[str]]:
        """
        Validate request fits in context window.

        Args:
            request: Request to validate
            model: Target model

        Returns:
            (is_valid, error_message)
        """
        limits = self.get_limits(model)
        estimated_input = self.estimate_input_tokens(request)
        requested_output = request.get("max_tokens", limits.max_output_tokens)
        total_needed = estimated_input + requested_output

        errors = []

        if estimated_input > limits.max_input_tokens:
            errors.append(
                f"Estimated input tokens ({estimated_input:,}) exceeds "
                f"model limit ({limits.max_input_tokens:,})"
            )

        if requested_output > limits.max_output_tokens:
            errors.append(
                f"Requested max_tokens ({requested_output:,}) exceeds "
                f"model limit ({limits.max_output_tokens:,})"
            )

        if total_needed > limits.total_context:
            errors.append(
                f"Total tokens needed ({total_needed:,}) exceeds "
                f"context window ({limits.total_context:,})"
            )

        if errors:
            return False, "; ".join(errors)

        # Warn if close to limit
        usage_ratio = estimated_input / limits.safe_input_limit
        if usage_ratio > 0.9:
            logger.warning(
                f"High context usage: {usage_ratio:.1%} of safe limit",
                extra={
                    "model": model,
                    "estimated_input": estimated_input,
                    "safe_limit": limits.safe_input_limit,
                }
            )

        return True, None

    def get_available_tokens(
        self,
        request: Dict[str, Any],
        model: str
    ) -> int:
        """
        Calculate available tokens for output.

        Args:
            request: Current request
            model: Target model

        Returns:
            Available tokens for output
        """
        limits = self.get_limits(model)
        used = self.estimate_input_tokens(request)
        available = limits.total_context - used
        return min(available, limits.max_output_tokens)

    def get_optimal_max_tokens(
        self,
        request: Dict[str, Any],
        model: str,
        min_tokens: int = 1024
    ) -> int:
        """
        Calculate optimal max_tokens for a request.

        Args:
            request: Current request
            model: Target model
            min_tokens: Minimum tokens to ensure

        Returns:
            Optimal max_tokens value
        """
        available = self.get_available_tokens(request, model)
        limits = self.get_limits(model)

        # Use available tokens but cap at model limit
        optimal = min(available, limits.max_output_tokens)

        # Ensure minimum
        return max(optimal, min_tokens)

    def adjust_request(
        self,
        request: Dict[str, Any],
        model: str
    ) -> Dict[str, Any]:
        """
        Adjust request to fit within context limits.

        Args:
            request: Request to adjust
            model: Target model

        Returns:
            Adjusted request
        """
        limits = self.get_limits(model)
        adjusted = request.copy()

        # Adjust max_tokens if needed
        current_max = request.get("max_tokens", limits.max_output_tokens)
        available = self.get_available_tokens(request, model)

        if current_max > available:
            adjusted["max_tokens"] = max(1024, available)
            logger.info(
                f"Adjusted max_tokens from {current_max} to {adjusted['max_tokens']}",
                extra={"model": model}
            )

        # Cap at model limit
        if adjusted.get("max_tokens", 0) > limits.max_output_tokens:
            adjusted["max_tokens"] = limits.max_output_tokens

        return adjusted

    def get_usage_stats(
        self,
        request: Dict[str, Any],
        model: str
    ) -> Dict[str, Any]:
        """
        Get detailed context usage statistics.

        Args:
            request: Request to analyze
            model: Target model

        Returns:
            Usage statistics dictionary
        """
        limits = self.get_limits(model)
        estimated_input = self.estimate_input_tokens(request)
        requested_output = request.get("max_tokens", limits.max_output_tokens)

        return {
            "model": model,
            "estimated_input_tokens": estimated_input,
            "requested_output_tokens": requested_output,
            "total_estimated": estimated_input + requested_output,
            "max_input_tokens": limits.max_input_tokens,
            "max_output_tokens": limits.max_output_tokens,
            "total_context": limits.total_context,
            "input_usage_percent": round(estimated_input / limits.max_input_tokens * 100, 1),
            "total_usage_percent": round(
                (estimated_input + requested_output) / limits.total_context * 100, 1
            ),
            "available_output_tokens": self.get_available_tokens(request, model),
        }


# =============================================================================
# Global Instance
# =============================================================================

_context_manager: Optional[ContextManager] = None


def get_context_manager() -> ContextManager:
    """Get or create global context manager"""
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextManager()
    return _context_manager

"""
Beta Features Handler

Handles Anthropic beta feature headers and provides compatibility
for features that may not be supported by all providers.
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass
from enum import Enum


logger = logging.getLogger("shin-gateway")


# =============================================================================
# Beta Feature Definitions
# =============================================================================

class FallbackBehavior(Enum):
    """How to handle unsupported beta features"""
    IGNORE = "ignore"      # Silently ignore
    EMULATE = "emulate"    # Try to emulate
    ERROR = "error"        # Return error


@dataclass
class BetaFeature:
    """Beta feature definition"""
    name: str
    full_name: str
    supported: bool
    fallback: FallbackBehavior
    description: str = ""


# Known beta features and how to handle them
BETA_FEATURES: Dict[str, BetaFeature] = {
    "prompt-caching-2024-07-31": BetaFeature(
        name="prompt-caching",
        full_name="prompt-caching-2024-07-31",
        supported=False,
        fallback=FallbackBehavior.IGNORE,
        description="Prompt caching for reduced latency",
    ),
    "max-tokens-3-5-sonnet-2024-07-15": BetaFeature(
        name="extended-output",
        full_name="max-tokens-3-5-sonnet-2024-07-15",
        supported=True,
        fallback=FallbackBehavior.EMULATE,
        description="Extended max_tokens for Claude 3.5 Sonnet",
    ),
    "computer-use-2024-10-22": BetaFeature(
        name="computer-use",
        full_name="computer-use-2024-10-22",
        supported=False,
        fallback=FallbackBehavior.ERROR,
        description="Computer use tools",
    ),
    "token-counting-2024-11-01": BetaFeature(
        name="token-counting",
        full_name="token-counting-2024-11-01",
        supported=True,
        fallback=FallbackBehavior.EMULATE,
        description="Token counting endpoint",
    ),
    "message-batches-2024-09-24": BetaFeature(
        name="batches",
        full_name="message-batches-2024-09-24",
        supported=False,
        fallback=FallbackBehavior.ERROR,
        description="Message batches for async processing",
    ),
    "pdfs-2024-09-25": BetaFeature(
        name="pdfs",
        full_name="pdfs-2024-09-25",
        supported=False,
        fallback=FallbackBehavior.IGNORE,
        description="PDF document support",
    ),
    "interleaved-thinking-2024-10-01": BetaFeature(
        name="interleaved-thinking",
        full_name="interleaved-thinking-2024-10-01",
        supported=False,
        fallback=FallbackBehavior.EMULATE,
        description="Interleaved thinking blocks",
    ),
    # Claude Code specific features
    "extended-thinking-2025-01-24": BetaFeature(
        name="extended-thinking",
        full_name="extended-thinking-2025-01-24",
        supported=False,
        fallback=FallbackBehavior.IGNORE,
        description="Extended thinking/reasoning for Claude Code",
    ),
    "tools-2024-04-04": BetaFeature(
        name="tools",
        full_name="tools-2024-04-04",
        supported=True,
        fallback=FallbackBehavior.EMULATE,
        description="Tool use support",
    ),
    "output-128k-2025-02-19": BetaFeature(
        name="output-128k",
        full_name="output-128k-2025-02-19",
        supported=True,
        fallback=FallbackBehavior.EMULATE,
        description="128k output tokens",
    ),
}


# =============================================================================
# Beta Features Handler
# =============================================================================

class BetaFeaturesHandler:
    """
    Handle Anthropic beta features.

    Parses anthropic-beta headers and determines how to handle
    each requested feature.
    """

    def __init__(self, custom_features: Optional[Dict[str, BetaFeature]] = None):
        self.features = {**BETA_FEATURES}
        if custom_features:
            self.features.update(custom_features)

    def parse_header(self, header_value: Optional[str]) -> List[str]:
        """
        Parse anthropic-beta header value.

        Args:
            header_value: Comma-separated list of beta features

        Returns:
            List of feature names
        """
        if not header_value:
            return []
        return [f.strip() for f in header_value.split(",") if f.strip()]

    def validate_features(
        self,
        features: List[str]
    ) -> tuple[Set[str], List[str], List[str]]:
        """
        Validate requested beta features.

        Args:
            features: List of requested feature names

        Returns:
            (supported_features, warnings, errors)
        """
        supported = set()
        warnings = []
        errors = []

        for feature in features:
            if feature in self.features:
                config = self.features[feature]

                if config.fallback == FallbackBehavior.ERROR and not config.supported:
                    errors.append(f"Beta feature not supported: {feature}")
                elif config.fallback == FallbackBehavior.IGNORE and not config.supported:
                    warnings.append(f"Beta feature ignored: {feature}")
                    supported.add(feature)  # Add but will be ignored
                else:
                    supported.add(feature)
            else:
                # Unknown feature - warn but don't fail
                warnings.append(f"Unknown beta feature: {feature}")

        return supported, warnings, errors

    def get_feature(self, name: str) -> Optional[BetaFeature]:
        """Get feature by name"""
        return self.features.get(name)

    def is_supported(self, name: str) -> bool:
        """Check if feature is supported"""
        feature = self.features.get(name)
        return feature.supported if feature else False

    def should_emulate(self, name: str) -> bool:
        """Check if feature should be emulated"""
        feature = self.features.get(name)
        if not feature:
            return False
        return not feature.supported and feature.fallback == FallbackBehavior.EMULATE

    def apply_features(
        self,
        request: Dict[str, Any],
        features: Set[str]
    ) -> Dict[str, Any]:
        """
        Apply beta feature modifications to request.

        Args:
            request: Original request
            features: Enabled beta features

        Returns:
            Modified request
        """
        modified = request.copy()

        # Extended output tokens
        if "max-tokens-3-5-sonnet-2024-07-15" in features:
            # Allow higher max_tokens values
            max_tokens = request.get("max_tokens", 0)
            if max_tokens > 8192:
                logger.debug(f"Extended output enabled: max_tokens={max_tokens}")

        # Token counting
        if "token-counting-2024-11-01" in features:
            # Enable token counting in response
            modified["_include_usage"] = True

        return modified

    def process_response(
        self,
        response: Dict[str, Any],
        features: Set[str]
    ) -> Dict[str, Any]:
        """
        Process response based on enabled features.

        Args:
            response: Original response
            features: Enabled beta features

        Returns:
            Modified response
        """
        modified = response.copy()

        # Add cache statistics if caching was requested
        if "prompt-caching-2024-07-31" in features:
            if "usage" in modified and isinstance(modified["usage"], dict):
                # Add empty cache stats
                modified["usage"]["cache_creation_input_tokens"] = 0
                modified["usage"]["cache_read_input_tokens"] = 0

        return modified


# =============================================================================
# Prompt Cache Handler
# =============================================================================

class PromptCacheHandler:
    """
    Handle Anthropic cache_control blocks.

    Strips cache_control for non-Anthropic providers and provides
    simulated cache statistics.
    """

    def strip_cache_control(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Remove cache_control from messages.

        Args:
            messages: List of messages with potential cache_control

        Returns:
            Messages with cache_control removed
        """
        cleaned = []
        for msg in messages:
            cleaned_msg = msg.copy()

            content = msg.get("content")
            if isinstance(content, list):
                cleaned_content = []
                for block in content:
                    if isinstance(block, dict):
                        # Remove cache_control key
                        cleaned_block = {
                            k: v for k, v in block.items()
                            if k != "cache_control"
                        }
                        cleaned_content.append(cleaned_block)
                    else:
                        cleaned_content.append(block)
                cleaned_msg["content"] = cleaned_content

            cleaned.append(cleaned_msg)

        return cleaned

    def strip_from_system(
        self,
        system: Any
    ) -> Any:
        """
        Remove cache_control from system prompt.

        Args:
            system: System prompt (string or list)

        Returns:
            System prompt with cache_control removed
        """
        if isinstance(system, str):
            return system

        if isinstance(system, list):
            cleaned = []
            for block in system:
                if isinstance(block, dict):
                    cleaned_block = {
                        k: v for k, v in block.items()
                        if k != "cache_control"
                    }
                    cleaned.append(cleaned_block)
                else:
                    cleaned.append(block)
            return cleaned

        return system

    def get_cache_stats(self) -> Dict[str, int]:
        """Get simulated cache statistics"""
        return {
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }


# =============================================================================
# Global Instances
# =============================================================================

_beta_handler: Optional[BetaFeaturesHandler] = None
_cache_handler: Optional[PromptCacheHandler] = None


def get_beta_handler() -> BetaFeaturesHandler:
    """Get or create global beta features handler"""
    global _beta_handler
    if _beta_handler is None:
        _beta_handler = BetaFeaturesHandler()
    return _beta_handler


def get_cache_handler() -> PromptCacheHandler:
    """Get or create global prompt cache handler"""
    global _cache_handler
    if _cache_handler is None:
        _cache_handler = PromptCacheHandler()
    return _cache_handler

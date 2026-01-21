"""
Model Capability Detection

Detects and validates model capabilities for proper request handling.
Ensures agents only use features supported by the target model.
"""

from __future__ import annotations
import logging
from typing import Dict, Set, Optional, Any
from dataclasses import dataclass, field


logger = logging.getLogger("shin-gateway")


# =============================================================================
# Capability Definitions
# =============================================================================

@dataclass
class ModelCapabilities:
    """Model capability specification"""
    supports_tools: bool = True
    supports_vision: bool = False
    supports_streaming: bool = True
    supports_system_prompt: bool = True
    supports_parallel_tools: bool = True
    supports_json_mode: bool = False
    supports_thinking: bool = False
    supports_prompt_caching: bool = False
    max_tools: int = 128
    max_images: int = 20
    max_context_tokens: int = 128000
    max_output_tokens: int = 8192
    tool_choice_modes: Set[str] = field(
        default_factory=lambda: {"auto", "any", "tool", "none"}
    )
    supported_image_types: Set[str] = field(
        default_factory=lambda: {"image/jpeg", "image/png", "image/gif", "image/webp"}
    )


# =============================================================================
# Model Capability Database
# =============================================================================

MODEL_CAPABILITIES: Dict[str, ModelCapabilities] = {
    # Claude models
    "claude-3-5-sonnet": ModelCapabilities(
        supports_tools=True,
        supports_vision=True,
        supports_streaming=True,
        supports_parallel_tools=True,
        supports_thinking=True,
        supports_prompt_caching=True,
        max_tools=128,
        max_context_tokens=200000,
        max_output_tokens=8192,
    ),
    "claude-3-5-haiku": ModelCapabilities(
        supports_tools=True,
        supports_vision=True,
        supports_streaming=True,
        supports_parallel_tools=True,
        supports_prompt_caching=True,
        max_tools=128,
        max_context_tokens=200000,
        max_output_tokens=8192,
    ),
    "claude-3-opus": ModelCapabilities(
        supports_tools=True,
        supports_vision=True,
        supports_streaming=True,
        supports_thinking=True,
        supports_prompt_caching=True,
        max_context_tokens=200000,
        max_output_tokens=4096,
    ),

    # GPT models
    "gpt-4o": ModelCapabilities(
        supports_tools=True,
        supports_vision=True,
        supports_streaming=True,
        supports_json_mode=True,
        supports_parallel_tools=True,
        max_tools=128,
        max_context_tokens=128000,
        max_output_tokens=16384,
    ),
    "gpt-4o-mini": ModelCapabilities(
        supports_tools=True,
        supports_vision=True,
        supports_streaming=True,
        supports_json_mode=True,
        supports_parallel_tools=True,
        max_context_tokens=128000,
        max_output_tokens=16384,
    ),
    "gpt-4-turbo": ModelCapabilities(
        supports_tools=True,
        supports_vision=True,
        supports_streaming=True,
        supports_json_mode=True,
        max_context_tokens=128000,
        max_output_tokens=4096,
    ),

    # Qwen models (via Ollama)
    "qwen2.5-coder": ModelCapabilities(
        supports_tools=True,
        supports_vision=False,
        supports_streaming=True,
        supports_parallel_tools=True,
        max_tools=64,
        max_context_tokens=32768,
        max_output_tokens=8192,
    ),
    "qwen2.5": ModelCapabilities(
        supports_tools=True,
        supports_vision=False,
        supports_streaming=True,
        max_tools=64,
        max_context_tokens=32768,
        max_output_tokens=8192,
    ),

    # Llama models
    "llama-3.3-70b": ModelCapabilities(
        supports_tools=True,
        supports_vision=False,
        supports_streaming=True,
        supports_parallel_tools=True,
        max_tools=32,
        max_context_tokens=128000,
        max_output_tokens=8192,
    ),
    "llama-3.1-8b": ModelCapabilities(
        supports_tools=True,
        supports_vision=False,
        supports_streaming=True,
        max_tools=16,
        max_context_tokens=128000,
        max_output_tokens=4096,
    ),
    "llama-3.2": ModelCapabilities(
        supports_tools=True,
        supports_vision=True,  # Vision models
        supports_streaming=True,
        max_context_tokens=128000,
        max_output_tokens=4096,
    ),

    # Deepseek
    "deepseek-coder": ModelCapabilities(
        supports_tools=True,
        supports_vision=False,
        supports_streaming=True,
        max_tools=32,
        max_context_tokens=65536,
        max_output_tokens=4096,
    ),
    "deepseek-v2": ModelCapabilities(
        supports_tools=True,
        supports_vision=False,
        supports_streaming=True,
        max_context_tokens=65536,
        max_output_tokens=8192,
    ),

    # Mistral
    "mistral": ModelCapabilities(
        supports_tools=True,
        supports_vision=False,
        supports_streaming=True,
        max_tools=32,
        max_context_tokens=32768,
        max_output_tokens=4096,
    ),
    "mixtral": ModelCapabilities(
        supports_tools=True,
        supports_vision=False,
        supports_streaming=True,
        max_tools=32,
        max_context_tokens=32768,
        max_output_tokens=4096,
    ),

    # Groq-hosted models
    "llama-3.3-70b-versatile": ModelCapabilities(
        supports_tools=True,
        supports_vision=False,
        supports_streaming=True,
        supports_parallel_tools=True,
        max_tools=32,
        max_context_tokens=128000,
        max_output_tokens=8192,
    ),
    "llama-3.1-8b-instant": ModelCapabilities(
        supports_tools=True,
        supports_vision=False,
        supports_streaming=True,
        max_tools=16,
        max_context_tokens=128000,
        max_output_tokens=4096,
    ),
}


# =============================================================================
# Capability Detector
# =============================================================================

class CapabilityDetector:
    """
    Detect and validate model capabilities.

    Provides capability information for request validation and
    feature gate decisions.
    """

    def __init__(self, custom_capabilities: Optional[Dict[str, ModelCapabilities]] = None):
        self.capabilities = {**MODEL_CAPABILITIES}
        if custom_capabilities:
            self.capabilities.update(custom_capabilities)

    def get_capabilities(self, model: str) -> ModelCapabilities:
        """
        Get capabilities for a model.

        Args:
            model: Model name or alias

        Returns:
            ModelCapabilities for the model
        """
        # Exact match
        if model in self.capabilities:
            return self.capabilities[model]

        # Normalize model name
        model_lower = model.lower()

        # Prefix/contains match
        for key, caps in self.capabilities.items():
            key_lower = key.lower()
            if model_lower.startswith(key_lower) or key_lower in model_lower:
                return caps

        # Default conservative capabilities
        logger.debug(f"No capabilities found for {model}, using defaults")
        return ModelCapabilities(
            supports_tools=True,
            supports_vision=False,
            supports_streaming=True,
            max_tools=16,
            max_context_tokens=8192,
            max_output_tokens=2048,
        )

    def validate_request(
        self,
        request: Dict[str, Any],
        model: str
    ) -> tuple[bool, Optional[str]]:
        """
        Validate request against model capabilities.

        Args:
            request: Request to validate
            model: Target model

        Returns:
            (is_valid, error_message)
        """
        caps = self.get_capabilities(model)
        errors = []

        # Check tools
        tools = request.get("tools", [])
        if tools:
            if not caps.supports_tools:
                errors.append(f"Model {model} does not support tools")
            elif len(tools) > caps.max_tools:
                errors.append(f"Too many tools ({len(tools)}), max is {caps.max_tools}")

        # Check vision
        if self._has_images(request):
            if not caps.supports_vision:
                errors.append(f"Model {model} does not support vision")
            else:
                image_count = self._count_images(request)
                if image_count > caps.max_images:
                    errors.append(f"Too many images ({image_count}), max is {caps.max_images}")

        # Check streaming
        if request.get("stream") and not caps.supports_streaming:
            errors.append(f"Model {model} does not support streaming")

        # Check max_tokens
        max_tokens = request.get("max_tokens", 0)
        if max_tokens > caps.max_output_tokens:
            errors.append(
                f"max_tokens ({max_tokens}) exceeds model limit ({caps.max_output_tokens})"
            )

        # Check tool choice
        tool_choice = request.get("tool_choice")
        if tool_choice:
            choice_type = tool_choice.get("type") if isinstance(tool_choice, dict) else None
            if choice_type and choice_type not in caps.tool_choice_modes:
                errors.append(f"Tool choice mode '{choice_type}' not supported")

        # Check thinking
        if request.get("thinking") and not caps.supports_thinking:
            # Not an error, just won't work as expected
            logger.warning(f"Model {model} does not support thinking, will be emulated")

        if errors:
            return False, "; ".join(errors)

        return True, None

    def _has_images(self, request: Dict[str, Any]) -> bool:
        """Check if request contains images"""
        for msg in request.get("messages", []):
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "image":
                            return True
                        if block.get("type") == "image_url":
                            return True
        return False

    def _count_images(self, request: Dict[str, Any]) -> int:
        """Count images in request"""
        count = 0
        for msg in request.get("messages", []):
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") in ("image", "image_url"):
                            count += 1
        return count

    def supports_feature(self, model: str, feature: str) -> bool:
        """
        Check if model supports a specific feature.

        Args:
            model: Model name
            feature: Feature name (e.g., 'tools', 'vision', 'streaming')

        Returns:
            True if supported
        """
        caps = self.get_capabilities(model)
        feature_map = {
            "tools": caps.supports_tools,
            "vision": caps.supports_vision,
            "streaming": caps.supports_streaming,
            "system": caps.supports_system_prompt,
            "parallel_tools": caps.supports_parallel_tools,
            "json_mode": caps.supports_json_mode,
            "thinking": caps.supports_thinking,
            "prompt_caching": caps.supports_prompt_caching,
        }
        return feature_map.get(feature, False)


# =============================================================================
# Global Instance
# =============================================================================

_capability_detector: Optional[CapabilityDetector] = None


def get_capability_detector() -> CapabilityDetector:
    """Get or create global capability detector"""
    global _capability_detector
    if _capability_detector is None:
        _capability_detector = CapabilityDetector()
    return _capability_detector


def get_capabilities(model: str) -> ModelCapabilities:
    """Convenience function to get model capabilities"""
    return get_capability_detector().get_capabilities(model)

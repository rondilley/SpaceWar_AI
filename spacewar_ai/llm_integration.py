"""
LLM Integration for Spacewar! AI

Supports:
- API key discovery from *.key.txt files and .env
- Multiple LLM providers (OpenAI, Anthropic, Groq, etc.)
- Local LLM via llama-cpp-python
- LLM-guided training (demonstrations, not real-time play)
- Provider health tracking with graceful fallback
"""

import os
import json
import time
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple
from enum import Enum
import logging

import numpy as np
import torch

from config import LLMConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Provider Status & Error Handling
# =============================================================================

class ProviderStatus(Enum):
    """Health status for LLM providers."""
    HEALTHY = "healthy"      # Working normally
    DEGRADED = "degraded"    # Rate limited or slow, but usable
    UNHEALTHY = "unhealthy"  # Server errors, temporarily unavailable
    DISABLED = "disabled"    # Auth failed or quota exceeded, don't retry


class LLMError(Exception):
    """Base exception for LLM errors."""
    pass


class RateLimitError(LLMError):
    """Rate limit (429) error - retryable."""
    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class QuotaExceededError(LLMError):
    """Quota exceeded - NOT retryable, disable provider."""
    pass


class AuthenticationError(LLMError):
    """Authentication failed - NOT retryable, disable provider."""
    pass


class ServerError(LLMError):
    """Server error (5xx) - retryable."""
    pass


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


def retry_with_backoff(
    func,
    config: RetryConfig = None,
    on_retry: Optional[callable] = None,
):
    """
    Execute function with exponential backoff retry.

    Args:
        func: Function to execute
        config: Retry configuration
        on_retry: Callback(attempt, delay, error) called before each retry

    Returns:
        Function result

    Raises:
        Last exception if all retries exhausted
    """
    if config is None:
        config = RetryConfig()

    last_exception = None

    for attempt in range(config.max_retries + 1):
        try:
            return func()
        except QuotaExceededError:
            # Don't retry quota errors
            raise
        except AuthenticationError:
            # Don't retry auth errors
            raise
        except (RateLimitError, ServerError) as e:
            last_exception = e

            if attempt >= config.max_retries:
                raise

            # Calculate delay
            if isinstance(e, RateLimitError) and e.retry_after:
                delay = e.retry_after
            else:
                delay = config.initial_delay * (config.exponential_base ** attempt)
                delay = min(delay, config.max_delay)

            if config.jitter:
                delay = delay * (0.5 + random.random())

            if on_retry:
                on_retry(attempt, delay, e)

            time.sleep(delay)
        except Exception as e:
            # Unknown error - don't retry
            raise

    if last_exception:
        raise last_exception


# =============================================================================
# API Key Management
# =============================================================================

@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""
    env_var: str
    base_url: Optional[str]
    fallback_model: str  # Used only if model discovery fails
    is_openai_compatible: bool = True


PROVIDER_CONFIGS: Dict[str, ProviderConfig] = {
    "openai": ProviderConfig(
        env_var="OPENAI_API_KEY",
        base_url="https://api.openai.com/v1",
        fallback_model="gpt-4o-mini",
    ),
    "anthropic": ProviderConfig(
        env_var="ANTHROPIC_API_KEY",
        base_url="https://api.anthropic.com",
        fallback_model="claude-sonnet-4-20250514",
        is_openai_compatible=False,
    ),
    "groq": ProviderConfig(
        env_var="GROQ_API_KEY",
        base_url="https://api.groq.com/openai/v1",
        fallback_model="llama-3.1-70b-versatile",
    ),
    "together": ProviderConfig(
        env_var="TOGETHER_API_KEY",
        base_url="https://api.together.xyz/v1",
        fallback_model="meta-llama/Llama-3-70b-chat-hf",
    ),
    "openrouter": ProviderConfig(
        env_var="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        fallback_model="anthropic/claude-3-haiku",
    ),
    "mistral": ProviderConfig(
        env_var="MISTRAL_API_KEY",
        base_url="https://api.mistral.ai/v1",
        fallback_model="mistral-small-latest",
    ),
    "deepseek": ProviderConfig(
        env_var="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com/v1",
        fallback_model="deepseek-chat",
    ),
    "google": ProviderConfig(
        env_var="GOOGLE_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        fallback_model="gemini-2.0-flash",
    ),
    "xai": ProviderConfig(
        env_var="XAI_API_KEY",
        base_url="https://api.x.ai/v1",
        fallback_model="grok-2-1212",
    ),
}

# =============================================================================
# Model Ranking - Higher score = better model
# =============================================================================

# Model quality rankings (higher = better)
# These are approximate rankings based on benchmarks and capabilities
MODEL_RANKINGS: Dict[str, int] = {
    # Anthropic - Flagship
    "claude-sonnet-4-20250514": 98,
    "claude-opus-4-20250514": 99,
    "claude-3-5-sonnet-20241022": 95,
    "claude-3-5-sonnet-latest": 95,
    "claude-3-5-sonnet-v2": 95,
    "claude-3-opus-20240229": 93,
    "claude-3-opus-latest": 93,
    "claude-3-sonnet-20240229": 85,
    "claude-3-5-haiku-latest": 80,
    "claude-3-5-haiku-20241022": 80,
    "claude-3-haiku-20240307": 70,

    # OpenAI - Flagship
    "gpt-4o": 94,
    "gpt-4o-2024-11-20": 94,
    "gpt-4o-2024-08-06": 93,
    "gpt-4-turbo": 91,
    "gpt-4-turbo-preview": 91,
    "gpt-4": 88,
    "gpt-4o-mini": 82,
    "gpt-4o-mini-2024-07-18": 82,
    "gpt-3.5-turbo": 70,
    "o1-preview": 97,
    "o1-mini": 88,

    # Google Gemini
    "gemini-2.0-flash": 90,
    "gemini-2.0-flash-exp": 90,
    "gemini-1.5-pro": 92,
    "gemini-1.5-pro-latest": 92,
    "gemini-1.5-flash": 85,
    "gemini-1.5-flash-latest": 85,
    "gemini-1.0-pro": 78,
    "gemini-pro": 78,

    # Groq (fast inference)
    "llama-3.3-70b-versatile": 88,
    "llama-3.1-70b-versatile": 86,
    "llama-3.1-8b-instant": 72,
    "llama3-70b-8192": 85,
    "llama3-8b-8192": 70,
    "mixtral-8x7b-32768": 80,
    "gemma2-9b-it": 75,

    # Mistral
    "mistral-large-latest": 90,
    "mistral-large-2411": 90,
    "mistral-medium-latest": 82,
    "mistral-small-latest": 75,
    "codestral-latest": 85,
    "open-mixtral-8x22b": 85,
    "open-mixtral-8x7b": 78,

    # DeepSeek
    "deepseek-chat": 85,
    "deepseek-coder": 83,
    "deepseek-reasoner": 90,

    # Together AI / Open models
    "meta-llama/Llama-3-70b-chat-hf": 85,
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": 87,
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": 92,
    "mistralai/Mixtral-8x22B-Instruct-v0.1": 85,
    "Qwen/Qwen2.5-72B-Instruct-Turbo": 88,

    # xAI Grok
    "grok-2-1212": 92,
    "grok-2-vision-1212": 90,
    "grok-beta": 85,
}


def get_model_score(model_id: str) -> int:
    """Get quality score for a model. Higher = better."""
    # Direct match
    if model_id in MODEL_RANKINGS:
        return MODEL_RANKINGS[model_id]

    # Partial match (model names often have version suffixes)
    model_lower = model_id.lower()
    for known_model, score in MODEL_RANKINGS.items():
        if known_model.lower() in model_lower or model_lower in known_model.lower():
            return score

    # Unknown model - assign default score based on name patterns
    if "opus" in model_lower or "405b" in model_lower:
        return 90
    if "grok-2" in model_lower:
        return 90
    if "grok" in model_lower:
        return 85
    if "sonnet" in model_lower or "pro" in model_lower or "large" in model_lower:
        return 85
    if "70b" in model_lower or "72b" in model_lower:
        return 82
    if "haiku" in model_lower or "flash" in model_lower or "mini" in model_lower:
        return 75
    if "8b" in model_lower or "7b" in model_lower or "small" in model_lower:
        return 65

    return 50  # Unknown model


def select_best_model(available_models: List[str]) -> Optional[str]:
    """Select the best model from a list of available models."""
    if not available_models:
        return None

    # Score each model and pick the best
    scored = [(model, get_model_score(model)) for model in available_models]
    scored.sort(key=lambda x: x[1], reverse=True)

    best_model, best_score = scored[0]
    logger.info(f"Selected best model: {best_model} (score: {best_score})")

    return best_model


# =============================================================================
# Model Discovery
# =============================================================================

def discover_models_openai_compatible(base_url: str, api_key: str) -> List[str]:
    """Discover available models from an OpenAI-compatible API."""
    import requests

    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(
            f"{base_url}/models",
            headers=headers,
            timeout=10,
        )

        if response.status_code == 200:
            data = response.json()
            models = []

            # Handle different response formats
            if "data" in data:
                models = [m.get("id", m.get("name", "")) for m in data["data"]]
            elif isinstance(data, list):
                models = [m.get("id", m.get("name", "")) for m in data]

            # Filter to chat/instruct models only, exclude non-chat and special models
            exclude_patterns = [
                "tts", "whisper", "embedding", "embed", "moderation",
                "dall-e", "image", "audio", "speech", "realtime",
                "transcription", "vision-preview",
                # Exclude o1/o3 reasoning models - they have special requirements
                # and may not be accessible on all API tiers
            ]

            # Models that start with o1 or o3 need special handling
            def is_accessible_model(model_name: str) -> bool:
                """Check if model is accessible on standard API tiers."""
                m = model_name.lower()
                # o1 and o3 models have special requirements (no system prompt,
                # different pricing tier, etc.) - prefer standard models
                if m.startswith("o1") or m.startswith("o3"):
                    return False
                return True

            chat_models = [
                m for m in models
                if m and
                any(kw in m.lower() for kw in [
                    "gpt", "claude", "gemini", "llama", "mixtral", "mistral",
                    "chat", "instruct", "turbo", "sonnet", "opus", "haiku",
                    "flash", "pro", "deepseek", "qwen"
                ]) and
                not any(excl in m.lower() for excl in exclude_patterns) and
                is_accessible_model(m)
            ]

            return chat_models if chat_models else models

    except Exception as e:
        logger.debug(f"Model discovery failed for {base_url}: {e}")

    return []


def discover_models_anthropic(api_key: str) -> List[str]:
    """Get available Anthropic models."""
    # Anthropic doesn't have a public list models endpoint
    # Return known models in order of capability
    return [
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
        "claude-3-5-sonnet-latest",
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-latest",
        "claude-3-5-haiku-latest",
        "claude-3-haiku-20240307",
    ]


def discover_models(provider: str, api_key: str) -> List[str]:
    """Discover available models for a provider."""
    provider_config = PROVIDER_CONFIGS.get(provider)
    if not provider_config:
        return []

    if provider == "anthropic":
        return discover_models_anthropic(api_key)

    if provider_config.base_url:
        models = discover_models_openai_compatible(provider_config.base_url, api_key)
        if models:
            return models

    # Return fallback model as single option
    return [provider_config.fallback_model]


def get_best_model_for_provider(provider: str, api_key: str) -> str:
    """Get the best available model for a provider."""
    models = discover_models(provider, api_key)

    if models:
        best = select_best_model(models)
        if best:
            return best

    # Fallback
    provider_config = PROVIDER_CONFIGS.get(provider)
    return provider_config.fallback_model if provider_config else "unknown"


class APIKeyManager:
    """
    Discovers and manages API keys from multiple sources.

    Priority:
    1. *.key.txt files in keys_dir (and parent directories)
    2. .env file in keys_dir
    3. Environment variables
    """

    # Alias mapping for common naming variations
    PROVIDER_ALIASES = {
        "claude": "anthropic",
        "gpt": "openai",
        "gemini": "google",
        "llama": "groq",
        "grok": "xai",
    }

    def __init__(self, keys_dir: str = "."):
        self.keys_dir = Path(keys_dir).resolve()
        self.keys: Dict[str, str] = {}
        self._discover_keys()

    def _get_search_dirs(self) -> List[Path]:
        """Get list of directories to search for key files."""
        dirs = [self.keys_dir]

        # Also search parent directory (for running from subdirs)
        parent = self.keys_dir.parent
        if parent != self.keys_dir:
            dirs.append(parent)

        # Search script's directory
        script_dir = Path(__file__).parent.parent.resolve()
        if script_dir not in dirs:
            dirs.append(script_dir)

        return dirs

    def _discover_keys(self):
        """Find all API keys from various sources."""
        # Source 1: *.key.txt files
        self._load_key_txt_files()

        # Source 2: .env file
        self._load_dotenv()

        # Source 3: Environment variables (fallback)
        self._load_from_environment()

        if self.keys:
            logger.info(f"Loaded API keys for: {list(self.keys.keys())}")
        else:
            logger.warning("No API keys found")

    def _load_key_txt_files(self):
        """Load keys from *.key.txt files."""
        patterns = ["*.key.txt", "*.api.key"]

        for search_dir in self._get_search_dirs():
            if not search_dir.exists():
                continue

            for pattern in patterns:
                for key_file in search_dir.glob(pattern):
                    # Extract provider name from filename
                    # e.g., "openai.key.txt" -> "openai"
                    # e.g., "anthropic.api.key" -> "anthropic"
                    stem = key_file.stem
                    if stem.endswith(".key"):
                        provider = stem[:-4]
                    elif stem.endswith(".api"):
                        provider = stem[:-4]
                    else:
                        provider = stem

                    provider = provider.lower()

                    # Apply alias mapping
                    provider = self.PROVIDER_ALIASES.get(provider, provider)

                    # Skip if already loaded (first found wins)
                    if provider in self.keys:
                        continue

                    try:
                        with open(key_file, "r") as f:
                            key = f.read().strip()
                            if key and not key.startswith("#"):
                                self.keys[provider] = key
                                logger.info(f"Loaded key for {provider} from {key_file}")
                    except Exception as e:
                        logger.warning(f"Failed to load {key_file}: {e}")

    def _load_dotenv(self):
        """Load keys from .env file."""
        for search_dir in self._get_search_dirs():
            dotenv_path = search_dir / ".env"

            if not dotenv_path.exists():
                continue

            try:
                with open(dotenv_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue

                        if "=" not in line:
                            continue

                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")

                        # Map env var names to providers
                        for provider, config in PROVIDER_CONFIGS.items():
                            if key == config.env_var and provider not in self.keys:
                                self.keys[provider] = value
                                logger.info(f"Loaded key for {provider} from {dotenv_path}")

            except Exception as e:
                logger.warning(f"Failed to load {dotenv_path}: {e}")

    def _load_from_environment(self):
        """Load keys from environment variables."""
        for provider, config in PROVIDER_CONFIGS.items():
            if provider not in self.keys:
                env_key = os.environ.get(config.env_var)
                if env_key:
                    self.keys[provider] = env_key
                    logger.debug(f"Loaded key for {provider} from environment")

    def get_key(self, provider: str) -> Optional[str]:
        """Get API key for a provider."""
        return self.keys.get(provider.lower())

    def get_available_providers(self) -> List[str]:
        """Get list of providers with available keys."""
        return list(self.keys.keys())

    def has_any_key(self) -> bool:
        """Check if any API key is available."""
        return len(self.keys) > 0


# =============================================================================
# Hardware Detection (for local LLM)
# =============================================================================

@dataclass
class HardwareProfile:
    """System hardware profile."""
    cpu_cores: int
    ram_gb: float
    gpu_name: Optional[str]
    gpu_vram_gb: Optional[float]
    gpu_type: str  # "cuda", "rocm", "metal", "vulkan", "cpu"
    gpu_backend: Optional[str]


def detect_hardware(vram_override: Optional[float] = None) -> HardwareProfile:
    """
    Detect system hardware capabilities.

    Args:
        vram_override: Manual override for GPU VRAM in GB (useful for shared memory systems)
    """
    import subprocess

    cpu_cores = os.cpu_count() or 4

    # RAM detection
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        ram_gb = 8.0

    gpu_name = None
    gpu_vram_gb = None
    gpu_type = "cpu"
    gpu_backend = None

    # Try NVIDIA CUDA first
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_type = "cuda"
            gpu_backend = f"CUDA {torch.version.cuda}"
        except Exception:
            pass

    # Try AMD ROCm
    if gpu_type == "cpu":
        try:
            # Check for ROCm via rocm-smi
            result = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram", "--json"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                # Parse ROCm output - handle different JSON formats
                for card_id, card_data in data.items():
                    if isinstance(card_data, dict):
                        # Try different key formats
                        for key in ["VRAM Total Memory (B)", "vram_total", "VRAM Total"]:
                            if key in card_data:
                                vram_val = card_data[key]
                                # Handle string or int values
                                if isinstance(vram_val, str):
                                    vram_val = int(vram_val.replace(",", "").strip())
                                gpu_vram_gb = vram_val / (1024**3)
                                gpu_type = "rocm"
                                gpu_backend = "ROCm"
                                break
                        if gpu_type == "rocm":
                            break

            # Alternative: try rocm-smi without JSON
            if gpu_type == "cpu":
                result = subprocess.run(
                    ["rocm-smi", "--showmeminfo", "vram"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    for line in result.stdout.split("\n"):
                        if "Total" in line and ("GB" in line or "MB" in line or "Memory" in line):
                            # Parse lines like "VRAM Total Memory (B): 103079215104"
                            # or "Total: 96 GB"
                            parts = line.split(":")
                            if len(parts) >= 2:
                                val_str = parts[-1].strip()
                                if "GB" in val_str.upper():
                                    gpu_vram_gb = float(val_str.upper().replace("GB", "").strip())
                                elif "MB" in val_str.upper():
                                    gpu_vram_gb = float(val_str.upper().replace("MB", "").strip()) / 1024
                                else:
                                    # Assume bytes
                                    try:
                                        gpu_vram_gb = int(val_str.replace(",", "")) / (1024**3)
                                    except ValueError:
                                        pass
                                if gpu_vram_gb and gpu_vram_gb > 0:
                                    gpu_type = "rocm"
                                    gpu_backend = "ROCm"
                                    break

            # Get GPU name
            result2 = subprocess.run(
                ["rocm-smi", "--showproductname"],
                capture_output=True, text=True, timeout=5
            )
            if result2.returncode == 0:
                for line in result2.stdout.split("\n"):
                    # Skip header/separator lines
                    if "=====" in line or not line.strip():
                        continue
                    # Look for actual product info
                    if ":" in line:
                        key, value = line.split(":", 1)
                        value = value.strip()
                        if value and ("Card" in key or "GPU" in key or "Product" in key or "series" in key.lower()):
                            gpu_name = value
                            break

            # Detect shared memory APUs (Strix Halo, Phoenix, etc.)
            # These have integrated GPUs that share system RAM
            is_shared_memory_apu = False
            if gpu_name:
                gpu_lower = gpu_name.lower()
                # Radeon 8000 series = Strix Halo integrated GPU
                # Radeon 700M series = Phoenix/Hawk Point integrated
                if any(x in gpu_lower for x in ["8060", "8050", "8040", "890m", "880m", "780m", "760m"]):
                    is_shared_memory_apu = True
                # Also check for generic integrated indicators
                if "graphics" in gpu_lower and not any(x in gpu_lower for x in ["rx ", "pro ", "instinct"]):
                    is_shared_memory_apu = True

            # For shared memory APUs, use portion of system RAM as VRAM
            if is_shared_memory_apu and (gpu_vram_gb is None or gpu_vram_gb < 4):
                # APUs can use a large portion of system RAM
                # Strix Halo (395+) can allocate up to 96GB to GPU
                # Conservative estimate: 60% of RAM, capped at 96GB
                estimated_vram = min(ram_gb * 0.6, 96.0)
                logger.info(f"Detected shared memory APU ({gpu_name})")
                logger.info(f"  Estimating {estimated_vram:.0f}GB usable VRAM from {ram_gb:.0f}GB system RAM")
                gpu_vram_gb = estimated_vram

        except Exception as e:
            logger.debug(f"ROCm detection error: {e}")

    # Try AMD via lspci if rocm-smi failed or got bad data
    if gpu_type == "cpu" or (gpu_type == "rocm" and (gpu_vram_gb is None or gpu_vram_gb < 1)):
        try:
            result = subprocess.run(
                ["lspci", "-v"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                # Collect all AMD GPUs, prefer discrete over integrated
                amd_gpus = []
                for line in result.stdout.split("\n"):
                    if ("VGA" in line or "Display" in line) and ("AMD" in line or "Radeon" in line):
                        amd_gpus.append(line)

                # Prefer discrete GPUs (skip integrated like "Radeon 8060S")
                for line in amd_gpus:
                    line_lower = line.lower()
                    # Skip integrated graphics (usually have lower model numbers or "integrated")
                    if any(x in line_lower for x in ["integrated", "8060", "680m", "780m", "igpu"]):
                        continue

                    gpu_name = line.split(":")[-1].strip()[:60]
                    gpu_type = "rocm"
                    gpu_backend = "ROCm (detected via lspci)"

                    # Estimate VRAM from model
                    if "w7900" in line_lower or "pro w7900" in line_lower:
                        gpu_vram_gb = 48.0
                    elif "w7800" in line_lower:
                        gpu_vram_gb = 32.0
                    elif "7900 xtx" in line_lower:
                        gpu_vram_gb = 24.0
                    elif "7900 xt" in line_lower or "7900xt" in line_lower:
                        gpu_vram_gb = 20.0
                    elif "7900" in line_lower:
                        gpu_vram_gb = 20.0
                    elif "6900" in line_lower:
                        gpu_vram_gb = 16.0
                    elif "6800" in line_lower:
                        gpu_vram_gb = 16.0
                    elif "instinct" in line_lower or "mi" in line_lower:
                        # MI series - high memory
                        if "mi300" in line_lower:
                            gpu_vram_gb = 192.0
                        elif "mi250" in line_lower:
                            gpu_vram_gb = 128.0
                        elif "mi210" in line_lower or "mi200" in line_lower:
                            gpu_vram_gb = 64.0
                        elif "mi100" in line_lower:
                            gpu_vram_gb = 32.0
                        else:
                            gpu_vram_gb = 32.0  # Default for MI series
                    elif "395" in line_lower:
                        # Likely referring to Strix Halo or similar high-end
                        gpu_vram_gb = 96.0  # Strix Halo shares system memory, estimate high
                    break

                # If no discrete found, use first AMD GPU
                if gpu_type == "cpu" and amd_gpus:
                    line = amd_gpus[0]
                    gpu_name = line.split(":")[-1].strip()[:60]
                    gpu_type = "rocm"
                    gpu_backend = "ROCm (integrated)"
                    gpu_vram_gb = 4.0  # Conservative estimate for integrated
        except Exception:
            pass

    # Check for Vulkan support (fallback for AMD)
    if gpu_type == "cpu":
        try:
            result = subprocess.run(
                ["vulkaninfo", "--summary"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and "AMD" in result.stdout:
                gpu_type = "vulkan"
                gpu_backend = "Vulkan"
                for line in result.stdout.split("\n"):
                    if "deviceName" in line:
                        gpu_name = line.split("=")[-1].strip()
                        break
        except Exception:
            pass

    # Apply VRAM override if specified
    if vram_override is not None and vram_override > 0:
        gpu_vram_gb = vram_override
        if gpu_type == "cpu":
            gpu_type = "rocm"  # Assume ROCm for manual override
            gpu_backend = "Manual override"
        logger.info(f"Using manual VRAM override: {vram_override:.1f}GB")

    # Log what we found
    if gpu_name:
        vram_str = f"{gpu_vram_gb:.1f}GB VRAM" if gpu_vram_gb else "unknown VRAM"
        logger.info(f"Detected GPU: {gpu_name} ({vram_str}, {gpu_backend})")
    else:
        logger.info(f"No GPU detected, using CPU mode ({cpu_cores} cores, {ram_gb:.1f}GB RAM)")

    return HardwareProfile(
        cpu_cores=cpu_cores,
        ram_gb=ram_gb,
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram_gb,
        gpu_type=gpu_type,
        gpu_backend=gpu_backend,
    )


# =============================================================================
# Local Model Catalog
# =============================================================================

@dataclass
class LocalModelSpec:
    """Specification for a local GGUF model."""
    repo_id: str
    filename: str
    size_gb: float
    min_vram_gb: float  # For full GPU offload
    min_ram_gb: float   # For CPU-only
    quality_score: int
    quantization: str
    description: str


# Comprehensive model catalog - ordered by quality (best first)
# NOTE: Only single-file models are listed here. Large models from official repos
# are often sharded - we use lmstudio-community or bartowski repos for single files.
LOCAL_MODEL_CATALOG: List[LocalModelSpec] = [
    # === Large Models (30-40B) - Single file from lmstudio-community ===
    LocalModelSpec(
        repo_id="lmstudio-community/Qwen2.5-32B-Instruct-GGUF",
        filename="Qwen2.5-32B-Instruct-Q8_0.gguf",
        size_gb=34.8, min_vram_gb=36.0, min_ram_gb=40.0,
        quality_score=92, quantization="Q8_0",
        description="Qwen 2.5 32B Q8 - Highest quality"
    ),
    LocalModelSpec(
        repo_id="lmstudio-community/Qwen2.5-32B-Instruct-GGUF",
        filename="Qwen2.5-32B-Instruct-Q6_K.gguf",
        size_gb=26.9, min_vram_gb=28.0, min_ram_gb=32.0,
        quality_score=91, quantization="Q6_K",
        description="Qwen 2.5 32B Q6 - High quality"
    ),
    LocalModelSpec(
        repo_id="lmstudio-community/Qwen2.5-32B-Instruct-GGUF",
        filename="Qwen2.5-32B-Instruct-Q4_K_M.gguf",
        size_gb=19.9, min_vram_gb=21.0, min_ram_gb=24.0,
        quality_score=89, quantization="Q4_K_M",
        description="Qwen 2.5 32B Q4 - Good quality/size"
    ),
    LocalModelSpec(
        repo_id="lmstudio-community/Qwen2.5-32B-Instruct-GGUF",
        filename="Qwen2.5-32B-Instruct-Q3_K_L.gguf",
        size_gb=17.2, min_vram_gb=18.0, min_ram_gb=20.0,
        quality_score=86, quantization="Q3_K_L",
        description="Qwen 2.5 32B Q3 - Smaller size"
    ),

    # === Medium-Large Models (14-27B) ===
    LocalModelSpec(
        repo_id="lmstudio-community/Qwen2.5-14B-Instruct-GGUF",
        filename="Qwen2.5-14B-Instruct-Q8_0.gguf",
        size_gb=15.7, min_vram_gb=17.0, min_ram_gb=18.0,
        quality_score=87, quantization="Q8_0",
        description="Qwen 2.5 14B Q8 - High quality"
    ),
    LocalModelSpec(
        repo_id="lmstudio-community/Qwen2.5-14B-Instruct-GGUF",
        filename="Qwen2.5-14B-Instruct-Q4_K_M.gguf",
        size_gb=9.0, min_vram_gb=10.0, min_ram_gb=12.0,
        quality_score=85, quantization="Q4_K_M",
        description="Qwen 2.5 14B Q4 - Balanced"
    ),
    LocalModelSpec(
        repo_id="lmstudio-community/gemma-2-27b-it-GGUF",
        filename="gemma-2-27b-it-Q4_K_M.gguf",
        size_gb=16.0, min_vram_gb=18.0, min_ram_gb=20.0,
        quality_score=84, quantization="Q4_K_M",
        description="Gemma 2 27B - Google's best open"
    ),

    # === Small-Medium Models (7-13B) - Most reliable ===
    LocalModelSpec(
        repo_id="lmstudio-community/Qwen2.5-7B-Instruct-GGUF",
        filename="Qwen2.5-7B-Instruct-Q8_0.gguf",
        size_gb=8.1, min_vram_gb=9.0, min_ram_gb=10.0,
        quality_score=80, quantization="Q8_0",
        description="Qwen 2.5 7B Q8 - High quality"
    ),
    LocalModelSpec(
        repo_id="lmstudio-community/Qwen2.5-7B-Instruct-GGUF",
        filename="Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        size_gb=4.7, min_vram_gb=5.5, min_ram_gb=6.0,
        quality_score=78, quantization="Q4_K_M",
        description="Qwen 2.5 7B - Fast and capable"
    ),
    LocalModelSpec(
        repo_id="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
        filename="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        size_gb=4.9, min_vram_gb=5.5, min_ram_gb=7.0,
        quality_score=77, quantization="Q4_K_M",
        description="Llama 3.1 8B - Meta's efficient"
    ),
    LocalModelSpec(
        repo_id="lmstudio-community/gemma-2-9b-it-GGUF",
        filename="gemma-2-9b-it-Q4_K_M.gguf",
        size_gb=5.5, min_vram_gb=6.0, min_ram_gb=8.0,
        quality_score=76, quantization="Q4_K_M",
        description="Gemma 2 9B - Excellent for size"
    ),
    LocalModelSpec(
        repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        filename="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        size_gb=4.4, min_vram_gb=5.0, min_ram_gb=6.0,
        quality_score=73, quantization="Q4_K_M",
        description="Mistral 7B v0.2 - Solid performer"
    ),

    # === Small Models (3-4B) - Fast inference ===
    LocalModelSpec(
        repo_id="lmstudio-community/Qwen2.5-3B-Instruct-GGUF",
        filename="Qwen2.5-3B-Instruct-Q8_0.gguf",
        size_gb=3.6, min_vram_gb=4.0, min_ram_gb=5.0,
        quality_score=68, quantization="Q8_0",
        description="Qwen 2.5 3B Q8 - Good quality"
    ),
    LocalModelSpec(
        repo_id="lmstudio-community/Qwen2.5-3B-Instruct-GGUF",
        filename="Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        size_gb=2.1, min_vram_gb=2.5, min_ram_gb=4.0,
        quality_score=65, quantization="Q4_K_M",
        description="Qwen 2.5 3B - Very fast"
    ),
    LocalModelSpec(
        repo_id="bartowski/Phi-3.5-mini-instruct-GGUF",
        filename="Phi-3.5-mini-instruct-Q4_K_M.gguf",
        size_gb=2.3, min_vram_gb=3.0, min_ram_gb=4.0,
        quality_score=67, quantization="Q4_K_M",
        description="Phi 3.5 Mini - Microsoft's efficient"
    ),

    # === Tiny Models (1-2B) - Ultra fast ===
    LocalModelSpec(
        repo_id="lmstudio-community/Qwen2.5-1.5B-Instruct-GGUF",
        filename="Qwen2.5-1.5B-Instruct-Q8_0.gguf",
        size_gb=1.7, min_vram_gb=2.0, min_ram_gb=3.0,
        quality_score=55, quantization="Q8_0",
        description="Qwen 2.5 1.5B - Ultra fast"
    ),
    LocalModelSpec(
        repo_id="lmstudio-community/Qwen2.5-0.5B-Instruct-GGUF",
        filename="Qwen2.5-0.5B-Instruct-Q8_0.gguf",
        size_gb=0.6, min_vram_gb=1.0, min_ram_gb=2.0,
        quality_score=45, quantization="Q8_0",
        description="Qwen 2.5 0.5B - Minimal resources"
    ),
]


def select_best_local_model(hardware: HardwareProfile) -> Optional[LocalModelSpec]:
    """Select the best local model that fits the hardware."""
    available_vram = (hardware.gpu_vram_gb or 0) * 0.85  # Leave 15% headroom
    available_ram = hardware.ram_gb * 0.7  # Leave 30% for system

    use_gpu = hardware.gpu_type in ("cuda", "rocm", "vulkan", "metal")

    suitable_models = []

    for model in LOCAL_MODEL_CATALOG:
        if use_gpu and available_vram >= model.min_vram_gb:
            # Can run on GPU
            suitable_models.append((model, model.quality_score, True))
        elif available_ram >= model.min_ram_gb:
            # Can run on CPU (with quality penalty)
            suitable_models.append((model, model.quality_score - 5, False))

    if not suitable_models:
        logger.warning("No suitable local model found for this hardware")
        return None

    # Sort by quality score (highest first)
    suitable_models.sort(key=lambda x: x[1], reverse=True)

    best_model, score, on_gpu = suitable_models[0]

    logger.info(f"Selected local model: {best_model.description}")
    logger.info(f"  Repo: {best_model.repo_id}")
    logger.info(f"  File: {best_model.filename} ({best_model.size_gb:.1f}GB)")
    logger.info(f"  Quality score: {score}, GPU offload: {on_gpu}")

    return best_model


def download_model(model: LocalModelSpec, models_dir: str = "./models") -> Optional[str]:
    """Download a model from HuggingFace Hub."""
    import subprocess
    from pathlib import Path

    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    model_file = models_path / model.filename

    if model_file.exists():
        # Verify file size is correct
        actual_size_gb = model_file.stat().st_size / (1024**3)
        if actual_size_gb >= model.size_gb * 0.95:  # Within 5% of expected size
            logger.info(f"Model already downloaded: {model_file} ({actual_size_gb:.1f}GB)")
            return str(model_file)
        else:
            logger.warning(f"Incomplete model file detected ({actual_size_gb:.1f}GB vs {model.size_gb:.1f}GB expected), re-downloading...")
            model_file.unlink()

    logger.info(f"Downloading model: {model.repo_id}/{model.filename}")
    logger.info(f"  Size: {model.size_gb:.1f}GB - this may take a while...")
    logger.info(f"  Download method: wget (v2)")  # Version marker to confirm new code is running

    # For large files, use wget which is more reliable than huggingface_hub
    # huggingface_hub has issues with XET chunked downloads hanging at 100%
    download_url = f"https://huggingface.co/{model.repo_id}/resolve/main/{model.filename}"
    logger.info(f"  URL: {download_url}")

    # Check if wget is available
    try:
        subprocess.run(["wget", "--version"], capture_output=True, check=True)
        wget_available = True
    except (FileNotFoundError, subprocess.CalledProcessError):
        wget_available = False
        logger.warning("wget not available, will use huggingface_hub (may hang on large files)")

    if wget_available:
        try:
            logger.info("Starting wget download...")
            # Use wget with resume support (-c), progress bar
            result = subprocess.run(
                [
                    "wget",
                    "-c",  # Continue/resume partial downloads
                    "--show-progress",
                    "--progress=bar:force",
                    "-O", str(model_file),
                    download_url,
                ],
                check=True,
                timeout=14400,  # 4 hour timeout for very large files
            )

            if model_file.exists():
                actual_size_gb = model_file.stat().st_size / (1024**3)
                if actual_size_gb >= model.size_gb * 0.95:
                    logger.info(f"Download complete: {model_file} ({actual_size_gb:.1f}GB)")
                    return str(model_file)
                else:
                    logger.error(f"Download incomplete: {actual_size_gb:.1f}GB vs {model.size_gb:.1f}GB expected")
                    return None
            else:
                logger.error("Download failed - file not created")
                return None

        except subprocess.TimeoutExpired:
            logger.error("Download timed out after 4 hours")
            return None
        except subprocess.CalledProcessError as e:
            logger.error(f"wget failed with exit code {e.returncode}")
            return None  # Don't fall back to huggingface_hub - it's broken for large files

    # Only use huggingface_hub if wget is not available
    try:
        from huggingface_hub import hf_hub_download

        # Download directly to cache and get the path
        cached_path = hf_hub_download(
            repo_id=model.repo_id,
            filename=model.filename,
        )

        logger.info(f"Download complete: {cached_path}")

        # Create symlink in models_dir for convenience (fast, no copy)
        if not model_file.exists():
            try:
                model_file.symlink_to(cached_path)
                logger.info(f"Created symlink: {model_file} -> {cached_path}")
            except OSError:
                # Symlink failed (Windows or permissions), just use cache path
                logger.info(f"Using cached model directly: {cached_path}")
                return cached_path

        return str(model_file) if model_file.exists() else cached_path

    except ImportError:
        logger.error("huggingface_hub not installed. Run: pip install huggingface_hub")
        return None
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        return None


def get_llama_cpp_settings(hardware: HardwareProfile, model: LocalModelSpec) -> Dict[str, Any]:
    """Get optimal llama.cpp settings for the hardware."""
    settings = {
        "n_ctx": 4096,  # Context size
        "n_threads": min(hardware.cpu_cores, 16),  # Cap threads
        "n_batch": 512,
        "verbose": False,
    }

    available_vram = (hardware.gpu_vram_gb or 0) * 0.85

    if hardware.gpu_type == "cuda":
        if available_vram >= model.min_vram_gb:
            settings["n_gpu_layers"] = -1  # Full GPU offload
        else:
            # Partial offload
            layers_ratio = available_vram / model.min_vram_gb
            settings["n_gpu_layers"] = int(80 * layers_ratio)  # Estimate

    elif hardware.gpu_type == "rocm":
        # ROCm/HIP support in llama-cpp-python
        if available_vram >= model.min_vram_gb:
            settings["n_gpu_layers"] = -1
        else:
            layers_ratio = available_vram / model.min_vram_gb
            settings["n_gpu_layers"] = int(80 * layers_ratio)

    elif hardware.gpu_type == "vulkan":
        # Vulkan backend
        if available_vram >= model.min_vram_gb:
            settings["n_gpu_layers"] = -1

    else:
        # CPU only
        settings["n_gpu_layers"] = 0
        # Use more threads for CPU
        settings["n_threads"] = hardware.cpu_cores

    return settings


# =============================================================================
# LLM Clients
# =============================================================================

class BaseLLMClient:
    """Base class for LLM clients with health tracking."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.system_prompt = self._get_system_prompt()
        self.status = ProviderStatus.HEALTHY
        self.provider_name = "unknown"
        self.model = "unknown"
        self._consecutive_failures = 0
        self._max_consecutive_failures = 3

    def is_available(self) -> bool:
        """Check if provider is available for use."""
        return self.status in (ProviderStatus.HEALTHY, ProviderStatus.DEGRADED)

    def mark_success(self):
        """Mark a successful request."""
        self._consecutive_failures = 0
        if self.status == ProviderStatus.DEGRADED:
            self.status = ProviderStatus.HEALTHY
            logger.info(f"{self.provider_name}: Status restored to HEALTHY")

    def mark_rate_limited(self):
        """Mark provider as rate limited."""
        self.status = ProviderStatus.DEGRADED
        logger.warning(f"{self.provider_name}: Rate limited, status DEGRADED")

    def mark_error(self):
        """Mark a general error."""
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._max_consecutive_failures:
            self.status = ProviderStatus.UNHEALTHY
            logger.error(f"{self.provider_name}: Too many failures, status UNHEALTHY")

    def mark_disabled(self, reason: str):
        """Mark provider as disabled (quota/auth issues)."""
        self.status = ProviderStatus.DISABLED
        logger.error(f"{self.provider_name}: DISABLED - {reason}")

    def _get_system_prompt(self) -> str:
        return """You are an expert AI pilot in the classic game Spacewar!.
You control a spaceship competing against an opponent around a central star with gravitational pull.

Given the game state, choose the optimal action:
0: No action (drift with current momentum)
1: Rotate left (counter-clockwise)
2: Rotate right (clockwise)
3: Thrust forward (uses fuel)
4: Fire torpedo (uses ammo, has cooldown)
5: Thrust + Fire (combined action)

Strategic considerations:
- Gravity pulls everything toward the central star - avoid collision!
- Lead your shots - torpedoes inherit your velocity
- Conserve fuel for critical maneuvers
- Conserve ammo - you have limited torpedoes
- Use gravity for slingshot maneuvers
- Predict opponent movement

Respond with ONLY a single digit (0-5) representing your chosen action."""

    def _format_state(self, state: Dict[str, Any]) -> str:
        """Format game state as natural language."""
        return f"""Current State:
Position: ({state['x']:.0f}, {state['y']:.0f})
Velocity: ({state['vx']:.1f}, {state['vy']:.1f})
Facing: {state['angle']:.0f} degrees
Fuel: {state['fuel']:.0f}%
Ammo: {state['ammo']} torpedoes
Fire cooldown ready: {state['can_fire']}

Star distance: {state['star_dist']:.0f} (DANGER if < 100)

Opponent:
- Relative position: ({state['opp_rel_x']:.0f}, {state['opp_rel_y']:.0f})
- Distance: {state['opp_dist']:.0f}
- Facing: {state['opp_angle']:.0f} degrees

Incoming torpedoes: {state['torpedo_count']}"""

    def get_action(self, state: Dict[str, Any]) -> int:
        """Get action from LLM. Override in subclasses."""
        raise NotImplementedError


class OpenAICompatibleClient(BaseLLMClient):
    """Client for OpenAI-compatible APIs (OpenAI, Groq, Together, etc.)."""

    def __init__(
        self,
        config: LLMConfig,
        api_key: str,
        base_url: str,
        model: str,
        provider_name: str = "openai",
    ):
        super().__init__(config)
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.provider_name = provider_name
        self.client = None
        self._init_client()

    def _init_client(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                max_retries=0,  # We handle retries ourselves
            )
        except ImportError:
            logger.error("openai package not installed. Run: pip install openai")
            raise

    def _categorize_error(self, error: Exception) -> Exception:
        """Categorize OpenAI errors - be conservative, don't disable providers aggressively."""
        # Check for specific OpenAI exception types if available
        error_type = type(error).__name__

        # Only check for very specific error codes, not string matching
        status_code = None
        if hasattr(error, 'status_code'):
            status_code = error.status_code
        elif hasattr(error, 'response') and hasattr(error.response, 'status_code'):
            status_code = error.response.status_code

        # Only disable for explicit insufficient_quota error code
        if status_code == 429:
            error_str = str(error).lower()
            # Only mark disabled if explicitly says "insufficient_quota" as the error code
            if "insufficient_quota" in error_str and "code" in error_str:
                self.mark_disabled("Quota exceeded")
                return QuotaExceededError(str(error))
            # Otherwise just rate limit - will retry
            self.mark_rate_limited()
            return RateLimitError(str(error))

        if status_code == 401:
            self.mark_disabled("Authentication failed")
            return AuthenticationError(str(error))

        if status_code in (500, 502, 503, 504):
            self.mark_error()
            return ServerError(str(error))

        # For any other error, just mark as error but don't disable
        self.mark_error()
        return error

    def _make_request(self, state_text: str) -> str:
        """Make the actual API request."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Game state:\n{state_text}\n\nChoose action (0-5):"},
            ],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            timeout=30,
        )
        return response.choices[0].message.content.strip()

    def get_action(self, state: Dict[str, Any]) -> int:
        """Query LLM for action with proper error handling."""
        if self.client is None:
            raise RuntimeError(f"{self.provider_name} client not initialized")

        state_text = self._format_state(state)

        def make_request():
            try:
                return self._make_request(state_text)
            except Exception as e:
                raise self._categorize_error(e)

        def on_retry(attempt, delay, error):
            logger.info(f"{self.provider_name}: Retry {attempt+1} in {delay:.1f}s ({type(error).__name__})")

        try:
            content = retry_with_backoff(
                make_request,
                config=RetryConfig(max_retries=2),
                on_retry=on_retry,
            )

            # Extract first digit
            for char in content:
                if char.isdigit():
                    action = int(char)
                    if 0 <= action <= 5:
                        self.mark_success()
                        return action

            self.mark_success()  # Request succeeded even if response was weird
            return 0

        except (QuotaExceededError, AuthenticationError):
            raise  # Re-raise permanent errors
        except Exception as e:
            logger.warning(f"{self.provider_name}: Query failed - {e}")
            raise


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic API with health tracking."""

    def __init__(self, config: LLMConfig, api_key: str, model: str):
        super().__init__(config)
        self.api_key = api_key
        self.model = model
        self.provider_name = "anthropic"
        self.client = None
        self._init_client()

    def _init_client(self):
        """Initialize Anthropic client."""
        try:
            from anthropic import Anthropic
            self.client = Anthropic(
                api_key=self.api_key,
                max_retries=0,  # We handle retries ourselves
            )
        except ImportError:
            logger.error("anthropic package not installed. Run: pip install anthropic")
            raise

    def _categorize_error(self, error: Exception) -> Exception:
        """Categorize Anthropic errors - be conservative, don't disable aggressively."""
        # Check for specific status codes
        status_code = None
        if hasattr(error, 'status_code'):
            status_code = error.status_code
        elif hasattr(error, 'response') and hasattr(error.response, 'status_code'):
            status_code = error.response.status_code

        if status_code == 429:
            self.mark_rate_limited()
            return RateLimitError(str(error))

        if status_code == 401:
            self.mark_disabled("Authentication failed")
            return AuthenticationError(str(error))

        if status_code in (500, 502, 503, 504, 529):  # 529 = overloaded
            self.mark_error()
            return ServerError(str(error))

        # For any other error, just mark as error but don't disable
        self.mark_error()
        return error

    def _make_request(self, state_text: str) -> str:
        """Make the actual API request."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.config.max_tokens,
            system=self.system_prompt,
            messages=[
                {"role": "user", "content": f"Game state:\n{state_text}\n\nChoose action (0-5):"},
            ],
        )
        return response.content[0].text.strip()

    def get_action(self, state: Dict[str, Any]) -> int:
        """Query Claude for action with proper error handling."""
        if self.client is None:
            raise RuntimeError(f"{self.provider_name} client not initialized")

        state_text = self._format_state(state)

        def make_request():
            try:
                return self._make_request(state_text)
            except Exception as e:
                raise self._categorize_error(e)

        def on_retry(attempt, delay, error):
            logger.info(f"{self.provider_name}: Retry {attempt+1} in {delay:.1f}s ({type(error).__name__})")

        try:
            content = retry_with_backoff(
                make_request,
                config=RetryConfig(max_retries=2),
                on_retry=on_retry,
            )

            # Extract first digit
            for char in content:
                if char.isdigit():
                    action = int(char)
                    if 0 <= action <= 5:
                        self.mark_success()
                        return action

            self.mark_success()
            return 0

        except QuotaExceededError:
            raise
        except AuthenticationError:
            raise
        except Exception as e:
            logger.warning(f"{self.provider_name}: Query failed - {e}")
            raise


class LocalLLMClient(BaseLLMClient):
    """Client for local LLM via llama-cpp-python with auto-download."""

    def __init__(
        self,
        config: LLMConfig,
        hardware: HardwareProfile,
        model_path: Optional[str] = None,
        model_spec: Optional[LocalModelSpec] = None,
        models_dir: str = "./models",
    ):
        super().__init__(config)
        self.hardware = hardware
        self.model_path = model_path
        self.model_spec = model_spec
        self.models_dir = models_dir
        self.llm = None
        self.provider_name = "local"
        self.model = "unknown"
        self._init_llm()

    def _init_llm(self):
        """Initialize llama-cpp-python with auto-download."""
        try:
            from llama_cpp import Llama
        except ImportError:
            logger.error("llama-cpp-python not installed. Run: pip install llama-cpp-python")
            logger.error("For AMD ROCm: CMAKE_ARGS='-DLLAMA_HIPBLAS=on' pip install llama-cpp-python")
            return

        # If no model path, auto-select and download
        if self.model_path is None:
            if self.model_spec is None:
                self.model_spec = select_best_local_model(self.hardware)

            if self.model_spec is None:
                logger.error("No suitable model for hardware")
                return

            # Download if needed
            self.model_path = download_model(self.model_spec, self.models_dir)
            if self.model_path is None:
                logger.error("Failed to download model")
                return

        # Get optimal settings for hardware
        if self.model_spec:
            settings = get_llama_cpp_settings(self.hardware, self.model_spec)
        else:
            # Fallback settings if no spec (manual model path)
            settings = {
                "n_ctx": self.config.local_context_size,
                "n_threads": min(self.hardware.cpu_cores, 16),
                "n_gpu_layers": -1 if self.hardware.gpu_type != "cpu" else 0,
                "verbose": False,
            }

        # Override context size from config if specified
        if self.config.local_context_size:
            settings["n_ctx"] = self.config.local_context_size

        try:
            logger.info(f"Loading local LLM: {self.model_path}")
            logger.info(f"Settings: n_gpu_layers={settings.get('n_gpu_layers')}, "
                       f"n_threads={settings.get('n_threads')}, n_ctx={settings.get('n_ctx')}")

            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=settings.get("n_ctx", 4096),
                n_threads=settings.get("n_threads", 8),
                n_gpu_layers=settings.get("n_gpu_layers", 0),
                n_batch=settings.get("n_batch", 512),
                verbose=settings.get("verbose", False),
            )

            # Set model name from spec or path
            if self.model_spec:
                self.model = self.model_spec.description
            elif self.model_path:
                self.model = Path(self.model_path).stem

            logger.info(f"Local LLM loaded successfully: {self.model_path}")

        except Exception as e:
            logger.error(f"Failed to load local LLM: {e}")
            self.llm = None

    def get_action(self, state: Dict[str, Any]) -> int:
        """Query local LLM for action."""
        if self.llm is None:
            return 0

        state_text = self._format_state(state)
        prompt = f"{self.system_prompt}\n\nGame state:\n{state_text}\n\nAction:"

        try:
            output = self.llm(
                prompt,
                max_tokens=5,
                temperature=self.config.temperature,
                stop=["\n", " ", "."],
            )

            content = output["choices"][0]["text"].strip()
            for char in content:
                if char.isdigit():
                    action = int(char)
                    if 0 <= action <= 5:
                        return action
            return 0

        except Exception as e:
            logger.warning(f"Local LLM query failed: {e}")
            return 0


# =============================================================================
# Multi-Provider LLM Client
# =============================================================================

class MultiProviderClient(BaseLLMClient):
    """
    LLM client that uses multiple providers with intelligent selection and fallback.

    - Uses all available providers
    - Tracks provider health status (HEALTHY, DEGRADED, UNHEALTHY, DISABLED)
    - Automatically falls back on errors
    - Permanently disables providers with quota/auth issues
    - Prioritizes healthy, faster providers
    """

    def __init__(self, config: LLMConfig, clients: Dict[str, BaseLLMClient]):
        super().__init__(config)
        self.clients = clients
        self.provider_name = "multi"
        self.provider_stats: Dict[str, Dict[str, float]] = {
            name: {"success": 0, "failure": 0, "total_time_ms": 0}
            for name in clients.keys()
        }
        self._provider_order = list(clients.keys())

    def _get_available_providers(self) -> List[str]:
        """Get list of available (non-disabled) providers."""
        return [
            name for name, client in self.clients.items()
            if client.is_available()
        ]

    def _get_best_provider(self, exclude: set = None) -> Optional[str]:
        """Select best available provider based on success rate and speed."""
        if exclude is None:
            exclude = set()

        available = [
            name for name in self._get_available_providers()
            if name not in exclude
        ]

        if not available:
            return None

        best_provider = None
        best_score = -1

        for name in available:
            client = self.clients[name]
            stats = self.provider_stats[name]

            total = stats["success"] + stats["failure"]

            # Prioritize healthy over degraded
            health_bonus = 50 if client.status == ProviderStatus.HEALTHY else 0

            if total == 0:
                # Untested provider - try it with medium priority
                score = 25 + health_bonus
            else:
                success_rate = stats["success"] / total
                avg_time = stats["total_time_ms"] / max(1, stats["success"])

                # Score: prioritize success rate, health status, then speed
                speed_score = 1000 / max(avg_time, 1)
                score = success_rate * 100 + health_bonus + speed_score * 0.1

            if score > best_score:
                best_score = score
                best_provider = name

        return best_provider

    def get_action(self, state: Dict[str, Any]) -> int:
        """Query LLM for action, with intelligent fallback across providers."""
        tried_providers = set()

        while True:
            provider_name = self._get_best_provider(exclude=tried_providers)

            if provider_name is None:
                # No more available providers
                break

            tried_providers.add(provider_name)
            client = self.clients[provider_name]

            start_time = time.time()
            try:
                action = client.get_action(state)
                elapsed_ms = (time.time() - start_time) * 1000

                # Record success
                self.provider_stats[provider_name]["success"] += 1
                self.provider_stats[provider_name]["total_time_ms"] += elapsed_ms

                return action

            except QuotaExceededError as e:
                # Provider permanently disabled - don't retry
                logger.error(f"{provider_name}: Quota exceeded, disabling provider")
                self.provider_stats[provider_name]["failure"] += 1
                # Status already set to DISABLED by client
                continue

            except AuthenticationError as e:
                # Provider permanently disabled - don't retry
                logger.error(f"{provider_name}: Auth failed, disabling provider")
                self.provider_stats[provider_name]["failure"] += 1
                # Status already set to DISABLED by client
                continue

            except (RateLimitError, ServerError) as e:
                # Temporary issue - try next provider
                elapsed_ms = (time.time() - start_time) * 1000
                self.provider_stats[provider_name]["failure"] += 1
                logger.warning(f"{provider_name}: Temporary error, trying next provider")
                continue

            except Exception as e:
                elapsed_ms = (time.time() - start_time) * 1000
                self.provider_stats[provider_name]["failure"] += 1
                logger.warning(f"{provider_name}: Error - {e}")
                continue

        # All providers failed or disabled
        available = self._get_available_providers()
        if not available:
            logger.error("All LLM providers are disabled (quota/auth issues)")
        else:
            logger.error(f"All {len(tried_providers)} providers failed this request")

        return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get provider statistics with health status."""
        stats = {}
        for name, s in self.provider_stats.items():
            client = self.clients[name]
            total = s["success"] + s["failure"]
            stats[name] = {
                "status": client.status.value,
                "success_rate": s["success"] / max(1, total),
                "total_calls": total,
                "avg_time_ms": s["total_time_ms"] / max(1, s["success"]) if s["success"] > 0 else 0,
            }
        return stats

    def get_available_count(self) -> int:
        """Get count of available providers."""
        return len(self._get_available_providers())


# =============================================================================
# LLM Agent Factory
# =============================================================================

def _create_single_client(
    config: LLMConfig,
    provider: str,
    api_key: str,
    model: Optional[str] = None,
) -> Tuple[Optional[BaseLLMClient], str]:
    """
    Create a single LLM client for a provider.

    Returns:
        Tuple of (client, model_name) - client may be None on failure
    """
    provider_config = PROVIDER_CONFIGS.get(provider)
    if provider_config is None:
        logger.warning(f"Unknown provider: {provider}")
        return None, ""

    # Discover best model if not specified
    if model is None:
        logger.info(f"Discovering best model for {provider}...")
        model = get_best_model_for_provider(provider, api_key)

    try:
        if provider == "anthropic":
            return AnthropicClient(config, api_key, model), model
        else:
            return OpenAICompatibleClient(
                config, api_key, provider_config.base_url, model,
                provider_name=provider,
            ), model
    except Exception as e:
        logger.warning(f"Failed to create client for {provider}: {e}")
        return None, model


def create_local_llm_client(
    config: LLMConfig,
    models_dir: str = "./models",
) -> Optional[LocalLLMClient]:
    """
    Create a local LLM client with auto-download.

    Detects hardware, selects the best model, downloads if needed,
    and initializes with optimal settings.
    """
    logger.info("Initializing local LLM...")

    # Detect hardware
    hardware = detect_hardware()
    logger.info(f"Hardware: {hardware.cpu_cores} cores, {hardware.ram_gb:.1f}GB RAM")
    if hardware.gpu_name:
        logger.info(f"GPU: {hardware.gpu_name} ({hardware.gpu_vram_gb:.1f}GB VRAM, {hardware.gpu_backend})")

    # Use specified model path if provided
    if config.local_model:
        logger.info(f"Using specified model: {config.local_model}")
        return LocalLLMClient(
            config=config,
            hardware=hardware,
            model_path=config.local_model,
            models_dir=models_dir,
        )

    # Auto-select and download best model
    model_spec = select_best_local_model(hardware)
    if model_spec is None:
        logger.error("No suitable model found for hardware")
        return None

    logger.info(f"Selected model: {model_spec.description}")
    logger.info(f"  Quality score: {model_spec.quality_score}, Size: {model_spec.size_gb:.1f}GB")

    return LocalLLMClient(
        config=config,
        hardware=hardware,
        model_spec=model_spec,
        models_dir=models_dir,
    )


def create_llm_client(
    config: LLMConfig,
    api_manager: Optional[APIKeyManager] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    include_local: bool = False,
    models_dir: str = "./models",
) -> Optional[BaseLLMClient]:
    """
    Create LLM client based on available providers.

    Args:
        config: LLM configuration
        api_manager: API key manager (optional if using local only)
        provider: Specific provider to use (None = use all available)
        model: Specific model to use (None = auto-select best)
        include_local: Include local LLM as a provider option
        models_dir: Directory for local model downloads

    If no provider specified, creates a multi-provider client using ALL available providers.
    If provider specified, creates a single-provider client.
    If include_local is True or config.use_local is True, local LLM is included.
    """

    if config.use_local:
        # Local LLM only mode
        return create_local_llm_client(config, models_dir)

    # Handle local-only request
    if provider == "local":
        return create_local_llm_client(config, models_dir)

    # Check API manager availability
    available_providers = []
    if api_manager is not None:
        available_providers = api_manager.get_available_providers()

    if not available_providers and not include_local:
        logger.error("No API providers available")
        return None

    # If specific provider requested, create single client
    if provider is not None and provider != "local":
        if api_manager is None:
            logger.error("API manager required for API providers")
            return None

        api_key = api_manager.get_key(provider)
        if api_key is None:
            logger.error(f"No API key for provider: {provider}")
            return None

        client, model_name = _create_single_client(config, provider, api_key, model)
        if client:
            logger.info(f"Using LLM: {provider} / {model_name}")
        return client

    # No provider specified - create multi-provider client with ALL available providers
    clients: Dict[str, BaseLLMClient] = {}

    # Sort providers by preference
    if available_providers:
        sorted_providers = sorted(
            available_providers,
            key=lambda p: config.preferred_providers.index(p) if p in config.preferred_providers else 999
        )

        for prov in sorted_providers:
            api_key = api_manager.get_key(prov)
            if api_key:
                client, model_name = _create_single_client(config, prov, api_key, model)
                if client:
                    clients[prov] = client
                    logger.info(f"Initialized LLM provider: {prov} / {model_name}")

    # Include local LLM as a provider option (as fallback or primary)
    if include_local:
        try:
            local_client = create_local_llm_client(config, models_dir)
            if local_client and local_client.llm is not None:
                clients["local"] = local_client
                logger.info("Initialized local LLM provider")
        except Exception as e:
            logger.warning(f"Failed to initialize local LLM: {e}")

    if not clients:
        logger.error("Failed to initialize any LLM providers")
        return None

    if len(clients) == 1:
        # Only one provider - return single client
        name = list(clients.keys())[0]
        logger.info(f"Using single LLM provider: {name}")
        return clients[name]

    # Multiple providers - return multi-provider client
    logger.info(f"Using {len(clients)} LLM providers with automatic selection and fallback")
    return MultiProviderClient(config, clients)


# =============================================================================
# LLM Demonstration Generator
# =============================================================================

class DemonstrationGenerator:
    """
    Generates expert demonstrations using LLM.

    Used for training-time guidance, not real-time play.
    Generates offline demonstrations that can be used for:
    - Imitation learning (behavioral cloning)
    - Reward shaping
    - Exploration guidance
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        env,  # SpacewarEnv
    ):
        self.llm = llm_client
        self.env = env

    def _obs_to_state_dict(self, obs: np.ndarray, env) -> Dict[str, Any]:
        """Convert observation array to state dictionary for LLM."""
        # Extract values from normalized observation
        # Indices based on environment._get_observation structure
        ship = env.ships[0]
        opponent = env.ships[1]

        return {
            "x": ship.position.x,
            "y": ship.position.y,
            "vx": ship.velocity.x,
            "vy": ship.velocity.y,
            "angle": ship.angle,
            "fuel": ship.fuel,
            "ammo": ship.ammo,
            "can_fire": ship.can_fire(),
            "star_dist": ship.position.distance_to(env.star.position),
            "opp_rel_x": opponent.position.x - ship.position.x,
            "opp_rel_y": opponent.position.y - ship.position.y,
            "opp_dist": ship.position.distance_to(opponent.position),
            "opp_angle": opponent.angle,
            "torpedo_count": len([t for t in env.torpedoes if t.owner_id != 0]),
        }

    def generate_episode(self) -> List[Tuple[np.ndarray, int, float]]:
        """
        Generate one episode of LLM demonstrations.

        Returns:
            List of (state, action, reward) tuples
        """
        demonstrations = []

        obs, info = self.env.reset()
        done = False
        truncated = False

        while not (done or truncated):
            # Convert observation to state dict for LLM
            state_dict = self._obs_to_state_dict(obs, self.env)

            # Get LLM action
            action = self.llm.get_action(state_dict)

            # Get opponent action (heuristic for now)
            opp_action = self.env._heuristic_action(1)

            # Step environment
            next_obs, reward, done, truncated, info = self.env.step_both(action, opp_action)

            demonstrations.append((obs.copy(), action, reward))
            obs = next_obs

        return demonstrations

    def generate_demonstrations(
        self,
        n_episodes: int,
        callback=None,
    ) -> List[List[Tuple[np.ndarray, int, float]]]:
        """
        Generate multiple episodes of demonstrations.

        Args:
            n_episodes: Number of episodes to generate
            callback: Optional callback(episode_idx, episode_data) called after each episode

        Returns:
            List of episodes, each episode is list of (state, action, reward)
        """
        all_demonstrations = []

        for i in range(n_episodes):
            episode = self.generate_episode()
            all_demonstrations.append(episode)

            if callback:
                callback(i, episode)

            logger.info(f"Generated demonstration episode {i+1}/{n_episodes}, length={len(episode)}")

        return all_demonstrations


# =============================================================================
# LLM-Guided Exploration
# =============================================================================

class LLMExplorationGuide:
    """
    Uses LLM to guide exploration during RL training.

    Instead of random exploration, uses LLM suggestions with some probability.
    This is more sample-efficient than pure random exploration.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        exploration_prob: float = 0.1,
    ):
        self.llm = llm_client
        self.exploration_prob = exploration_prob
        self._query_count = 0
        self._cache: Dict[str, int] = {}

    def _state_to_cache_key(self, state_dict: Dict[str, Any]) -> str:
        """Create cache key from state (quantized for similarity)."""
        # Quantize values for caching similar states
        return f"{int(state_dict['x']/50)}_{int(state_dict['y']/50)}_{int(state_dict['angle']/30)}"

    def should_use_llm(self) -> bool:
        """Decide whether to use LLM for this step."""
        return np.random.random() < self.exploration_prob

    def get_exploration_action(
        self,
        state_dict: Dict[str, Any],
        use_cache: bool = True,
    ) -> int:
        """Get LLM-guided exploration action."""
        # Check cache first
        if use_cache:
            cache_key = self._state_to_cache_key(state_dict)
            if cache_key in self._cache:
                return self._cache[cache_key]

        # Query LLM
        action = self.llm.get_action(state_dict)
        self._query_count += 1

        # Cache result
        if use_cache:
            self._cache[cache_key] = action

        return action

    def get_stats(self) -> Dict[str, Any]:
        """Get exploration statistics."""
        return {
            "query_count": self._query_count,
            "cache_size": len(self._cache),
            "cache_hit_rate": 1 - (self._query_count / max(1, len(self._cache) + self._query_count)),
        }


# =============================================================================
# LLM Strategy Generator (Code as Policy)
# =============================================================================

STRATEGY_PROMPT = '''You are an expert game AI developer. Write a Python strategy function for the classic game Spacewar!.

GAME RULES:
- Two spaceships compete around a central star with gravitational pull
- Ships can: rotate left/right, thrust forward, fire torpedoes
- Collision with star or torpedo = death
- Limited fuel and ammunition

INPUT: A dictionary 'state' with these keys:
- x, y: Ship position (screen center is ~400, 300)
- vx, vy: Ship velocity
- angle: Ship facing direction in degrees (0-360)
- fuel: Remaining fuel (0-100)
- ammo: Remaining torpedoes (integer)
- can_fire: Boolean, True if fire cooldown is ready
- star_dist: Distance to the central star (DANGER if < 100!)
- opp_rel_x, opp_rel_y: Opponent position relative to ship
- opp_dist: Distance to opponent
- opp_angle: Opponent's facing direction

OUTPUT: Return an integer 0-5:
- 0: No action (drift)
- 1: Rotate left (counter-clockwise)
- 2: Rotate right (clockwise)
- 3: Thrust forward (uses fuel)
- 4: Fire torpedo (uses ammo, has cooldown)
- 5: Thrust + Fire (combined)

STRATEGY TIPS:
- Gravity pulls toward center - use it for slingshot maneuvers
- Lead your shots - torpedoes inherit your velocity
- Conserve fuel for critical maneuvers
- Stay at medium distance - too close is dangerous, too far wastes ammo
- Avoid the star at all costs!

Write ONLY the function, no explanations. Use simple math (no imports needed).
The function must be named 'strategy' and take one argument 'state'.

Example structure:
```python
def strategy(state):
    # Your logic here
    return action  # 0-5
```
'''


@dataclass
class GeneratedStrategy:
    """A strategy generated by an LLM."""
    provider: str
    model: str
    code: str
    function: callable
    generation_time: float


class LLMStrategyGenerator:
    """
    Generates executable strategy code from LLM.

    This follows the "Code as Policies" approach from research:
    - Ask LLM ONCE to generate Python code
    - Execute the code locally (fast, no API calls during play)
    - Much more efficient than per-action API calls
    """

    def __init__(self, llm_client: BaseLLMClient):
        self.llm = llm_client
        self.provider_name = getattr(llm_client, 'provider_name', 'unknown')
        self.model = getattr(llm_client, 'model', 'unknown')

    def _extract_code(self, response: str) -> str:
        """Extract Python code from LLM response."""
        # Try to find code block
        if "```python" in response:
            start = response.find("```python") + 9
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()

        if "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()

        # Look for function definition
        if "def strategy" in response:
            lines = response.split("\n")
            code_lines = []
            in_function = False
            indent_level = 0

            for line in lines:
                if "def strategy" in line:
                    in_function = True
                    indent_level = len(line) - len(line.lstrip())
                    code_lines.append(line)
                elif in_function:
                    if line.strip() == "":
                        code_lines.append(line)
                    elif line.startswith(" " * (indent_level + 1)) or line.startswith("\t"):
                        code_lines.append(line)
                    elif line.strip().startswith("#"):
                        code_lines.append(line)
                    elif not line.strip():
                        continue
                    else:
                        # End of function
                        break

            if code_lines:
                return "\n".join(code_lines)

        # Return as-is if we can't extract
        return response.strip()

    def _validate_and_compile(self, code: str) -> callable:
        """Validate and compile the strategy code."""
        # Check for dangerous operations
        dangerous = ['import ', 'exec(', 'eval(', 'open(', '__', 'subprocess',
                     'os.', 'sys.', 'file', 'input(', 'compile(']
        code_lower = code.lower()
        for d in dangerous:
            if d in code_lower:
                raise ValueError(f"Code contains forbidden operation: {d}")

        # Ensure it has the strategy function
        if "def strategy" not in code:
            raise ValueError("Code must define a 'strategy' function")

        # Compile in restricted namespace
        import math
        namespace = {
            'abs': abs,
            'min': min,
            'max': max,
            'int': int,
            'float': float,
            'bool': bool,
            'round': round,
            'len': len,
            'range': range,
            'sum': sum,
            'math': math,
            'sqrt': math.sqrt,
            'sin': math.sin,
            'cos': math.cos,
            'atan2': math.atan2,
            'pi': math.pi,
            'degrees': math.degrees,
            'radians': math.radians,
        }

        try:
            exec(code, namespace)
        except SyntaxError as e:
            raise ValueError(f"Syntax error in generated code: {e}")
        except Exception as e:
            raise ValueError(f"Error compiling code: {e}")

        if 'strategy' not in namespace:
            raise ValueError("Code did not define 'strategy' function")

        strategy_func = namespace['strategy']

        # Test with a sample state
        test_state = {
            'x': 400, 'y': 300, 'vx': 0, 'vy': 0, 'angle': 0,
            'fuel': 100, 'ammo': 10, 'can_fire': True,
            'star_dist': 200, 'opp_rel_x': 100, 'opp_rel_y': 50,
            'opp_dist': 112, 'opp_angle': 180
        }

        try:
            result = strategy_func(test_state)
            if not isinstance(result, (int, float)) or not (0 <= int(result) <= 5):
                raise ValueError(f"Strategy returned invalid action: {result}")
        except Exception as e:
            raise ValueError(f"Strategy failed test execution: {e}")

        return strategy_func

    def generate(self) -> Optional[GeneratedStrategy]:
        """
        Generate a strategy by querying the LLM once.

        Returns:
            GeneratedStrategy with executable function, or None on failure
        """
        logger.info(f"Generating strategy from {self.provider_name}/{self.model}...")
        start_time = time.time()

        try:
            # Query LLM for strategy code
            if hasattr(self.llm, 'llm') and self.llm.llm is not None:
                # LocalLLMClient - use llama-cpp's completion API
                prompt = f"<|im_start|>user\n{STRATEGY_PROMPT}<|im_end|>\n<|im_start|>assistant\n"
                output = self.llm.llm(
                    prompt,
                    max_tokens=2000,
                    temperature=0.7,
                    stop=["<|im_end|>", "<|im_start|>"],
                )
                code_response = output["choices"][0]["text"]
            elif hasattr(self.llm, 'client') and self.llm.client is not None:
                # API client - use raw client for custom prompt
                if self.provider_name == 'anthropic':
                    response = self.llm.client.messages.create(
                        model=self.model,
                        max_tokens=2000,
                        messages=[{"role": "user", "content": STRATEGY_PROMPT}],
                    )
                    code_response = response.content[0].text
                else:
                    # OpenAI-compatible
                    response = self.llm.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": STRATEGY_PROMPT}],
                        max_tokens=2000,
                        temperature=0.7,
                    )
                    code_response = response.choices[0].message.content
            else:
                raise RuntimeError("LLM client not initialized")

            # Extract code from response
            code = self._extract_code(code_response)
            logger.debug(f"Extracted code:\n{code}")

            # Validate and compile
            strategy_func = self._validate_and_compile(code)

            generation_time = time.time() - start_time
            logger.info(f"Strategy generated successfully in {generation_time:.1f}s")

            return GeneratedStrategy(
                provider=self.provider_name,
                model=self.model,
                code=code,
                function=strategy_func,
                generation_time=generation_time,
            )

        except Exception as e:
            logger.error(f"Failed to generate strategy from {self.provider_name}: {e}")
            return None

    @staticmethod
    def create_fallback_strategy() -> GeneratedStrategy:
        """Create a simple fallback strategy if LLM fails."""
        code = '''
def strategy(state):
    # Fallback heuristic strategy

    # Priority 1: Avoid star
    if state['star_dist'] < 120:
        return 3  # Thrust away

    # Priority 2: Fire at close opponent
    if state['opp_dist'] < 200 and state['can_fire'] and state['ammo'] > 0:
        # Simple aim check
        import math
        dx, dy = state['opp_rel_x'], state['opp_rel_y']
        target_angle = math.degrees(math.atan2(dy, dx)) % 360
        angle_diff = (target_angle - state['angle'] + 180) % 360 - 180
        if abs(angle_diff) < 30:
            return 4  # Fire
        elif angle_diff > 0:
            return 1  # Rotate left
        else:
            return 2  # Rotate right

    # Priority 3: Approach opponent
    if state['opp_dist'] > 300 and state['fuel'] > 20:
        return 3  # Thrust toward

    # Priority 4: Aim at opponent
    import math
    dx, dy = state['opp_rel_x'], state['opp_rel_y']
    target_angle = math.degrees(math.atan2(dy, dx)) % 360
    angle_diff = (target_angle - state['angle'] + 180) % 360 - 180
    if abs(angle_diff) > 10:
        return 1 if angle_diff > 0 else 2

    return 0  # Drift
'''
        import math
        namespace = {'math': math}
        exec(code, namespace)

        return GeneratedStrategy(
            provider='fallback',
            model='heuristic',
            code=code,
            function=namespace['strategy'],
            generation_time=0.0,
        )


def generate_strategies_for_providers(
    config: LLMConfig,
    api_manager: 'APIKeyManager',
    providers: Optional[List[str]] = None,
) -> Dict[str, GeneratedStrategy]:
    """
    Generate strategies from multiple LLM providers.

    Args:
        config: LLM configuration
        api_manager: API key manager
        providers: List of providers to use, or None for all available

    Returns:
        Dict mapping provider name to GeneratedStrategy
    """
    if providers is None:
        providers = api_manager.get_available_providers()

    strategies = {}

    for provider in providers:
        api_key = api_manager.get_key(provider)
        if not api_key:
            continue

        try:
            client, model = _create_single_client(config, provider, api_key)
            if client is None:
                continue

            generator = LLMStrategyGenerator(client)
            strategy = generator.generate()

            if strategy:
                strategies[provider] = strategy
                logger.info(f"Generated strategy for {provider}: {len(strategy.code)} chars")
            else:
                logger.warning(f"Failed to generate strategy for {provider}, using fallback")
                strategies[provider] = LLMStrategyGenerator.create_fallback_strategy()
                strategies[provider].provider = provider
                strategies[provider].model = "fallback"

        except Exception as e:
            logger.error(f"Error generating strategy for {provider}: {e}")
            strategies[provider] = LLMStrategyGenerator.create_fallback_strategy()
            strategies[provider].provider = provider

    return strategies


# =============================================================================
# Iterative Strategy Refinement (Eureka-style)
# =============================================================================

REFINEMENT_PROMPT_TEMPLATE = '''You previously wrote this Spacewar! strategy:

```python
{current_code}
```

PERFORMANCE RESULTS after {episodes} episodes:
- Win rate: {win_rate:.1%}
- Average reward: {avg_reward:.1f}
- Best reward: {best_reward:.1f}
- Worst reward: {worst_reward:.1f}
- Wins: {wins}, Losses: {losses}, Draws: {draws}

COMMON ISSUES OBSERVED:
{issues}

IMPROVEMENT NEEDED: The strategy is {assessment}.

Please write an IMPROVED strategy that addresses these issues. Consider:
1. If win rate is low, improve attack accuracy and timing
2. If dying to star often, prioritize star avoidance
3. If running out of fuel/ammo, be more conservative
4. If opponent is winning, study their likely patterns

Write ONLY the improved function. Keep the same signature: def strategy(state):

```python
def strategy(state):
    # Your improved logic here
    return action  # 0-5
```
'''


@dataclass
class StrategyPerformance:
    """Tracks performance metrics for a strategy."""
    episodes: int = 0
    total_reward: float = 0.0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    rewards: List[float] = field(default_factory=list)
    death_by_star: int = 0
    death_by_torpedo: int = 0
    out_of_fuel: int = 0
    out_of_ammo: int = 0

    @property
    def avg_reward(self) -> float:
        return self.total_reward / max(1, self.episodes)

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses + self.draws
        return self.wins / max(1, total)

    @property
    def best_reward(self) -> float:
        return max(self.rewards) if self.rewards else 0.0

    @property
    def worst_reward(self) -> float:
        return min(self.rewards) if self.rewards else 0.0

    def add_episode(self, reward: float, won: bool, lost: bool, info: Dict[str, Any] = None):
        """Record an episode result."""
        self.episodes += 1
        self.total_reward += reward
        self.rewards.append(reward)

        if won:
            self.wins += 1
        elif lost:
            self.losses += 1
        else:
            self.draws += 1

        # Track failure modes from info
        if info:
            if info.get('death_cause') == 'star':
                self.death_by_star += 1
            elif info.get('death_cause') == 'torpedo':
                self.death_by_torpedo += 1
            if info.get('out_of_fuel'):
                self.out_of_fuel += 1
            if info.get('out_of_ammo'):
                self.out_of_ammo += 1

    def reset(self):
        """Reset for new iteration."""
        self.episodes = 0
        self.total_reward = 0.0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.rewards = []
        self.death_by_star = 0
        self.death_by_torpedo = 0
        self.out_of_fuel = 0
        self.out_of_ammo = 0

    def get_issues_summary(self) -> str:
        """Generate a summary of common issues."""
        issues = []
        total = max(1, self.episodes)

        if self.death_by_star > total * 0.1:
            issues.append(f"- Dying to star collision too often ({self.death_by_star}/{total} episodes)")
        if self.death_by_torpedo > total * 0.3:
            issues.append(f"- Getting hit by torpedoes frequently ({self.death_by_torpedo}/{total})")
        if self.out_of_fuel > total * 0.2:
            issues.append(f"- Running out of fuel ({self.out_of_fuel}/{total}) - need fuel conservation")
        if self.out_of_ammo > total * 0.3:
            issues.append(f"- Wasting ammo ({self.out_of_ammo}/{total}) - need better aim")
        if self.win_rate < 0.3:
            issues.append("- Very low win rate - strategy is ineffective")
        if self.avg_reward < 0:
            issues.append("- Negative average reward - taking too much damage")

        if not issues:
            issues.append("- No major issues detected, but can still improve")

        return "\n".join(issues)

    def get_assessment(self) -> str:
        """Get overall assessment."""
        if self.win_rate >= 0.7:
            return "performing well but could be optimized"
        elif self.win_rate >= 0.5:
            return "adequate but needs improvement"
        elif self.win_rate >= 0.3:
            return "underperforming and needs significant changes"
        else:
            return "failing badly and needs a complete rethink"


class IterativeStrategyTrainer:
    """
    Iteratively refines LLM-generated strategies based on training feedback.

    Implements Eureka-style iterative refinement:
    1. Generate initial strategy
    2. Train for N episodes
    3. Evaluate performance
    4. If not improving, ask LLM to refine strategy
    5. Repeat
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        episodes_per_iteration: int = 500,
        max_iterations: int = 10,
        improvement_threshold: float = 0.05,  # 5% improvement required
        min_win_rate: float = 0.6,  # Target win rate
    ):
        self.llm = llm_client
        self.provider_name = getattr(llm_client, 'provider_name', 'unknown')
        self.model = getattr(llm_client, 'model', 'unknown')

        self.episodes_per_iteration = episodes_per_iteration
        self.max_iterations = max_iterations
        self.improvement_threshold = improvement_threshold
        self.min_win_rate = min_win_rate

        self.generator = LLMStrategyGenerator(llm_client)

        # Current state
        self.current_strategy: Optional[GeneratedStrategy] = None
        self.current_performance = StrategyPerformance()
        self.iteration = 0
        self.best_win_rate = 0.0
        self.best_strategy: Optional[GeneratedStrategy] = None

        # History
        self.history: List[Dict[str, Any]] = []

    def initialize(self) -> Optional[GeneratedStrategy]:
        """Generate initial strategy."""
        logger.info(f"Generating initial strategy from {self.provider_name}...")
        self.current_strategy = self.generator.generate()

        if self.current_strategy:
            self.iteration = 1
            logger.info(f"Initial strategy generated ({len(self.current_strategy.code)} chars)")
            return self.current_strategy
        else:
            logger.warning("Failed to generate initial strategy, using fallback")
            self.current_strategy = LLMStrategyGenerator.create_fallback_strategy()
            self.iteration = 1
            return self.current_strategy

    def record_episode(self, reward: float, won: bool, lost: bool, info: Dict[str, Any] = None):
        """Record an episode result."""
        self.current_performance.add_episode(reward, won, lost, info)

    def should_refine(self) -> bool:
        """Check if we should refine the strategy."""
        perf = self.current_performance

        # Not enough episodes yet
        if perf.episodes < self.episodes_per_iteration:
            return False

        # Already at max iterations
        if self.iteration >= self.max_iterations:
            logger.info(f"Max iterations ({self.max_iterations}) reached")
            return False

        # Already performing well
        if perf.win_rate >= self.min_win_rate:
            logger.info(f"Target win rate achieved ({perf.win_rate:.1%} >= {self.min_win_rate:.1%})")
            return False

        # Check if improving
        if perf.win_rate > self.best_win_rate + self.improvement_threshold:
            # Good improvement, but continue if not at target
            self.best_win_rate = perf.win_rate
            self.best_strategy = self.current_strategy
            logger.info(f"New best win rate: {perf.win_rate:.1%}")

        return True

    def refine_strategy(self) -> Optional[GeneratedStrategy]:
        """Ask LLM to refine the strategy based on performance feedback."""
        if not self.current_strategy:
            return self.initialize()

        perf = self.current_performance

        # Save history
        self.history.append({
            'iteration': self.iteration,
            'win_rate': perf.win_rate,
            'avg_reward': perf.avg_reward,
            'episodes': perf.episodes,
            'code': self.current_strategy.code,
        })

        # Build refinement prompt
        prompt = REFINEMENT_PROMPT_TEMPLATE.format(
            current_code=self.current_strategy.code,
            episodes=perf.episodes,
            win_rate=perf.win_rate,
            avg_reward=perf.avg_reward,
            best_reward=perf.best_reward,
            worst_reward=perf.worst_reward,
            wins=perf.wins,
            losses=perf.losses,
            draws=perf.draws,
            issues=perf.get_issues_summary(),
            assessment=perf.get_assessment(),
        )

        logger.info(f"Iteration {self.iteration} complete: "
                   f"win_rate={perf.win_rate:.1%}, avg_reward={perf.avg_reward:.1f}")
        logger.info(f"Requesting strategy refinement from {self.provider_name}...")

        try:
            # Query LLM for refined strategy
            if hasattr(self.llm, 'llm') and self.llm.llm is not None:
                # LocalLLMClient - use llama-cpp's completion API
                llm_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                output = self.llm.llm(
                    llm_prompt,
                    max_tokens=2000,
                    temperature=0.7,
                    stop=["<|im_end|>", "<|im_start|>"],
                )
                code_response = output["choices"][0]["text"]
            elif hasattr(self.llm, 'client') and self.llm.client is not None:
                if self.provider_name == 'anthropic':
                    response = self.llm.client.messages.create(
                        model=self.model,
                        max_tokens=2000,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    code_response = response.content[0].text
                else:
                    # OpenAI-compatible
                    response = self.llm.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=2000,
                        temperature=0.7,
                    )
                    code_response = response.choices[0].message.content
            else:
                raise RuntimeError("LLM client not initialized")

            # Extract and validate new code
            code = self.generator._extract_code(code_response)
            strategy_func = self.generator._validate_and_compile(code)

            # Update current strategy
            self.current_strategy = GeneratedStrategy(
                provider=self.provider_name,
                model=self.model,
                code=code,
                function=strategy_func,
                generation_time=0.0,
            )

            self.iteration += 1
            self.current_performance.reset()

            logger.info(f"Strategy refined (iteration {self.iteration}, {len(code)} chars)")
            return self.current_strategy

        except Exception as e:
            logger.error(f"Failed to refine strategy: {e}")
            # Keep current strategy
            self.current_performance.reset()
            self.iteration += 1
            return self.current_strategy

    def get_strategy_function(self) -> callable:
        """Get the current strategy function."""
        if self.current_strategy:
            return self.current_strategy.function
        return LLMStrategyGenerator.create_fallback_strategy().function

    def get_best_strategy(self) -> Optional[GeneratedStrategy]:
        """Get the best performing strategy so far."""
        return self.best_strategy or self.current_strategy

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all iterations."""
        return {
            'provider': self.provider_name,
            'model': self.model,
            'total_iterations': self.iteration,
            'best_win_rate': self.best_win_rate,
            'history': self.history,
            'best_strategy_code': self.best_strategy.code if self.best_strategy else None,
        }

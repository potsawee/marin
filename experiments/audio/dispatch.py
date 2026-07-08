# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fray dispatch for audio train runs (grug pattern: LocalClient runs in-process on local GPUs)."""

import logging
import os
import re

from fray.cluster import ResourceConfig
from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, create_environment
from marin.training.run_environment import extras_for_resources
from marin.training.training import resolve_training_env

from experiments.audio.train_audio_lm import AudioTrainConfig, main

logger = logging.getLogger(__name__)

_FORWARDED_ENV_PREFIXES = ("XLA_FLAGS", "NCCL_", "JAX_", "LEVANTER_")
_FORWARDED_ENV_EXCLUDE = ("JAX_PLATFORMS",)


def _forwarded_env_vars() -> dict[str, str]:
    return {
        k: v for k, v in os.environ.items() if k.startswith(_FORWARDED_ENV_PREFIXES) and k not in _FORWARDED_ENV_EXCLUDE
    }


def dispatch_audio_training_run(
    *,
    run_id: str,
    config: AudioTrainConfig,
    resources: ResourceConfig,
    max_retries_failure: int = 3,
) -> None:
    """Submit one audio train run through Fray and wait for completion."""
    safe_run_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", run_id)
    env_vars = resolve_training_env(base_env=_forwarded_env_vars(), resources=resources)
    request = JobRequest(
        name=f"audio-train-{safe_run_id}",
        entrypoint=Entrypoint.from_callable(main, args=[config]),
        resources=resources,
        environment=create_environment(env_vars=env_vars, extras=extras_for_resources(resources)),
        max_retries_failure=max_retries_failure,
    )
    logger.info("Dispatching audio training via Fray: %s", request.name)
    job = current_client().submit(request)
    job.wait(raise_on_failure=True)

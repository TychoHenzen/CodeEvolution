from __future__ import annotations

import logging
import subprocess
import time
from urllib.error import URLError
from urllib.request import urlopen

logger = logging.getLogger(__name__)


class LlamaServer:
    """Manages a llama-server subprocess lifecycle."""

    def __init__(self, config):
        self._config = config
        self._process: subprocess.Popen | None = None

    def _build_args(self) -> list[str]:
        cfg = self._config
        args = [
            cfg.server_path,
            "-m", cfg.model_path,
            "--port", str(cfg.port),
            "-ngl", str(cfg.gpu_layers),
            "-c", str(cfg.context_size),
            "-t", str(cfg.threads),
            "--cache-type-k", cfg.cache_type_k,
            "--cache-type-v", cfg.cache_type_v,
        ]
        if cfg.flash_attn:
            args.append("--flash-attn")
        return args

    def start(self, timeout: float = 120) -> None:
        args = self._build_args()
        logger.info("Starting llama-server: %s", " ".join(args))

        self._process = subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        # Check it didn't exit immediately
        time.sleep(0.5)
        if self._process.poll() is not None:
            stderr = self._process.stderr.read().decode(errors="replace")
            raise RuntimeError(
                f"llama-server exited immediately (code {self._process.returncode}): {stderr}"
            )

        self._wait_until_ready(timeout)
        logger.info("llama-server ready on port %d", self._config.port)

    def _wait_until_ready(self, timeout: float) -> None:
        url = f"http://localhost:{self._config.port}/health"
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                resp = urlopen(url, timeout=2)
                if resp.status == 200:
                    return
            except (URLError, OSError):
                pass
            if self._process.poll() is not None:
                stderr = self._process.stderr.read().decode(errors="replace")
                raise RuntimeError(
                    f"llama-server exited during startup (code {self._process.returncode}): {stderr}"
                )
            time.sleep(1)
        raise TimeoutError(
            f"llama-server not ready after {timeout}s on port {self._config.port}"
        )

    def stop(self) -> None:
        if self._process is None:
            return
        logger.info("Stopping llama-server (pid %d)", self._process.pid)
        self._process.terminate()
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        self._process = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

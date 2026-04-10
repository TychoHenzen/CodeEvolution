from __future__ import annotations

import json
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
        self._stderr_log = None

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
            args.extend(["--flash-attn", "on"])
        return args

    def start(self, timeout: float = 300) -> None:
        args = self._build_args()
        logger.info("Starting llama-server: %s", " ".join(args))

        # Send stderr to a temp file instead of PIPE to avoid deadlocks
        # when llama-server writes verbose model-loading logs.
        import tempfile
        self._stderr_log = tempfile.NamedTemporaryFile(
            mode="w+", suffix=".log", prefix="llama-server-", delete=False,
        )

        self._process = subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=self._stderr_log,
        )

        # Check it didn't exit immediately
        time.sleep(1)
        if self._process.poll() is not None:
            stderr = self._read_stderr_tail()
            raise RuntimeError(
                f"llama-server exited immediately (code {self._process.returncode}): {stderr}"
            )

        self._wait_until_ready(timeout)
        logger.info("llama-server ready on port %d", self._config.port)

    def _read_stderr_tail(self, max_bytes: int = 4096) -> str:
        """Read the last max_bytes from the stderr log file."""
        if self._stderr_log is None:
            return ""
        try:
            self._stderr_log.flush()
            self._stderr_log.seek(0, 2)  # seek to end
            size = self._stderr_log.tell()
            start = max(0, size - max_bytes)
            self._stderr_log.seek(start)
            return self._stderr_log.read()
        except (OSError, ValueError):
            return ""

    def _wait_until_ready(self, timeout: float) -> None:
        url = f"http://localhost:{self._config.port}/health"
        deadline = time.monotonic() + timeout
        last_status = None
        while time.monotonic() < deadline:
            try:
                resp = urlopen(url, timeout=2)
                body = resp.read()
                if resp.status == 200:
                    try:
                        data = json.loads(body)
                        status = data.get("status", "unknown")
                    except (json.JSONDecodeError, ValueError):
                        status = "ok"
                    if status == "ok" or status == "no slot available":
                        return
                    if status != last_status:
                        logger.info("llama-server health: %s", status)
                        last_status = status
            except (URLError, OSError):
                pass
            if self._process.poll() is not None:
                stderr = self._read_stderr_tail()
                raise RuntimeError(
                    f"llama-server exited during startup (code {self._process.returncode}): {stderr}"
                )
            time.sleep(2)
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
        if self._stderr_log is not None:
            import os
            name = self._stderr_log.name
            self._stderr_log.close()
            try:
                os.unlink(name)
            except OSError:
                pass
            self._stderr_log = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

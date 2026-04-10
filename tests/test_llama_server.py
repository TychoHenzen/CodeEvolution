import subprocess
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from codeevolve.llama_server import LlamaServer


@dataclass
class _TestConfig:
    server_path: str = "llama-server"
    model_path: str = "/models/test.gguf"
    port: int = 8080
    gpu_layers: int = 30
    context_size: int = 4096
    threads: int = 8
    cache_type_k: str = "q8_0"
    cache_type_v: str = "q8_0"
    flash_attn: bool = True


@pytest.fixture
def server_config():
    return _TestConfig()


@patch("codeevolve.llama_server.urlopen")
@patch("codeevolve.llama_server.subprocess.Popen")
def test_start_launches_process(mock_popen, mock_urlopen, server_config):
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_popen.return_value = mock_proc
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = b'{"status":"ok"}'
    mock_urlopen.return_value = mock_resp

    server = LlamaServer(server_config)
    server.start()

    args = mock_popen.call_args[0][0]
    assert "llama-server" in args[0]
    assert "-m" in args
    assert "/models/test.gguf" in args
    assert "--port" in args
    assert "8080" in args
    assert "-ngl" in args
    assert "30" in args
    assert "-c" in args
    assert "4096" in args
    assert "-t" in args
    assert "8" in args
    assert "--flash-attn" in args

    server.stop()
    mock_proc.terminate.assert_called_once()


@patch("codeevolve.llama_server.urlopen")
@patch("codeevolve.llama_server.subprocess.Popen")
def test_context_manager(mock_popen, mock_urlopen, server_config):
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_popen.return_value = mock_proc
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = b'{"status":"ok"}'
    mock_urlopen.return_value = mock_resp

    with LlamaServer(server_config) as srv:
        assert srv._process is not None

    mock_proc.terminate.assert_called_once()


@patch("codeevolve.llama_server.urlopen")
@patch("codeevolve.llama_server.subprocess.Popen")
def test_start_raises_on_early_exit(mock_popen, mock_urlopen, server_config):
    mock_proc = MagicMock()
    mock_proc.poll.return_value = 1
    mock_proc.stderr = MagicMock()
    mock_proc.stderr.read.return_value = b"error loading model"
    mock_popen.return_value = mock_proc

    server = LlamaServer(server_config)
    with pytest.raises(RuntimeError, match="llama-server exited"):
        server.start()


@patch("codeevolve.llama_server.urlopen")
@patch("codeevolve.llama_server.subprocess.Popen")
def test_no_flash_attn(mock_popen, mock_urlopen, server_config):
    server_config.flash_attn = False
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_popen.return_value = mock_proc
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = b'{"status":"ok"}'
    mock_urlopen.return_value = mock_resp

    server = LlamaServer(server_config)
    server.start()

    args = mock_popen.call_args[0][0]
    assert "--flash-attn" not in args

    server.stop()

"""A2A Process Manager — spawns the A2A server as a subprocess if no external URL is provided.

On `fiber start`, if A2A_SERVER_URL is not set, this manager:
1. Spawns the a2a_server as a subprocess on the configured port
2. Waits for the /health endpoint to respond
3. Sets A2A_SERVER_URL in the environment so the processor agent can discover it
4. Kills the subprocess on shutdown
"""
import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_PORT = 3008
HEALTH_TIMEOUT = 30  # seconds to wait for health check


class A2AProcessManager:
    """Manages the A2A server subprocess lifecycle."""

    def __init__(self, port: int = None, a2a_server_dir: str = None):
        self.port = port or int(os.getenv("A2A_PORT", str(DEFAULT_PORT)))
        self.a2a_server_dir = a2a_server_dir or os.getenv("A2A_SERVER_DIR", "")
        self._process: Optional[asyncio.subprocess.Process] = None
        self._url = f"http://localhost:{self.port}"

    @property
    def url(self) -> str:
        return self._url

    async def start(self) -> bool:
        """Start the A2A server subprocess and wait for it to be healthy.

        Returns True if the server started successfully, False otherwise.
        """
        # If A2A_SERVER_URL is already set, assume external server — skip spawn
        existing_url = os.getenv("A2A_SERVER_URL")
        if existing_url:
            self._url = existing_url
            logger.info(f"A2A server URL already configured: {existing_url}")
            return await self._health_check()

        # Resolve server directory
        server_dir = self._resolve_server_dir()
        if not server_dir:
            logger.warning("A2A server directory not found — A2A delegation will not be available")
            return False

        logger.info(f"Starting A2A server subprocess on port {self.port} from {server_dir}")

        env = os.environ.copy()
        env["PORT"] = str(self.port)

        try:
            self._process = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "a2a_server.main",
                cwd=str(server_dir),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            logger.info(f"A2A server subprocess started (PID {self._process.pid})")

            # Set env var so processor agent discovers it
            os.environ["A2A_SERVER_URL"] = self._url

            # Wait for health
            if await self._wait_healthy():
                logger.info(f"A2A server healthy at {self._url}")
                return True
            else:
                logger.error("A2A server failed health check — killing subprocess")
                await self.stop()
                return False

        except Exception as e:
            logger.error(f"Failed to start A2A server subprocess: {e}")
            return False

    async def stop(self):
        """Stop the A2A server subprocess."""
        if self._process and self._process.returncode is None:
            logger.info(f"Stopping A2A server subprocess (PID {self._process.pid})")
            try:
                self._process.terminate()
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning("A2A server did not terminate gracefully — killing")
                    self._process.kill()
                    await self._process.wait()
            except Exception as e:
                logger.error(f"Error stopping A2A server: {e}")
            self._process = None

        # Clean up env var if we set it
        if os.getenv("A2A_SERVER_URL") == self._url:
            os.environ.pop("A2A_SERVER_URL", None)

    async def _wait_healthy(self) -> bool:
        """Poll /health until it responds or timeout."""
        import httpx

        for i in range(HEALTH_TIMEOUT * 2):  # Check every 0.5s
            # Check if process died
            if self._process and self._process.returncode is not None:
                stderr = ""
                if self._process.stderr:
                    stderr = (await self._process.stderr.read()).decode(errors="replace")
                logger.error(f"A2A server exited with code {self._process.returncode}: {stderr[:500]}")
                return False

            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(2.0)) as client:
                    resp = await client.get(f"{self._url}/health")
                    if resp.status_code == 200:
                        return True
            except (httpx.ConnectError, httpx.ReadTimeout):
                pass

            await asyncio.sleep(0.5)

        return False

    async def _health_check(self) -> bool:
        """Single health check against the configured URL."""
        import httpx

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                resp = await client.get(f"{self._url}/health")
                return resp.status_code == 200
        except Exception:
            return False

    def _resolve_server_dir(self) -> Optional[Path]:
        """Find the a2a_server package directory."""
        if self.a2a_server_dir:
            p = Path(self.a2a_server_dir)
            if p.exists():
                return p

        # Check common locations relative to the fiberwise project
        candidates = [
            Path(__file__).resolve().parents[3] / "a2a-server",
            Path(__file__).resolve().parents[3] / "agent-a2a-server",
            Path(os.getenv("A2A_SERVER_DIR", "")) if os.getenv("A2A_SERVER_DIR") else None,
        ]

        for c in candidates:
            if c and c.exists() and (c / "a2a_server" / "main.py").exists():
                return c

        return None

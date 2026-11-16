import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Annotated, Literal
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger("openai_images_api")


class NoAuth(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["none"] = "none"


class BearerAuth(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["bearer"]
    token_env: str


class BasicAuth(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["basic"]
    username_env: str
    password_env: str


class BackendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(default="", pattern=r"^[a-zA-Z0-9_-]*$")
    url: str
    auth: NoAuth | BearerAuth | BasicAuth = Field(default_factory=NoAuth, discriminator="type")
    headers_env: dict[str, str] = Field(default_factory=dict)
    model_overrides: dict[
        Literal["t2i_4", "t2i_8", "edit_4", "edit_8"],
        Annotated[str, Field(min_length=1)],
    ] = Field(default_factory=dict)


@dataclass(eq=False)
class Backend:
    name: str
    url: str = field(repr=False)
    headers: dict[str, str] = field(repr=False)
    auth: httpx.BasicAuth | None = field(default=None, repr=False)
    model_overrides: dict[str, str] = field(default_factory=dict)

    def client(self):
        return httpx.AsyncClient(
            base_url=self.url, headers=self.headers, auth=self.auth,
            timeout=httpx.Timeout(60, connect=5),
        )


def load_backends() -> list[Backend]:
    raw = os.getenv("COMFY_BACKENDS")
    try:
        configs = json.loads(raw) if raw is not None else [
            {"url": os.getenv("COMFY_URL", "http://127.0.0.1:8188")}
        ]
    except ValueError:
        raise RuntimeError("COMFY_BACKENDS must be a JSON array") from None
    if not isinstance(configs, list) or not configs:
        raise RuntimeError("COMFY_BACKENDS must contain at least one backend")

    backends = []
    names = set()
    urls = set()
    for index, config in enumerate(configs, 1):
        try:
            settings = BackendConfig.model_validate(config)
            parsed = urlsplit(settings.url)
            if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                    or parsed.username is not None or parsed.query or parsed.fragment):
                raise ValueError()
            _ = parsed.port
            backend = Backend(
                settings.name or f"comfy-{index}", settings.url.rstrip("/"), {},
                model_overrides=settings.model_overrides,
            )
            if backend.name in names or backend.url in urls:
                raise ValueError()
            for header, env in settings.headers_env.items():
                backend.headers[header] = required_env(env)
            if isinstance(settings.auth, BearerAuth):
                if "authorization" in {key.lower() for key in backend.headers}:
                    raise ValueError()
                backend.headers["Authorization"] = f"Bearer {required_env(settings.auth.token_env)}"
            if isinstance(settings.auth, BasicAuth):
                if "authorization" in {key.lower() for key in backend.headers}:
                    raise ValueError()
                backend.auth = httpx.BasicAuth(required_env(settings.auth.username_env),
                                               required_env(settings.auth.password_env))
            httpx.Headers(backend.headers)
        except (ValueError, ValidationError):
            raise RuntimeError(f"Invalid configuration for ComfyUI backend #{index}") from None
        names.add(backend.name)
        urls.add(backend.url)
        backends.append(backend)
    return backends


def required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError()
    return value


class NoBackendAvailable(Exception):
    pass


async def backend_healthy(client: httpx.AsyncClient) -> bool:
    try:
        response = await client.get("system_stats", timeout=5)
        return response.status_code == 200 and isinstance(response.json(), dict)
    except (httpx.HTTPError, ValueError):
        return False


class BackendPool:
    def __init__(self, backends: list[Backend]):
        self.backends = backends.copy()
        self.reset()

    def reset(self):
        self.available = asyncio.Condition()
        self.in_use = set()

    async def health(self, backend: Backend) -> bool:
        async with backend.client() as client:
            return await backend_healthy(client)

    @asynccontextmanager
    async def lease(self):
        remaining = set(self.backends)
        while remaining:
            async with self.available:
                await self.available.wait_for(lambda: remaining - self.in_use)
                backend = next(b for b in self.backends if b in remaining and b not in self.in_use)
                self.backends.remove(backend)
                self.backends.append(backend)
                self.in_use.add(backend)
            try:
                async with backend.client() as client:
                    if await backend_healthy(client):
                        yield backend, client
                        return
                logger.warning("ComfyUI backend %s failed its health check", backend.name)
                remaining.remove(backend)
            finally:
                async with self.available:
                    self.in_use.remove(backend)
                    self.available.notify_all()
        raise NoBackendAvailable

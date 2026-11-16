import asyncio
import io
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from PIL import Image


class ImagesAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("OPENAI_API_KEY", "test-key")
        import openai_images_api

        cls.api = openai_images_api

    def test_size_and_tenth_reference(self):
        self.assertEqual(self.api.dimensions("705X1024"), (704, 1024))
        for invalid in ("zero", "0x1024", "-32x1024", "99999x1024"):
            with self.assertRaises(self.api.HTTPException) as err:
                self.api.dimensions(invalid)
            self.assertEqual(err.exception.status_code, 400)
        graph = self.api.graph("test", 704, 1024, 42, 4, [str(i) for i in range(10)])
        for index in range(1, 11):
            node_id, _ = graph["4"]["inputs"][f"images.image_{index}"]
            if index == 1:
                node_id, _ = graph[node_id]["inputs"]["image"]
            self.assertEqual(graph[node_id]["inputs"]["image"], str(index - 1))

    def test_deferred_edit_through_authenticated_backend(self):
        from comfy_backend import BackendPool, load_backends

        with patch.dict(os.environ, {
            "COMFY_BACKENDS": json.dumps([
                {"url": "http://offline", "model_overrides": {"edit_8": "unavailable.safetensors"}},
                {
                    "url": "https://worker/prefix",
                    "auth": {"type": "bearer", "token_env": "COMFY_TEST_TOKEN"},
                    "model_overrides": {"edit_8": "qwen-warm/edit-8.safetensors"},
                },
            ]),
            "COMFY_TEST_TOKEN": "upstream-token",
        }):
            pool = BackendPool(load_backends())

        output = io.BytesIO()
        Image.new("RGB", (64, 64)).save(output, format="PNG")
        png = output.getvalue()
        calls = []

        def upstream(request):
            if request.url.host == "offline":
                self.assertEqual(request.url.path, "/system_stats")
                self.assertNotIn("authorization", request.headers)
                raise httpx.ConnectError("offline", request=request)
            self.assertEqual(request.headers["authorization"], "Bearer upstream-token")
            path = request.url.path
            calls.append(path)
            if path == "/prefix/system_stats":
                return httpx.Response(200, json={})
            if path == "/prefix/upload/image":
                self.assertIn(png, request.read())
                return httpx.Response(200, json={"name": "renamed.png", "subfolder": "uploads"})
            if path == "/prefix/prompt":
                graph = json.loads(request.content)["prompt"]
                self.assertEqual(graph["21"]["inputs"]["image"], "uploads/renamed.png")
                self.assertEqual(graph["1"]["inputs"]["unet_name"], "qwen-warm/edit-8.safetensors")
                return httpx.Response(200, json={"prompt_id": "p1"})
            if path == "/prefix/history/p1":
                return httpx.Response(200, json={"p1": {"outputs": {"8": {"images": [
                    {"filename": "result.png", "subfolder": "render", "type": "output"}
                ]}}}})
            if path == "/prefix/view":
                self.assertEqual(request.url.params["filename"], "result.png")
                return httpx.Response(200, content=png, headers={"Content-Type": "image/png"})
            self.fail(f"Unexpected upstream request {path}")

        async_client = httpx.AsyncClient
        with tempfile.TemporaryDirectory() as tmp, patch.multiple(
            self.api, POOL=pool, INPUT_DIR=Path(tmp) / "inputs",
            OUTPUT_DIR=Path(tmp) / "output", TASK_DB=Path(tmp) / "tasks.sqlite3",
        ), patch("comfy_backend.httpx.AsyncClient", side_effect=lambda **kw: async_client(
            transport=httpx.MockTransport(upstream), **kw
        )), TestClient(self.api.app) as client:
            headers = {"Authorization": f"Bearer {self.api.API_KEY}"}
            response = client.post("/v1/images/edits", headers=headers,
                                   data={"prompt": "edit", "deferred": "true", "output_format": "webp"},
                                   files={"image": ("input.png", png, "image/png")})
            self.assertEqual(response.status_code, 202, response.text)
            poll_url = response.json()["poll_url"]
            for _ in range(60):
                response = client.get(poll_url, headers=headers)
                if response.status_code != 202:
                    break
                time.sleep(0.05)
            self.assertEqual(response.json()["status"], "completed", response.text)
            file_response = client.get(response.json()["data"][0]["url"])
            self.assertEqual(file_response.status_code, 200, file_response.text if file_response.status_code != 200 else "")
            self.assertEqual(client.get(response.json()["data"][0]["url"].rsplit(".", 1)[0]).status_code, 404)
            self.assertEqual(Image.open(io.BytesIO(file_response.content)).format, "WEBP")
            self.assertEqual(list(self.api.INPUT_DIR.iterdir()), [])
        self.assertEqual(calls[-4:], ["/prefix/upload/image", "/prefix/prompt", "/prefix/history/p1", "/prefix/view"])


class BackendScheduling(unittest.IsolatedAsyncioTestCase):
    async def test_independent_workers_with_basic_and_custom_auth(self):
        from comfy_backend import BackendPool, load_backends

        config = [
            {"name": "basic", "url": "https://basic",
             "auth": {"type": "basic", "username_env": "TEST_USER", "password_env": "TEST_PASSWORD"}},
            {"name": "header", "url": "https://header", "headers_env": {"X-API-Key": "TEST_TOKEN"}},
        ]
        with patch.dict(os.environ, {"COMFY_BACKENDS": json.dumps(config), "TEST_USER": "worker",
                                     "TEST_PASSWORD": "secret", "TEST_TOKEN": "custom-key"}):
            pool = BackendPool(load_backends())

        def upstream(request):
            if request.url.host == "basic":
                self.assertEqual(request.headers["authorization"], "Basic d29ya2VyOnNlY3JldA==")
                self.assertNotIn("x-api-key", request.headers)
            else:
                self.assertEqual(request.headers["x-api-key"], "custom-key")
                self.assertNotIn("authorization", request.headers)
            return httpx.Response(200, json={})

        async_client = httpx.AsyncClient
        held = set()
        both_running = asyncio.Event()

        async def submit():
            async with pool.lease() as (backend, client):
                held.add(backend.name)
                if len(held) == 2:
                    both_running.set()
                await asyncio.wait_for(both_running.wait(), timeout=2)
                self.assertEqual((await client.get("prompt")).status_code, 200)

        with patch("comfy_backend.httpx.AsyncClient", side_effect=lambda **kw: async_client(
            transport=httpx.MockTransport(upstream), **kw
        )):
            await asyncio.gather(submit(), submit())
        self.assertEqual(held, {"basic", "header"})

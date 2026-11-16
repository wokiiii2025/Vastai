import asyncio
import base64
import copy
import hmac
import io
import logging
import json
import sqlite3
import os
import random
import time
import uuid
from contextlib import asynccontextmanager, closing
from pathlib import Path
from typing import Annotated, Literal, Sequence

import httpx
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError

from comfy_backend import BackendPool, NoBackendAvailable, load_backends

POOL = BackendPool(load_backends())
API_KEY = os.environ["OPENAI_API_KEY"]
if not API_KEY.strip():
    raise RuntimeError("OPENAI_API_KEY must contain a non-empty key")
OUTPUT_DIR = Path(os.getenv("COMFY_OUTPUT_DIR", "/workspace/ComfyUI/output")).resolve()
INPUT_DIR = Path(os.getenv("COMFY_INPUT_DIR", "/workspace/ComfyUI/input")).resolve()
TIMEOUT = int(os.getenv("IMAGE_TIMEOUT", "1800"))
MAX_OUTPUT_DIMENSION = int(os.getenv("IMAGE_MAX_DIMENSION", "4096"))
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
if TIMEOUT < 1 or not 32 <= MAX_OUTPUT_DIMENSION <= 16384:
    raise RuntimeError("Invalid IMAGE_TIMEOUT or IMAGE_MAX_DIMENSION")
MODEL = os.getenv("QWEN_MODEL", "NoctQ_V4_int8_convrot.safetensors")
TEXT_ENCODER = os.getenv("QWEN_TEXT_ENCODER", "qwen3vl_8b_int8_convrot_heretic.safetensors")
VAE = os.getenv("QWEN_VAE", "qwen_image_2.1_vae_bf16.safetensors")
TURBO8_LORA = os.getenv("QWEN_TURBO8_LORA", "turbo8_lora_step2500.safetensors")
LIGHTNING4_LORA = os.getenv("QWEN_LIGHTNING4_LORA", "qwen_image_2.1_lightning_4step.safetensors")
NSFW_EDIT_LORA = os.getenv("QWEN_NSFW_EDIT_LORA", "Qwen-Image-2.1 NSFW Image Edit.safetensors")
logger = logging.getLogger("openai_images_api")
MAX_REFERENCE_IMAGES = 10
MAX_IMAGE_BYTES = 30 * 1024 * 1024
MAX_INPUT_DIMENSION = 4096
MAX_INPUT_PIXELS = 16 * 1024 * 1024
ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
OUTPUT_FORMATS = {"png", "jpg", "jpeg", "webp"}
OUTPUT_MEDIA_TYPES = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}
GENERATION_MODEL_ID = "t2i-v1"
EDIT_MODEL_ID = "i2i-edit-v1"


bearer_scheme = HTTPBearer(description="请输入 API Key；系统会自动添加 Bearer 前缀。", auto_error=False)


async def auth(credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)] = None):
    if credentials is None or credentials.scheme.lower() != "bearer" or not hmac.compare_digest(credentials.credentials.encode(), API_KEY.encode()):
        raise HTTPException(401, "API 密钥无效", headers={"WWW-Authenticate": "Bearer"})


class GenerationRequest(BaseModel):
    model: str = GENERATION_MODEL_ID
    prompt: str = Field(min_length=1)
    n: int = Field(1, ge=1, le=4)
    size: str = "1024x1024"
    response_format: Literal["url", "b64_json"] = "url"
    output_format: Literal["png", "jpg", "jpeg", "webp"] = "png"
    deferred: bool = False
    user: str | None = None
    seed: int | None = Field(default=None, ge=0, le=2**64 - 1)
    steps: Literal[4, 8] | None = None


def dimensions(size: str) -> tuple[int, int]:
    try:
        w, h = (int(v) for v in size.lower().split("x", 1))
    except ValueError:
        raise HTTPException(400, "size must be WxH format") from None
    if not (1 <= w <= MAX_OUTPUT_DIMENSION and 1 <= h <= MAX_OUTPUT_DIMENSION):
        raise HTTPException(400, f"width and height must be between 1 and {MAX_OUTPUT_DIMENSION}")
    return (max(32, w // 32 * 32), max(32, h // 32 * 32))


def graph(
    prompt: str, width: int, height: int, seed: int, steps: int,
    input_names: Sequence[str] = (), use_lora: bool = False,
    preserve_input_size: bool = False,
) -> dict:
    if steps not in (4, 8):
        raise HTTPException(400, "steps must be 4 or 8")
    acceleration_lora = LIGHTNING4_LORA if steps == 4 else TURBO8_LORA
    nodes = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": MODEL, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": TEXT_ENCODER, "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "4": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["2", 0], "prompt": prompt, "negative_prompt": "", "resolution": 0}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "10": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": acceleration_lora, "strength_model": 1.0}},
        "11": {"class_type": "ModelSamplingFlux", "inputs": {"model": ["10", 0], "max_shift": 0.9 if steps == 4 else 0.6935, "base_shift": 0.5, "width": width, "height": height}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": "openai/images-v1"}},
    }
    # Use the same explicit KSampler controls for generation and editing.
    nodes["6"] = {"class_type": "KSampler", "inputs": {"model": ["11", 0], "positive": ["4", 0], "negative": ["4", 1], "latent_image": ["5", 0], "seed": seed, "steps": steps, "cfg": 2.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}}
    if input_names:
        nodes.pop("5")
        nodes["4"]["inputs"]["vae"] = ["3", 0]
        for index, input_name in enumerate(input_names, start=1):
            node_id = str(20 + index)
            nodes[node_id] = {"class_type": "LoadImage", "inputs": {"image": input_name}}
            image_output: list[str | int] = [node_id, 0]
            if index == 1 and not preserve_input_size:
                nodes["31"] = {"class_type": "ImageScale", "inputs": {"image": [node_id, 0], "upscale_method": "lanczos", "width": width, "height": height, "crop": "disabled"}}
                image_output = ["31", 0]
            nodes["4"]["inputs"][f"images.image_{index}"] = image_output
        nodes["6"]["inputs"]["latent_image"] = ["4", 2]
    if use_lora:
        nodes["12"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["10", 0], "lora_name": NSFW_EDIT_LORA, "strength_model": 1.0}}
        nodes["11"]["inputs"]["model"] = ["12", 0]
    return nodes


async def run_one(client: httpx.AsyncClient, workflow: dict) -> Path:
    response = await client.post("prompt", json={"prompt": workflow, "client_id": str(uuid.uuid4())})
    if response.status_code == 400:
        logger.error("ComfyUI rejected workflow: %s", response.text[:1500])
    response.raise_for_status()
    prompt_id = response.json()["prompt_id"]
    try:
        async with asyncio.timeout(TIMEOUT):
            while True:
                history = await client.get(f"history/{prompt_id}")
                history.raise_for_status()
                item = history.json().get(prompt_id)
                if item:
                    break
                await asyncio.sleep(1)
    except TimeoutError:
        raise HTTPException(504, "图像生成超时") from None

    if item.get("status", {}).get("status_str") == "error":
        logger.error("ComfyUI execution failed for %s: %s", prompt_id, item["status"])
        raise HTTPException(502, "图像处理失败")
    images = [img for node in item.get("outputs", {}).values() for img in node.get("images", [])
              if img.get("type") == "output"]
    if not images:
        raise HTTPException(502, "图像服务没有返回输出")
    image = images[0]
    output = await client.get("view", params={
        "filename": image["filename"], "subfolder": image.get("subfolder", ""), "type": "output",
    })
    output.raise_for_status()
    with Image.open(io.BytesIO(output.content)) as parsed:
        if parsed.format != "PNG":
            raise HTTPException(502, "图像服务返回了意外的格式")
        parsed.verify()
    output_dir = OUTPUT_DIR / "api-results"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{uuid.uuid4().hex}.png"
    path.write_bytes(output.content)
    return path


async def run_images(graphs, response_format, output_format, base_url):
    try:
        async with POOL.lease() as (backend, client):
            graphs = copy.deepcopy(graphs)
            uploaded = {}
            results = []
            for workflow in graphs:
                if backend.model_overrides:
                    sampler = workflow["6"]["inputs"]
                    operation = "edit" if sampler["latent_image"] == ["4", 2] else "t2i"
                    model_name = backend.model_overrides.get(f"{operation}_{sampler['steps']}")
                    if model_name:
                        workflow["1"]["inputs"]["unet_name"] = model_name
                for node in workflow.values():
                    if node["class_type"] != "LoadImage":
                        continue
                    name = node["inputs"]["image"]
                    if name not in uploaded:
                        with (INPUT_DIR / name).open("rb") as source:
                            response = await client.post("upload/image", data={
                                "type": "input", "subfolder": "openai-api",
                            }, files={"image": (name, source)})
                        response.raise_for_status()
                        meta = response.json()
                        uploaded[name] = "/".join(filter(None, [meta.get("subfolder"), meta["name"]]))
                    node["inputs"]["image"] = uploaded[name]
                path = await run_one(client, workflow)
                results.append(response_item(path, response_format, output_format, base_url))
            return {"created": int(time.time()), "data": results}
    except NoBackendAvailable:
        raise HTTPException(503, "没有可用的 ComfyUI 服务") from None
    except httpx.TimeoutException:
        raise HTTPException(504, "连接 ComfyUI 超时") from None
    except (httpx.HTTPError, ValueError, KeyError, TypeError, OSError, Image.DecompressionBombError) as err:
        logger.error("ComfyUI request failed on %s (%s)", backend.name, type(err).__name__)
        raise HTTPException(502, "图像服务返回错误") from None


def convert_output(path: Path, output_format: str) -> Path:
    if output_format == "png":
        return path
    converted = path.with_suffix(f".{output_format}")
    with Image.open(path) as source:
        if output_format in ("jpg", "jpeg"):
            source.convert("RGB").save(converted, format="JPEG", quality=95, optimize=True)
        elif output_format == "webp":
            source.save(converted, format="WEBP", quality=95, method=6)
        else:
            raise HTTPException(400, "unsupported output format")
    return converted


def response_item(path: Path, response_format: str, output_format: str, base_url: str):
    converted = path
    try:
        converted = convert_output(path, output_format)
        if response_format == "b64_json":
            return {"b64_json": base64.b64encode(converted.read_bytes()).decode()}
    finally:
        if converted != path or response_format == "b64_json":
            path.unlink(missing_ok=True)
        if response_format == "b64_json":
            converted.unlink(missing_ok=True)
    path = converted
    token = base64.urlsafe_b64encode(path.relative_to(OUTPUT_DIR).as_posix().encode()).decode().rstrip("=")
    signature = hmac.digest(API_KEY.encode(), token.encode(), "sha256").hex()
    return {"url": base_url + f"/v1/files/{token}.{signature}"}


# Persistent queue, shared by all backend workers in this API process.
TASK_DB = Path(os.getenv("IMAGE_TASK_DB", "/workspace/qwen-openai-api/tasks.sqlite3"))

def task_db():
    TASK_DB.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(TASK_DB)
    db.row_factory = sqlite3.Row
    return closing(db)

@asynccontextmanager
async def lifespan(_app):
    POOL.reset()
    with task_db() as db, db:
        db.execute("CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, status TEXT, created_at INTEGER, updated_at INTEGER, payload TEXT, result TEXT, error TEXT)")
        db.execute(
            "UPDATE tasks SET status='failed', error=?, updated_at=? WHERE status='in_progress'",
            (json.dumps({"code": "service_restarted", "message": "服务重启中断任务，请重新提交。"}, ensure_ascii=False), int(time.time())),
        )
    workers = [asyncio.create_task(process_tasks()) for _ in POOL.backends]
    try:
        yield
    finally:
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)


app = FastAPI(title="OpenAI 兼容图像接口", description="支持文生图、单图编辑和多参考图编辑。输入支持 PNG、JPG、JPEG、WebP；输出格式可由 output_format 指定。使用 Bearer API Key 鉴权。", version="1.4.0", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_tags=[{"name": "模型", "description": "查询可供外部调用的稳定模型别名。"}, {"name": "图像", "description": "文生图与单图/多参考图编辑。"}])


def task_view(row):
    value = {k: row[k] for k in ("id", "status", "created_at", "updated_at")}
    value["object"] = "image.request"
    if row["result"]:
        value.update(json.loads(row["result"]))
    if row["error"]:
        value["error"] = json.loads(row["error"])
    if row["status"] == "queued":
        with task_db() as db, db:
            value["queue_position"] = db.execute("SELECT count(*) FROM tasks WHERE status='queued' AND rowid < (SELECT rowid FROM tasks WHERE id=?)", (row["id"],)).fetchone()[0]
    return value

async def process_tasks():
    while True:
        with task_db() as db, db:
            row = db.execute("SELECT * FROM tasks WHERE status='queued' ORDER BY rowid LIMIT 1").fetchone()
            if row:
                db.execute("UPDATE tasks SET status='in_progress', updated_at=? WHERE id=?", (int(time.time()), row["id"]))
        if row is None:
            await asyncio.sleep(0.5)
            continue
        payload = {}
        try:
            payload = json.loads(row["payload"])
            result = await run_images(
                payload["graphs"], payload["response_format"], payload["output_format"],
                PUBLIC_BASE_URL or payload["base_url"].rstrip("/"),
            )
            with task_db() as db, db:
                db.execute("UPDATE tasks SET status='completed', result=?, payload='{}', updated_at=? WHERE id=?",
                           (json.dumps(result), int(time.time()), row["id"]))
        except Exception as err:
            logger.exception("异步图像任务失败：%s", row["id"])
            message = err.detail if isinstance(err, HTTPException) else "图像处理失败，请稍后重试。"
            error = json.dumps({"code": "generation_failed", "message": message}, ensure_ascii=False)
            with task_db() as db, db:
                db.execute("UPDATE tasks SET status='failed', error=?, payload='{}', updated_at=? WHERE id=?",
                           (error, int(time.time()), row["id"]))
        finally:
            cleanup_inputs(payload.get("input_names", []))


def cleanup_inputs(input_names):
    for name in input_names:
        try:
            (INPUT_DIR / name).unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove staged reference image %s", name)


async def execute_images(graphs, response_format, output_format, request, deferred=False, input_names=()):
    base_url = PUBLIC_BASE_URL or str(request.base_url).rstrip("/")
    if deferred or "respond-async" in request.headers.get("prefer", "").lower():
        ident = "img_req_" + uuid.uuid4().hex
        now = int(time.time())
        payload = {"graphs":graphs,"response_format":response_format,"output_format":output_format,"base_url":base_url,"input_names":input_names}
        with task_db() as db, db:
            if db.execute("SELECT count(*) FROM tasks WHERE status IN ('queued','in_progress')").fetchone()[0] >= 100:
                raise HTTPException(429, "任务队列已满,请稍后重试", headers={"Retry-After":"10"})
            db.execute("INSERT INTO tasks VALUES (?,?,?,?,?,?,?)",(ident,"queued",now,now,json.dumps(payload),None,None))
            row = db.execute("SELECT * FROM tasks WHERE id=?",(ident,)).fetchone()
        result = task_view(row)
        result["poll_url"] = base_url + "/v1/images/requests/" + ident
        return JSONResponse(result,status_code=202,headers={"Location":result["poll_url"],"Retry-After":"3","Preference-Applied":"respond-async"})
    return await run_images(graphs, response_format, output_format, base_url)


@app.get("/v1/images/requests/{request_id}", dependencies=[Depends(auth)], tags=["图像"], summary="查询异步任务状态与结果（兼容扩展）")
async def get_image_request(request_id: str):
    with task_db() as db, db:
        row = db.execute("SELECT * FROM tasks WHERE id=?",(request_id,)).fetchone()
    if row is None:
        return JSONResponse({"error":{"message":"任务不存在","type":"invalid_request_error","code":"request_not_found"}},status_code=404)
    pending = row["status"] in ("queued","in_progress")
    return JSONResponse(task_view(row),status_code=202 if pending else 200,headers={"Cache-Control":"no-store",**({"Retry-After":"3"} if pending else {})})


@app.get("/docs", include_in_schema=False)
async def scalar_docs():
    return HTMLResponse("""<!doctype html><html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>图像 API 文档</title></head><body><script id=\"api-reference\" data-url=\"/openapi.json\" data-configuration='{"theme":"purple","layout":"modern","hideModels":false,"defaultHttpClient":{"targetKey":"shell","clientKey":"curl"},"authentication":{"preferredSecurityScheme":"HTTPBearer"}}'></script><script src=\"https://cdn.jsdelivr.net/npm/@scalar/api-reference\"></script></body></html>""")


@app.get("/redoc", include_in_schema=False)
async def redoc_redirect():
    return HTMLResponse("""<!doctype html><html><head><meta charset=\"utf-8\"><meta http-equiv=\"refresh\" content=\"0;url=/docs\"><title>图像 API 文档</title></head><body><a href=\"/docs\">打开现代 API 文档</a></body></html>""")


@app.get("/health")
async def health():
    backends = list(POOL.backends)
    checks = await asyncio.gather(*(POOL.health(backend) for backend in backends))
    healthy = sum(checks)
    return {
        "状态": "正常" if healthy == len(backends) else "降级",
        "默认步数": 8,
        "最多参考图": MAX_REFERENCE_IMAGES,
        "backends": {"healthy": healthy, "total": len(backends)},
    }


@app.get("/readyz", include_in_schema=False)
async def readiness():
    return {"status": "ok"}


@app.get("/v1/models", dependencies=[Depends(auth)], tags=["模型"], summary="查询模型别名")
async def models():
    now = int(time.time())
    return {"object": "list", "data": [{"id": GENERATION_MODEL_ID, "object": "model", "created": now, "owned_by": "local"}, {"id": EDIT_MODEL_ID, "object": "model", "created": now, "owned_by": "local"}]}


@app.post("/v1/images/generations", dependencies=[Depends(auth)], tags=["图像"], summary="文生图")
async def generations(body: GenerationRequest, request: Request):
    w, h = dimensions(body.size)
    if body.model != GENERATION_MODEL_ID:
        raise HTTPException(400, "unsupported generation model")
    graphs = [graph(body.prompt, w, h, (body.seed + i) % 2**64 if body.seed is not None else random.randrange(0, 2**63), body.steps or 8) for i in range(body.n)]
    return await execute_images(graphs,body.response_format,body.output_format,request,body.deferred)


@app.post("/v1/images/edits", dependencies=[Depends(auth)], tags=["图像"], summary="单图或多参考图编辑")
async def edits(
    request: Request,
    image: Annotated[list[UploadFile], File(description="参考图。多图时重复使用同名 image 字段,按上传顺序对应 <image1> 到 <image10>。")],
    prompt: Annotated[str, Form(min_length=1)],
    model: Annotated[str, Form()] = EDIT_MODEL_ID,
    n: Annotated[int, Form(ge=1, le=4)] = 1,
    size: Annotated[str, Form(description="auto 沿用第一张参考图的宽高比例和近似尺寸；也可传 1024x1024 等显式尺寸。")] = "auto",
    response_format: Annotated[Literal["url", "b64_json"], Form()] = "url",
    output_format: Annotated[Literal["png", "jpg", "jpeg", "webp"], Form(description="输出图像格式；jpeg 与 jpg 等价。")] = "png",
    user: Annotated[str | None, Form()] = None,
    seed: Annotated[int | None, Form(ge=0, le=2**64 - 1)] = None,
    steps: Annotated[int, Form()] = 8,
    deferred: Annotated[bool, Form(description="异步提交任务,通过任务查询接口轮询；默认同步。")] = False,
):
    if steps not in (4, 8):
        raise HTTPException(400, "steps must be 4 or 8")
    if model != EDIT_MODEL_ID:
        raise HTTPException(400, "unsupported edit model")
    if not 1 <= len(image) <= MAX_REFERENCE_IMAGES:
        raise HTTPException(400, "expected 1 to 10 reference images")

    preserve_input_size = size.lower() == "auto"
    if not preserve_input_size:
        w, h = dimensions(size)

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_names: list[str] = []
    first_dimensions: tuple[int, int] | None = None
    try:
        for index, upload in enumerate(image, start=1):
            raw = await upload.read(MAX_IMAGE_BYTES + 1)
            if not raw:
                raise HTTPException(400, "reference image is empty")
            if len(raw) > MAX_IMAGE_BYTES:
                raise HTTPException(413, f"第 {index} 张参考图超过 30MB")
            suffix = Path(upload.filename or "input.png").suffix.lower()
            if suffix not in ALLOWED_IMAGE_SUFFIXES:
                raise HTTPException(400, "expected a PNG, JPEG or WebP image")
            try:
                with Image.open(io.BytesIO(raw)) as parsed_image:
                    image_width, image_height = parsed_image.size
                    if parsed_image.format not in {"PNG", "JPEG", "WEBP"}:
                        raise ValueError("unsupported image format")
                    parsed_image.verify()
            except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
                raise HTTPException(400, "invalid reference image")
            if image_width > MAX_INPUT_DIMENSION or image_height > MAX_INPUT_DIMENSION:
                raise HTTPException(400, "reference image exceeds maximum dimensions")
            if image_width * image_height > MAX_INPUT_PIXELS:
                raise HTTPException(400, "reference image exceeds maximum pixel count")
            if index == 1:
                first_dimensions = (image_width, image_height)
            name = f"openai_{uuid.uuid4().hex}{suffix}"
            input_names.append(name)
            (INPUT_DIR / name).write_bytes(raw)

        if preserve_input_size:
            assert first_dimensions is not None
            w, h = (max(32, min(MAX_OUTPUT_DIMENSION // 32, round(v / 32)) * 32) for v in first_dimensions)
        graphs = [graph(prompt, w, h, (seed + i) % 2**64 if seed is not None else random.randrange(0, 2**63), steps, input_names, use_lora=True, preserve_input_size=preserve_input_size) for i in range(n)]
        result = await execute_images(graphs, response_format, output_format, request, deferred, input_names)
        if isinstance(result, JSONResponse) and result.status_code == 202:
            input_names = []
        return result
    finally:
        cleanup_inputs(input_names)


@app.get("/v1/files/{token}")
async def files(token: str):
    encoded, _, signature = token.partition(".")
    try:
        rel = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)).decode()
        path = (OUTPUT_DIR / rel).resolve()
    except (ValueError, OSError):
        raise HTTPException(404) from None
    output_dir = OUTPUT_DIR.resolve()
    if signature:
        expected = hmac.digest(API_KEY.encode(), encoded.encode(), "sha256").hex()
        if not hmac.compare_digest(signature.encode(), expected.encode()):
            raise HTTPException(404)
        allowed = path.parent == output_dir / "api-results"
    else:
        # Existing URLs predate signatures and refer to SaveImage's API prefix.
        allowed = path.parent == output_dir / "openai" and path.name.startswith("images-v1_")
    if not allowed or path.suffix.lower().lstrip(".") not in OUTPUT_FORMATS or not path.is_file():
        raise HTTPException(404)
    return FileResponse(path)

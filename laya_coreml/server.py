"""Jev-compatible HTTP server for local Core ML Laya decisions.

Wraps :class:`laya_coreml.Agent` on a FastAPI application that speaks the
``POST /v1/systemone`` protocol (single request, batch, and ``GET /v1/models``),
so an existing Jev / laya client can point at a local Apple-Silicon machine with
only a base-url change. The model loads once at startup; each request is a
single forward pass and generates no tokens.

Install the runtime with ``pip install 'laya-coreml[serve]'`` (adds FastAPI and
uvicorn), then ``laya-serve --model <bundle-dir>``. The request/response contract
matches the hosted Laya Studio / TypeSafe Jev OpenAPI at
https://api.laya.studio/v1/openapi: ``{"state", "questions"}`` in, one forward
pass, ``{"model", "answers", "usage"}`` out.
"""

from __future__ import annotations

import argparse
import os
import secrets
import threading
import uuid
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from . import __version__
from .agent import load

__all__ = ["create_app", "load_agent", "model_info", "main"]


# --- Error contract ---------------------------------------------------------
# The hosted API answers with ``{"error": {"code", "message"}}`` and a status
# code. ServeError is the single source of truth for that mapping; raising one
# inside any endpoint produces the correct body.

class ServeError(Exception):
    status: int = 500
    code: str = "internal_error"


class InvalidRequest(ServeError):
    status = 400
    code = "invalid_request_error"


class InvalidQuestion(ServeError):
    """A question failed validation or exceeded the export's shape budget."""

    status = 422
    code = "invalid_question_error"


class InferenceError(ServeError):
    status = 500
    code = "internal_error"


class ModelUnavailable(ServeError):
    status = 503
    code = "unavailable_error"


class AuthenticationError(ServeError):
    status = 401
    code = "authentication_error"

# --- Request models ---------------------------------------------------------
# ``state`` may be text, a JSON object, or a list of conversation turns; the
# agent serialises it. Question *content* is validated by the agent (one source
# of truth), so here we only fix the protocol-level shape.

class DecisionRequest(BaseModel):
    model_config = {"extra": "ignore"}

    state: Any = ""
    questions: Dict[str, Any]
    model: Optional[str] = None
    lang: Optional[str] = None


class BatchRequest(BaseModel):
    requests: List[DecisionRequest] = Field(default_factory=list, max_length=64)


# --- Agent + model info -----------------------------------------------------

def load_agent(
    model: str,
    *,
    compute_units: str = "cpu_gpu",
    revision: Optional[str] = None,
    offline: bool = False,
    allow_unvalidated_gpu: bool = False,
):
    """Load the Core ML agent once, mapping load failures to a 503 body."""
    try:
        return load(
            model,
            revision=revision,
            local_files_only=offline,
            compute_units=compute_units,
            allow_unvalidated_gpu=allow_unvalidated_gpu,
        )
    except Exception as exc:  # normalise any load failure to a Jev error body
        raise ModelUnavailable(f"Could not load model {model!r}: {exc}") from exc


def model_info(agent) -> dict:
    """Describe the served checkpoint for ``GET /v1/models``."""
    manifest = getattr(agent, "manifest", {}) or {}
    return {
        "id": "laya-coreml",
        "object": "model",
        "served_model_name": "laya-coreml",
        "origin": manifest.get("source"),
        "revision": manifest.get("revision"),
        "precision": manifest.get("precision"),
        "format": manifest.get("format"),
        "created": 0,
    }


# --- App --------------------------------------------------------------------

def create_app(agent, api_key: Optional[str] = None) -> FastAPI:
    """Build the FastAPI app around an already-loaded agent.

    A single lock serialises forward passes: Core ML's ``MLModel.predict`` is not
    guaranteed safe to call concurrently on one instance, and a local decision
    model pays nothing for that serialisation.

    Pass ``api_key`` (or set ``LAYA_API_KEY``) to require
    ``Authorization: Bearer <key>`` on every endpoint, including ``GET
    /v1/models``. Omit it and the server stays open (localhost default).
    """

    bearer = HTTPBearer(auto_error=False)

    async def require_key(
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer),
    ) -> None:
        if api_key is None:
            return
        token = credentials.credentials if credentials is not None else None
        if token is None or not secrets.compare_digest(token, api_key):
            raise AuthenticationError("Invalid or missing API key.")

    info = model_info(agent)
    lock = threading.Lock()

    app = FastAPI(
        title="laya-coreml",
        description="Jev-compatible local serving for the Laya Core ML decision model.",
        version=__version__,
    )
    def envelope(result: dict) -> dict:
        return {
            "id": uuid.uuid4().hex,
            "model": result.get("model") or info["id"],
            "answers": result.get("answers", {}),
            "usage": result.get("usage", {"input_tokens": 0, "output_tokens": 0}),
            "routing": {"model": info["id"], "reason": "local-coreml"},
        }

    def run(state: Any, questions: dict) -> dict:
        # ValueError -> bad question / over-capacity (422); non-finite output ->
        # inference fault (500). Both come from the agent's predict path.
        try:
            with lock:
                result = agent.predict(state, questions)
        except ServeError:
            raise
        except ValueError as exc:
            raise InvalidQuestion(str(exc)) from exc
        except FloatingPointError as exc:
            raise InferenceError(str(exc)) from exc
        return envelope(result)

    @app.exception_handler(ServeError)
    async def _handle_serve_error(request: Request, exc: ServeError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status, content={"error": {"code": exc.code, "message": str(exc)}}
        )

    @app.post("/v1/systemone", dependencies=[Depends(require_key)])
    def systemone(req: DecisionRequest) -> dict:
        return run(req.state, req.questions)

    @app.post("/v1/systemone/batch", dependencies=[Depends(require_key)])
    def systemone_batch(req: BatchRequest) -> dict:
        # One result (or one {error}) per request, in order.
        results: list = []
        for sub in req.requests:
            try:
                results.append(run(sub.state, sub.questions))
            except ServeError as exc:
                results.append({"error": {"code": exc.code, "message": str(exc)}})
            except ValueError as exc:
                results.append(
                    {"error": {"code": "invalid_question_error", "message": str(exc)}}
                )
            except FloatingPointError as exc:
                results.append({"error": {"code": "internal_error", "message": str(exc)}})
        return {"results": results}

    @app.get("/v1/models", dependencies=[Depends(require_key)])
    def list_models() -> dict:
        return {"object": "list", "data": [info]}

    return app


# --- CLI --------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="laya-serve",
        description="Serve a local Laya Core ML bundle over the Jev /v1/systemone API.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Local model bundle directory, or a Hugging Face model id "
        "(default: the package default, which downloads from the Hub).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000).")
    parser.add_argument(
        "--compute-units",
        default="cpu_gpu",
        choices=["all", "cpu", "cpu_gpu", "cpu_ne"],
        help="Core ML compute units (default: cpu_gpu).",
    )
    parser.add_argument("--revision", default=None, help="Pin a Hugging Face revision.")
    parser.add_argument(
        "--offline", action="store_true", help="Only use local files / cached Hub snapshots."
    )
    parser.add_argument(
        "--allow-unvalidated-gpu",
        action="store_true",
        help="Serve a range-shape export on CPU_AND_GPU despite the fidelity guard.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Require 'Authorization: Bearer <key>' on all endpoints. "
        "Defaults to $LAYA_API_KEY when set; omit both to leave the server open.",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="uvicorn log level (default: info).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    import uvicorn

    args = build_parser().parse_args(argv)
    model = args.model or "aac6fef/laya-multilingual-coreml"
    agent = load_agent(
        model,
        compute_units=args.compute_units,
        revision=args.revision,
        offline=args.offline,
        allow_unvalidated_gpu=args.allow_unvalidated_gpu,
    )
    app = create_app(agent, api_key=args.api_key or os.environ.get("LAYA_API_KEY"))
    info = model_info(agent)
    print(
        f"laya-serve: serving {model!r} on http://{args.host}:{args.port}  ({info['id']})\n"
        f"  POST /v1/systemone | POST /v1/systemone/batch | GET /v1/models\n"
        f"  OpenAPI docs: http://{args.host}:{args.port}/docs\n"
        f"  auth: {'Bearer key required' if (args.api_key or os.environ.get('LAYA_API_KEY')) else 'open (no --api-key)'}",
        flush=True,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)

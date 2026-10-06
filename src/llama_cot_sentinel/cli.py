"""llama-cot-sentinel serve --backend http://127.0.0.1:8081 --config examples/sentinel-medical.json --port 8002"""
from __future__ import annotations

import argparse
import logging
import os
from typing import List, Optional

from . import __version__
from .core import SentinelConfig


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.environ.get("COT_SENTINEL_" + name, default)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llama-cot-sentinel", description="Step-checked reasoning for llama-server: a decision model checks "
                                "every reasoning step and the answer, with rollback / reflection when a rule is broken.")
    p.add_argument("--version", action="version", version="%(prog)s " + __version__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="run the OpenAI-compatible service")
    s.add_argument("--backend", default=_env("BACKEND", "http://127.0.0.1:8080"), help="llama-server base URL (needs /apply-template and /completion)")
    s.add_argument("--config", default=_env("CONFIG"), required=_env("CONFIG") is None, help="rules and settings (see examples/sentinel-medical.json)")
    s.add_argument("--critic-url", default=_env("CRITIC_URL"), help="override the config's SystemOne decision-model URL (e.g. Clef-Flash on llama-server)")
    s.add_argument("--host", default=_env("HOST", "127.0.0.1"))
    s.add_argument("--port", type=int, default=int(_env("PORT", "8002")))
    s.add_argument("--timeout", type=float, default=float(_env("TIMEOUT", "900")), help="seconds per backend request")
    s.add_argument("--log-level", default=_env("LOG_LEVEL", "INFO"))
    return p


def main(argv: Optional[List[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    logging.basicConfig(level=a.log_level.upper(), format="%(asctime)s %(levelname)s %(message)s")
    cfg = SentinelConfig.load(a.config)
    if a.critic_url:
        cfg.critic_url = a.critic_url
    import uvicorn

    from .server import create_app

    logging.getLogger("llama_cot_sentinel").info("sentinel %s: backend %s, critic %s, checks %s", __version__, a.backend, cfg.critic_url,
                                                  ", ".join(c.id for c in cfg.checks))
    uvicorn.run(create_app(cfg, a.backend, a.timeout), host=a.host, port=a.port, log_level=a.log_level.lower())
    return 0

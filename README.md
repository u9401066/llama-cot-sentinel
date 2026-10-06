# llama-cot-sentinel

**Step-checked chain of thought for llama-server.** While a reasoning model thinks, a fast *decision model* (Cloudflare
[Clef / Clef-Flash](https://huggingface.co/Cloudflare/clef-flash), or any server with the SystemOne `/v1/systemone` API) checks
every reasoning step against your rules. A step that breaks a rule is withdrawn and the model thinks again from the last good step
with a first-person correction, or a reflection ("Wait, …") is added; the final answer is held until it passes. An OpenAI-compatible
service, so llama-server's Web UI or any OpenAI client can use it.

Built for clinical reasoning (rules: *red flag missed*, *contraindication unchecked*, *premature closure*), but the rules are
plain yes/no questions in a JSON file. **A research tool, not a validated clinical safety device** — see [Limits](#limits).

```
client ──► llama-cot-sentinel :8002 ──/apply-template, /completion (stream)──► llama-server (reasoning model)
                   │   each reasoning step / answer paragraph
                   └──► /v1/systemone (one noul question per rule, ~50 ms) ──► decision model (e.g. Clef-Flash on llama-server)

pass ─► released to the client        rollback rule ─► step discarded, restart from accepted text + correction
                                       inject rule   ─► step kept + "Wait, …" reflection, generation continues
answer ─► held, checked per paragraph; still failing after the limits ─► released with a warning naming every flagged rule
```

## Install and run

```bash
pip install git+https://github.com/u9401066/llama-cot-sentinel
# a reasoning model on llama-server (e.g. :8081) and a decision model, e.g. Clef-Flash (llama.cpp >= b11433):
llama-server -m Clef-Flash-Q4_K_M.gguf -c 8192 -b 2048 -ub 2048 -np 1 --cache-ram 0 --port 8000
llama-cot-sentinel serve --backend http://127.0.0.1:8081 --config examples/sentinel-medical.json --port 8002
curl -s localhost:8002/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"58M, sudden chest pain for 30 min, sweating. First steps?"}]}'
```

* `POST /v1/chat/completions` — every request is answered with checked reasoning (stream or not). A leading trigger prefix such
  as `med:` (config `triggers`) is removed, so a gateway can route by prefix. Tools and earlier reasoning in the request are ignored.
  Streams carry `reasoning_content` (released steps and `[sentinel: …]` notes), `content`, and on the last chunk llama-server-style
  `timings` plus a `sentinel` report: `interventions`, `flagged`, `steps_checked`, `critic_ms`, `attempts`.
* `GET /v1/models`, `GET /health`.
* Library: `SentinelRun(cfg, backend_client, SystemOneCritic(cfg), messages).events()`.
* Behind [llama-mcp-router](https://github.com/u9401066/llama-mcp-router) (≥ 0.12): `--routes` with
  `{"prefix": "med:", "model": "sentinel", "url": "http://127.0.0.1:8002"}` sends `med:` chats from the same Web UI here.

## Configuration ([examples/sentinel-medical.json](examples/sentinel-medical.json))

| key | default | meaning |
|---|---|---|
| `critic_url`, `critic_model`, `critic_timeout` | `http://127.0.0.1:8000`, –, 20 | the SystemOne decision model |
| `checks[]` | – | `id`, `question` (yes = rule broken), `threshold` (0.5), `action` (`rollback` \| `inject`), `correction` (first-person text), `label` |
| `triggers`, `model_id` | `["med:"]`, `sentinel` | prefixes removed from the first user message; the model id listed |
| `temperature`, `top_p`, `max_tokens` | 0.6, 0.9, 8192 | generation (a request may override temperature / top_p / `chat_template_kwargs`) |
| `min_step_chars`, `max_step_chars` | 60, 1200 | a step = a paragraph or line of at least `min_step_chars`; long ones are cut at a sentence end |
| `check_answer` | true | hold and check the final answer |
| `max_per_check`, `max_interventions` | 2, 4 | per rule / per answer; afterwards steps pass with a note and the answer gets a `warning` |
| `case_chars`, `context_chars` | 2000, 1200 | how much of the user's messages / accepted reasoning the critic sees |
| `system`, `annotate`, `warning` | –, true, … | default system prompt; `[sentinel: …]` notes; warning text (`{labels}`) |

Rolling back is a new `/completion` request with the accepted text as prompt — llama-server's prefix cache keeps it cheap (the
accepted reasoning is re-read at ~2,800 tokens/s on a 27B ternary model), and it works for hybrid/recurrent models (Qwen3.5) whose
state cannot be cut at an arbitrary token. If the critic is unreachable, steps pass unchecked (fail open).

## Results (RTX 4090: Bonsai 27B generates, Clef-Flash Q4_K_M judges; details in [benchmarks/RESULTS.md](benchmarks/RESULTS.md))

| critic, 42 labeled steps (中文 + English) | red flag | contraindication | closure | correct steps passed | per step |
|---|---|---|---|---|---|
| **Clef-Flash, the 3 checks of the example** | 6/6 | 6/6 | 3/3 | **20/22** | **49 ms** |
| the 27B itself as a JSON judge (first 32 steps) | 6/6 | 6/6 | 3/3 | **3/17** | **8 s** |

On 8 live cases: no false interventions on the 5 normal questions (two runs, +0–8 s); with a system prompt pushing towards benign
answers, answers were corrected (e.g. "no ECG, come back in a week" → immediate ECG + troponin for possible ACS) or released with a
warning.

## Limits

* **The critic's knowledge.** A general decision model catches clear violations, not *rationalised* ones: an answer that listed
  succinylcholine for RSI on day 6 of a major burn "because hyperkalaemia risk is lower by day 6" (wrong) scored 0.29–0.40; the bare
  step "I will use succinylcholine" scores 0.81. Doses are not checked reliably (nitrofurantoin 500 mg: 0.05) — use a formulary /
  calculator. Domain-tuned critics or knowledge-grounded checks are the next step.
* **Steering is advice.** A strong system instruction ("prefer benign explanations") can win over corrections; the sentinel then
  marks the answer instead of releasing it silently.
* **Granularity.** Steps are paragraphs/lines; a violation spread over several steps may pass each one.
* Latency: ~0.05–0.3 s of critic time per step (1–9 s per answer here), plus re-reading and regeneration per intervention.

## Tests and benchmarks

```bash
pip install -e ".[test]" && pytest -q
python benchmarks/critic_eval.py --systemone http://127.0.0.1:8000 [--with-dose] [--llm http://127.0.0.1:8081]
python benchmarks/live_demo.py --router http://127.0.0.1:8002 --llm http://127.0.0.1:8081
```

MIT license.

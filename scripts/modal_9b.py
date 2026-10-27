"""The same pilot on Qwen/Qwen3.5-9B (the model the user named for proving the data, 2026-10-24), v2 data only.

Qwen3.5 is a vision-language model with hybrid linear attention, so the LoRA touches only the language model's
projection layers and is merged into the weights after training; vLLM then serves the merged model like the base one
(no LoRA path). Thinking is off in both arms (the training targets carry no thinking). Same decoding as v2: greedy,
16k-token cap, repetition penalty 1.05.

  .venv/bin/modal run scripts/sft_fto/modal_9b.py::before   # -> tmp/sft_fto/out_before_9b.jsonl
  .venv/bin/modal run scripts/sft_fto/modal_9b.py::train    # -> tmp/sft_fto/train_log_9b.json, /vol/merged_9b
  .venv/bin/modal run scripts/sft_fto/modal_9b.py::after    # -> tmp/sft_fto/out_after_9b.jsonl
"""
import json
from pathlib import Path

import modal

MODEL = "Qwen/Qwen3.5-9B"
VOL = modal.Volume.from_name("sft-fto", create_if_missing=True)
CACHE = "/vol"
MERGED = f"{CACHE}/merged_9b"
NO_THINK = {"enable_thinking": False}
app = modal.App("sft-fto-9b")
_here = Path(__file__).resolve()
LOCAL = _here.parents[2] / "tmp/sft_fto" if len(_here.parents) > 2 else Path("/tmp")   # the container sees only /root/modal_9b.py

ENV = {"HF_HOME": f"{CACHE}/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"}
infer_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "vllm==0.31.0", "transformers==5.17.0", "huggingface_hub[hf_transfer]").env(ENV | {"VLLM_USE_FLASHINFER_SAMPLER": "0"})   # its sampler JIT-builds with nvcc; greedy needs none
train_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.13.0", "transformers==5.17.0", "trl==1.15.0", "peft==0.21.2", "accelerate==1.15.0", "datasets==5.0.1",
    "flash-linear-attention==0.5.2", "huggingface_hub[hf_transfer]").env(ENV)


@app.function(image=infer_image, gpu="H100", volumes={CACHE: VOL}, timeout=3 * 3600)
def generate(rows: list[dict], path: str) -> list[dict]:
    from vllm import LLM, SamplingParams
    VOL.reload()
    llm = LLM(path, tokenizer=MODEL, max_model_len=100_000, gpu_memory_utilization=0.9, enable_prefix_caching=False,
              limit_mm_per_prompt={"image": 0, "video": 0})
    sp = SamplingParams(temperature=0.0, max_tokens=16_384, repetition_penalty=1.05)
    outs = llm.chat([r["messages"] for r in rows], sp, chat_template_kwargs=NO_THINK)
    return [{"id": r["id"], "set": r.get("set"), "task": r.get("task"), "deliverable": r.get("deliverable"), "text": o.outputs[0].text,
             "finish": o.outputs[0].finish_reason, "prompt_tokens": len(o.prompt_token_ids)} for r, o in zip(rows, outs)]


@app.function(image=train_image, gpu="H100", volumes={CACHE: VOL}, timeout=5 * 3600)
def train_lora(rows: list[dict], epochs: int = 1, max_length: int = 32_768, lr: float = 5e-5, merged: str = MERGED) -> dict:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from trl import SFTConfig, SFTTrainer
    tok = AutoTokenizer.from_pretrained(MODEL)
    keep = [r for r in rows if len(tok.apply_chat_template(r["messages"], tokenize=True, **NO_THINK)) <= max_length]   # nothing truncated
    data = Dataset.from_list([{"prompt": r["messages"][:2], "completion": r["messages"][2:], "chat_template_kwargs": NO_THINK} for r in keep])
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16)
    cfg = SFTConfig(output_dir=f"{CACHE}/run_9b", num_train_epochs=epochs, per_device_train_batch_size=1, gradient_accumulation_steps=8,
                    learning_rate=lr, lr_scheduler_type="cosine", warmup_steps=1, logging_steps=1, save_strategy="no",
                    bf16=True, gradient_checkpointing=True, max_length=max_length, completion_only_loss=True, report_to=[])
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
    tr = SFTTrainer(model=model, args=cfg, train_dataset=data, peft_config=lora, processing_class=tok)
    out = tr.train()
    tr.model.merge_and_unload().save_pretrained(merged)
    tok.save_pretrained(merged)
    VOL.commit()
    return {"rows": len(rows), "kept": len(keep), "train_loss": out.training_loss, "steps": out.global_step, "log": [x for x in tr.state.log_history if "loss" in x]}


def _rows(name: str) -> list[dict]:
    return [json.loads(x) for x in (LOCAL / name).read_text().splitlines()]


def _save(res: list[dict], name: str) -> None:
    (LOCAL / name).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in res))
    print(json.dumps([{k: x[k] for k in ("id", "finish")} | {"chars": len(x["text"])} for x in res], indent=1))


@app.local_entrypoint()
def before():
    _save(generate.remote(_rows("test.jsonl"), MODEL), "out_before_9b.jsonl")


@app.local_entrypoint()
def train():
    res = train_lora.remote(_rows("train_v2.jsonl"))
    (LOCAL / "train_log_9b.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("rows", "kept", "train_loss", "steps")}))


@app.local_entrypoint()
def after():
    _save(generate.remote(_rows("test.jsonl"), MERGED), "out_after_9b.jsonl")


# stronger run on the same data: three epochs at 1e-4 (the one-epoch run moved loss only 0.78 -> 0.72)
@app.local_entrypoint()
def strong():
    res = train_lora.remote(_rows("train_v2.jsonl"), 3, 32_768, 1e-4, f"{CACHE}/merged_9b_strong")
    (LOCAL / "train_log_9b_strong.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("rows", "kept", "train_loss", "steps")}))
    _save(generate.remote(_rows("test.jsonl"), f"{CACHE}/merged_9b_strong"), "out_after_9b_strong.jsonl")


# ---- RL (GRPO-style, user 2026-10-27): rollouts here, Sonnet reward locally (scripts/sft_fto/rl.py), policy step here ----
@app.function(image=infer_image, gpu="H100", volumes={CACHE: VOL}, timeout=3 * 3600)
def rollouts(rows: list[dict], path: str, n: int = 4, temperature: float = 0.8) -> list[dict]:
    from vllm import LLM, SamplingParams
    VOL.reload()
    llm = LLM(path, tokenizer=MODEL, max_model_len=40_000, gpu_memory_utilization=0.9, limit_mm_per_prompt={"image": 0, "video": 0})
    sp = SamplingParams(n=n, temperature=temperature, max_tokens=12_288, repetition_penalty=1.05, seed=0)
    outs = llm.chat([r["messages"][:2] for r in rows], sp, chat_template_kwargs=NO_THINK)
    return [{"id": r["id"], "k": k, "text": c.text, "finish": c.finish_reason} for r, o in zip(rows, outs) for k, c in enumerate(o.outputs)]


@app.function(image=train_image, gpu="H100", volumes={CACHE: VOL}, timeout=4 * 3600)
def pg_step(items: list[dict], adapter_in: str | None, tag: str, lr: float = 2e-5, accum: int = 16) -> dict:
    """One on-policy pass: loss = -advantage x mean log-prob of the completion's tokens, LoRA only. Saves the adapter
    and a merged copy for the next round's rollouts."""
    import random

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    VOL.reload()
    tok = AutoTokenizer.from_pretrained(MODEL)
    base = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")
    if adapter_in:
        model = PeftModel.from_pretrained(base, adapter_in, is_trainable=True)
    else:
        model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
                                                target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"))
    inner = model.get_base_model()
    inner.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    inner.enable_input_require_grads()
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    random.Random(0).shuffle(items)
    log, n = [], 0
    for i, it in enumerate(items):
        p_ids = tok.apply_chat_template(it["messages"][:2], add_generation_prompt=True, tokenize=True, **NO_THINK)
        p_ids = p_ids if isinstance(p_ids, list) else list(p_ids["input_ids"])   # transformers 5 returns a BatchEncoding
        c_ids = tok(it["text"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
        ids = torch.tensor([p_ids + c_ids], device="cuda")
        h = inner.model(input_ids=ids).last_hidden_state[0, len(p_ids) - 1:-1]   # logits only over the completion
        logp = torch.log_softmax(inner.lm_head(h).float(), -1).gather(-1, ids[0, len(p_ids):, None]).squeeze(-1)
        loss = -it["adv"] * logp.mean() / accum
        loss.backward()
        n += 1
        log.append({"i": i, "adv": it["adv"], "logp": logp.mean().item()})
        if n == accum or i == len(items) - 1:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            n = 0
    out = f"{CACHE}/rl/{tag}"
    model.save_pretrained(f"{out}/adapter")
    model.merge_and_unload().save_pretrained(f"{out}/merged")
    tok.save_pretrained(f"{out}/merged")
    VOL.commit()
    return {"tag": tag, "items": len(items), "log": log}

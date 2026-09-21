"""Remote (A100) scoring of PokerBench-format prompts with YiPz/llama3-8b-pokerbench-sft via vLLM.
For each prompt: chat template (system from the model card) + assistant prefix '<action>'; read the next-token top-25
logprobs and pick the first token of each action word -> per-decision action-type log-probs; also keep the greedy text."""
import time, sys, pandas as pd, numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
MODEL = 'YiPz/llama3-8b-pokerbench-sft'
SYS = 'You are an expert poker player. Respond with your action in <action></action> tags.'
tok = AutoTokenizer.from_pretrained(MODEL)
import argparse
ap = argparse.ArgumentParser(); ap.add_argument('n_head', nargs='?', type=int, default=0); ap.add_argument('--in', dest='inp', default='prompts_unique.parquet')
ap.add_argument('--shard', default='0/1'); ap.add_argument('--out', default='scores.parquet'); args = ap.parse_args()
df = pd.read_parquet(args.inp)
si, sn = [int(x) for x in args.shard.split('/')]; df = df.iloc[si::sn].reset_index(drop=True)
if args.n_head: df = df.head(args.n_head)
print(f'input {args.inp} shard {si}/{sn}: {len(df)} prompts -> {args.out}', flush=True)
def build(p):
    s = tok.apply_chat_template([{'role': 'system', 'content': SYS}, {'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True)
    return s + '<action>'
prompts = [build(p) for p in df.prompt.tolist()]
pre = tok.encode('<action>', add_special_tokens=False); cands = {}
for w in ['fold', 'check', 'call', 'bet', 'raise', 'allin', 'all']:
    ids = tok.encode('<action>' + w, add_special_tokens=False)
    cands[w] = ids[len(pre)] if ids[:len(pre)] == pre and len(ids) > len(pre) else tok.encode(w, add_special_tokens=False)[0]
print('prefix tokens', pre, 'candidate first tokens', {w: (i, tok.decode([i])) for w, i in cands.items()}, flush=True)
print('example prompt tail:', repr(prompts[0][-200:]), flush=True)
llm = LLM(model=MODEL, dtype='bfloat16', max_model_len=1536, gpu_memory_utilization=0.92, enable_prefix_caching=True, max_logprobs=30)   # default max_logprobs=20 < 25 requested
sp = SamplingParams(max_tokens=8, temperature=0, logprobs=25)
t0 = time.time(); outs = llm.generate(prompts, sp); dt = time.time() - t0
rows = []
for o in outs:
    lps = o.outputs[0].logprobs; lp0 = lps[0] if lps else {}
    floor = min(v.logprob for v in lp0.values()) if lp0 else -30.0
    rec = {f'lp_{w}': (lp0[i].logprob if i in lp0 else floor - 1.0) for w, i in cands.items()}
    rec['in_top'] = int(sum(i in lp0 for i in cands.values())); rec['gen'] = o.outputs[0].text; rows.append(rec)
res = pd.DataFrame(rows); res['hand_id'] = df.hand_id.values; res['action_no'] = df.action_no.values
res.to_parquet(args.out, index=False)
print(f'done {len(res)} prompts in {dt:.0f}s; gen sample: {res.gen.head(5).tolist()}; in_top mean {res.in_top.mean():.2f}', flush=True)
import os; sys.stdout.flush(); sys.stderr.flush(); os._exit(0)   # vLLM v1 engine process otherwise hangs at interpreter exit (smoke run held the GPU 25 min)

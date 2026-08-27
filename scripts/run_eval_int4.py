#!/usr/bin/env python3
"""VCC-Bench: Rule vs INT4 Model vs Perplexity comparison — Qwen2.5-0.5B-Instruct (fp16)"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from vncompress.evaluation import VCCBench, VCCBenchConfig
from vncompress.compressors.no_model import (
    NoModelToneCompressor, NoModelCombinedCompressor, NoModelBaselineCompressor,
)
from vncompress.tone_aware.vietnamese_tones import TONE_NAME_TO_ID, get_tone_analyzer
from vncompress.tone_aware.tone_scoring import PhonologicalConsistencyLoss

ta = get_tone_analyzer()

def tpr(original_ids, compressed_ids, tok):
    def tones(ids):
        r = []
        for tid in ids:
            ts = tok.decode([tid]).replace('\u2581',' ').replace('\u0130',' ').strip()
            tn = ta.get_dominant_tone(ts[:20])
            r.append(TONE_NAME_TO_ID.get(tn or 'ngang', 0))
        return r
    o, c = tones(original_ids), tones(compressed_ids)
    oc = sum(1 for t in o if t > 0)
    return sum(1 for t in c if t > 0) / max(oc, 1) if oc > 0 else 1.0

print("=" * 64)
print("VCC-Bench: INT4 Model Comparison")
print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB)")
print("=" * 64)

print("\nLoading model...")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token

m = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True, torch_dtype=torch.float16,
)  # no device_map: see docs/benchmark.md (caching_allocator_warmup segfault)
m = m.to('cuda')
m.eval(); m.config.output_hidden_states = True

probe = PhonologicalConsistencyLoss(hidden_dim=m.config.hidden_size)
probe = probe.to(m.device).half().eval()

print(f"  VRAM: {torch.cuda.max_memory_allocated()/1024**3:.1f} GB")
torch.cuda.reset_peak_memory_stats()

MAX_TOKENS = 2048

def model_tone_compress(ids, ratio):
    n = min(len(ids), MAX_TOKENS); s = ids[:n]
    tl = max(int(n/ratio), 4); k = 2
    it = torch.tensor([s]).to(m.device)
    with torch.no_grad(): hs = m(it, output_hidden_states=True).hidden_states[-1]
    w = probe.score_importance(hs).squeeze(0).cpu().clamp(0.5, 3.0)
    mb = max(0, tl-2*k)
    mw = w[k:n-k] if n>2*k else w
    keep = set(range(k)) | set(range(n-k, n))
    if mb > 0 and len(mw) > 0:
        _, top = torch.topk(mw, min(mb, len(mw)))
        keep.update(k+i for i in sorted(top.tolist()))
    r = [s[i] for i in sorted(keep) if i<n]
    return r + ids[MAX_TOKENS:] if len(ids)>MAX_TOKENS else r

def model_ppl_compress(ids, ratio):
    n = min(len(ids), MAX_TOKENS); s = ids[:n]
    tl = max(int(n/ratio), 4); k = 2
    it = torch.tensor([s]).to(m.device)
    with torch.no_grad(): lo = m(it, labels=it).logits.squeeze(0).float()
    ppl = torch.cat([torch.zeros(1), F.cross_entropy(lo[:-1], it[0,1:], reduction='none').cpu()])
    ppl = 1.0 + ppl; ppl.clamp_(0.5, 3.0)
    mb = max(0, tl-2*k)
    mp = ppl[k:n-k] if n>2*k else ppl
    keep = set(range(k)) | set(range(n-k, n))
    if mb > 0 and len(mp) > 0:
        _, top = torch.topk(mp, min(mb, len(mp)))
        keep.update(k+i for i in sorted(top.tolist()))
    r = [s[i] for i in sorted(keep) if i<n]
    return r + ids[MAX_TOKENS:] if len(ids)>MAX_TOKENS else r

print("Loading VCC-Bench...")
dp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vcc_bench_data", "vcc_bench_v1.json")
cfg = VCCBenchConfig(compression_ratios=[2.0,4.0])
bench = VCCBench.load_from_json(dp, cfg)

nm = {
    "rule_tone": ("Tone-Aware (rule, 0 VRAM)", NoModelToneCompressor(tok)),
    "rule_comb": ("Combined (rule, 0 VRAM)", NoModelCombinedCompressor(tok)),
    "baseline": ("Random Baseline", NoModelBaselineCompressor(tok, mode="random")),
    "first_n": ("First-N Baseline", NoModelBaselineCompressor(tok, mode="first")),
}
mm = [
    ("model_ppl", "Perplexity (INT4 model)", model_ppl_compress),
    ("model_tone", "Tone Probe (INT4 model)", model_tone_compress),
]

all_r = {}; ts = time.time()
for task, samples in bench.samples.items():
    if task not in cfg.tasks: continue
    n = min(3, len(samples)); all_r[task] = {}
    for ratio in cfg.compression_ratios:
        for mn, (desc, comp) in nm.items():
            crs,tprs,tms=[],[],[]
            for s in samples[:n]:
                ids=tok.encode(s.context, add_special_tokens=False)
                if len(ids)<10: continue
                try:
                    t0=time.time(); r=comp.compress(ids,target_ratio=ratio)
                    dt=(time.time()-t0)*1000
                    crs.append(len(ids)/max(len(r.compressed_ids),1))
                    tprs.append(tpr(ids,r.compressed_ids,tok)); tms.append(dt)
                except: pass
            if crs: all_r[task][f"{mn}_r{ratio}"]={"cr":sum(crs)/len(crs),"tpr":sum(tprs)/len(tprs),"ms":sum(tms)/len(tms),"n":len(crs)}
        for mn, desc, fn in mm:
            crs,tprs,tms=[],[],[]
            for s in samples[:n]:
                ids=tok.encode(s.context, add_special_tokens=False)
                if len(ids)<10: continue
                try:
                    t0=time.time(); comp=fn(ids,ratio)
                    dt=(time.time()-t0)*1000
                    crs.append(len(ids)/max(len(comp),1))
                    tprs.append(tpr(ids,comp,tok)); tms.append(dt)
                except: pass
            if crs: all_r[task][f"{mn}_r{ratio}"]={"cr":sum(crs)/len(crs),"tpr":sum(tprs)/len(tprs),"ms":sum(tms)/len(tms),"n":len(crs)}

elapsed=time.time()-ts; peak=torch.cuda.max_memory_allocated()/1024**3

print("\n"+"="*72)
print(f"RESULTS  |  Qwen2.5-0.5B  |  {elapsed:.0f}s  |  VRAM peak: {peak:.1f} GB")
print("="*72)
disp={"rule_tone":"Tone-Aware (rule)","rule_comb":"Combined (rule)","baseline":"Random","first_n":"First-N","model_ppl":"Perplexity (INT4)","model_tone":"Tone Probe (INT4)"}
order=["first_n","baseline","rule_tone","rule_comb","model_ppl","model_tone"]
for r in ["2.0","4.0"]:
    print(f"\n  RATIO {r}x:")
    print(f"  {'Method':<25s} {'CR':>6s} {'TPR':>7s} {'Time':>8s}")
    print(f"  {'-'*25} {'-'*6} {'-'*7} {'-'*8}")
    for mn in order:
        a,b,c=[],[],[]
        for tk in all_r:
            d=all_r[tk].get(f"{mn}_r{r}",{})
            if d.get("n",0)>0: a.append(d["cr"]);b.append(d["tpr"]);c.append(d["ms"])
        if a: print(f"  {disp[mn]:<25s} {sum(a)/len(a):>6.2f} {sum(b)/len(b):>7.3f} {sum(c)/len(c):>7.0f}ms")

print("\n  PER-TASK TPR @4x:")
print(f"  {'Task':<22s} {'Rule':>7s} {'PPL':>7s} {'Probe':>7s} {'Random':>7s}")
print(f"  {'-'*22} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
for tk in sorted(all_r.keys()):
    a=all_r[tk].get("rule_tone_r4.0",{}).get("tpr",0)
    b=all_r[tk].get("model_ppl_r4.0",{}).get("tpr",0)
    c=all_r[tk].get("model_tone_r4.0",{}).get("tpr",0)
    d=all_r[tk].get("baseline_r4.0",{}).get("tpr",0)
    print(f"  {tk[:22]:<22s} {a:>7.3f} {b:>7.3f} {c:>7.3f} {d:>7.3f}")

os.makedirs("./results_int4",exist_ok=True)
json.dump(all_r, open("./results_int4/comparison.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)
print("\nSaved: ./results_int4/comparison.json")

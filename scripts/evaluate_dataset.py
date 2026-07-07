#!/usr/bin/env python3
"""Self-evaluation of VCC-Bench dataset quality."""
import json, os, sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'vcc_bench_data')

def main():
    path = os.path.join(DATA_DIR, 'vcc_bench_v1.json')
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = data['samples']
    meta = data['metadata']

    print('=' * 60)
    print('VCC-Bench v1.0 -- Self-Evaluation Report')
    print('=' * 60)

    name = meta.get('name', '?')
    ver = meta.get('version', '?')
    date = meta.get('date', '?')
    lic = meta.get('license', '?')
    print(f'\nDataset: {name} v{ver}')
    print(f'Date: {date}')
    print(f'License: {lic}')
    print(f'Total samples: {len(samples)}')

    print(f'\n--- Task Distribution ---')
    task_counts = Counter(s['task'] for s in samples)
    for task, count in sorted(task_counts.items()):
        bar = '#' * (count // 5)
        print(f'  {task:<30s} {count:>4d}  {bar}')

    print(f'\n--- Context Length Distribution (chars) ---')
    for task in sorted(task_counts.keys()):
        lengths = [s.get('char_length', len(s.get('context', ''))) for s in samples if s['task'] == task]
        if lengths:
            lengths.sort()
            n = len(lengths)
            print(f'  {task}:')
            print(f'    Count={n}, Min={lengths[0]}, P25={lengths[n//4]}, Med={lengths[n//2]}, P75={lengths[3*n//4]}, Max={lengths[-1]}')
            print(f'    Avg={sum(lengths)//n}')

    print(f'\n--- Domain Diversity (Long-Doc QA) ---')
    domains = Counter(s.get('domain', '?') for s in samples if s['task'] == 'long_document_qa')
    print(f'  Unique domains: {len(domains)}')
    for d, c in domains.most_common(10):
        print(f'    {d}: {c}')

    print(f'\n--- Conversation Scenarios ---')
    scenarios = Counter(s.get('scenario', '?') for s in samples if s['task'] == 'multi_turn_conversation')
    print(f'  Unique scenarios: {len(scenarios)}')
    for s, c in scenarios.most_common():
        print(f'    {s}: {c}')

    print(f'\n--- Quality Assessment ---')
    query_lens = [len(s['query']) for s in samples]
    ref_lens = [len(s.get('reference_answer', '')) for s in samples]
    ctx_lens = [len(s.get('context', '')) for s in samples]
    print(f'  Context lengths: min={min(ctx_lens)}, max={max(ctx_lens)}, avg={sum(ctx_lens)//len(ctx_lens)}')
    print(f'  Query lengths: min={min(query_lens)}, max={max(query_lens)}, avg={sum(query_lens)//len(query_lens)}')
    print(f'  Reference lengths: min={min(ref_lens)}, max={max(ref_lens)}, avg={sum(ref_lens)//len(ref_lens)}')

    empty_contexts = sum(1 for s in samples if not s.get('context', '').strip())
    empty_queries = sum(1 for s in samples if not s.get('query', '').strip())
    empty_refs = sum(1 for s in samples if not s.get('reference_answer', '').strip())
    print(f'  Empty contexts: {empty_contexts}')
    print(f'  Empty queries: {empty_queries}')
    print(f'  Empty references: {empty_refs}')

    print(f'\n--- Vietnamese Content Verification ---')
    from vncompress.tone_aware.vietnamese_tones import is_vietnamese
    vi_count = 0
    for s in samples:
        ctx = s.get('context', '')[:1000]
        if is_vietnamese(ctx):
            vi_count += 1
    print(f'  Samples with Vietnamese text: {vi_count}/{len(samples)} ({100*vi_count//len(samples)}%)')

    print(f'\n--- VERDICT ---')
    issues = []
    if empty_contexts > 0:
        issues.append(f'{empty_contexts} empty contexts')
    if empty_queries > 0:
        issues.append(f'{empty_queries} empty queries')
    if empty_refs > 0:
        issues.append(f'{empty_refs} empty references')
    if task_counts.get('needle_in_haystack', 0) < 10:
        issues.append('needle_in_haystack < 10 samples')
    if task_counts.get('agent_tool_calling', 0) < 5:
        issues.append('agent_tool_calling < 5 samples')
    if len(domains) < 5:
        issues.append(f'only {len(domains)} document domains (min 5)')

    if issues:
        print('ISSUES: ' + ', '.join(issues))
    else:
        print('PASSED -- Dataset ready for benchmarking.')

    print(f'\n--- File Sizes ---')
    for f in sorted(os.listdir(DATA_DIR)):
        if f.endswith('.json'):
            size = os.path.getsize(os.path.join(DATA_DIR, f))
            samples_in_file = 0
            if f.startswith('vcc_bench_') and not f.startswith('vcc_bench_v1'):
                with open(os.path.join(DATA_DIR, f), 'r', encoding='utf-8') as ff:
                    task_data = json.load(ff)
                    samples_in_file = len(task_data.get('samples', []))
            info = f' ({samples_in_file} samples)' if samples_in_file else ''
            print(f'  {f}: {size/1024:.1f} KB{info}')

if __name__ == '__main__':
    main()

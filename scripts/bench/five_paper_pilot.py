"""Five-whole-papers-per-call development pilot; no test papers are loaded.

Prepare and commit the plan before running. Run only one configuration per fresh
output file. This pilot does not authorize or launch model servers.
"""
import argparse
import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
from openai import OpenAI

import baseline_qwen as baseline
from baseline_evidence import digest
from nli_variants import load_val
from stage_c_nli_qwen import TAUS, TYPES, candidates, evc
from stage_c_rubric_val import RUBRICS

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'docs/experiments/five_paper_pilot_2026-11-21'
REFERENCE = DEST / 'reference_27b_val_strict.jsonl'
PLAN = DEST / 'plan.json'
SEED, MAX_OUTPUT = 20261121, 8192
IDS = ['2401.17544', '2201.02967', '2207.09511', '2301.03441', '2507.19233']


def sha(path):
    data = path.read_bytes()
    if path.suffix in {'.py', '.json', '.jsonl'}:
        data = data.replace(b'\r\n', b'\n')
    return hashlib.sha256(data).hexdigest()


def data():
    rows = baseline.load_rows('gold', 'dev')
    paper_ids = sorted({r['pid'] for r in rows})
    if random.Random(SEED).sample(paper_ids, 5) != IDS:
        raise ValueError('The preselected development sample changed')
    selected = [r for pid in IDS for r in rows if r['pid'] == pid]
    if not selected or {r['pid'] for r in selected} != set(IDS):
        raise ValueError('Selected papers missing')
    _, val = load_val('val')
    scores = np.load(ROOT / 'data/gold/nli_val_scores.npz')
    typed = np.stack([evc(scores, h) for h in TYPES], 1)
    pool = candidates(val, typed, TAUS['r90'])
    if len(pool) != 2278:
        raise ValueError('Original validation candidate set changed')
    chosen = [r for r in pool if r['id'] in IDS]
    return selected, pool, chosen


def group_prompt(rows):
    prompt = baseline.build_prompt('gold', 'strict') + (
        '\n\nThere are five separate papers below. Lines starting "PAPER " are boundaries, not sentences. '
        'Judge each numbered sentence only in its own paper; do not use context from a different paper.')
    text, keys, last = [], [], None
    for row in rows:
        if row['pid'] != last:
            last = row['pid']
            text.append(f"PAPER {last}")
        keys.append([row['pid'], row['idx']])
        text.append(f"[{len(keys)-1}] {row['sentence']}")
    return prompt, '\n'.join(text), keys


def make_plan():
    rows, pool, chosen = data()
    system, user, keys = group_prompt(rows)
    raw = [json.loads(x) for x in REFERENCE.read_text(encoding='utf8').splitlines() if x.strip()]
    if len(raw) != len(pool) or {r['i'] for r in raw} != set(range(len(pool))):
        raise ValueError('Saved 27B validation verdicts are incomplete or duplicated')
    files = ['data/gold/gap_gold_final.jsonl', 'data/gold/splits.json', 'data/paper_texts.jsonl',
             'data/gold/nli_val_scores.npz', 'docs/experiments/nli_variants.json',
             str(REFERENCE.relative_to(ROOT)), 'scripts/bench/five_paper_pilot.py']
    plan = {'purpose': '5 full papers per call vs 20-candidate funnel; development pilot only',
            'sample_rule': 'random.Random(20261121).sample(sorted(arxiv validation paper ids), 5)',
            'paper_ids': IDS, 'rows_sha256': digest(rows), 'keys_sha256': digest(keys),
            'full_paper_system_sha256': digest(system), 'full_paper_user_sha256': digest(user),
            'funnel_system_sha256': digest(RUBRICS['strict']), 'selected_candidate_keys_sha256': digest([[r['id'],r['idx']] for r in chosen]),
            'mode': 'high_precision', 'stage_b_tau': TAUS['r90'], 'full_papers_per_call': 5,
            'funnel_candidates_per_call': 20, 'seed': 0, 'temperature': 0, 'thinking': False,
            'full_paper_max_output_tokens': MAX_OUTPUT,
            'models': {'27b': {'repo': 'Qwen/Qwen3.8-27B-FP8', 'revision': '017b9c7af6b5689d5dd426a76e0bc077eb5ca20a'},
                       '4b': {'repo': 'Qwen/Qwen3.5-4B', 'revision': '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'}},
            'comparison': '27B uses original full-validation batch-20 strict funnel verdicts; 4B funnel will run on the selected candidates in batches of 20. All five-paper outputs are scored on all sentences. No input labels are sent.',
            'metrics': 'Sentence-level TP, FP, FN, precision, recall, F1, exact failures and token usage; five papers are too few for inferential claims.',
            'files_sha256': {p: sha(ROOT / p) for p in files},
            'hash_note': 'Text file SHA-256 values normalize CRLF to LF; NPZ bytes are not normalized.'}
    return plan


def require_frozen():
    plan = json.loads(PLAN.read_text(encoding='utf8'))
    if plan != make_plan():
        raise SystemExit('Plan, data, prompts, candidate selection or code changed; do not run')
    paths = [str(PLAN.relative_to(ROOT)), str(Path(__file__).resolve().relative_to(ROOT)), str(REFERENCE.relative_to(ROOT))]
    for path in paths:
        if subprocess.run(['git','status','--porcelain','--',path],cwd=ROOT,capture_output=True,text=True).stdout.strip():
            raise SystemExit('Commit the exact plan, runner and reference before inference')
        if subprocess.run(['git','cat-file','-e',f'HEAD:{path}'],cwd=ROOT,capture_output=True).returncode:
            raise SystemExit('Plan, runner or reference not in HEAD')
    return plan


def client():
    base, key = os.environ.get('QWEN_BASE_URL'), os.environ.get('QWEN_API_KEY')
    if not base or not key:
        raise SystemExit('Set QWEN_BASE_URL and QWEN_API_KEY in the process environment')
    return OpenAI(base_url=base, api_key=key, timeout=1800, max_retries=0)


def metrics(rows, kept):
    totals = [sum(v[j] for v in baseline.per_paper(rows, kept).values()) for j in range(3)]
    p, r, f1 = baseline.prf(*totals)
    return {'tp': totals[0], 'fp': totals[1], 'positives': totals[2],
            'precision': round(p, 4), 'recall': round(r, 4), 'f1': round(f1, 4)}


def run_full(model, served):
    plan = require_frozen()
    rows, _, _ = data()
    system, user, keys = group_prompt(rows)
    output = DEST / f'{model}_five_papers.json'
    if output.exists():
        raise SystemExit(f'{output} already exists; preserve existing evidence')
    c = client()
    if served not in {x.id for x in c.models.list().data}:
        raise SystemExit('Requested served model is not available')
    schema = {'type':'object','additionalProperties':False,'required':['positives'],
              'properties':{'positives':{'type':'array','items':{'type':'integer'}}}}
    attempts=[]
    for _ in range(2):
        t=time.time(); item=None
        try:
            resp=c.chat.completions.create(model=served, temperature=0, max_tokens=MAX_OUTPUT,
                messages=[{'role':'system','content':system},{'role':'user','content':user}],
                response_format={'type':'json_schema','json_schema':{'name':'positives','schema':schema,'strict':True}},
                extra_body={'seed':0,'chat_template_kwargs':{'enable_thinking':False}})
            item={'seconds':round(time.time()-t,3),'finish_reason':resp.choices[0].finish_reason,
                  'usage':resp.usage.model_dump() if resp.usage else None,'response':resp.choices[0].message.content}
            attempts.append(item)
            if item['finish_reason']!='stop':continue
            parsed=json.loads(item['response']); ids=parsed.get('positives')
            if set(parsed)!={'positives'} or not isinstance(ids,list) or any(type(i)!=int or not 0<=i<len(keys) for i in ids) or len(set(ids))!=len(ids):
                item['validation_error']='invalid, duplicate or out-of-range ids';continue
            kept={tuple(keys[i]) for i in ids}
            result={'status':'ok','model':model,'served_model':served,'plan_sha256':sha(PLAN),
                    'model_revision_claim':plan['models'][model], 'attempts':attempts,'kept':[list(k) for k in sorted(kept)],
                    'metrics':metrics(rows,kept)}
            output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
            print(json.dumps({'status':'ok','model':model,'metrics':result['metrics'],'calls':len(attempts),'usage':item['usage']}),flush=True)
            return
        except Exception as exc:
            error={'error_type':type(exc).__name__,'status_code':getattr(exc,'status_code',None)}
            if item is None: attempts.append({'seconds':round(time.time()-t,3), **error})
            else: item['validation_error']=error
    output.write_text(json.dumps({'status':'failed','model':model,'plan_sha256':sha(PLAN),'attempts':attempts},indent=2)+'\n',encoding='utf8')
    raise SystemExit('All attempts failed; evidence saved without treating failures as negative answers')


def score_reference():
    require_frozen()
    rows,pool,_ = data()
    raw = [json.loads(x) for x in REFERENCE.read_text(encoding='utf8').splitlines() if x.strip()]
    kept={(pool[r['i']]['id'],pool[r['i']]['idx']) for r in raw if r['keep']}
    universe={(r['pid'],r['idx']) for r in rows}
    print(json.dumps({'model':'27b','reference':'original validation strict funnel',
                      'metrics':metrics(rows,kept & universe)},indent=2))


def run_funnel_4b(served):
    require_frozen()
    rows, _, chosen = data()
    output = DEST / '4b_funnel_batches.jsonl'
    summary = DEST / '4b_funnel_summary.json'
    if output.exists() or summary.exists():
        raise SystemExit('4B funnel output exists; preserve it and inspect before rerunning')
    c = client()
    if served not in {x.id for x in c.models.list().data}:
        raise SystemExit('Requested served model is not available')
    schema={'type':'object','additionalProperties':False,'required':['results'],
            'properties':{'results':{'type':'array','items':{'type':'object','additionalProperties':False,
             'required':['id','keep'],'properties':{'id':{'type':'integer'},'keep':{'type':'boolean'}}}}}}
    kept=set()
    with output.open('x',encoding='utf8') as fh:
        for start in range(0,len(chosen),20):
            batch=chosen[start:start+20]
            items=[{'i':start+j,'type':r['gap_type'],'context':r['paragraph_text'][:600],
                    'sentence':r['gap_sentence'][:400]} for j,r in enumerate(batch)]
            want={i['i'] for i in items}; attempts=[]; verdict=None
            for _ in range(2):
                t=time.time(); item=None
                try:
                    resp=c.chat.completions.create(model=served,temperature=0,max_tokens=256+400*len(items),
                        messages=[{'role':'system','content':RUBRICS['strict']},
                                  {'role':'user','content':json.dumps(items,ensure_ascii=False)}],
                        response_format={'type':'json_schema','json_schema':{'name':'gap_verdicts','schema':schema,'strict':True}},
                        extra_body={'seed':0,'chat_template_kwargs':{'enable_thinking':False}})
                    item={'seconds':round(time.time()-t,3),'finish_reason':resp.choices[0].finish_reason,
                          'usage':resp.usage.model_dump() if resp.usage else None,'response':resp.choices[0].message.content}
                    attempts.append(item)
                    if item['finish_reason']!='stop':continue
                    parsed=json.loads(item['response'])
                    result=parsed.get('results',[])
                    if (set(parsed)!={'results'} or len(result)!=len(want) or
                        any(set(r)!={'id','keep'} or type(r['id']) is not int or type(r['keep']) is not bool for r in result) or
                        {r['id'] for r in result}!=want):
                        item['validation_error']='incomplete, duplicate or invalid results';continue
                    verdict=result;break
                except Exception as exc:
                    error={'error_type':type(exc).__name__,'status_code':getattr(exc,'status_code',None)}
                    if item is None: attempts.append({'seconds':round(time.time()-t,3), **error})
                    else: item['validation_error']=error
            record={'start':start,'keys':[[r['id'],r['idx']] for r in batch], 'attempts':attempts,
                    'status':'ok' if verdict is not None else 'failed','verdicts':verdict}
            fh.write(json.dumps(record,ensure_ascii=False)+'\n');fh.flush()
            if verdict is None:raise SystemExit(f'Funnel batch {start} failed; evidence saved')
            kept.update((chosen[r['id']]['id'],chosen[r['id']]['idx']) for r in verdict if r['keep'])
            print(f'4B funnel: {min(start+20,len(chosen))}/{len(chosen)} candidates',flush=True)
    result={'status':'ok','model':'4b','served_model':served,'plan_sha256':sha(PLAN),
            'candidate_count':len(chosen),'kept':[list(k) for k in sorted(kept)],'metrics':metrics(rows,kept)}
    summary.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'status':'ok','model':'4b','funnel_metrics':result['metrics']}),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=('prepare','reference','full','funnel4'))
    ap.add_argument('--model',choices=('27b','4b'))
    ap.add_argument('--served-model')
    args=ap.parse_args()
    if args.command=='prepare':
        if PLAN.exists(): raise SystemExit('Plan already exists; use a fresh pilot directory')
        DEST.mkdir(parents=True,exist_ok=True)
        plan=make_plan(); PLAN.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf8')
        print('Prepared plan for',len(plan['paper_ids']),'development papers; commit before inference')
    elif args.command=='reference':score_reference()
    elif args.command=='funnel4':
        if not args.served_model:ap.error('funnel4 requires --served-model')
        run_funnel_4b(args.served_model)
    elif args.command=='full':
        if not args.model or not args.served_model:ap.error('full requires --model and --served-model')
        run_full(args.model,args.served_model)


if __name__ == '__main__':main()

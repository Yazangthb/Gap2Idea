"""Five-paper 27B prompt tuning on development papers and separate dev check.

Prepare/commit plan first; run all four tuning prompts for both methods; select
by shortlist F1 with specified tie-breakers; commit selection before check.
One model request per five-paper group per method, failures recorded as failures.
"""
import json
import random
import subprocess
import sys
import time

import numpy as np
import five_paper_pilot as p
from baseline_evidence import digest
from nli_variants import load_val
from stage_c_nli_qwen import TAUS,TYPES,candidates,evc

DEST=p.ROOT/'docs/experiments/five_paper_dev_tuning_2026-11-22'
PLAN=DEST/'plan.json'; SELECTION=DEST/'selection.json'
NAMES=('neutral','strict','keep_biased','short')
BOUNDARY='\n\nThere are five separate papers below. Lines starting "PAPER " are boundaries, not sentences. Judge each numbered sentence only in its own paper.'
SHORT=('For each numbered sentence, return its ID in the positives list if it states a research limitation, '
       'future-work direction, or an unsolved problem in the field or prior work.')
OLD=('The input is consecutive sentences from scientific papers, in reading order, one per line as "[id] sentence". '
     'Judge every sentence on its own, using the neighbouring sentences only to disambiguate.')
NEW=('The input is sentences from scientific papers, in reading order, one per line as "[id] sentence". '
     'Judge every sentence on its own.')


def make_data():
    raw=p.baseline.load_rows('gold','dev'); eligible=sorted({r['pid'] for r in raw}-set(p.IDS))
    rng=random.Random(20261122); rng.shuffle(eligible)
    if len(eligible)<10: raise ValueError('Not enough unused arXiv dev papers')
    _,val=load_val('val'); npz=np.load(p.ROOT/'data/gold/nli_val_scores.npz')
    typed=np.stack([evc(npz,h) for h in TYPES],1); all_candidates=candidates(val,typed,TAUS['r90'])
    result={}
    for split,pids in [('tune',eligible[:5]),('check',eligible[5:10])]:
        rows=[r for pid in pids for r in raw if r['pid']==pid]
        candidates_keys={(r['id'],r['idx']) for r in all_candidates if r['id'] in pids}
        keys=[(r['pid'],r['idx']) for r in rows]
        if len(set(keys))!=len(keys) or not candidates_keys<=set(keys): raise ValueError('Data or candidate keys mismatch')
        result[split]={'paper_ids':pids,'rows':rows,'keys':keys,'candidate_keys':candidates_keys}
    return result


def system(name):
    if name=='short': return SHORT+BOUNDARY
    text=p.baseline.build_prompt('gold',name)
    if OLD not in text: raise ValueError('Unexpected prompt')
    return text.replace(OLD,NEW)+BOUNDARY


def user_text(data,method):
    valid=set(data['keys']) if method=='full' else data['candidate_keys']
    ids={k:i for i,k in enumerate(data['keys'])}; lines=[]; last=None
    for r in data['rows']:
        key=(r['pid'],r['idx'])
        if key not in valid: continue
        if r['pid']!=last: last=r['pid']; lines.append('PAPER '+last)
        lines.append(f'[{ids[key]}] {r["sentence"]}')
    return valid,'\n'.join(lines)


def plan_for(data):
    return {'purpose':'Select a prompt on five arXiv development papers, then evaluate the selected prompt on a separate five-paper development group; not a blind test',
      'split_rule':'Exclude earlier five-paper pilot; shuffle sorted remaining arXiv development paper ids with random.Random(20261122); first five tune, next five check',
      'candidate_selection':'Stage A junk filter + typed NLI Stage B, frozen r90 threshold',
      'selection_rule':'Highest tuning-set funnel F1, tie break higher precision, then higher recall, then prompt name alphabetically',
      'prompts':{n:{'text':system(n),'sha256':digest(system(n))} for n in NAMES},
      'methods':'same prompt, model and format; one Qwen request for five full papers vs one for their Stage A+B candidates',
      'model':'qwen3.8-27b','temperature':0,'seed':0,'thinking':False,'max_output_tokens':8192,
      'run_order':'all four tuning prompts x both methods; select and commit; selected prompt x both methods on separate check papers',
      'splits':{k:{'paper_ids':d['paper_ids'],'rows_sha256':digest(d['rows']),
                   'keys_sha256':digest(d['keys']),'candidate_keys_sha256':digest(sorted(d['candidate_keys'])),
                   'sentences':len(d['rows']),'candidates':len(d['candidate_keys']),
                   'full_input_sha256':digest(user_text(d,'full')[1]),'funnel_input_sha256':digest(user_text(d,'funnel')[1])}
                for k,d in data.items()},
      'source_files_sha256':{f:p.sha(p.ROOT/f) for f in ('scripts/bench/five_paper_dev_tuning.py',
          'scripts/bench/five_paper_pilot.py','data/gold/gap_gold_final.jsonl','data/gold/splits.json',
          'data/paper_texts.jsonl','data/gold/nli_val_scores.npz','docs/experiments/nli_variants.json')}}


def check_committed(path):
    rel=path.relative_to(p.ROOT).as_posix()
    if (subprocess.run(['git','status','--porcelain','--',rel],cwd=p.ROOT,capture_output=True,text=True).stdout.strip()
        or subprocess.run(['git','cat-file','-e',f'HEAD:{rel}'],cwd=p.ROOT,capture_output=True).returncode):
        raise SystemExit(f'Commit {rel} first')


def frozen():
    p.require_frozen(); data=make_data()
    if not PLAN.exists() or json.loads(PLAN.read_text(encoding='utf8'))!=plan_for(data): raise SystemExit('Plan or data changed')
    check_committed(PLAN);check_committed(p.ROOT/'scripts/bench/five_paper_dev_tuning.py')
    return data


def run(split,name,method):
    if split not in ('tune','check') or name not in NAMES or method not in ('full','funnel'): raise SystemExit('Invalid config')
    data=frozen(); d=data[split]
    if split=='check':
        check_committed(SELECTION); chosen=json.loads(SELECTION.read_text(encoding='utf8'))
        if chosen['prompt']!=name or chosen['plan_sha256']!=p.sha(PLAN): raise SystemExit('Check prompt does not match selection')
    path=DEST/f'{split}_{name}_{method}.json'
    if path.exists(): raise SystemExit('Existing output preserved')
    valid,user=user_text(d,method); prompt=system(name); c=p.client(); model='qwen3.8-27b'
    if model not in {x.id for x in c.models.list().data}: raise SystemExit('Model unavailable')
    schema={'type':'object','additionalProperties':False,'required':['positives'],
            'properties':{'positives':{'type':'array','items':{'type':'integer'}}}}
    attempts=[]; result=None
    for _ in range(2):
        start=time.time(); item=None
        try:
            resp=c.chat.completions.create(model=model,temperature=0,max_tokens=8192,
                messages=[{'role':'system','content':prompt},{'role':'user','content':user}],
                response_format={'type':'json_schema','json_schema':{'name':'positives','schema':schema,'strict':True}},
                extra_body={'seed':0,'chat_template_kwargs':{'enable_thinking':False}})
            item={'seconds':round(time.time()-start,3),'finish_reason':resp.choices[0].finish_reason,
                  'usage':resp.usage.model_dump() if resp.usage else None,'response':resp.choices[0].message.content};attempts.append(item)
            if item['finish_reason']!='stop': continue
            parsed=json.loads(item['response']); ids=parsed.get('positives')
            if (set(parsed)!={'positives'} or not isinstance(ids,list) or len(set(ids))!=len(ids) or
                any(type(i) is not int or i<0 or i>=len(d['keys']) or d['keys'][i] not in valid for i in ids)):
                item['validation_error']='invalid, duplicate or non-input ids'; continue
            kept={d['keys'][i] for i in ids}
            result={'status':'ok','split':split,'prompt':name,'method':method,'plan_sha256':p.sha(PLAN),
                    'input_sentence_count':len(valid),'prompt_sha256':digest(prompt),'input_sha256':digest(user),
                    'attempts':attempts,'kept':[list(k) for k in sorted(kept)],'metrics':p.metrics(d['rows'],kept)}
            break
        except Exception as e:
            err={'error_type':type(e).__name__,'status_code':getattr(e,'status_code',None)}
            if item is None: attempts.append({'seconds':round(time.time()-start,3),**err})
            else: item['validation_error']=err
    if result is None:
        result={'status':'failed','split':split,'prompt':name,'method':method,'plan_sha256':p.sha(PLAN),'attempts':attempts}
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print(json.dumps({'split':split,'prompt':name,'method':method,'status':result['status'],'metrics':result.get('metrics'),
                      'usage':attempts[-1].get('usage')}),flush=True)
    if result['status']!='ok': raise SystemExit('Failed; evidence preserved')


def select():
    frozen(); candidates_data=[]
    for name in NAMES:
        for method in ('full','funnel'):
            path=DEST/f'tune_{name}_{method}.json'; check_committed(path)
            r=json.loads(path.read_text(encoding='utf8'))
            if r['status']!='ok' or r['plan_sha256']!=p.sha(PLAN): raise SystemExit('Incomplete tuning run')
            if method=='funnel': candidates_data.append((name,r['metrics']))
    if SELECTION.exists(): raise SystemExit('Selection already exists')
    candidates_data.sort(key=lambda x:(-x[1]['f1'],-x[1]['precision'],-x[1]['recall'],x[0]))
    result={'prompt':candidates_data[0][0],'plan_sha256':p.sha(PLAN),
            'rule':'Highest funnel F1; ties precision, recall, alphabetical name','ranking':candidates_data}
    SELECTION.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result),flush=True)


if __name__=='__main__':
    command=sys.argv[1]
    if command=='prepare':
        if PLAN.exists(): raise SystemExit('Plan exists')
        DEST.mkdir(parents=True,exist_ok=True); PLAN.write_text(json.dumps(plan_for(make_data()),indent=2,ensure_ascii=False)+'\n',encoding='utf8')
        print('Plan prepared; commit before inference')
    elif command=='run': run(*sys.argv[2:])
    elif command=='select': select()
    else: raise SystemExit('Unknown command')

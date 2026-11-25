"""Exploratory original/high NLI threshold consensus on the prior 25 papers.

Both Qwen decisions use the same neutral prompt and ±1 neighbors. The original
threshold requests were saved earlier; this script makes one new high-threshold
request per five-paper group and intersects their output IDs. This technique was
selected post hoc on development data; this 25-paper test pool is not pristine.
"""
import json
import sys
import time

import five_paper_context_pilot as context
import five_paper_fp8_comparison as base
import five_paper_pilot as p
from baseline_evidence import digest
from nli_variants import load_val

DEST=p.ROOT/'docs/experiments/five_paper_threshold_consensus_2026-11-25'
PLAN=DEST/'plan.json';SUMMARY=DEST/'summary.json';HIGH_TAU=.05


def groups():
    source=base.frozen();_,rows=load_val('test');scores=base.typed(base.TEST_FT).max(1)
    lookup={(r['pid'],r['idx']):i for i,r in enumerate(rows)};out=[]
    for g in source:
        high={k for k in g['keys'] if not rows[lookup[k]]['junk'] and scores[lookup[k]]>=HIGH_TAU}
        if not high<=g['fine']:raise ValueError('High-threshold candidates must be a subset of original')
        data=dict(g);data['fine']=high;out.append(data)
    return out


def plan_for(gs):
    return {'purpose':'Exploratory high-threshold + original-threshold output intersection on prior 25 papers',
      'selection':'post-hoc: intersection was noticed after inspecting both threshold development results',
      'input_caveat':'25-paper test pool already analyzed previously, not independent or blind',
      'model':base.MODEL,'model_revision':'017b9c7af6b5689d5dd426a76e0bc077eb5ca20a',
      'prompt_sha256':digest(base.prompts.system('neutral')),'high_threshold':HIGH_TAU,
      'original_plan_sha256':p.sha(base.PLAN),'runner_sha256':p.sha(p.ROOT/'scripts/bench/five_paper_threshold_consensus.py'),
      'decision':'retain only sentence IDs accepted in both the existing original-threshold Qwen run and new high-threshold run',
      'cost':'two Qwen requests per group, plus NLI; original calls already exist',
      'groups':[{'paper_ids':g['paper_ids'],'high_candidates':len(g['fine']),
                 'high_input_sha256':digest(base.input_for(g,'fine')[1]),
                 'original_result_sha256':p.sha(base.DEST/f'group_{i:02d}_fine.json')} for i,g in enumerate(gs,1)],
      'temperature':0,'seed':0,'thinking':False,'max_output_tokens':8192}


def frozen():
    gs=groups();context.check_tracked(PLAN);context.check_tracked(p.ROOT/'scripts/bench/five_paper_threshold_consensus.py')
    for i in range(1,6):context.check_tracked(base.DEST/f'group_{i:02d}_fine.json')
    if json.loads(PLAN.read_text(encoding='utf8'))!=plan_for(gs):raise SystemExit('Plan or input changed')
    return gs


def run(n):
    if n not in range(1,6):raise SystemExit('Invalid group')
    g=frozen()[n-1];valid,user=base.input_for(g,'fine');keys=g['keys'];path=DEST/f'group_{n:02d}_high.json'
    if path.exists():raise SystemExit('Preserve existing output')
    c=p.client()
    if base.MODEL not in {x.id for x in c.models.list().data}:raise SystemExit('Expected model unavailable')
    schema={'type':'object','additionalProperties':False,'required':['positives'],
            'properties':{'positives':{'type':'array','items':{'type':'integer'}}}}
    attempts=[];result=None
    for _ in range(2):
        start=time.time();item=None
        try:
            resp=c.chat.completions.create(model=base.MODEL,temperature=0,max_tokens=8192,
                messages=[{'role':'system','content':base.prompts.system('neutral')},{'role':'user','content':user}],
                response_format={'type':'json_schema','json_schema':{'name':'positives','schema':schema,'strict':True}},
                extra_body={'seed':0,'chat_template_kwargs':{'enable_thinking':False}})
            item={'seconds':round(time.time()-start,3),'finish_reason':resp.choices[0].finish_reason,
                  'usage':resp.usage.model_dump() if resp.usage else None,'response':resp.choices[0].message.content};attempts.append(item)
            if item['finish_reason']!='stop':continue
            obj=json.loads(item['response']);ids=obj.get('positives')
            if (set(obj)!={'positives'} or not isinstance(ids,list) or
                any(type(i) is not int or i<0 or i>=len(keys) or keys[i] not in valid for i in ids)):
                item['validation_error']='invalid or non-input ids';continue
            item['duplicate_id_count']=len(ids)-len(set(ids));kept={keys[i] for i in ids}
            result={'status':'ok','group':n,'plan_sha256':p.sha(PLAN),'input_sha256':digest(user),
                    'candidate_count':len(valid),'kept':[list(k) for k in sorted(kept)],'attempts':attempts,
                    'metrics':p.metrics(g['rows'],kept)};break
        except Exception as e:
            err={'error_type':type(e).__name__,'status_code':getattr(e,'status_code',None)}
            if item is None:attempts.append({'seconds':round(time.time()-start,3),**err})
            else:item['validation_error']=err
    if result is None:result={'status':'failed','group':n,'plan_sha256':p.sha(PLAN),'attempts':attempts}
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print(json.dumps({'group':n,'status':result['status'],'metrics':result.get('metrics'),
                      'usage':attempts[-1].get('usage')}),flush=True)


def analyze():
    gs=frozen();rows=[r for g in gs for r in g['rows']];original=set();high=set();intersection=set();group_results=[];it=ot=0
    for i,g in enumerate(gs,1):
        path=DEST/f'group_{i:02d}_high.json';context.check_tracked(path)
        h=json.loads(path.read_text(encoding='utf8'));o=json.loads((base.DEST/f'group_{i:02d}_fine.json').read_text(encoding='utf8'))
        if h['status']!='ok' or o['status']!='ok' or h['plan_sha256']!=p.sha(PLAN):raise SystemExit('Incomplete or mismatched run')
        A={tuple(k) for k in o['kept']};B={tuple(k) for k in h['kept']};original|=A;high|=B;intersection|=A&B
        group_results.append({'group':i,'original':p.metrics(g['rows'],A),'high':p.metrics(g['rows'],B),
                              'consensus':p.metrics(g['rows'],A&B)})
        for attempt in h['attempts']:
            u=attempt.get('usage') or {};it+=u.get('prompt_tokens',0);ot+=u.get('completion_tokens',0)
    summary={'plan_sha256':p.sha(PLAN),'original':p.metrics(rows,original),'high':p.metrics(rows,high),
             'consensus':p.metrics(rows,intersection),'groups':group_results,
             'additional_qwen_input_tokens':it,'additional_qwen_output_tokens':ot,
             'high_target_sentences':sum(len(g['fine']) for g in gs)}
    if SUMMARY.exists():raise SystemExit('Preserve summary')
    SUMMARY.write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8');print(json.dumps(summary),flush=True)


if __name__=='__main__':
    cmd=sys.argv[1]
    if cmd=='prepare':
        if PLAN.exists():raise SystemExit('Plan exists')
        gs=groups();DEST.mkdir(parents=True,exist_ok=True);PLAN.write_text(json.dumps(plan_for(gs),indent=2)+'\n',encoding='utf8')
        print('Prepared consensus plan; commit before inference')
    elif cmd=='run':run(int(sys.argv[2]))
    elif cmd=='analyze':analyze()
    else:raise SystemExit('Unknown command')

"""Matched test-split check of context layouts vs saved full-paper baseline."""
import json, os, sys, urllib.request
import five_paper_context_format_34 as dev
import five_paper_68_verdicts as base
import five_paper_pilot as p
import five_paper_dev_tuning as d
from baseline_evidence import digest

DEST=p.ROOT/'docs/experiments/five_paper_context_test_34_2026-11-26'
PLAN=DEST/'plan.json'; PRE=DEST/'preflight.json'; NAMES=dev.NAMES
RUNNER=p.ROOT/'scripts/bench/five_paper_context_test_34.py'

def plan():
    groups=base.frozen()[8:]
    return {'purpose':'Compare existing full-paper baseline and existing fine-NLI baseline with surrounding and no-context layouts on the same 34 test papers.',
      'limitations':'Previously studied test papers; prompt, format and layout exploration has used this dataset; labels are LLM-generated and not fully human-validated. Exploratory, not an independent test.',
      'baseline':'Reuses previous matched full and fine-NLI responses; no new full-paper calls. Same groups, IDs, model and per-candidate verdict prompt.',
      'layouts':'Fine-NLI surrounding: Previous, numbered target, Next; no-context: numbered target only. Current fine baseline: numbered target, followed by previous and next in Context line.',
      'prompt':base.prompt.prompt('verdicts'),'model':base.MODEL,'max_output_tokens':16384,'temperature':0,'seed':0,'thinking':False,
      'format_validation':'Every actual candidate receives exactly one boolean decision. Extra IDs are excluded and recorded; missing or duplicate input IDs fail.',
      'previous_plan_sha256':p.sha(base.PLAN),'dev_layout_runner_sha256':p.sha(p.ROOT/'scripts/bench/five_paper_context_format_34.py'),
      'runner_sha256':p.sha(RUNNER),
      'baseline_sha256':{str(i):{m:p.sha(base.DEST/f'group{i:02d}_{m}.json') for m in ('full','fine')} for i in range(9,17)},
      'groups':[{'original_group':i+9,'paper_ids':g['paper_ids'],'domain':g['domain'],'rows_sha256':digest(g['rows']),
        'candidate_sha256':digest(sorted(g['fine'])),'candidates':len(g['fine']),
        'input_sha256':{n:digest(dev.render(g,n)) for n in NAMES}} for i,g in enumerate(groups)]}

def frozen():
    d.check_committed(PLAN);d.check_committed(RUNNER)
    if json.loads(PLAN.read_text(encoding='utf8'))!=plan():raise ValueError('Plan changed')
    return base.load()[0][8:]

def configure():
    dev.DEST=DEST;dev.PLAN=PLAN;dev.PRE=PRE

def preflight():
    groups=frozen(); entries=[]; key=os.environ['QWEN_API_KEY'];url=os.environ['QWEN_BASE_URL'].removesuffix('/v1')+'/tokenize'
    for i,g in enumerate(groups,1):
        for n in NAMES:
            payload=json.dumps({'model':base.MODEL,'messages':[{'role':'system','content':base.prompt.prompt('verdicts')},{'role':'user','content':dev.render(g,n)}],'add_generation_prompt':True},ensure_ascii=False).encode()
            req=urllib.request.Request(url,data=payload,headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=120) as r:o=json.load(r)
            item={'group':i,'layout':n,'tokens':o['count'],'fits':o['count']+16384+512<=o.get('max_model_len',262144)};entries.append(item);print(json.dumps(item),flush=True)
    if PRE.exists():raise ValueError('Preflight exists')
    PRE.write_text(json.dumps({'plan_sha256':p.sha(PLAN),'status':'ok' if all(x['fits'] for x in entries) else 'too_large','entries':entries},indent=2)+'\n',encoding='utf8')
    if not all(x['fits'] for x in entries):raise SystemExit('Context too large')

def run_all():
    groups=frozen();d.check_committed(PRE);pre=json.loads(PRE.read_text(encoding='utf8'))
    if pre['status']!='ok' or pre['plan_sha256']!=p.sha(PLAN):raise ValueError('Bad preflight')
    configure();c=p.client()
    if base.MODEL not in {x.id for x in c.models.list().data}:raise ValueError('Model missing')
    for i in range(1,9):
        for n in NAMES:
            path=DEST/f'group{i:02d}_{n}.json'
            if path.exists():
                r=json.loads(path.read_text(encoding='utf8'))
                if r['status']=='ok' and r['plan_sha256']==p.sha(PLAN):continue
                raise ValueError('Existing failed result; investigate')
            dev.run(groups,i,n,c)

def summary():
    groups=frozen();result={}
    for domain in ('all','arxiv_cs','biomed_sal'):
        result[domain]={}
        for name in ('full','baseline')+NAMES:
            rows=[];keep=set();ti=to=0;sentences=0;extras=[]
            for i,g in enumerate(groups,1):
                if domain!='all' and g['domain']!=domain:continue
                path=(base.DEST/f'group{i+8:02d}_{"full" if name=="full" else "fine"}.json') if name in ('full','baseline') else DEST/f'group{i:02d}_{name}.json'
                r=json.loads(path.read_text(encoding='utf8'));expected=p.sha(base.PLAN) if name in ('full','baseline') else p.sha(PLAN)
                if r['status']!='ok' or r['plan_sha256']!=expected:raise ValueError('Failed output or plan mismatch')
                rows+=g['rows'];keep.update(tuple(k) for k in r['kept']);sentences+=len(g['keys']) if name=='full' else len(g['fine'])
                for a in r['attempts']:
                    u=a.get('usage') or {};ti+=u.get('prompt_tokens',0);to+=u.get('completion_tokens',0)
                    if a.get('excluded_extra_entries'):extras.append({'group':i,'ids':a['excluded_extra_entries']})
            result[domain][name]={'metrics':p.metrics(rows,keep),'sentences_passed':sentences,'input_tokens':ti,'output_tokens':to,'extras':extras}
    out={'plan_sha256':p.sha(PLAN),'results':result};print(json.dumps(out,indent=2),flush=True)
    if (DEST/'summary.json').exists():raise ValueError('Summary exists')
    (DEST/'summary.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')

if __name__=='__main__':
    if sys.argv[1]=='prepare':
        if PLAN.exists():raise ValueError('Plan exists')
        DEST.mkdir(parents=True,exist_ok=True);PLAN.write_text(json.dumps(plan(),indent=2,ensure_ascii=False)+'\n',encoding='utf8');print('Prepared')
    elif sys.argv[1]=='preflight':preflight()
    elif sys.argv[1]=='run':run_all()
    elif sys.argv[1]=='summary':summary()

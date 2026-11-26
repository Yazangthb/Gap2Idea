"""Compare current, surrounding, and no-neighbor layouts on 34 dev papers."""
import json, os, sys, time, urllib.request
import five_paper_68_verdicts as base
import five_paper_pilot as p
import five_paper_dev_tuning as d
from baseline_evidence import digest

DEST=p.ROOT/'docs/experiments/five_paper_context_format_34_2026-11-26'
PLAN=DEST/'plan.json'; PRE=DEST/'preflight.json'; NAMES=('surrounding','no_context')

def render(g,name):
    lines=[]; last=None; rows=g['rows'];keys=g['keys']
    for i,r in enumerate(rows):
        if keys[i] not in g['fine']:continue
        if r['pid']!=last:last=r['pid'];lines.append('PAPER '+last)
        if name=='surrounding':
            prev=rows[i-1]['sentence'][-800:] if i and rows[i-1]['pid']==r['pid'] else ''
            nxt=rows[i+1]['sentence'][:800] if i+1<len(rows) and rows[i+1]['pid']==r['pid'] else ''
            lines.append(f'Previous: {prev}\n[{i}] {r["sentence"]}\nNext: {nxt}')
        elif name=='no_context': lines.append(f'[{i}] {r["sentence"]}')
        else:raise ValueError('Unknown layout')
    return '\n'.join(lines)

def plan():
    groups=base.frozen()[:8]
    return {'purpose':'Compare input layout and no-neighbor context using existing fine-tuned NLI candidates and per-candidate verdicts on 34 dev papers.',
      'limitations':'All papers previously studied in development; labels are LLM-generated; format and context quantity are intentionally manipulated, not model or candidate selection.',
      'baseline':'Saved 68-paper fine-tuned NLI verdicts, same group IDs 1-8. Target followed by Context: Previous ... Next ...',
      'surrounding':'Previous: previous original sentence; numbered target on its own next line; Next: next original sentence. Neighbor clipping unchanged at 800 characters.',
      'no_context':'Only numbered targets, no neighbors.',
      'prompt':base.prompt.prompt('verdicts'),'model':base.MODEL,'max_output_tokens':16384,'temperature':0,'seed':0,'thinking':False,
      'validation':'Every candidate ID must have exactly one boolean decision. Extra IDs are recorded and discarded only if every candidate has exactly one decision; otherwise fail.',
      'original_plan_sha256':p.sha(base.PLAN),'runner_sha256':p.sha(p.ROOT/'scripts/bench/five_paper_context_format_34.py'),
      'baseline_sha256':{str(i):p.sha(base.DEST/f'group{i:02d}_fine.json') for i in range(1,9)},
      'groups':[{'split':g['split'],'domain':g['domain'],'paper_ids':g['paper_ids'],'rows_sha256':digest(g['rows']),
       'candidate_sha256':digest(sorted(g['fine'])),'candidate_count':len(g['fine']),
       'input_sha256':{n:digest(render(g,n)) for n in NAMES}} for g in groups]}

def frozen():
    d.check_committed(PLAN);d.check_committed(p.ROOT/'scripts/bench/five_paper_context_format_34.py')
    if json.loads(PLAN.read_text(encoding='utf8'))!=plan():raise ValueError('Plan changed')
    return base.load()[0][:8]

def preflight():
    groups=frozen(); entries=[];key=os.environ['QWEN_API_KEY'];url=os.environ['QWEN_BASE_URL'].removesuffix('/v1')+'/tokenize'
    for i,g in enumerate(groups,1):
        for n in NAMES:
            payload=json.dumps({'model':base.MODEL,'messages':[{'role':'system','content':base.prompt.prompt('verdicts')},{'role':'user','content':render(g,n)}],'add_generation_prompt':True},ensure_ascii=False).encode()
            req=urllib.request.Request(url,data=payload,headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=120) as r:o=json.load(r)
            entry={'group':i,'layout':n,'tokens':o['count'],'fits':o['count']+16384+512<=o.get('max_model_len',262144)};entries.append(entry);print(json.dumps(entry),flush=True)
    if PRE.exists():raise ValueError('Preflight exists')
    PRE.write_text(json.dumps({'plan_sha256':p.sha(PLAN),'entries':entries,'status':'ok' if all(x['fits'] for x in entries) else 'too_large'},indent=2)+'\n',encoding='utf8')
    if not all(x['fits'] for x in entries):raise SystemExit('Context too large')

def run(groups,i,n,c):
    path=DEST/f'group{i:02d}_{n}.json'
    if path.exists():raise ValueError('Existing output preserved')
    g=groups[i-1]; keys=g['keys']; valid=g['fine']; input_ids={j for j,k in enumerate(keys) if k in valid};user=render(g,n)
    schema={'type':'object','additionalProperties':False,'required':['results'],'properties':{'results':{'type':'array','items':{'type':'object','additionalProperties':False,'required':['id','keep'],'properties':{'id':{'type':'integer'},'keep':{'type':'boolean'}}}}}}
    attempts=[];out=None
    for _ in range(2):
        start=time.time();a=None
        try:
            r=c.chat.completions.create(model=base.MODEL,temperature=0,max_tokens=16384,messages=[{'role':'system','content':base.prompt.prompt('verdicts')},{'role':'user','content':user}],response_format={'type':'json_schema','json_schema':{'name':'verdicts','schema':schema,'strict':True}},extra_body={'seed':0,'chat_template_kwargs':{'enable_thinking':False}})
            a={'seconds':round(time.time()-start,2),'finish_reason':r.choices[0].finish_reason,'usage':r.usage.model_dump() if r.usage else None,'response':r.choices[0].message.content};attempts.append(a)
            if a['finish_reason']!='stop':raise ValueError('Truncated')
            obj=json.loads(a['response']);ent=obj['results']
            if set(obj)!={'results'} or not isinstance(ent,list):raise ValueError('Bad schema')
            if any(not isinstance(x,dict) or set(x)!={'id','keep'} or type(x['id'])!=int or type(x['keep'])!=bool for x in ent):raise ValueError('Bad item')
            good=[x for x in ent if x['id'] in input_ids];extra=[x for x in ent if x['id'] not in input_ids]
            ids=[x['id'] for x in good]
            if set(ids)!=input_ids or len(ids)!=len(set(ids)):raise ValueError('Missing or duplicate input ID')
            keep={keys[x['id']] for x in good if x['keep']};a['excluded_extra_entries']=extra
            out={'status':'ok','group':i,'layout':n,'plan_sha256':p.sha(PLAN),'input_sha256':digest(user),
             'kept':[list(x) for x in sorted(keep)],'metrics':p.metrics(g['rows'],keep),'attempts':attempts};break
        except Exception as err:
            if a is None:attempts.append({'seconds':round(time.time()-start,2),'error_type':type(err).__name__})
            else:a['error_type']=type(err).__name__;a['error']=str(err)[:200]
    if out is None:out={'status':'failed','group':i,'layout':n,'plan_sha256':p.sha(PLAN),'attempts':attempts}
    path.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print(json.dumps({'group':i,'layout':n,'status':out['status'],'metrics':out.get('metrics'),'usage':attempts[-1].get('usage')}),flush=True)
    if out['status']!='ok':raise SystemExit('Failed; output saved')

def run_all():
    groups=frozen();d.check_committed(PRE);pre=json.loads(PRE.read_text(encoding='utf8'))
    if pre['status']!='ok' or pre['plan_sha256']!=p.sha(PLAN):raise ValueError('Bad preflight')
    c=p.client()
    if base.MODEL not in {x.id for x in c.models.list().data}:raise ValueError('Model missing')
    for i in range(1,9):
        for name in NAMES:run(groups,i,name,c)

def summary():
    groups=frozen();result={}
    for domain in ('all','arxiv_cs','biomed_sal'):
        result[domain]={}
        for name in ('baseline',)+NAMES:
            rows=[];keep=set();ti=to=0;sentences=0;extras=[]
            for i,g in enumerate(groups,1):
                if domain!='all' and g['domain']!=domain:continue
                path=base.DEST/f'group{i:02d}_fine.json' if name=='baseline' else DEST/f'group{i:02d}_{name}.json'
                r=json.loads(path.read_text(encoding='utf8')); expected=p.sha(base.PLAN) if name=='baseline' else p.sha(PLAN)
                if r['status']!='ok' or r['plan_sha256']!=expected:raise ValueError('Failed output or plan mismatch')
                rows+=g['rows'];keep.update(tuple(k) for k in r['kept']);sentences+=len(g['fine'])
                for a in r['attempts']:
                    u=a.get('usage') or {};ti+=u.get('prompt_tokens',0);to+=u.get('completion_tokens',0)
                    if a.get('excluded_extra_entries'):extras.append({'group':i,'ids':a['excluded_extra_entries']})
            result[domain][name]={'metrics':p.metrics(rows,keep),'candidate_count':sentences,'input_tokens':ti,'output_tokens':to,'extras':extras}
    out={'plan_sha256':p.sha(PLAN),'results':result}; print(json.dumps(out,indent=2),flush=True)
    target=DEST/'summary.json'
    if target.exists():raise ValueError('Summary exists')
    target.write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')

if __name__=='__main__':
    if sys.argv[1]=='prepare':
        if PLAN.exists():raise ValueError('Plan exists')
        DEST.mkdir(parents=True,exist_ok=True);PLAN.write_text(json.dumps(plan(),indent=2,ensure_ascii=False)+'\n',encoding='utf8');print('Prepared')
    elif sys.argv[1]=='preflight':preflight()
    elif sys.argv[1]=='run':run_all()
    elif sys.argv[1]=='summary':summary()

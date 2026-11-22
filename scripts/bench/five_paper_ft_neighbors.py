"""One-call Qwen 27B review of fine-tuned NLI candidates with ±1 neighbors."""
import json
import math
import sys
import time

import numpy as np
import five_paper_context_pilot as q
import five_paper_context_v100 as v
import five_paper_dev_tuning as d
import five_paper_pilot as p
from baseline_evidence import digest
from nli_variants import load_val
from stage_c_nli_qwen import TYPES,evc,candidates

DEST=p.ROOT/'docs/experiments/five_paper_ft_neighbors_2026-11-22'
PLAN=DEST/'plan.json'; RESULT=DEST/'27b_v100_ft_neighbors.json'
SCORES=p.ROOT/'data/gold/nli_ft/nli_ft_gold_val_scores.npz'


def build():
    selected=d.make_data()['check'];rows=selected['rows'];keys=selected['keys']
    _,val=load_val('val');z=np.load(SCORES);typed=np.stack([evc(z,h) for h in TYPES],1);sc=typed.max(1)
    positives=np.array([r['label']!='none' for r in val]);eligible=np.array([not r['junk'] for r in val])
    ranked=np.sort(sc[positives&eligible])[::-1];tau=float(ranked[math.ceil(.9*positives.sum())-1])
    pool=candidates(val,typed,tau); wanted={(r['id'],r['idx']) for r in pool if r['id'] in selected['paper_ids']}
    if not wanted<=set(keys):raise ValueError('Unmatched candidates')
    lines=[];last=None
    for i,r in enumerate(rows):
        if keys[i] not in wanted:continue
        if r['pid']!=last:last=r['pid'];lines.append('PAPER '+last)
        before=rows[i-1]['sentence'][-q.CAP:] if i and rows[i-1]['pid']==r['pid'] else ''
        after=rows[i+1]['sentence'][:q.CAP] if i+1<len(rows) and rows[i+1]['pid']==r['pid'] else ''
        lines.append(f'[{i}] {r["sentence"]}\nContext: Previous: {before} Next: {after}')
    return selected,wanted,tau,'\n'.join(lines)


def plan_for(selected,wanted,tau,user):
    return {'purpose':'Match the existing single-request five-paper V100 Q4 neighbor experiment, replacing only zero-shot NLI candidates with fine-tuned candidates',
      'model':v.MODEL,'ollama_digest':v.DIGEST,'prompt':'neutral','prompt_sha256':digest(d.system('neutral')),
      'candidate_rule':'Stage A plus max typed fine-tuned NLI entailment/(entailment+contradiction), threshold chosen for 90% recall on all 34 validation papers',
      'threshold':tau,'candidate_count':len(wanted),'keys_sha256':digest(sorted(wanted)),
      'input_sha256':digest(user),'sample_papers':selected['paper_ids'],
      'source_score_sha256':p.sha(SCORES),'source_dev_plan_sha256':p.sha(d.PLAN),
      'comparison_result_sha256':p.sha(q.DEST/'27b_v100_q4_neighbors.json'),
      'runner_sha256':p.sha(p.ROOT/'scripts/bench/five_paper_ft_neighbors.py'),
      'neighbors':'previous original sentence last 800 characters; next original sentence first 800 characters, within the same paper',
      'temperature':0,'seed':0,'thinking':False,'max_output_tokens':8192,
      'caveat':'Sample belongs to the validation set used for fine-tuned checkpoint/threshold selection; exploratory LLM labels'}


def main(cmd):
    selected,wanted,tau,user=build()
    if cmd=='prepare':
        if PLAN.exists():raise SystemExit('Plan exists')
        DEST.mkdir(parents=True,exist_ok=True)
        PLAN.write_text(json.dumps(plan_for(selected,wanted,tau,user),indent=2)+'\n',encoding='utf8')
        print('Prepared FT plan; commit before inference. Candidates:',len(wanted),'threshold:',tau);return
    if cmd!='run':raise SystemExit('Unknown command')
    q.frozen();q.check_tracked(v.CONFIG);q.check_tracked(q.DEST/'27b_v100_q4_neighbors.json')
    q.check_tracked(PLAN);q.check_tracked(p.ROOT/'scripts/bench/five_paper_ft_neighbors.py')
    if json.loads(PLAN.read_text(encoding='utf8'))!=plan_for(selected,wanted,tau,user):raise SystemExit('Plan changed')
    if RESULT.exists():raise SystemExit('Preserve existing evidence')
    c=p.client()
    if v.MODEL not in {x.id for x in c.models.list().data}:raise SystemExit('V100 model unavailable')
    schema={'type':'object','additionalProperties':False,'required':['positives'],
            'properties':{'positives':{'type':'array','items':{'type':'integer'}}}}
    attempts=[];result=None;system=d.system('neutral');keys=selected['keys']
    for _ in range(2):
        start=time.time();item=None
        try:
            resp=c.chat.completions.create(model=v.MODEL,temperature=0,max_tokens=8192,
                messages=[{'role':'system','content':system},{'role':'user','content':user}],
                response_format={'type':'json_schema','json_schema':{'name':'positives','schema':schema,'strict':True}},
                extra_body={'seed':0,'chat_template_kwargs':{'enable_thinking':False}})
            item={'seconds':round(time.time()-start,3),'finish_reason':resp.choices[0].finish_reason,
                  'usage':resp.usage.model_dump() if resp.usage else None,'response':resp.choices[0].message.content};attempts.append(item)
            if item['finish_reason']!='stop':continue
            obj=json.loads(item['response']);ids=obj.get('positives')
            if (set(obj)!={'positives'} or not isinstance(ids,list) or
                any(type(i) is not int or i<0 or i>=len(keys) or keys[i] not in wanted for i in ids)):
                item['validation_error']='invalid or non-input ids';continue
            item['duplicate_id_count']=len(ids)-len(set(ids));kept={keys[i] for i in ids}
            result={'status':'ok','model':v.MODEL,'plan_sha256':p.sha(PLAN),'candidate_count':len(wanted),
                    'kept':[list(k) for k in sorted(kept)],'attempts':attempts,'metrics':p.metrics(selected['rows'],kept)};break
        except Exception as e:
            err={'error_type':type(e).__name__,'status_code':getattr(e,'status_code',None)}
            if item is None:attempts.append({'seconds':round(time.time()-start,3),**err})
            else:item['validation_error']=err
    if result is None:result={'status':'failed','plan_sha256':p.sha(PLAN),'attempts':attempts}
    RESULT.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print(json.dumps({'status':result['status'],'metrics':result.get('metrics'),'candidates':len(wanted),
                      'usage':attempts[-1].get('usage')}),flush=True)
    if result['status']!='ok':raise SystemExit('Failed; evidence preserved')


if __name__=='__main__':main(sys.argv[1])

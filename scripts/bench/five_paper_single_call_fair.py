"""Matched one-request comparison: five full papers vs A+B shortlist.

Both use the same neutral instruction, model, ordering, original global sentence
IDs, format, decoding and output limit. Only the included sentences differ.
Run `full` and `funnel` once each; never hide invalid or incomplete outputs.
"""
import json
import subprocess
import sys
import time

import five_paper_pilot as p
from baseline_evidence import digest


def build():
    rows,_,chosen=p.data()
    candidate_keys={(r['id'],r['idx']) for r in chosen}
    keys=[(r['pid'],r['idx']) for r in rows]
    ids={k:i for i,k in enumerate(keys)}
    if len(keys)!=len(set(keys)) or len(candidate_keys)!=len(chosen) or not candidate_keys<=set(keys):
        raise SystemExit('Duplicate or unmatched sentence keys')
    prompt=p.baseline.build_prompt('gold','neutral')
    old=('The input is consecutive sentences from scientific papers, in reading order, one per line as "[id] sentence". '
         'Judge every sentence on its own, using the neighbouring sentences only to disambiguate.')
    new=('The input is sentences from scientific papers, in reading order, one per line as "[id] sentence". '
         'Judge every sentence on its own.')
    if old not in prompt: raise SystemExit('Unexpected base prompt')
    system=prompt.replace(old,new)+('\n\nThere are five separate papers below. Lines starting "PAPER " are boundaries, '
                                    'not sentences. Judge each numbered sentence only in its own paper.')
    def text_for(subset):
        lines=[]; last=None
        for r in rows:
            key=(r['pid'],r['idx'])
            if key not in subset: continue
            if r['pid']!=last: last=r['pid']; lines.append('PAPER '+last)
            lines.append(f'[{ids[key]}] {r["sentence"]}')
        return '\n'.join(lines)
    return rows,keys,system,{'full':(set(keys),text_for(set(keys))),
                             'funnel':(candidate_keys,text_for(candidate_keys))}


def main(name):
    if name not in ('full','funnel'): raise SystemExit('Choose full or funnel')
    p.require_frozen(); rel='scripts/bench/five_paper_single_call_fair.py'
    if (subprocess.run(['git','status','--porcelain','--',rel],cwd=p.ROOT,capture_output=True,text=True).stdout.strip()
        or subprocess.run(['git','cat-file','-e',f'HEAD:{rel}'],cwd=p.ROOT,capture_output=True).returncode):
        raise SystemExit('Commit the exact comparison before inference')
    rows,keys,system,inputs=build(); valid,user=inputs[name]
    path=p.DEST/f'27b_fair_single_call_{name}.json'
    if path.exists(): raise SystemExit('Preserve existing evidence')
    c=p.client(); model='qwen3.8-27b'
    if model not in {x.id for x in c.models.list().data}: raise SystemExit('Expected model unavailable')
    schema={'type':'object','additionalProperties':False,'required':['positives'],
            'properties':{'positives':{'type':'array','items':{'type':'integer'}}}}
    attempts=[]; result=None
    for _ in range(2):
        t=time.time(); item=None
        try:
            resp=c.chat.completions.create(model=model,temperature=0,max_tokens=p.MAX_OUTPUT,
                messages=[{'role':'system','content':system},{'role':'user','content':user}],
                response_format={'type':'json_schema','json_schema':{'name':'positives','schema':schema,'strict':True}},
                extra_body={'seed':0,'chat_template_kwargs':{'enable_thinking':False}})
            item={'seconds':round(time.time()-t,3),'finish_reason':resp.choices[0].finish_reason,
                  'usage':resp.usage.model_dump() if resp.usage else None,'response':resp.choices[0].message.content}
            attempts.append(item)
            if item['finish_reason']!='stop': continue
            parsed=json.loads(item['response']); ids=parsed.get('positives')
            if (set(parsed)!={'positives'} or not isinstance(ids,list) or len(set(ids))!=len(ids) or
                any(type(i) is not int or i<0 or i>=len(keys) or keys[i] not in valid for i in ids)):
                item['validation_error']='invalid, duplicate, or non-input IDs'; continue
            kept={keys[i] for i in ids}
            result={'status':'ok','method':name,'model':model,'plan_sha256':p.sha(p.PLAN),
                    'system_prompt':system,'system_sha256':digest(system),'user_sha256':digest(user),
                    'input_sentence_count':len(valid),'attempts':attempts,
                    'kept':[list(k) for k in sorted(kept)],'metrics':p.metrics(rows,kept)}
            break
        except Exception as e:
            error={'error_type':type(e).__name__,'status_code':getattr(e,'status_code',None)}
            if item is None: attempts.append({'seconds':round(time.time()-t,3),**error})
            else: item['validation_error']=error
    if result is None:
        result={'status':'failed','method':name,'model':model,'plan_sha256':p.sha(p.PLAN),
                'system_prompt':system,'system_sha256':digest(system),'user_sha256':digest(user),
                'input_sentence_count':len(valid),'attempts':attempts}
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'method':name,'status':result['status'],'metrics':result.get('metrics'),
                      'calls':len(attempts),'usage':attempts[-1].get('usage')}),flush=True)
    if result['status']!='ok': raise SystemExit('Comparison failed; evidence preserved')


if __name__=='__main__': main(sys.argv[1])

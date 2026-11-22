"""Exploratory 4B direct fallback: one complete paper per call, no funnel."""
import json
import time

import five_paper_pilot as p
from baseline_evidence import digest


def main():
    p.require_frozen()
    rows, _, _ = p.data()
    path = p.DEST / '4b_one_paper_direct.jsonl'
    result_path = p.DEST / '4b_one_paper_direct_summary.json'
    if path.exists() or result_path.exists():
        raise SystemExit('Existing evidence must be preserved')
    c = p.client()
    model = 'qwen3.5-4b-bf16:latest'
    schema = {'type': 'object', 'additionalProperties': False, 'required': ['positives'],
              'properties': {'positives': {'type': 'array', 'items': {'type': 'integer'}}}}
    kept = set(); failed = []
    with path.open('x', encoding='utf8') as f:
        for pid in p.IDS:
            subset = [r for r in rows if r['pid'] == pid]
            system, user, keys = p.group_prompt(subset)
            system = system.replace('There are five separate papers below.', 'There is one paper below.')
            rec = {'paper_id': pid, 'sentence_count': len(subset), 'user_sha256': digest(user)}
            t = time.time()
            try:
                resp = c.chat.completions.create(model=model, temperature=0, max_tokens=p.MAX_OUTPUT,
                    messages=[{'role':'system','content':system},{'role':'user','content':user}],
                    response_format={'type':'json_schema','json_schema':{'name':'positives','schema':schema,'strict':True}},
                    extra_body={'seed':0,'chat_template_kwargs':{'enable_thinking':False}})
                rec.update({'seconds': round(time.time()-t, 3), 'finish_reason':resp.choices[0].finish_reason,
                            'usage':resp.usage.model_dump() if resp.usage else None,
                            'response':resp.choices[0].message.content})
                if rec['finish_reason'] != 'stop': raise ValueError('output limit')
                parsed=json.loads(rec['response']); ids=parsed.get('positives')
                if (set(parsed)!={'positives'} or not isinstance(ids,list) or
                    any(type(i) is not int or i<0 or i>=len(keys) for i in ids) or len(set(ids))!=len(ids)):
                    raise ValueError('invalid, duplicate or out-of-range IDs')
                kept.update(tuple(keys[i]) for i in ids)
                rec['status']='ok'
            except Exception as e:
                failed.append(pid); rec['status']='failed'; rec['error_type']=type(e).__name__
            f.write(json.dumps(rec,ensure_ascii=False)+'\n'); f.flush()
            print(json.dumps({'paper_id':pid,'status':rec['status'],'seconds':rec.get('seconds')}),flush=True)
    summary = {'status':'failed' if failed else 'ok','failed_papers':failed,'model':model,
               'plan_sha256':p.sha(p.PLAN),'calls':len(p.IDS),'kept':[list(x) for x in sorted(kept)]}
    if not failed: summary['metrics']=p.metrics(rows,kept)
    result_path.write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'status':summary['status'],'metrics':summary.get('metrics')}),flush=True)


if __name__ == '__main__':
    main()

"""Continue frozen 68-paper study, recording post-hoc recovery of extra IDs.

Original output-validation stopped at group 2, full: both retries returned
all 1729 required IDs once plus one invented, false ID (1729). The original
responses remain unchanged. This wrapper accepts a failed response only if
every input ID is present exactly once and every entry has a valid schema;
out-of-input IDs are excluded and recorded. Missing, repeated, conflicting or
malformed input verdicts still fail. This recovery rule was adopted after
observing the first failure and must be disclosed alongside results.
"""
import json
from collections import Counter
import five_paper_68_verdicts as e
import five_paper_pilot as p
import five_paper_dev_tuning as d
from baseline_evidence import digest

RUNNER=p.ROOT/'scripts/bench/five_paper_68_recover.py'

def recover(groups,i,method):
    original=e.DEST/f'group{i:02d}_{method}.json'; target=e.DEST/f'group{i:02d}_{method}_recovered.json'
    r=json.loads(original.read_text(encoding='utf8'))
    if r.get('status')!='failed' or r.get('plan_sha256')!=p.sha(e.PLAN): raise ValueError('Not a matching failed run')
    g=groups[i-1]; valid,user=e.prior.input_for(g,method); keys=g['keys']; input_ids={j for j,k in enumerate(keys) if k in valid}
    for attempt_no,a in enumerate(r['attempts'],1):
        try:
            if a.get('finish_reason')!='stop': continue
            obj=json.loads(a['response']); entries=obj['results']
            if set(obj)!={'results'} or not isinstance(entries,list): continue
            if any(not isinstance(x,dict) or set(x)!={'id','keep'} or type(x['id'])!=int or type(x['keep'])!=bool for x in entries): continue
            good=[x for x in entries if x['id'] in input_ids]; extra=[x for x in entries if x['id'] not in input_ids]
            counts=Counter(x['id'] for x in good)
            if set(counts)!=input_ids or any(v!=1 for v in counts.values()) or not extra: continue
            kept={keys[x['id']] for x in good if x['keep']}
            out={'status':'ok_recovered','group':i,'method':method,'plan_sha256':p.sha(e.PLAN),'input_sha256':digest(user),
              'original_output_sha256':p.sha(original),'attempt_used':attempt_no,'excluded_extra_entries':extra,
              'kept':[list(k) for k in sorted(kept)],'metrics':p.metrics(g['rows'],kept)}
            if target.exists():
                if json.loads(target.read_text(encoding='utf8'))!=out: raise ValueError('Existing recovery differs')
            else: target.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
            print(json.dumps({'group':i,'method':method,'status':'ok_recovered','excluded':extra,'metrics':out['metrics']}),flush=True)
            return
        except (KeyError,TypeError,json.JSONDecodeError): continue
    raise SystemExit('Cannot safely recover: missing/duplicate/invalid input verdicts or incomplete response')

def main():
    d.check_committed(RUNNER); groups=e.frozen(); d.check_committed(e.PRE)
    if json.loads(e.PRE.read_text(encoding='utf8'))['plan_sha256']!=p.sha(e.PLAN): raise ValueError('Preflight mismatch')
    c=p.client()
    if e.MODEL not in {x.id for x in c.models.list().data}: raise ValueError('Model unavailable')
    for i in range(1,len(groups)+1):
        for method in e.METHODS:
            original=e.DEST/f'group{i:02d}_{method}.json'
            if original.exists() and json.loads(original.read_text(encoding='utf8')).get('status')=='failed': recover(groups,i,method); continue
            try: e.run(groups,i,method,c)
            except SystemExit:
                if original.exists(): recover(groups,i,method)
                else: raise

if __name__=='__main__':main()

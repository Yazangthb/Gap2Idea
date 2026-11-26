"""Summarize completed 68-paper outputs, including documented recovery."""
import json
import five_paper_68_verdicts as e
import five_paper_pilot as p

def compute():
    groups=e.frozen(); totals={}
    strata={'all':lambda g:True,'val':lambda g:g['split']=='val','test':lambda g:g['split']=='test',
      'val_arxiv':lambda g:g['split']=='val' and g['domain']=='arxiv_cs',
      'val_biomed':lambda g:g['split']=='val' and g['domain']=='biomed_sal',
      'test_arxiv':lambda g:g['split']=='test' and g['domain']=='arxiv_cs',
      'test_biomed':lambda g:g['split']=='test' and g['domain']=='biomed_sal'}
    for label,select in strata.items():
        totals[label]={}
        for method in e.METHODS:
            rows=[]; kept=set(); tokens_in=tokens_out=0; candidates=0; incomplete=[]; recovered=[]; paper_count=0
            for i,g in enumerate(groups,1):
                if not select(g):continue
                original=e.DEST/f'group{i:02d}_{method}.json'
                if not original.exists():incomplete.append(i);continue
                r=json.loads(original.read_text(encoding='utf8'))
                if r.get('plan_sha256')!=p.sha(e.PLAN): raise ValueError('Plan mismatch')
                for attempt in r['attempts']:
                    usage=attempt.get('usage') or {};tokens_in+=usage.get('prompt_tokens',0);tokens_out+=usage.get('completion_tokens',0)
                if r['status']=='failed':
                    recovery=e.DEST/f'group{i:02d}_{method}_recovered.json'
                    if not recovery.exists():incomplete.append(i);continue
                    r=json.loads(recovery.read_text(encoding='utf8'))
                    if r['original_output_sha256']!=p.sha(original) or r['plan_sha256']!=p.sha(e.PLAN): raise ValueError('Recovery mismatch')
                    recovered.append({'group':i,'excluded':r['excluded_extra_entries']})
                elif r['status']!='ok':raise ValueError('Bad status')
                rows.extend(g['rows']);kept.update(tuple(k) for k in r['kept']);paper_count+=len(g['paper_ids'])
                candidates+=len(e.prior.input_for(g,method)[0])
            totals[label][method]={'complete':not incomplete,'missing_groups':incomplete,'papers_completed':paper_count,
             'sentences_passed':candidates,'all_attempt_input_tokens':tokens_in,'all_attempt_output_tokens':tokens_out,
             'recovered':recovered,'metrics':p.metrics(rows,kept) if rows else None}
    return {'plan_sha256':p.sha(e.PLAN),'strata':totals}

if __name__=='__main__':
    summary=compute(); print(json.dumps(summary,indent=2))
    if all(x['complete'] for x in summary['strata']['all'].values()):
        target=e.DEST/'summary.json'
        if target.exists():raise ValueError('Summary exists')
        target.write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')

"""Build a self-contained-data interactive fragment from saved NLI test scores."""
import argparse
import json
from pathlib import Path

import numpy as np
import plot_nli_thresholds as static

ROOT=Path(__file__).resolve().parents[2]
TEMPLATE=ROOT/'docs/experiments/nli_threshold_curves/interactive-template.html'


def build(out):
    y_val,ok_val=static.arrays('val'); y,ok=static.arrays('test')
    if len(y)!=11741 or int(y.sum())!=664:raise ValueError('Unexpected test dataset')
    models={}; configs={}
    for key,name in [('zero','zero-shot'),('fine','fine-tuned')]:
        s=static.scores(static.FILES[name]['test']);val=static.scores(static.FILES[name]['val'])
        default=.35411001352163435 if key=='zero' else static.val_threshold(val,y_val,ok_val)
        all_scores=sorted((float(x) for x in s[ok]),reverse=True)
        positive=sorted((float(x) for x in s[ok&y]),reverse=True)
        models[key]={'all':all_scores,'positive':positive,'default_threshold':default}
        rows=[]
        for threshold in np.geomspace(1e-5,1,220):
            count,precision,recall=static.score_at(s,y,ok,threshold)
            rows.append([float(threshold),None if not count else float(precision*100),float(recall*100)])
        configs[key]={'id':'nli-threshold-chart','title':'Precision and recall by threshold','showHeader':False,
            'description':name+' ModernBERT on 34 test papers, after cleanup.',
            'height':300,'spec':{'type':'line','x':'threshold','y':['Precision','Recall'],
            'xScale':'log','xDomain':[0.00001,1],'yDomain':[0,100],
            'xLabel':'NLI threshold (log scale)','yLabel':'Percent','showLegend':True},
            'columns':['threshold','Precision','Recall'],'rows':rows}
    template=TEMPLATE.read_text(encoding='utf8')
    for placeholder,obj in [('__ZERO_CONFIG__',configs['zero']),('__FINE_CONFIG__',configs['fine']),('__SCORE_DATA__',models)]:
        if template.count(placeholder)!=1:raise ValueError('Missing or duplicate placeholder: '+placeholder)
        template=template.replace(placeholder,json.dumps(obj,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c'))
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(template,encoding='utf8')
    print('fragment bytes',out.stat().st_size,'zero default',models['zero']['default_threshold'],
          'fine default',models['fine']['default_threshold'])
    # Verify both marked values against the table before delivering.
    for key,name in [('zero','zero-shot'),('fine','fine-tuned')]:
        count,precision,recall=static.score_at(static.scores(static.FILES[name]['test']),y,ok,models[key]['default_threshold'])
        print(key,'sentences',count,'precision',round(precision,4),'recall',round(recall,4))
    assert static.score_at(static.scores(static.FILES['zero-shot']['test']),y,ok,models['zero']['default_threshold'])[0]==2522
    assert static.score_at(static.scores(static.FILES['fine-tuned']['test']),y,ok,models['fine']['default_threshold'])[0]==1998


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);args=ap.parse_args();build(args.output)

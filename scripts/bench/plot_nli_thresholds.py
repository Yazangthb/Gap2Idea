"""Plot held-out gold-test NLI precision, recall and candidate counts vs threshold.

Marker thresholds were chosen on validation, not from the plotted test curves.
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from nli_variants import load_val

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/experiments/nli_threshold_curves'
H=('lim_c','fw_c','gap_c')
FILES={'zero-shot':{'val':ROOT/'data/gold/nli_val_scores.npz',
                    'test':ROOT/'data/gold/nli_test_scores.npz'},
       'fine-tuned':{'val':ROOT/'data/gold/nli_ft/nli_ft_gold_val_scores.npz',
                     'test':ROOT/'data/gold/nli_ft/nli_ft_gold_test_scores.npz'}}


def scores(path):
    z=np.load(path)
    return np.stack([z[f'sent__{h}'][:,0]/(z[f'sent__{h}'][:,0]+z[f'sent__{h}'][:,2]+1e-9) for h in H],axis=1).max(axis=1)


def arrays(part):
    _,r=load_val(part)
    return np.array([x['label']!='none' for x in r]),np.array([not x['junk'] for x in r])


def val_threshold(s,y,ok):
    rank=np.sort(s[ok&y])[::-1]
    return float(rank[math.ceil(.9*y.sum())-1])


def score_at(s,y,ok,threshold):
    keep=ok&(s>=threshold);n=int(keep.sum());tp=int((keep&y).sum())
    return n,tp/(n or 1),tp/int(y.sum())


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    y_val,ok_val=arrays('val');y,ok=arrays('test')
    assert len(y)==11741 and int(y.sum())==664 and int(ok.sum())==9604
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,
                         'axes.titleweight':'bold','axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,
                         'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})
    fig,ax=plt.subplots(2,2,figsize=(12.8,8.4),gridspec_kw={'height_ratios':[1.25,1]},sharex='col')
    fig.subplots_adjust(left=.075,right=.98,top=.79,bottom=.20,wspace=.22,hspace=.13)
    fig.suptitle('NLI threshold: precision, recall and text passed',x=.075,y=.965,ha='left',fontsize=18,fontweight='bold')
    fig.text(.075,.914,'34 held-out test papers · 11,741 sentences · 664 LLM-labeled gaps · after cleanup',
             ha='left',fontsize=10.5,color='#465467')
    fig.text(.075,.885,'Markers use development-selected thresholds; curves show test-set behavior.',
             ha='left',fontsize=9.5,color='#465467')
    for col,name in enumerate(('zero-shot','fine-tuned')):
        s=scores(FILES[name]['test']);sv=scores(FILES[name]['val']);assert len(s)==len(y)
        thresholds=np.geomspace(0.01 if name=='zero-shot' else .00001,1,750)
        values=np.array([score_at(s,y,ok,t) for t in thresholds]); top,bottom=ax[0,col],ax[1,col]
        top.set_title('Zero-shot ModernBERT' if name=='zero-shot' else 'Fine-tuned ModernBERT',loc='left',fontsize=12,pad=9)
        top.plot(thresholds,100*values[:,1],color='#6e55a3',lw=2.4,label='Precision')
        top.plot(thresholds,100*values[:,2],color='#087f8c',lw=2.4,label='Recall')
        bottom.plot(thresholds,values[:,0],color='#cb6e30',lw=2.4,label='Sentences passed')
        markers=([('More recall',val_threshold(sv,y_val,ok_val),'#8b97a6'),('Current',.35411001352163435,'#263548'),('Fewer sentences',.6693316727769015,'#8b97a6')]
                 if name=='zero-shot' else [('Fine-tuned choice',val_threshold(sv,y_val,ok_val),'#263548')])
        for label,t,color in markers:
            n,precision,recall=score_at(s,y,ok,t)
            for a in (top,bottom):a.axvline(t,color=color,lw=1.1,ls='--',alpha=.8,zorder=0)
            top.scatter([t,t],[precision*100,recall*100],color=['#6e55a3','#087f8c'],s=30,zorder=3,edgecolors='white',linewidths=.5)
            bottom.scatter([t],[n],color='#cb6e30',s=34,zorder=3,edgecolors='white',linewidths=.5)
            print(name,label,'threshold',t,'sentences',n,'precision',round(precision,4),'recall',round(recall,4))
        if name=='zero-shot':
            bottom.text(.03,.06,'τ=0.232  ·  3,277 sentences\nτ=0.354  ·  2,522 sentences  (current)\nτ=0.669  ·  1,633 sentences',
                        transform=bottom.transAxes,fontsize=8.6,va='bottom',color='#465467',
                        bbox={'facecolor':'white','edgecolor':'none','alpha':.92,'pad':3})
        else:
            bottom.text(.03,.07,'Selected: τ=0.0021\n1,998 sentences',transform=bottom.transAxes,
                        fontsize=8.6,va='bottom',color='#465467',
                        bbox={'facecolor':'white','edgecolor':'none','alpha':.92,'pad':3})
        for a in (top,bottom):
            a.set_xscale('log');a.set_xlim(thresholds[0],1);a.grid(axis='y',color='#e4e8ed',lw=.7);a.set_axisbelow(True)
        top.set_ylim(0,105);top.set_yticks([0,25,50,75,100]);bottom.set_ylim(0,10200);bottom.set_yticks([0,2500,5000,7500,10000])
        bottom.set_xlabel('NLI score threshold (log scale)')
    ax[0,0].set_ylabel('Precision / recall (%)');ax[1,0].set_ylabel('Sentences passed to 27B')
    fig.legend(handles=[Line2D([0],[0],color='#6e55a3',lw=2.5,label='Precision'),
                        Line2D([0],[0],color='#087f8c',lw=2.5,label='Recall'),
                        Line2D([0],[0],color='#cb6e30',lw=2.5,label='Sentences passed')],
               loc='upper center',bbox_to_anchor=(.53,.86),ncol=3,frameon=False,fontsize=9.5)
    fig.text(.075,.055,'Recall is measured against all 664 labeled gaps, including those removed by cleanup.\n'
             'Threshold scales differ between models; these are candidate-stage results, before 27B filtering.',
             fontsize=8.5,color='#465467',va='bottom')
    for ext in ('png','pdf'):
        fig.savefig(OUT/f'nli_threshold_tradeoffs.{ext}',dpi=180)
    plt.close(fig)


if __name__=='__main__':main()

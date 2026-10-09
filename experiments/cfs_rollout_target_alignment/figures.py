"""Vector-first journal figures; no web screenshots or plotting-dependent storage."""
import csv,logging
logging.getLogger('fontTools').setLevel(logging.WARNING)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import t as student_t
from experiment_core.storage import read_json,atomic_json
from .common import OUTPUT,write_csv,key
from .diagnostics import sign

COLORS=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9','#000000']
NAMES={'P0':'M0','P1':'Fluid','P2':'MC Table','P3':'Linear O','P4':'Linear A','P5':'Cheb O','P6':'Cheb A',
    'P7':'Spline O','P8':'Spline A','P9':'Bayes O','P10':'Bayes A','R0':'RH4','R1':'RH8','D':'Oracle H8'}
plt.rcParams.update({'font.family':'DejaVu Serif','mathtext.fontset':'stix','font.size':9,
    'axes.spines.top':False,'axes.spines.right':False,'axes.labelsize':9,'axes.titlesize':10,
    'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,
    'svg.fonttype':'none','savefig.dpi':600,'axes.linewidth':.7,'lines.linewidth':1.2})

def readcsv(name):
    with (OUTPUT/'processed'/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def finish(fig,name,rows,config,atlas):
    fig.tight_layout()
    fig.savefig(OUTPUT/'figures'/f'{name}.png',dpi=600,bbox_inches='tight')
    fig.savefig(OUTPUT/'figures'/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(OUTPUT/'figures'/f'{name}.svg',bbox_inches='tight')
    write_csv(OUTPUT/'processed'/f'{name}_source.csv',rows)
    atlas.savefig(fig,bbox_inches='tight');plt.close(fig)
    config['figures'][name]=dict(source=f'processed/{name}_source.csv',formats=['600dpi PNG','embedded-font vector PDF','SVG'],
        data_unit='USD unless axis explicitly scales to USD1000',layout='journal single/double column; original arrays, not raster screenshot')

def ci(values):
    x=np.asarray(values,float)
    if len(x)<2:return float(x.mean()),0.
    return float(x.mean()),float(student_t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)))

def phase1(config,atlas):
    raw=read_json(OUTPUT/'targets/phase1.json');rows=[r for r in raw if not r['baseline']]
    compact=[dict(query_id=r['query_id'],seed=r['scenario_seed'],time=r['time'],duration=r['candidate_duration'],
        route_type=r['route_type'],freeze=r['Y_freeze'],roll=r['Y_roll'],difference=r['Y_roll']-r['Y_freeze']) for r in rows]
    fig,ax=plt.subplots(figsize=(3.5,3.2))
    x=np.array([r['freeze'] for r in compact])/1000;y=np.array([r['roll'] for r in compact])/1000
    ax.scatter(x,y,s=10,alpha=.65,c=COLORS[0],edgecolors='black',linewidths=.15,rasterized=False)
    lo=min(x.min(),y.min());hi=max(x.max(),y.max());ax.plot([lo,hi],[lo,hi],'k--',lw=.8)
    ax.axhline(0,c='.7',lw=.5);ax.axvline(0,c='.7',lw=.5)
    ax.set(xlabel='Freeze advantage ($1000)',ylabel='Rolling advantage ($1000)',title='Same state, action and future')
    finish(fig,'01_freeze_vs_roll_scatter',compact,config,atlas)
    for name,field,xlabel in [('02_target_difference_by_time','time','Decision period'),('03_target_difference_by_duration','duration','Candidate duration (periods)')]:
        fig,ax=plt.subplots(figsize=(3.5,3));plot=[]
        bins=sorted({r[field] for r in compact})
        if field=='duration':bins=[1,4,8,12,100]
        for b in bins:
            if field=='time':group=[r for r in compact if r[field]==b];label=str(b)
            else:
                previous=0 if b==1 else bins[bins.index(b)-1]
                group=[r for r in compact if previous<r[field]<=b];label='1' if b==1 else f'{previous+1}-{b}' if b<100 else '>12'
            if not group:continue
            scen=[np.mean([abs(r['difference']) for r in group if r['seed']==seed]) for seed in sorted({r['seed'] for r in group})]
            m,h=ci(scen);plot.append(dict(bin=label,mean_absolute_difference=m,CI_half=h,n_scenarios=len(scen)))
        xx=np.arange(len(plot));means=[r['mean_absolute_difference']/1000 for r in plot];err=[r['CI_half']/1000 for r in plot]
        ax.errorbar(xx,means,yerr=err,fmt='o-',c=COLORS[0],capsize=3)
        ax.set_xticks(xx,[r['bin'] for r in plot]);ax.set(xlabel=xlabel,ylabel='Mean absolute mismatch ($1000)',title='Equal-scenario means; t95 intervals')
        finish(fig,name,plot,config,atlas)
    ranks=readcsv('candidate_ranking_metrics.csv')
    for name,metric in [('04_candidate_ranking_flip','pair'),('05_top1_disagreement','top')]:
        fig,ax=plt.subplots(figsize=(3.5,3));plot=[]
        for t in [3,8,12,16,20]:
            rr=[r for r in ranks if int(r['time'])==t]
            if not rr:continue
            if metric=='pair':
                den=sum(int(r['comparable_pairs']) for r in rr);a=sum(int(r['ranking_flips']) for r in rr)/den if den else 0;b=None
            else:a=np.mean([int(r['top1_disagreement']) for r in rr]);b=np.mean([int(r['top1_disjoint']) for r in rr])
            plot.append(dict(time=t,rate=float(a),disjoint_rate=None if b is None else float(b),groups=len(rr),n_scenarios=len({r['scenario_seed'] for r in rr})))
        x=np.arange(len(plot));ax.bar(x-.12,[r['rate'] for r in plot],width=.24,color=COLORS[0],edgecolor='black',lw=.5,label='Strict inversion' if metric=='pair' else 'Canonical top1')
        if metric=='top':ax.bar(x+.12,[r['disjoint_rate'] for r in plot],width=.24,color=COLORS[1],edgecolor='black',lw=.5,hatch='//',label='Disjoint top sets')
        ax.set_xticks(x,[str(r['time']) for r in plot]);ax.set(xlabel='Decision period',ylabel='Rate',ylim=(0,1));ax.legend(frameon=False)
        finish(fig,name,plot,config,atlas)
    fig,ax=plt.subplots(figsize=(3.5,3));plot=[]
    for kind in sorted({r['route_type'] for r in rows}):
        rr=[r for r in rows if r['route_type']==kind];sx=sign(np.array([r['Y_freeze'] for r in rr]),1e-6);sy=sign(np.array([r['Y_roll'] for r in rr]),1e-6)
        plot.append(dict(route_type=kind,inclusive=float(np.mean(sx!=sy)),strict_opposition=float(np.mean(sx*sy==-1)),n_candidates=len(rr),n_scenarios=len({r['scenario_seed'] for r in rr})))
    x=np.arange(len(plot));ax.bar(x-.15,[r['inclusive'] for r in plot],width=.3,color=COLORS[0],edgecolor='black',lw=.5,label='Including zero transitions')
    ax.bar(x+.15,[r['strict_opposition'] for r in plot],width=.3,color=COLORS[1],hatch='//',edgecolor='black',lw=.5,label='Strict opposite signs')
    ax.set_xticks(x,[r['route_type'].replace('_',' ') for r in plot]);ax.set(ylabel='Sign-change rate',ylim=(0,1));ax.legend(frameon=False,loc='upper left',fontsize=7)
    finish(fig,'06_sign_flip_by_route_type',plot,config,atlas)
    fig,ax=plt.subplots(figsize=(3.5,3));d=np.array([r['difference'] for r in compact])/1000
    ax.hist(d,bins=25,color=COLORS[0],edgecolor='black',lw=.5);ax.axvline(0,c='black',ls='--',lw=.8)
    ax.set(xlabel='Rolling minus freeze advantage ($1000)',ylabel='Candidate count',title='Descriptive, not independent replicates')
    finish(fig,'07_target_difference_distribution',compact,config,atlas)

def facets(cfg):
    fig,axes=plt.subplots(2,2,figsize=(7.2,4.8))
    for ax,c in zip(axes.flat,cfg['conditions']):ax.set_title(key(c).replace('california_corridor','CA').replace('western_multicluster','W'))
    return fig,list(axes.flat)

def comparison_figure(cfg,stats,family,name,config,atlas):
    fig,axes=facets(cfg);source=[]
    for ax,c in zip(axes,cfg['conditions']):
        rr=[r for r in stats if r['condition']==key(c) and r['comparison_family']==family]
        x=np.arange(len(rr));y=np.array([r['mean'] for r in rr])/1000
        lo=np.array([r['CI95_lower'] for r in rr])/1000;hi=np.array([r['CI95_upper'] for r in rr])/1000
        ax.errorbar(x,y,yerr=np.vstack((y-lo,hi-y)),fmt='o',c=COLORS[0],capsize=3);ax.axhline(0,c='black',ls='--',lw=.7)
        ax.set_xticks(x,[NAMES.get(r['algorithm'],r['algorithm']) for r in rr],rotation=45,ha='right')
        ax.set(ylabel={'VminusZero':'V minus M0 ($1000)','ACTIONminusOLD':'ACTION minus OLD ($1000)',
            'ACTIONminusTable':'V minus Table ($1000)'}[family]);source.extend(rr)
    finish(fig,name,source,config,atlas)

def phase2(cfg,config,atlas):
    metrics=read_json(OUTPUT/'processed/heldout_value_metrics.json');summary=read_json(OUTPUT/'processed/policy_statistics.json')
    prediction_ci={(r['condition'],r['method']):r for r in readcsv('heldout_pooled_RMSE_intervals.csv')}
    fig,axes=facets(cfg);source=[]
    methods=['linear','chebyshev','spline','bayesian']
    for ax,c in zip(axes,cfg['conditions']):
        for fs,offset,color,hatch in [('OLD',-.16,COLORS[0],None),('ACTION',.16,COLORS[1],'//')]:
            rr=[next(r for r in metrics if r['condition']==key(c) and r['method']==m+'-'+fs and r['scope']=='all') for m in methods]
            positions=np.arange(4)+offset
            yy=np.array([r['RMSE']/1000 for r in rr])
            bounds=[prediction_ci[(r['condition'],r['method'])] for r in rr]
            lower=np.array([float(r['bootstrap_CI95_lower'])/1000 for r in bounds])
            upper=np.array([float(r['bootstrap_CI95_upper'])/1000 for r in bounds])
            ax.bar(positions,yy,width=.32,color=color,hatch=hatch,edgecolor='black',lw=.5,label=fs)
            ax.vlines(positions,lower,upper,color='black',lw=.7)
            ax.plot(positions,lower,'_',c='black',ms=5);ax.plot(positions,upper,'_',c='black',ms=5)
            source.extend(r|dict(RMSE_bootstrap_lower=float(b['bootstrap_CI95_lower']),RMSE_bootstrap_upper=float(b['bootstrap_CI95_upper'])) for r,b in zip(rr,bounds))
        ax.set_xticks(range(4),['Linear','Cheb','Spline','Bayes']);ax.set(ylabel='Held-out RMSE ($1000)')
        ax.set_title(ax.get_title()+f" (n={rr[0]['n_scenarios']})")
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.055),ncol=2,frameon=False)
    fig.suptitle('Parent-block intervals; shared donor dependence unadjusted',fontsize=8,y=1.11)
    finish(fig,'08_old_vs_action_prediction',source,config,atlas)
    config['figures']['08_old_vs_action_prediction']['uncertainty']='Conditional parent-cluster bootstrap,10000; shared donor-node dependence unadjusted. Independent own-future sensitivity separately saved.'
    pp=readcsv('heldout_paired_metrics.csv');fig,axes=facets(cfg);source=[]
    for ax,c in zip(axes,cfg['conditions']):
        rr=[r for r in pp if r['condition']==key(c) and r['metric']=='pairwise_ranking_accuracy']
        yy=np.array([float(r['mean']) for r in rr]);lo=np.array([float(r['CI95_lower']) for r in rr]);hi=np.array([float(r['CI95_upper']) for r in rr])
        ax.errorbar(np.arange(len(rr)),yy,yerr=np.vstack((yy-lo,hi-yy)),fmt='o',c=COLORS[0],capsize=3);ax.axhline(0,c='black',ls='--',lw=.7)
        ax.set_xticks(range(len(rr)),[r['method'].replace('chebyshev','Cheb').replace('bayesian','Bayes') for r in rr]);ax.set(ylabel='Rank accuracy: ACTION - OLD');source.extend(rr)
    fig.suptitle('Parent-block intervals; shared donor dependence unadjusted',fontsize=8,y=1.04)
    finish(fig,'09_old_vs_action_ranking',source,config,atlas)
    for family,name in [('VminusZero','10_policy_paired_gain'),('ACTIONminusOLD','11_action_minus_old_profit'),('ACTIONminusTable','12_table_vs_complex_v')]:
        comparison_figure(cfg,summary['comparisons'],family,name,config,atlas)
    fig,axes=facets(cfg);source=[]
    for ax,c in zip(axes,cfg['conditions']):
        rr=[next(r for r in summary['summary'] if r['condition']==key(c) and r['policy']==p) for p in ['P0','R0','R1']]
        x=np.arange(3);yy=np.array([r['mean'] for r in rr])/1000;hh=np.array([r['CI95_upper']-r['mean'] for r in rr])/1000
        ax.bar(x,yy,yerr=hh,capsize=3,color=[COLORS[0],COLORS[1],COLORS[2]],edgecolor='black',lw=.5)
        ax.set_xticks(x,['M0','RH4','RH8']);ax.set(ylabel='Mean realized profit ($1000)');source.extend(rr)
    finish(fig,'13_rh4_rh8_reference',source,config,atlas)
    runs=read_json(OUTPUT/'processed/formal_policies.json');fig,axes=facets(cfg);source=[]
    for ax,c in zip(axes,cfg['conditions']):
        rr=[r for r in runs if r['condition']==key(c)];lookup={(r['seed'],r['policy']):r for r in rr};plot=[]
        for p in ['P0','P2','P6','P8','P10','R0','R1']:
            mean,half=ci([lookup[s,'D']['realized_profit']-lookup[s,p]['realized_profit'] for s in cfg['formal_seeds']])
            plot.append(dict(condition=key(c),policy=p,mean_difference=mean,CI_half=half))
        ax.errorbar(range(len(plot)),[r['mean_difference']/1000 for r in plot],yerr=[r['CI_half']/1000 for r in plot],fmt='o',c=COLORS[0],capsize=3)
        ax.set_xticks(range(len(plot)),[NAMES[r['policy']] for r in plot],rotation=45,ha='right');ax.axhline(0,c='black',ls='--',lw=.7)
        ax.set(ylabel='H8 preview gap ($1000)');source.extend(plot)
    finish(fig,'14_oracle_gap',source,config,atlas)
    fig,axes=facets(cfg);source=[]
    for ax,c in zip(axes,cfg['conditions']):
        rr=[r for r in summary['summary'] if r['condition']==key(c)]
        for i,r in enumerate(rr):
            ax.scatter(r['mean_wall_seconds'],r['mean']/1000,c=COLORS[i%7],marker=['o','s','^','D','v','P','X','<','>','p','*','h','H','d'][i],
                s=28,edgecolor='black',linewidth=.4,label=NAMES[r['policy']]);source.append(r)
        ax.set(xscale='log',xlabel='Mean online day wall time (s)',ylabel='Mean profit ($1000)')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=7,fontsize=7,frameon=False,bbox_to_anchor=(.5,-.075))
    finish(fig,'15_runtime_profit_frontier',source,config,atlas)

def toy(config,atlas):
    audit=read_json(OUTPUT/'qa/toy_accounting_audit.json');fig,axes=plt.subplots(1,3,figsize=(7.2,2.8))
    ax=axes[0];ax.plot([0,1,2],[0,0,0],c='black');ax.scatter([0,1,2],[0,0,0],s=350,c='white',edgecolor='black',zorder=3)
    for i,label in enumerate('ABC'):ax.text(i,0,label,ha='center',va='center',zorder=4)
    ax.annotate('R1: A to C, g=50',(2,.1),(.2,.8),arrowprops=dict(arrowstyle='->',color=COLORS[1]),fontsize=8)
    ax.annotate('R2: A to B, g=40',(1,-.1),(.1,-.8),arrowprops=dict(arrowstyle='->',color=COLORS[0]),fontsize=8)
    ax.set_ylim(-1.2,1.2);ax.set_xlim(-.3,2.3);ax.axis('off');ax.set_title('(a) Two current routes')
    ax=axes[1];ax.bar(['R1','R2'],[50,40],color=[COLORS[1],COLORS[0]],edgecolor='black');ax.set(ylabel='Current business profit',title='(b) M0 selects R1')
    ax=axes[2];ax.bar([0-.15,1-.15],[70,-100],width=.3,color=COLORS[0],edgecolor='black',label='Rolling')
    ax.bar([0+.15,1+.15],[70,0],width=.3,color=COLORS[1],hatch='//',edgecolor='black',label='Freeze')
    ax.set_xticks([0,1],['Short R2','Delayed R1']);ax.axhline(0,c='black',lw=.7);ax.set(ylabel='Relative future advantage',title='(c) Freeze hides delayed loss');ax.legend(frameon=False,fontsize=7)
    finish(fig,'16_toy_control_and_accounting',[audit['C']],config,atlas)

def main():
    cfg=read_json(OUTPUT/'config/main.json')
    config=dict(engine='Matplotlib vector backend; independent of experiment storage',dpi=600,font='DejaVu Serif / STIX mathematics',
        pdf_fonttype=42,svg_fonttype='text',palette=COLORS,colorblind_friendly=True,black_white='markers, hatches, dashed zero/identity lines',
        column_inches=[3.5,7.2],uncertainty='Scenario-level t95; figure08 whole-scenario bootstrap; Phase1 ranking/sign pooled rates descriptive without row-level CI',figures={})
    with PdfPages(OUTPUT/'figures/figure_atlas.pdf') as atlas:
        phase1(config,atlas)
        if (OUTPUT/'processed/policy_statistics.json').exists():phase2(cfg,config,atlas)
        toy(config,atlas)
    atomic_json(OUTPUT/'figures/plotting_config.json',config)

if __name__=='__main__':main()

"""Five same-source SVG/600dpi scientific plot groups, no dashboard screenshots."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import TARGET,NAMES,sha

STEMS=('01-profit','02-validation','03-critic','04-gradients','05-state-stochastic')
COLORS=dict(A='#0072B2',B='#E69F00',C='#009E73',D='#CC79A7',MC='#666666',M0='#999999')
STYLES=dict(A='-',B='--',C='-.',D=':')
LABELS=dict(zip(NAMES,('CA: 80 orders / 3 vehicles','W40: 40 orders / 8 vehicles','W80: 80 orders / 3 vehicles')))


def rows(name):
    with (TARGET/(name+'.csv')).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def run():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                         'axes.spines.right':False,'svg.fonttype':'none','savefig.dpi':600})
    paired=rows('paired_results');tr=rows('training_diagnostics');cr=rows('critic_diagnostics')
    gr=rows('gradient_diagnostics');ac=rows('actor_state_sensitivity')
    configs=[]
    def save(fig,stem,sources,caption):
        fig.tight_layout(rect=(0,.12,1,1) if stem==STEMS[1] else None)
        fig.savefig(TARGET/'figures'/(stem+'.svg'))
        fig.savefig(TARGET/'figures'/(stem+'.png'),dpi=600)
        config=dict(stem=stem,caption=caption,source_CSV={s+'.csv':sha(TARGET/(s+'.csv')) for s in sources},
            size_inches=list(fig.get_size_inches()),dpi=600,colors=COLORS,styles=STYLES,
            code_file=str(Path(__file__)),code_sha256=sha(Path(__file__)))
        (TARGET/'figures'/(stem+'.config.json')).write_text(json.dumps(config,indent=2),encoding='utf-8')
        configs.append(config);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(10,3.1))
    for ax,name in zip(axes,NAMES):
        group=[r for r in paired if r['condition']==name]
        methods=('M0','MC','A','B','C','D');ys=[np.mean([float(r[m+'_profit_USD']) for r in group])/1000 for m in methods]
        bars=ax.bar(methods,ys,color=[COLORS[m] for m in methods],edgecolor='black',linewidth=.4)
        for b,h in zip(bars,('','//','','//','xx','..')):b.set_hatch(h)
        ax.set_title(LABELS[name]);ax.set_ylabel('Realized profit (1000 USD)')
    save(fig,STEMS[0],['paired_results'],'Descriptive mean; paired crossed adjusted intervals in statistical_comparisons.csv, not bar overlap.')
    fig,axes=plt.subplots(1,3,figsize=(10,3.1))
    for ax,name in zip(axes,NAMES):
        for v in ('A','B','C','D'):
            means=[]
            for u in (8,16):
                rs=[float(r['validation_mean_USD'])/1000 for r in tr if r['condition']==name and r['variant']==v and int(r['update'])==u]
                means.append(np.mean(rs))
            ax.plot([8,16],means,STYLES[v],marker='o',color=COLORS[v],label=v)
        ax.set_xlabel('PPO update');ax.set_ylabel('Deterministic validation (1000 USD)');ax.set_title(LABELS[name])
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=4,bbox_to_anchor=(.5,.01),frameon=False)
    save(fig,STEMS[1],['training_diagnostics'],'Mean over three models on the same16validation scenes; no test selection.')
    fig,axes=plt.subplots(1,3,figsize=(10,3.1))
    for ax,name in zip(axes,NAMES):
        for v in ('A','B','D'):
            y=[np.mean([float(r['EV']) for r in cr if r['condition']==name and r['variant']==v and r['period']=='-1'
                       and r['target']=='MC_RTG' and int(r['update'])==u]) for u in range(1,17)]
            ax.plot(range(1,17),y,STYLES[v],color=COLORS[v],label=v)
        ax.axhline(0,color='black',lw=.5);ax.set_xlabel('PPO update');ax.set_ylabel('MC return-to-go EV');ax.set_title(LABELS[name]);ax.legend()
    save(fig,STEMS[2],['critic_diagnostics'],'Pre-update on-policy MC target EV; changing visitation, C has no Critic.')
    fig,axes=plt.subplots(2,3,figsize=(10,5.2))
    for j,name in enumerate(NAMES):
        for field,label,line in (('shared_actor_norm','Actor','-'),('shared_weighted_critic_norm','Weighted Critic','--')):
            y=[np.mean([float(r[field]) for r in gr if r['condition']==name and r['variant']=='B' and int(r['update'])==u]) for u in range(1,17)]
            axes[0,j].plot(range(1,17),y,line,label=label)
        axes[0,j].set_yscale('symlog',linthresh=.001);axes[0,j].set_ylabel('Shared gradient norm (B)');axes[0,j].legend(fontsize=8)
        ys=[]
        for u in range(1,17):
            values=[float(r['shared_cosine']) for r in gr if r['condition']==name and r['variant']=='B' and int(r['update'])==u and r['shared_cosine']]
            ys.append(float(np.mean(values)) if values else np.nan)
        axes[1,j].plot(range(1,17),ys);axes[1,j].axhline(0,color='black',lw=.5);axes[1,j].set_ylim(-1.05,1.05)
        axes[1,j].set_ylabel('Shared cosine (B)');axes[1,j].set_xlabel('PPO update');axes[0,j].set_title(LABELS[name])
    save(fig,STEMS[3],['gradient_diagnostics'],'Actual first accepted minibatch/update shared encoder/backbone only. A historical split missing; D has no shared parameters. Zero-norm cosine missing.')
    fig,axes=plt.subplots(2,3,figsize=(10,5.2))
    for j,name in enumerate(NAMES):
        for v in ('A','B','C','D'):
            y=[np.mean([float(r['D_pi']) for r in ac if r['condition']==name and r['variant']==v and int(r['update'])==u]) for u in (8,16)]
            axes[0,j].plot([8,16],y,STYLES[v],marker='o',color=COLORS[v],label=v)
        axes[0,j].set_xlabel('PPO update');axes[0,j].set_ylabel('Common-state D_pi');axes[0,j].legend(ncol=2);axes[0,j].set_title(LABELS[name])
        selected=[r for r in tr if r['condition']==name and r['stochastic_validation_USD']]
        y=[np.mean([float(r['stochastic_validation_USD'])-float(r['paired_deterministic_validation4_USD']) for r in selected if r['variant']==v])/1000 for v in ('A','B','C','D')]
        axes[1,j].bar(['A','B','C','D'],y,color=[COLORS[v] for v in ('A','B','C','D')],edgecolor='black',linewidth=.4)
        axes[1,j].axhline(0,color='black',lw=.5);axes[1,j].set_ylabel('Stochastic - deterministic\nvalidation (1000 USD)')
    save(fig,STEMS[4],['actor_state_sensitivity','training_diagnostics'],'Same12reference states. Stochastic diagnostic: same4scenes/seed and frozen policy, one draw/scene, not confirmatory.')
    return configs


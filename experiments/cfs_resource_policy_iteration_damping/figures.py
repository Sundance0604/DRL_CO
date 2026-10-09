"""Paper figures with portable data/config; Matplotlib is a replaceable backend."""
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from experiment_core.storage import read_json,atomic_json
from experiments.cfs_resource_policy_iteration.analysis import describe
from .common import OUTPUT,ALPHAS,key,write_csv,file_hash

PALETTE={'undamped':'#333333','025':'#0072B2','050':'#009E73','075':'#D55E00'}
MARKERS={'undamped':'x','025':'o','050':'s','075':'^'}
NAMES=['A2_01_profit_path_by_alpha','A2_02_table_relative_change','A2_03_action_disagreement',
    'A2_04_state_distribution_shift','A2_05_CA_profit_trajectory','A2_06_W40_profit_trajectory',
    'A2_07_raw_vs_damped_cell_paths','A2_08_oscillation_heatmap','A2_09_runtime_by_alpha']

def short(cid):
    return ('CA' if cid.startswith('california') else 'W')+cid.split('-n')[1].replace('-k','/')

def style():
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'mathtext.fontset':'stix',
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':.65,'lines.linewidth':1.1,
        'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'savefig.dpi':600})

def csvrows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def generate(cfg):
    style();out=OUTPUT/'figures';out.mkdir(exist_ok=True)
    selected=read_json(OUTPUT/'manifests/alpha_freeze.json')['selected_alpha']
    cids=[key(c) for c in cfg['conditions']]
    runs=read_json(OUTPUT/'processed/diagnostic_runs.json')
    profits=[]
    for cid in cids:
        for a in ['undamped']+list(ALPHAS):
            for n in range(5):
                label=f'U{n}' if a=='undamped' or n==0 else f'D{n}'
                rr=[r for r in runs if r['condition']==cid and r['policy']==label and
                    (r['alpha'] is None if label.startswith('U') else r['alpha']==a)]
                d=describe([r['realized_profit'] for r in rr],cfg)
                profits.append(dict(condition=cid,alpha=a,iteration=n,**d,stage='diagnostic'))
    tables=read_json(OUTPUT/'processed/table_metrics.json')
    actions=read_json(OUTPUT/'processed/diagnostic_action_summary.json')
    shifts=read_json(OUTPUT/'processed/diagnostic_state_shifts.json')
    states=[]
    for cid,a,n in __import__('itertools').product(cids,['undamped']+list(ALPHAS),range(1,5)):
        rr=[r for r in shifts if r['condition']==cid and r['alpha']==a and r['iteration']==n]
        vals=[r['absolute_SMD'] for r in rr if r['absolute_SMD'] is not None]
        states.append(dict(condition=cid,alpha=a,iteration=n,mean=float(np.mean(vals)) if vals else None,
            undefined_SMD=sum(r['absolute_SMD'] is None for r in rr),aggregation='descriptive equal time/hub/feature mean; not independent samples'))
    stats=read_json(OUTPUT/'processed/formal_statistics.json')
    oscillations=read_json(OUTPUT/'processed/oscillations.json')
    runtime=read_json(OUTPUT/'processed/runtime.json');saved=[]
    atlas=PdfPages(out/'A2_figure_atlas.pdf')
    def export(fig,name,data,meta):
        src=out/(name+'.source.csv');write_csv(src,data)
        atomic_json(out/(name+'.plot.json'),dict(schema_version='scientific-plot/v1',
            figure=name,backend='matplotlib',source_file=src.name,source_sha256=file_hash(src),
            width_inches=fig.get_figwidth(),height_inches=fig.get_figheight(),dpi=600,
            font='DejaVu Serif / STIX',palette=PALETTE,marker=MARKERS,
            font_embedding='PDF TrueType Type42',selected_alpha=selected,**meta))
        fig.savefig(out/(name+'.pdf'));fig.savefig(out/(name+'.svg'));fig.savefig(out/(name+'.png'),dpi=600)
        atlas.savefig(fig);saved.append(name);plt.close(fig)
    def panel_lines(name,data,y,ylabel,meta):
        fig,axs=plt.subplots(2,2,figsize=(7.2,5.2),layout='constrained')
        for cid,ax in zip(cids,axs.flat):
            for a in ['undamped']+list(ALPHAS):
                rr=sorted([r for r in data if r['condition']==cid and r['alpha']==a],key=lambda r:r['iteration'])
                if not rr:continue
                ax.plot([r['iteration'] for r in rr],[r[y] for r in rr],color=PALETTE[a],marker=MARKERS[a],
                    linestyle='--' if a=='undamped' else '-',label='undamped' if a=='undamped' else f'alpha={ALPHAS[a]:g}')
            ax.set(title=short(cid),xlabel='Policy improvement step',ylabel=ylabel)
            ax.set_xticks(sorted(set(r['iteration'] for r in data)));ax.grid(alpha=.16)
        handles,labels=axs.flat[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False,fontsize=8)
        export(fig,name,data,meta)
    panel_lines(NAMES[0],profits,'mean','Diagnostic mean profit (USD)',
        dict(statistical_unit='20 independent scenarios per condition',interval='source t95/bootstrap retained; path line shows means',formal_used=False))
    tt=[r for r in tables if r['kind']=='deployed']
    panel_lines(NAMES[1],tt,'relative_L2','Relative L2 table change',dict(grid='same native time/hub/q<=K; nonterminal',state_distribution_confound=True))
    aa=[r for r in actions if r['kind']=='adjacent']
    panel_lines(NAMES[2],aa,'mean','Same-state full action\ndisagreement (fraction)',dict(reference='U0 all24 periods; daily scenario fractions; n20'))
    panel_lines(NAMES[3],states,'mean','Mean absolute state SMD',dict(undefined='excluded from mean and counts retained, never zero-filled',interval='descriptive repeated-state diagnostic'))
    for name,cid in zip(NAMES[4:6],[next(c for c in cids if c.startswith('california')),next(c for c in cids if '-n40-' in c)]):
        fig,axs=plt.subplots(1,2,figsize=(7.2,3.25),layout='constrained');data=[]
        for stage,ax in zip(['diagnostic','formal'],axs):
            ss=read_json(OUTPUT/'processed'/f'{stage}_statistics.json')['summary']
            for family,color,marker in [('U','#333333','x'),('D',PALETTE[selected],MARKERS[selected])]:
                rr=sorted([r for r in ss if r['condition']==cid and r['policy'].startswith(family)],key=lambda r:int(r['policy'][1:]))
                x=[int(r['policy'][1:]) for r in rr];y=np.array([r['mean'] for r in rr])
                ax.errorbar(x,y,yerr=[y-[r['CI95_lower'] for r in rr],[r['CI95_upper'] for r in rr]-y],
                    color=color,marker=marker,linestyle='--' if family=='U' else '-',capsize=2,label='undamped' if family=='U' else f'damped alpha={ALPHAS[selected]:g}')
                data.extend(rr)
            ax.set(title=f'{short(cid)} ({stage})',xlabel='Policy improvement step',ylabel='Realized profit (USD)')
            ax.set_xticks(range(5));ax.grid(alpha=.16)
        handles,labels=axs[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False,fontsize=8)
        export(fig,name,data,dict(interval='scenario-level marginal mean t95; paired differences separate',sample_size={'diagnostic':20,'formal':40}))
    fig,axs=plt.subplots(2,2,figsize=(7.2,7.2),layout='constrained');data=[]
    colors=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00']
    for cid,ax in zip(cids,axs.flat):
        rr=sorted([r for r in oscillations if r['condition']==cid and r['alpha']==selected],key=lambda r:r['range_USD'],reverse=True)[:5]
        for i,r in enumerate(rr):
            color=colors[i];label=f'{r["hub"]}, q={r["q"]}, t={r["time"]}'
            ax.plot(range(4),[r[f'raw_V{n}'] for n in range(4)],color=color,linestyle='--',alpha=.7,
                marker=['o','s','^','D','v'][i],markersize=3,markerfacecolor='none')
            ax.plot(range(4),[r[f'damped_V{n}'] for n in range(4)],color=color,marker=['o','s','^','D','v'][i],label=label)
            data.append(r)
        ax.set(title=short(cid),xlabel='Value iteration n',ylabel='Marginal resource value (USD)');ax.set_xticks(range(4));ax.grid(alpha=.16)
        ax.legend(loc='upper center',bbox_to_anchor=(.5,-.25),fontsize=6.5,frameon=False,ncol=1)
    fig.suptitle('Raw: dashed, open markers; deployed: solid, filled markers',fontsize=9)
    export(fig,NAMES[6],data,dict(cell_selection='five largest deployed ranges, selected alpha; raw dashed / deployed solid',descriptive=True))
    fig,axs=plt.subplots(2,2,figsize=(7.2,7.2),layout='constrained');data=[]
    for cid,ax in zip(cids,axs.flat):
        rr=[r for r in oscillations if r['condition']==cid and r['alpha']==selected]
        times=sorted({r['time'] for r in rr});hq=sorted({(r['hub'],r['q']) for r in rr});lookup={(r['hub'],r['q'],r['time']):r['range_USD'] for r in rr}
        matrix=np.array([[lookup[h,q,t] for t in times] for h,q in hq])
        mesh=ax.pcolormesh(np.arange(len(times)+1),np.arange(len(hq)+1),matrix,cmap='cividis',shading='flat',rasterized=False)
        ax.set(title=short(cid),xlabel='Native time node',ylabel='Hub (q increases within block)')
        ax.set_xticks(np.arange(len(times))+.5,times)
        hubs=sorted({h for h,q in hq});ticks=[];labels=[]
        for h in hubs:
            ids=[i for i,(hh,q) in enumerate(hq) if hh==h]
            ticks.append(float(np.mean(ids)+.5));labels.append(f'{h}\nq=1..{len(ids)}')
        ax.set_yticks(ticks,labels,fontsize=7)
        cb=fig.colorbar(mesh,ax=ax,label='Deployed range (USD)');cb.solids.set_rasterized(False);data.extend(rr)
    export(fig,NAMES[7],data,dict(scope='all native nonterminal cells q<=K; selected alpha; per-condition color scales',no_cycle_claim=True))
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.5),layout='constrained');data=[]
    for a in ALPHAS:
        rr=[r for r in runtime if r['stage']=='offline' and r['alpha']==a]
        row=dict(alpha=a,alpha_value=ALPHAS[a],E2_E3_wall_seconds=sum(r['wall_seconds'] for r in rr),
            E2_E3_worker_seconds=sum(r['worker_seconds'] for r in rr),reused_E0_E1=True)
        data.append(row)
    axs[0].bar([f'{ALPHAS[r["alpha"]]:g}' for r in data],[r['E2_E3_wall_seconds']/3600 for r in data],color=[PALETTE[r['alpha']] for r in data])
    axs[0].set(xlabel='Alpha',ylabel='Fresh E(D2)+E(D3) wall hours',title='Offline evaluation (three workers)')
    on=[]
    for a in ALPHAS:
        rr=[r for r in runtime if r['stage']=='diagnostic' and r['alpha']==a and r['policy']=='D4']
        on.append(float(np.mean([r['online_seconds'] for r in rr])))
    axs[1].bar([f'{ALPHAS[a]:g}' for a in ALPHAS],on,color=[PALETTE[a] for a in ALPHAS])
    axs[1].set(xlabel='Alpha',ylabel='Mean diagnostic D4 online seconds',title='Online day (four regimes pooled)')
    for row,v in zip(data,on):row['mean_D4_online_seconds']=v
    export(fig,NAMES[8],data,dict(timers='wall differs from worker sum; reused shared E0/E1 cost disclosed separately; no cash ROI',parallel_workers=3))
    atlas.close();assert saved==NAMES
    atomic_json(out/'figure_manifest.json',dict(names=saved,atlas='A2_figure_atlas.pdf',atlas_pages=9,
        sha256={p.name:file_hash(p) for p in out.iterdir() if p.is_file() and p.name!='figure_manifest.json'}))
    return saved

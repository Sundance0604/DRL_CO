"""Six prespecified scientific figure groups; no numerical/selection mutation."""
import csv
import json
import textwrap
from collections import Counter, defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from experiment_core.storage import read_json, atomic_json
from .common import OUTPUT, sha, now
from .analysis import statistics

COLORS=['#0072B2','#D55E00','#009E73','#CC79A7']
MARKERS=['o','s','^','D']

def short(cid):
    return 'CA80/3' if cid.startswith('california') else 'W40/8' if '-n40-' in cid else 'W80/3'

def source(stem,rows):
    assert rows, 'No placeholder scientific figures'
    path=OUTPUT/'figures'/(stem+'.source.csv')
    cols=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows)

def style():
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'axes.labelsize':9,
        'axes.titlesize':10,'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':.7,'lines.linewidth':1.1,
        'savefig.dpi':600,'figure.dpi':120})

def interval(ax,x,row,color,marker,vertical=False):
    m=row['mean_USD'];lo=row['t95_low_USD'];hi=row['t95_high_USD']
    error=np.asarray([[m-lo],[hi-m]])
    if vertical:ax.errorbar(x,m,yerr=error,color=color,marker=marker,capsize=3,linestyle='none',mfc='white')
    else:ax.errorbar(m,x,xerr=error,color=color,marker=marker,capsize=3,linestyle='none',mfc='white')

def run():
    style();result=read_json(OUTPUT/'processed/analysis.json');names=[];missing=[];configs={}
    directory=OUTPUT/'figures';directory.mkdir(exist_ok=True)
    atlas=directory/'joint_action_value_atlas.pdf'
    def save(stem,fig,data,caption):
        assert data
        fig.subplots_adjust(bottom=.24 if stem=='J02_price_Q_ordering' else .19,top=.88,wspace=.42,hspace=.4)
        fig.text(.5,.025,textwrap.fill(caption,width=115),ha='center',va='bottom',fontsize=7)
        for ext in ['pdf','svg','png']:fig.savefig(directory/(stem+'.'+ext),dpi=600)
        pdf.savefig(fig);source(stem,data)
        c=dict(created_at=now(),figure=stem,caption=caption,size_inches=list(fig.get_size_inches()),
            dpi=600,vector_formats=['PDF','SVG'],fonts='embedded DejaVu Serif',
            colors=COLORS,markers=MARKERS,unit='scenario unless explicitly labelled',
            statistics_source_sha256=sha(OUTPUT/'processed/analysis.json'),
            renderer_sha256=sha(OUTPUT/'figures.py'))
        atomic_json(directory/(stem+'.config.json'),c);names.append(stem);configs[stem]=c;plt.close(fig)
    with PdfPages(atlas) as pdf:
        rr=result['primary'];fig,ax=plt.subplots(figsize=(7.16,4.6))
        for i,r in enumerate(rr):
            j=0 if r['source']=='M0' else 1;interval(ax,i,r,COLORS[j],MARKERS[j])
        ax.set_yticks(range(len(rr)),[short(r['condition'])+' / '+r['source'] for r in rr])
        ax.axvline(0,color='.4',linestyle='--',linewidth=.7);ax.invert_yaxis()
        ax.set_xlabel('Independent-assessment profit difference (USD)');ax.set_title('32-path selector minus price selector')
        fig.subplots_adjust(left=.22)
        save('J01_independent_assessment',fig,rr,'40 scenarios per row; paired unadjusted t95 intervals. Six-test Holm p values reported separately.')

        rows=[r for r in read_json(OUTPUT/'processed/joint_action_scores.json') if r['stage']=='formal']
        by=defaultdict(list)
        for r in rows:by[r['state_id']].append(r)
        data=[]
        for group in by.values():
            baseline=next(r for r in group if 'M0' in r['labels'])
            for r in group:data.append({k:r[k] for k in ['state_id','condition','source','scenario_seed','action_hash']}|
                dict(L_delta_USD=r['L_V_USD']-baseline['L_V_USD'],Q_delta_USD=r['Q_assessment_mean_USD']-baseline['Q_assessment_mean_USD'],
                     MC_SE_USD=r['Q_MC_SE_USD'],labels=r['labels']))
        fig,axes=plt.subplots(1,3,figsize=(7.16,3.65))
        for ax,cid in zip(axes,sorted({r['condition'] for r in data})):
            group=[r for r in data if r['condition']==cid]
            for j,src in enumerate(['M0','pi1']):
                g=[r for r in group if r['source']==src]
                ax.scatter([r['L_delta_USD'] for r in g],[r['Q_delta_USD'] for r in g],s=13,
                    marker=MARKERS[j],facecolors='none',edgecolors=COLORS[j],alpha=.6,label=src)
            allv=[r[k] for r in group for k in ['L_delta_USD','Q_delta_USD']];a,b=min(allv),max(allv)
            ax.plot([a,b],[a,b],'--',color='.5',lw=.6);ax.set_title(short(cid));ax.set_xlabel('Price score relative\nto M0 (USD)')
        axes[0].set_ylabel('Assessment Q relative to M0 (USD)');axes[-1].legend(loc='best',title='State source')
        save('J02_price_Q_ordering',fig,data,'Frozen candidate subset; point rankings contain MC error. Reporting L excludes the explicit solver tie-break.')

        data=[r for r in read_json(OUTPUT/'processed/action_score_residuals.json') if r['stage']=='formal' and r['comparison']=='select32-price']
        divergent=[r for r in data if r['releases_left']!=r['releases_right']]
        if divergent:
            for r in divergent:
                ca=Counter((x['time'],x['hub']) for x in r['releases_left']);cb=Counter((x['time'],x['hub']) for x in r['releases_right'])
                r['release_L1_count']=sum(abs(ca[k]-cb[k]) for k in ca.keys()|cb.keys())
                r['terminal_release_difference']=sum(x['terminal'] for x in r['releases_left'])-sum(x['terminal'] for x in r['releases_right'])
            fig,axes=plt.subplots(1,3,figsize=(7.16,3.6))
            for j,cid in enumerate(sorted({r['condition'] for r in divergent})):
                g=[r for r in divergent if r['condition']==cid]
                axes[0].scatter([r['g_difference_USD'] for r in g],[r['bonus_difference_USD'] for r in g],s=15,
                    marker=MARKERS[j],facecolors='none',edgecolors=COLORS[j],label=short(cid))
                axes[1].scatter([r['release_L1_count'] for r in g],[r['Q_difference_USD'] for r in g],s=15,marker=MARKERS[j],facecolors='none',edgecolors=COLORS[j])
                axes[2].scatter([r['L_difference_USD'] for r in g],[r['mixed_residual_USD'] for r in g],s=15,marker=MARKERS[j],facecolors='none',edgecolors=COLORS[j])
            for ax in axes:ax.axhline(0,color='.5',lw=.6)
            axes[0].set(xlabel='Booked g difference (USD)',ylabel='Slot bonus difference (USD)')
            axes[1].set(xlabel='Release group L1 difference',ylabel='Assessment gain (USD)')
            axes[2].set(xlabel='Price L difference (USD)',ylabel='Mixed Q-minus-L residual (USD)')
            axes[0].legend(loc='best')
            save('J03_release_mechanisms',fig,divergent,'All formal release-divergent pairs retained; no profit-based case selection. Residual is not pure nonseparability.')
        else:missing.append(dict(group='J03_release_mechanisms',reason='No formal release-divergent pairs'))

        interaction=read_json(OUTPUT/'processed/resource_interactions.json') if (OUTPUT/'processed/resource_interactions.json').exists() else []
        groups=defaultdict(list)
        for r in interaction:
            if r['stage']=='formal':groups[r['condition'],r['source'],r['scenario_seed']].append(r['mean_USD'])
        data=[dict(condition=c,source=s,scenario_seed=k,interaction_USD=float(np.mean(v)),absolute_interaction_USD=float(np.mean(np.abs(v)))) for (c,s,k),v in groups.items()]
        if data:
            fig,axes=plt.subplots(1,2,figsize=(7.16,3.65));summary=[]
            for i,(cid,src) in enumerate(sorted({(r['condition'],r['source']) for r in data})):
                vals=[r['interaction_USD'] for r in data if (r['condition'],r['source'])==(cid,src)]
                st=statistics(vals);summary.append(dict(condition=cid,source=src,**st));interval(axes[0],i,st,COLORS[i%2],MARKERS[i%2])
                aa=[r['absolute_interaction_USD'] for r in data if (r['condition'],r['source'])==(cid,src)]
                interval(axes[1],i,statistics(aa),COLORS[i%2],MARKERS[i%2])
            for ax in axes:ax.set_yticks(range(len(summary)),[short(r['condition'])+'/'+r['source'] for r in summary]);ax.axvline(0,color='.5',ls='--',lw=.6)
            axes[0].set_xlabel('Mean interaction (USD)');axes[1].set_xlabel('Mean absolute interaction (USD)')
            save('J04_resource_interactions',fig,data,'Scenario averages of eligible preselected current-free-resource pairs; unadjusted t95. Cross-time injection not implemented.')
        else:missing.append(dict(group='J04_resource_interactions',reason='No eligible formal interaction pairs'))

        data=result['bank_statistics'];fig,axes=plt.subplots(1,3,figsize=(7.16,3.65))
        for ax,cid in zip(axes,sorted({r['condition'] for r in data})):
            for r in data:
                if r['condition']==cid:interval(ax,r['bank'],r,COLORS[r['bank']-1],MARKERS[r['bank']-1],vertical=True)
            ax.axhline(0,color='.5',ls='--',lw=.6);ax.set_xticks(range(1,5));ax.set_xlabel('Construction bank');ax.set_title(short(cid))
        axes[0].set_ylabel('Paired deployment gain over M0 (USD)')
        save('J05_independent_banks',fig,data,'Each bank: 60 construction scenarios and the SAME 40 deployment scenarios. Conditional t95; four banks are descriptive, not n=160.')

        runtime=read_json(OUTPUT/'processed/runtime.json');data=[]
        stages=['pilot','diagnostic','formal']
        for stage in stages:
            r=read_json(OUTPUT/'manifests'/f'{stage}_joint.json');data.append(dict(category=stage+' MC pipeline',seconds=r['wall_seconds'],scope='wall inclusive',overlap='includes replay/solver/I/O'))
            r=read_json(OUTPUT/'manifests'/f'{stage}_actions.json');data.append(dict(category=stage+' candidates',seconds=r['wall_seconds'],scope='wall',overlap='includes candidate solves'))
        banks=read_json(OUTPUT/'manifests/banks_tables.json')
        data.extend(dict(category='bank'+str(r['bank'])+' construction',seconds=r['wall_seconds'],scope='wall inclusive',overlap='includes fitting/replay/storage') for r in banks['runtime'])
        d=read_json(OUTPUT/'manifests/banks_deployment.json');data.append(dict(category='600 deployment runs',seconds=d['wall_seconds'],scope='wall inclusive',overlap='includes replay/solver/storage'))
        fig,axes=plt.subplots(1,2,figsize=(7.16,4.7))
        axes[0].barh(range(len(data)),[r['seconds']/3600 for r in data],color=COLORS[0]);axes[0].set_yticks(range(len(data)),[r['category'] for r in data]);axes[0].set_xlabel('Stage wall time (h)');axes[0].invert_yaxis()
        metrics=['pipeline_inclusive_seconds','solver_seconds','replay_seconds']
        vals=[np.median([r[k] for r in runtime if r['stage']=='formal']) for k in metrics]
        axes[1].bar(range(3),vals,color=COLORS[:3]);axes[1].set_xticks(range(3),['Pipeline\n(inclusive)','Solver\n(nested)','Replay\n(nested)']);axes[1].set_ylabel('Median formal state worker time (s)')
        fig.subplots_adjust(left=.21,wspace=.65)
        save('J06_runtime',fig,data,'MC is offline diagnostic, NOT a real-time latency claim. Inclusive pipeline, solver and replay must not be added together.')
    files={p.name:sha(p) for p in directory.iterdir() if p.is_file() and p.name!='figure_manifest.json'}
    atomic_json(directory/'figure_manifest.json',dict(created_at=now(),names=names,atlas=atlas.name,missing_groups=missing,
        expected_groups=6,sha256=files,formats=['PDF','SVG','PNG600dpi','source CSV','config JSON']))
    return names

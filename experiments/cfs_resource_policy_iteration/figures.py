"""Sixteen vector-first publication figures with inspectable source CSV."""
import logging,textwrap
logging.getLogger('fontTools').setLevel(logging.WARNING)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from experiment_core.storage import read_json,atomic_json
from .common import OUTPUT,key,write_csv,file_hash

COLORS=['#0072B2','#D55E00','#009E73','#CC79A7','#000000']
STYLES=['o-','s--','^-.','D:','x-']
LABELS=['CA80/3','W40/8','W80/3','W20/8']
plt.rcParams.update({'font.family':'DejaVu Serif','mathtext.fontset':'stix','font.size':9,
    'axes.spines.top':False,'axes.spines.right':False,'axes.labelsize':9,'axes.titlesize':10,
    'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,
    'svg.fonttype':'none','savefig.dpi':600,'axes.linewidth':.7,'lines.linewidth':1.2})

def facets(cfg):
    fig,axes=plt.subplots(2,2,figsize=(7.2,4.8))
    for ax,name in zip(axes.flat,LABELS):ax.set_title(name)
    return fig,list(axes.flat)

def save(fig,name,rows,config,atlas,note):
    fig.tight_layout();fig.savefig(OUTPUT/'figures'/f'{name}.png',dpi=600,bbox_inches='tight')
    fig.savefig(OUTPUT/'figures'/f'{name}.pdf',bbox_inches='tight');fig.savefig(OUTPUT/'figures'/f'{name}.svg',bbox_inches='tight')
    source=f'figures/source_csv/{name}.csv';write_csv(OUTPUT/source,rows)
    axes_config=[]
    for ax in fig.axes:
        axes_config.append(dict(title=ax.get_title(),xlabel=ax.get_xlabel(),ylabel=ax.get_ylabel(),
            xscale=ax.get_xscale(),yscale=ax.get_yscale(),xlim=list(ax.get_xlim()),ylim=list(ax.get_ylim()),
            xticks=list(map(float,ax.get_xticks())),yticks=list(map(float,ax.get_yticks())),
            xticklabels=[t.get_text() for t in ax.get_xticklabels()],yticklabels=[t.get_text() for t in ax.get_yticklabels()],
            legend=[] if ax.get_legend() is None else [t.get_text() for t in ax.get_legend().get_texts()],
            scalar_color_maps=[dict(cmap=c.cmap.name,vmin=c.norm.vmin,vmax=c.norm.vmax) for c in ax.collections if c.get_array() is not None]))
    per=dict(source=source,source_sha256=file_hash(OUTPUT/source),note=note,atlas_page=len(config['figures'])+1,
        formats=['PNG600dpi','vector PDF embedded fonts','SVG'],
        size_inches=list(fig.get_size_inches()),dpi=600,bbox_inches='tight',axes=axes_config,
        colors=COLORS,line_marker_styles=STYLES,font='DejaVu Serif/STIX',matplotlib_version=matplotlib.__version__,numpy_version=np.__version__)
    config['figures'][name]=per;atomic_json(OUTPUT/'figures/plot_config'/f'{name}.json',per)
    # Individual plots stay caption-free for journal manuscripts; the local
    # atlas labels every plate and its statistical meaning explicitly.
    fig.text(.01,1.025,f"{per['atlas_page']:02d}  {name.replace('_',' ').title()}",ha='left',va='bottom',fontsize=10,fontweight='bold')
    fig.text(.01,-.025,textwrap.fill(note,width=105),ha='left',va='top',fontsize=7.5)
    atlas.savefig(fig,bbox_inches='tight');plt.close(fig)

def main():
    cfg=read_json(OUTPUT/'config/main.json');stats=read_json(OUTPUT/'processed/policy_statistics.json')
    tab=read_json(OUTPUT/'processed/table_diagnostics.json');states=read_json(OUTPUT/'processed/state_diagnostics.json')
    actions=read_json(OUTPUT/'processed/action_summary.json');formalactions=read_json(OUTPUT/'processed/formal_action_summary.json')
    runtime=read_json(OUTPUT/'processed/runtime_breakdown.json')
    for p in ['source_csv','plot_config']:(OUTPUT/'figures'/p).mkdir(exist_ok=True)
    config=dict(engine='Matplotlib vector backend, decoupled from experiment storage',dpi=600,font='DejaVu Serif/STIX',
        colorblind=True,black_white='distinct markers and line styles; Cividis heatmaps with monotone luminance and labeled color scale',
        column_inches=[3.5,7.2],uncertainty='40 scenario-paired t95 for gains; 20 scenario-equal daily rates for actions; table/state descriptives',figures={})
    with PdfPages(OUTPUT/'figures/figure_atlas.pdf') as atlas:
        fig,axes=facets(cfg);source=[]
        for ax,c in zip(axes,cfg['conditions']):
            rr=[next(r for r in stats['summary'] if r['condition']==key(c) and r['policy']==f'P{n}') for n in range(5)]
            y=np.array([r['mean'] for r in rr])/1000;err=np.array([r['CI95_upper']-r['mean'] for r in rr])/1000
            ax.errorbar(range(5),y,yerr=err,fmt='o-',capsize=3,color=COLORS[0]);ax.set_xticks(range(5));ax.set(xlabel='Policy iteration',ylabel='Realized profit ($1000)');source+=rr
        save(fig,'profit_by_iteration',source,config,atlas,'n=40 per policy; mean t95. Mean order is not significance.')
        fig,axes=facets(cfg);source=[]
        for ax,c in zip(axes,cfg['conditions']):
            rr=[next(r for r in stats['comparisons'] if r['condition']==key(c) and r['comparison']==f'P{n}-P{n-1}') for n in range(1,5)]
            y=np.array([r['mean'] for r in rr])/1000;err=np.array([r['CI95_upper']-r['mean'] for r in rr])/1000
            ax.errorbar(range(1,5),y,yerr=err,fmt='o-',capsize=3,color=COLORS[1]);ax.axhline(0,color='black',ls='--',lw=.7)
            ax.set_xticks(range(1,5));ax.set(xlabel='Improvement step',ylabel='Paired adjacent gain ($1000)');source+=rr
        save(fig,'paired_profit_gain_by_iteration',source,config,atlas,'Same 40 scenarios; unadjusted t95. Holm p saved in source table.')
        for name,field,label in [('table_rmse_change_by_iteration','RMSE','RMSE of table change (USD)'),
            ('table_relative_change_by_iteration','relative_L2','Relative L2 change'),
            ('table_spearman_by_iteration','Spearman','Spearman correlation'),
            ('table_sign_flip_by_iteration','sign_flip_rate','Three-way sign change rate')]:
            fig,axes=facets(cfg);source=[]
            for ax,c in zip(axes,cfg['conditions']):
                rr=[r for r in tab['changes'] if r['condition']==key(c) and r['scope']=='observed_shared' and r['segment']=='all']
                ax.plot([r['to_V'] for r in rr],[np.nan if r[field] is None else r[field] for r in rr],'o-',color=COLORS[0]);source+=rr
                ax.set_xticks([1,2,3],['V1-V0','V2-V1','V3-V2']);ax.set(ylabel=label)
                if field in ['Spearman','sign_flip_rate']:ax.set_ylim((-1.05,1.05) if field=='Spearman' else (0,1))
            save(fig,name,source,config,atlas,'Shared observed cells only; no terminal/unsupported q. Includes changing on-policy state mix.')
        fig,axes=facets(cfg);source=[]
        for ax,c in zip(axes,cfg['conditions']):
            for pool,stage,color,fmt in [(actions,'diagnostic n=20',COLORS[2],'o-'),(formalactions,'formal n=40',COLORS[1],'s--')]:
                rr=sorted([r for r in pool if r['condition']==key(c)],key=lambda r:r['iteration'])
                y=np.array([r['mean'] for r in rr]);err=np.array([r['CI95_upper']-r['mean'] for r in rr])
                ax.errorbar(range(1,5),y,yerr=err,fmt=fmt,capsize=3,color=color,label=stage)
                source+=[r|dict(stage=stage) for r in rr]
            ax.set_ylim(-.05,1.05);ax.set_xticks(range(1,5))
            ax.set(xlabel='Policy improvement step',ylabel='Same-state action disagreement')
        axes[0].legend(frameon=False)
        save(fig,'policy_action_disagreement_by_iteration',source,config,atlas,'All t in fixed pi0 banks, diagnostic n20 and formal n40 independent scenario-level fractions. Formal is corroboration only, not tuning; not own-trajectory differences.')
        fig,axes=facets(cfg);source=[]
        for ax,c in zip(axes,cfg['conditions']):
            for f,color,fmt in zip(['idle','backlog','incoming'],COLORS,STYLES):
                values=[]
                for n in range(4):
                    rr=[r for r in states['shifts'] if r['condition']==key(c) and r['from_pi']==n and r['feature']==f]
                    valid=[r['absolute_SMD'] for r in rr if r['absolute_SMD'] is not None]
                    mean=float(np.mean(valid));values.append(mean);source.append(dict(condition=key(c),step=n+1,feature=f,mean_absolute_SMD=mean,
                        cells=len(rr),valid_SMD_cells=len(valid),undefined_SMD_cells=len(rr)-len(valid)))
                ax.plot(range(1,5),values,fmt,c=color,label=f)
            ax.set_xticks(range(1,5));ax.set(xlabel='Policy improvement step',ylabel='Mean absolute standardized shift')
        axes[0].legend(frameon=False)
        save(fig,'state_distribution_shift_by_iteration',source,config,atlas,'20 scenarios per distribution; equal hub/time cells, descriptive SMD, undefined constant-denominator cells kept separate.')
        models={(n,key(c)):read_json(OUTPUT/'tables'/f'V{n}'/(key(c)+'.json')) for n in range(4) for c in cfg['conditions']}
        for n in range(4):
            fig,axes=facets(cfg);source=[]
            for ax,c in zip(axes,cfg['conditions']):
                cid=key(c);m=models[n,cid];hubs=sorted(m['values']);times=m['values'][hubs[0]]['1']['times']
                matrix=np.array([m['values'][h]['1']['values'] for h in hubs])/1000
                bound=max(abs(v) for nn in range(4) for h in hubs for v in models[nn,cid]['values'][h]['1']['values'])/1000
                # pcolormesh remains vector polygons in PDF/SVG, unlike imshow.
                im=ax.pcolormesh(np.arange(len(times)+1)-.5,np.arange(len(hubs)+1)-.5,matrix,
                    shading='flat',cmap='cividis',vmin=-max(bound,1),vmax=max(bound,1),rasterized=False)
                ax.set_xlim(-.5,len(times)-.5);ax.set_ylim(len(hubs)-.5,-.5)
                ax.set_xticks(range(len(times)),times);ax.set_yticks(range(len(hubs)),hubs,fontsize=7);ax.set(xlabel='Time',ylabel='Hub')
                colorbar=fig.colorbar(im,ax=ax,label='q=1 value ($1000)',fraction=.046,pad=.04)
                # Matplotlib normally rasterizes a dense colorbar even when the
                # heatmap mesh is vector; keep both entirely vector here.
                if colorbar.solids is not None:colorbar.solids.set_rasterized(False)
                source.extend(dict(condition=cid,iteration=n,time=t,hub=h,q=1,value_USD=m['values'][h]['1']['values'][i]) for h in hubs for i,t in enumerate(times))
            save(fig,f'hub_time_value_heatmap_V{n}',source,config,atlas,'Absolute q1 resource price; fixed per-condition color scale across V0-V3; terminal free resource zero.')
        fig,axes=facets(cfg);source=[]
        for ax,c in zip(axes,cfg['conditions']):
            cid=key(c);hubs=sorted(models[0,cid]['values']);times=models[0,cid]['values'][hubs[0]]['1']['times'];t=min(times,key=lambda t:abs(t-12));i=times.index(t)
            for n in range(4):
                yy=[np.mean([models[n,cid]['values'][h][str(q)]['values'][i] for h in hubs])/1000 for q in range(1,c['fleet']+1)]
                ax.plot(range(1,c['fleet']+1),yy,STYLES[n],c=COLORS[n],label=f'V{n}')
                source.extend(dict(condition=cid,iteration=n,time=t,q=q,value_USD=y*1000,aggregation='equal-hub mean') for q,y in enumerate(yy,1))
            ax.set(xlabel='Free resource slot q',ylabel='Mean hub marginal at t=12 ($1000)');ax.set_xticks(range(1,c['fleet']+1))
        axes[0].legend(frameon=False,ncol=2)
        save(fig,'q_marginal_curves_by_iteration',source,config,atlas,'No monotonic enforcement; equal-hub mean is visualization only, cell-level audits stored separately.')
        fig,axes=facets(cfg);fig.set_size_inches(7.2,6.5);source=[]
        for ax,c in zip(axes,cfg['conditions']):
            rr=sorted([r for r in tab['oscillations'] if r['condition']==key(c)],key=lambda r:(-r['range_USD'],r['hub'],r['time'],r['q']))[:5]
            for i,r in enumerate(rr):
                ax.plot(range(4),[r[f'V{n}']/1000 for n in range(4)],STYLES[i],c=COLORS[i],label=f"{r['hub']},t{r['time']},q{r['q']}");source.append(r)
            ax.set_xticks(range(4));ax.axhline(0,c='black',ls='--',lw=.7);ax.set(xlabel='Table iteration',ylabel='Resource marginal ($1000)')
            ax.legend(frameon=False,fontsize=6,loc='upper center',bbox_to_anchor=(.5,-.30),ncol=2,columnspacing=.8,handlelength=2)
        save(fig,'oscillation_cells',source,config,atlas,'Five largest descriptive ranges per condition; not a selected statistical test, all cells saved.')
        fig,axes=facets(cfg);source=[]
        for ax,c in zip(axes,cfg['conditions']):
            rr=[next(r for r in stats['summary'] if r['condition']==key(c) and r['policy']==p) for p in ['P4','R0','R1','D']]
            y=np.array([r['mean'] for r in rr])/1000;err=np.array([r['CI95_upper']-r['mean'] for r in rr])/1000
            ax.bar(range(4),y,yerr=err,capsize=3,color=COLORS[:4],edgecolor='black',lw=.5,hatch=['','//','xx','..'])
            ax.set_xticks(range(4),['pi4','RH4','RH8','Preview H8'],rotation=15);ax.set(ylabel='Realized profit ($1000)');source+=rr
        save(fig,'final_policy_vs_RH_reference',source,config,atlas,'Mean t95, n40; preview is nondeployable information reference, not certified optimum.')
        fig,axes=plt.subplots(1,2,figsize=(7.2,3.0));source=[]
        offline=[r for r in runtime if r['stage']=='offline_evaluation'];axes[0].bar(range(4),[r['wall_seconds']/60 for r in offline],color=COLORS[0],edgecolor='black')
        axes[0].set_xticks(range(4),['E(pi0)','E(pi1)','E(pi2)','E(pi3)']);axes[0].set(ylabel='Offline wall time (min)',title='Three-worker pipeline');source+=offline
        for i,c in enumerate(cfg['conditions']):
            rr=[r for r in stats['summary'] if r['condition']==key(c) and r['policy'].startswith('P')]
            rr.sort(key=lambda r:r['policy']);axes[1].plot(range(5),[r['mean_online_seconds'] for r in rr],STYLES[i],c=COLORS[i],label=LABELS[i]);source+=rr
        axes[1].set_xticks(range(5));axes[1].set(xlabel='Policy iteration',ylabel='Mean online day (s)',title='Frozen slot-price deployment');axes[1].legend(frameon=False,fontsize=7)
        save(fig,'runtime_by_iteration',source,config,atlas,'Offline wall excludes separately reported final aggregation; online excludes replay/storage. Nested timers not added.')
    assert len(config['figures'])==16
    atomic_json(OUTPUT/'figures/plotting_config.json',config);print('16 figure groups and vector atlas saved.',flush=True)

if __name__=='__main__':main()

"""Eleven task figures, vector atlas and reproducible source/config artifacts."""
import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from experiment_core.storage import read_json,atomic_json
from experiments.cfs_resource_policy_iteration_damping.figures import style,short,csvrows
from .common import OUTPUT,key,write_csv,file_hash
from .policies import LABELS

NAMES=['B_01_pressure_ratio_distribution','B_02_bucket_thresholds','B_03_bucket_occupancy',
    'B_04_value_by_bucket_heatmap','B_05_same_tuq_bucket_contrast','B_06_within_cell_variance_reduction',
    'B_07_base_vs_bucket_profit','B_08_paired_profit_gain','B_09_action_disagreement','B_10_fallback_rate','B_11_runtime']
COLORS=['#0072B2','#009E73','#D55E00'];HATCHES=['///','...','xxx']
POLICY_COLORS=['#666666','#0072B2','#009E73','#D55E00','#CC79A7','#E69F00']

def generate(cfg):
    style();out=OUTPUT/'figures';out.mkdir(exist_ok=True)
    cids=[key(c) for c in cfg['conditions']];priors=read_json(OUTPUT/'tables/priors.json')
    pressure=csvrows(OUTPUT/'processed/pressure_construction.csv')
    for r in pressure:r['R']=float(r['R'])
    support=csvrows(OUTPUT/'processed/support_audit.csv')
    contrasts=read_json(OUTPUT/'processed/bucket_contrast.json')
    occupancy=read_json(OUTPUT/'processed/formal_bucket_occupancy.json')
    variance=read_json(OUTPUT/'processed/within_cell_variance.json')
    corroboration=read_json(OUTPUT/'processed/variance_corroboration.json')
    stats=read_json(OUTPUT/'processed/formal_statistics.json')
    actions=read_json(OUTPUT/'processed/formal_action_summary.json')
    fallback=read_json(OUTPUT/'processed/formal_fallback_summary.json')
    runtime=read_json(OUTPUT/'processed/formal_runtime.json')
    atlas=PdfPages(out/'B_figure_atlas.pdf');saved=[]
    def export(fig,name,data,meta):
        # Solve layout once. Repeated PDF/SVG/PNG draws otherwise reflow axes,
        # making atlas and standalone pages needlessly different.
        fig.canvas.draw()
        fig.set_layout_engine('none')
        src=out/(name+'.source.csv');write_csv(src,data)
        atomic_json(out/(name+'.plot.json'),dict(schema_version='scientific-plot/v1',figure=name,
            backend='matplotlib',source_file=src.name,source_sha256=file_hash(src),
            width_inches=fig.get_figwidth(),height_inches=fig.get_figheight(),dpi=600,
            font='DejaVu Serif / STIX',palette=COLORS,black_white_discrimination='hatch/marker/line style',
            font_embedding='PDF TrueType Type42',**meta))
        for ext in ['pdf','svg','png']:fig.savefig(out/(name+'.'+ext),dpi=600)
        atlas.savefig(fig);saved.append(name);plt.close(fig)
    def four(height=5.4):return plt.subplots(2,2,figsize=(7.2,height),layout='constrained')
    fig,axs=four()
    for cid,ax in zip(cids,axs.flat):
        rr=[r['R'] for r in pressure if r['condition']==cid]
        ax.hist(rr,bins=30,density=True,color=COLORS[0],edgecolor='white',linewidth=.3)
        for cut,linestyle in [('c33','--'),('c67',':')]:ax.axvline(priors[cid]['thresholds'][cut],color='black',linestyle=linestyle,label=cut)
        ax.set(title=short(cid),xlabel='Construction pressure ratio R',ylabel='Density');ax.legend(frameon=False)
    export(fig,NAMES[0],pressure,dict(source='construction M0 original pre-injection states only; not independent inferential samples',bins=30))
    fig,ax=plt.subplots(figsize=(7.2,3.4),layout='constrained');cuts=[]
    for i,cut in enumerate(['c33','c67']):
        vals=[priors[cid]['thresholds'][cut] for cid in cids]
        ax.bar(np.arange(4)+(i-.5)*.3,vals,width=.3,color=COLORS[i],hatch=HATCHES[i],edgecolor='black',linewidth=.4,label=cut)
        cuts.extend(dict(condition=cid,threshold=cut,value=v,tied_thresholds=priors[cid]['thresholds']['tied_thresholds']) for cid,v in zip(cids,vals))
    ax.set(xticks=range(4),xticklabels=[short(cid) for cid in cids],ylabel='Frozen pressure threshold',xlabel='Condition');ax.legend(frameon=False)
    export(fig,NAMES[1],cuts,dict(quantiles=[1/3,2/3],source='construction only; ties retained, balanced may be empty'))
    fig,axs=four()
    for cid,ax in zip(cids,axs.flat):
        bottom=np.zeros(6)
        for b in range(3):
            vals=np.array([next(r for r in occupancy if r['condition']==cid and r['policy']==p and r['bucket']==b)['mean'] for p in LABELS])
            ax.bar(range(6),vals,bottom=bottom,color=COLORS[b],hatch=HATCHES[b],edgecolor='white',linewidth=.4,label=['surplus','balanced','scarce'][b]);bottom+=vals
        ax.set(title=short(cid),xticks=range(6),xticklabels=LABELS,ylabel='Formal mean time/hub occupancy',ylim=(0,1))
    handles,labels=axs.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside upper center',ncol=3,frameon=False,fontsize=7)
    export(fig,NAMES[2],occupancy,dict(n=40,unit='scenario; equal 24 periods x5 hubs within scenario; source t95 retained'))
    fig,axs=plt.subplots(4,3,figsize=(7.2,10.2),layout='constrained');data=[]
    fig.suptitle('Conditional prices (x = base fallback; circle = negative)',fontsize=9)
    for row,cid in enumerate(cids):
        rr=[r for r in contrasts if r['condition']==cid];times=sorted({r['time'] for r in rr});hq=sorted({(r['hub'],r['q']) for r in rr})
        vv=[r['deployed_value_USD'] for r in rr];lim=max(abs(min(vv)),abs(max(vv)),1.)
        for b in range(3):
            ax=axs[row,b];lookup={(r['hub'],r['q'],r['time']):r for r in rr if r['bucket']==b}
            mm=np.array([[lookup[h,q,t]['deployed_value_USD'] for t in times] for h,q in hq])
            mesh=ax.pcolormesh(np.arange(len(times)+1),np.arange(len(hq)+1),mm,cmap='BrBG',vmin=-lim,vmax=lim,rasterized=False)
            ax.set(title=f'{short(cid)}, b={b}',xlabel='Time node',ylabel='Hub / q block' if b==0 else '')
            ax.set_xticks(np.arange(len(times))+.5,times)
            hubs=sorted({h for h,q in hq});ticks=[]
            for h in hubs:ticks.append(np.mean([i+.5 for i,(hh,q) in enumerate(hq) if hh==h]))
            # Expose the q ordering instead of requiring the source CSV to
            # decode an otherwise unlabeled hub/q block.
            qlabels=[h+'\nq='+','.join(str(q) for hh,q in hq if hh==h) for h in hubs]
            ax.set_yticks(ticks,qlabels,fontsize=5.5)
            if b>0:ax.tick_params(axis='y',labelleft=False)
            # Every fallback cell is marked, rather than portrayed as supported.
            for yi,(h,q) in enumerate(hq):
                for xi,t in enumerate(times):
                    if lookup[h,q,t]['fallback']:ax.text(xi+.5,yi+.5,'x',ha='center',va='center',fontsize=4,color='black')
                    if lookup[h,q,t]['deployed_value_USD']<0:
                        ax.plot(xi+.5,yi+.5,marker='o',markersize=1.8,markerfacecolor='none',
                            markeredgecolor='black',markeredgewidth=.4,linestyle='none')
        cb=fig.colorbar(mesh,ax=list(axs[row,:]),label='USD',fraction=.025,pad=.015)
        cb.solids.set_rasterized(False);cb.ax.tick_params(labelsize=7)
        data.extend(rr)
    export(fig,NAMES[3],data,dict(values='actual conditional lookup or explicit SAME base fallback; x=unsupported',
        sign_encoding='unfilled circle denotes negative, so sign is not color-only in grayscale',
        scale='same symmetric USD limits across three buckets within condition',q_order='ascending within each hub block'))
    fig,axs=four(6.4);data=[];chosen_keys=set()
    for cid,ax in zip(cids,axs.flat):
        gg={}
        for r in contrasts:
            if r['condition']==cid:gg.setdefault((r['time'],r['hub'],r['q']),[]).append(r)
        eligible=[rr for rr in gg.values() if all(r['sample_count']>=10 for r in rr)]
        chosen=sorted(eligible,key=lambda rr:max(r['raw_conditional_mean_USD'] for r in rr)-min(r['raw_conditional_mean_USD'] for r in rr),reverse=True)[:3]
        for i,rr in enumerate(chosen):
            rr=sorted(rr,key=lambda r:r['bucket']);r=rr[0]
            ax.errorbar(range(3),[r['raw_conditional_mean_USD'] for r in rr],yerr=[2*r['MC_SE'] for r in rr],
                color=COLORS[i],marker=['o','s','^'][i],linestyle=['-','--',':'][i],capsize=2,label=f't={r["time"]}, {r["hub"]}, q={r["q"]}')
            data.extend(rr)
            chosen_keys.add((cid,r['time'],r['hub'],r['q']))
        if not chosen:ax.text(.5,.5,'No cell with all three buckets n>=10',transform=ax.transAxes,ha='center',wrap=True)
        ax.set(title=short(cid),xticks=range(3),xticklabels=['surplus','balanced','scarce'],ylabel='Raw conditional marginal value (USD)')
        if chosen:ax.legend(frameon=False,fontsize=7)
    # Preserve even unsupported/unselected rows, including the no-eligible-cell
    # case; a reproducible CSV must not silently become an empty source file.
    data=[r|dict(plot_selected=(r['condition'],r['time'],r['hub'],r['q']) in chosen_keys) for r in contrasts]
    export(fig,NAMES[4],data,dict(selection='three largest construction bucket contrasts with ALL three n>=10 per condition; plot_selected marks drawn cells',
        uncertainty='plus/minus 2 conditional MC_SE, not formal confidence interval; selected descriptive extremes',monotonicity='not enforced'))
    fig,axs=four();data=[]
    for cid,ax in zip(cids,axs.flat):
        rr=[r for r in variance if r['condition']==cid];held=next(r for r in corroboration if r['condition']==cid)
        vals=[np.mean([r['unconditional_variance_USD2'] for r in rr]),np.mean([r['weighted_within_bucket_variance_USD2'] for r in rr]),held['base_MSE_USD2'],held['bucket_MSE_USD2']]
        ax.bar(range(4),np.array(vals)/1e6,color=[COLORS[0],COLORS[1],COLORS[0],COLORS[1]],hatch='///',edgecolor='black',linewidth=.4)
        ax.set(title=short(cid),xticks=range(4),xticklabels=['Train\nbase var','Train\nwithin var','Held-out\nbase MSE','Held-out\nbucket MSE'],ylabel='Squared USD (millions)')
        data.extend(dict(condition=cid,quantity=k,value_USD2=v,heldout_corroborated=held['corroborated']) for k,v in zip(['train_base_variance','train_within_variance','heldout_base_MSE','heldout_bucket_MSE'],vals))
    export(fig,NAMES[5],data,dict(train='descriptive; weighted within variance decreases mechanically',
        heldout='frozen construction predictions on 20 independent diagnostic scenarios; paired MSE inference in processed/variance_corroboration.csv',cell_weight='equal same-tuq native cells'))
    fig,axs=four()
    for cid,ax in zip(cids,axs.flat):
        rr=[next(r for r in stats['summary'] if r['condition']==cid and r['policy']==p) for p in LABELS];y=np.array([r['mean'] for r in rr])
        ax.bar(range(6),y/1000,color=POLICY_COLORS,edgecolor='black',linewidth=.4,hatch='///')
        ax.errorbar(range(6),y/1000,yerr=[(y-[r['CI95_lower'] for r in rr])/1000,([r['CI95_upper'] for r in rr]-y)/1000],fmt='none',color='black',capsize=2)
        ax.set(title=short(cid),xticks=range(6),xticklabels=LABELS,ylabel='Mean formal realized profit (1000 USD)')
    export(fig,NAMES[6],stats['summary'],dict(n=40,interval='marginal scenario t95, paired contrasts separate',preview='D is PreviewH8, NOT full-day bound'))
    fig,ax=plt.subplots(figsize=(7.2,3.6),layout='constrained');data=[]
    for i,cid in enumerate(cids):
        r=next(r for r in stats['paired'] if r['condition']==cid and r['comparison']=='B2-B1');data.append(r)
        ax.plot([r['bootstrap95_lower']/1000,r['bootstrap95_upper']/1000],[i+.08,i+.08],color=COLORS[1],linestyle='--',linewidth=2)
        ax.errorbar(r['mean']/1000,i,xerr=[[max(0,r['mean']-r['CI95_lower'])/1000],[max(0,r['CI95_upper']-r['mean'])/1000]],fmt='o',color=COLORS[0],capsize=3)
    ax.axvline(0,color='black',linewidth=.8);ax.set(yticks=range(4),yticklabels=[short(c) for c in cids],xlabel='Paired B2 - B1 profit (1000 USD)')
    ax.plot([],[],color=COLORS[0],marker='o',label='paired t95');ax.plot([],[],color=COLORS[1],linestyle='--',label='paired bootstrap95');ax.legend(frameon=False)
    export(fig,NAMES[7],data,dict(n=40,bootstrap=10000,interval='unadjusted; Holm p across4 in source',unit='same frozen scenarios'))
    fig,axs=four();fields=['full_action_disagree','routes_disagree','assignments_disagree','release_hub_disagree','wait_reposition_disagree']
    for cid,ax in zip(cids,axs.flat):
        for i,comparison in enumerate(['B1-B0','B2-B1']):
            rr=[next(r for r in actions if r['condition']==cid and r['comparison']==comparison and r['component']==f) for f in fields]
            ax.bar(np.arange(5)+(i-.5)*.35,[r['mean'] for r in rr],width=.35,color=COLORS[i],hatch=HATCHES[i],edgecolor='black',linewidth=.4,label=comparison)
        ax.set(title=short(cid),xticks=range(5),xticklabels=['Full','Route','Assign','Release','Wait'],ylabel='Same-state disagreement fraction',ylim=(0,1))
    axs.flat[0].legend(frameon=False,fontsize=7)
    export(fig,NAMES[8],actions,dict(reference='B0 exact full state, both sides separately solve; all24periods; n40 scenario fractions',canonicalization='exact, no rounding or ID quotient'))
    fig,ax=plt.subplots(figsize=(7.2,3.5),layout='constrained');means=np.array([r['mean'] for r in fallback])
    ax.bar(range(4),means,color=COLORS[1],hatch='///',edgecolor='black',linewidth=.4)
    ax.errorbar(range(4),means,yerr=[means-[r['CI95_lower'] for r in fallback],[r['CI95_upper'] for r in fallback]-means],fmt='none',color='black',capsize=3)
    ax.axhline(.5,color='black',linestyle='--',label='preregistered 0.5 ceiling')
    low=min(-.03,min(r['CI95_lower'] for r in fallback)-.02);high=max(1.03,max(r['CI95_upper'] for r in fallback)+.02)
    ax.set(xticks=range(4),xticklabels=[short(r['condition']) for r in fallback],ylabel='Mean daily B2 query fallback fraction',ylim=(low,high));ax.legend(frameon=False)
    export(fig,NAMES[9],fallback,dict(denominator='ALL pre-solve coefficient queries, including terminal queries; NOT only selected slots',
        active_rate='nonterminal pooled rate separately retained in source; CI=scenario marginal t95'))
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.7),layout='constrained');data=[]
    for cid in cids:
        rr=[r for r in runtime if r['condition']==cid and r['policy']=='B2'];data.append(dict(kind='online',condition=cid,policy='B2',seconds=float(np.mean([r['online_seconds'] for r in rr]))))
    axs[0].bar(range(4),[r['seconds'] for r in data],color=COLORS[1],hatch='///',edgecolor='black',linewidth=.4)
    axs[0].set(xticks=range(4),xticklabels=[short(c) for c in cids],ylabel='Mean formal B2 online seconds',title='Online day')
    for stage in ['construction','diagnostic']:
        r=read_json(OUTPUT/'manifests'/f'{stage}_target_manifest.json');data.append(dict(kind='offline',stage=stage,seconds=r['wall_seconds'],worlds=r['worlds'],worker_seconds=r['worker_seconds']))
    axs[1].bar(range(2),[r['seconds']/3600 for r in data if r['kind']=='offline'],color=[COLORS[0],COLORS[1]],hatch='///',edgecolor='black',linewidth=.4)
    axs[1].set(xticks=range(2),xticklabels=['Construction','Diagnostic'],ylabel='Shared M0 target wall hours',title='Offline targets (three workers)')
    export(fig,NAMES[10],data,dict(online='complete timeline with cooldown, timers nested, not additive',offline='same target bank reused for both base/bucket; no duplicated cost'))
    atlas.close();assert saved==NAMES
    atomic_json(out/'figure_manifest.json',dict(names=saved,atlas='B_figure_atlas.pdf',atlas_pages=11,
        sha256={p.name:file_hash(p) for p in out.iterdir() if p.is_file() and p.name!='figure_manifest.json'}))
    return saved

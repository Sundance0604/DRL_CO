"""Non-numerical plotting/export QA/final gates; no parameter or sample changes."""
import argparse
import csv
import json
import xml.etree.ElementTree as ET
import numpy as np
from .analysis import rows,summary
from .common import *


def plot():
    load()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'svg.fonttype':'none'})
    data=rows();s=summary(data);colors=['#0072B2','#D55E00'];names=['old','new'];labels=['Baseline','Modified']
    def save(fig,name,sources):
        fig.tight_layout();fig.savefig(TARGET/'figures'/f'{name}.svg');fig.savefig(TARGET/'figures'/f'{name}.png',dpi=600)
        write_json(TARGET/'figures'/f'{name}.json',dict(source_csv=sources,dpi=600,size_inches=list(fig.get_size_inches()),
            generator_sha256=sha(__file__),scope='W40, training seed30001; descriptive development, no confidence or significance claims'))
        plt.close(fig)
    fig,axs=plt.subplots(2,1,figsize=(6,5),sharex=True)
    for name,color,label in zip(names,colors,labels):
        ts=[r for r in data['training_comparison'] if r['method']==name]
        ks=[r for r in data['kl_comparison'] if r['method']==name and r['last_check']]
        axs[0].plot([r['update'] for r in ts],[r['effective_steps'] for r in ts],marker='o' if name=='old' else 's',color=color,label=label,linestyle='-' if name=='old' else '--')
        axs[1].plot([r['update'] for r in ks],[r['approximate_KL_actual'] for r in ks],marker='o' if name=='old' else 's',color=color,linestyle='-' if name=='old' else '--')
    axs[0].axhline(8,color='.4',linestyle=':');axs[0].set_ylim(0,9);axs[0].set_ylabel('Optimized minibatches');axs[0].legend(loc='upper center',ncol=2,bbox_to_anchor=(.5,1.3))
    axs[1].axhline(.03,color='.4',linestyle=':',label='stop threshold');axs[1].set_ylabel('Actual sampled KL at last check');axs[1].set_xlabel('PPO update');axs[1].set_xticks(range(1,9))
    save(fig,'01_KL_and_steps',['training_comparison.csv','kl_comparison.csv'])
    fig,axs=plt.subplots(2,1,figsize=(6,5),sharex=True,sharey=True)
    for ax,name,label in zip(axs,names,labels):
        rs=[r for r in data['kl_comparison'] if r['method']==name and r['last_check']];x=[r['update'] for r in rs];alpha=[r['alpha_exact_KL'] for r in rs];beta=[r['beta_exact_KL'] for r in rs]
        ax.bar(x,alpha,color='#0072B2',label='alpha');ax.bar(x,beta,bottom=alpha,color='#E69F00',hatch='//',label='beta')
        ax.set_ylabel('Exact KL');ax.set_title(label+' (actual last check)',fontsize=10)
    axs[0].legend(loc='upper left',ncol=2);axs[1].set_xticks(range(1,9));axs[1].set_xlabel('PPO update')
    save(fig,'02_exact_KL_components',['kl_comparison.csv'])
    fig,axs=plt.subplots(1,2,figsize=(6,3.2))
    axs[0].bar(labels,[s[n]['beta_mean_absolute'] for n in names],color=colors,hatch='//');axs[0].set_ylabel('Mean absolute deterministic beta')
    axs[1].bar(labels,[s[n]['active_switches']/7 for n in names],color=colors,hatch='//');axs[1].set_ylim(0,1);axs[1].set_ylabel('q-active switch fraction vs beta0')
    for i,n in enumerate(names):axs[1].text(i,.035,f'{s[n]["active_switches"]}/7',ha='center')
    save(fig,'03_beta_and_switches',['beta_action_diagnostics.csv','q_active_comparison.csv'])
    fig,ax=plt.subplots(figsize=(6,3.6));profits=data['paired_development_profit'];x=np.arange(4)
    for i,r in enumerate(profits):ax.plot([i-.06,i+.06],[r['old_profit_USD'],r['new_profit_USD']],color='.6',linewidth=1)
    for offset,name,color,label,marker in zip([-.06,.06],names,colors,labels,['o','s']):ax.scatter(x+offset,[r[name+'_profit_USD'] for r in profits],color=color,label=label,marker=marker,s=38)
    ax.set_xticks(x,[str(r['scenario_seed']) for r in profits]);ax.set_xlabel('Development scenario seed');ax.set_ylabel('Full-cooldown profit (USD)')
    ax.legend(loc='upper center',ncol=2,bbox_to_anchor=(.5,1.2))
    save(fig,'04_paired_development_profit',['paired_development_profit.csv'])
    write_json(TARGET/'qa/figures.json',dict(status='PASSED',hashes={str(p):sha(p) for p in (TARGET/'figures').iterdir() if p.is_file()}))


def verify():
    m=load();data=rows();computed=summary(data)
    if computed!=read_json(TARGET/'qa/summary.json'):raise RuntimeError('summary reconstruction mismatch')
    for name,expected in data.items():
        with (TARGET/(name+'.csv')).open(encoding='utf-8-sig',newline='') as f:actual=list(csv.DictReader(f))
        if len(actual)!=len(expected):raise RuntimeError('CSV count differs')
        for a,e in zip(actual,expected):
            for k,v in e.items():
                if v is None:
                    if a[k]!='':raise RuntimeError('NA changed')
                elif isinstance(v,(int,float)) and not isinstance(v,bool):
                    if float(a[k])!=v:raise RuntimeError('CSV number differs '+k)
                elif a[k]!=str(v):raise RuntimeError('CSV value differs '+k)
    verify_hashes(Path('/'),m['history_hashes'])
    verify_hashes(Path('/'),read_json(TARGET/'qa/figures.json')['hashes'])
    for p in (TARGET/'figures').glob('*.svg'):ET.parse(p)
    raw={p.relative_to(TARGET).as_posix():sha(p) for folder in ('training','snapshots','probes','evaluation','data') for p in (TARGET/folder).rglob('*') if p.is_file()}
    write_json(TARGET/'qa/exports.json',dict(status='PASSED',all_CSV_cells_recalculated=True,no_p_values=True,raw_hashes=raw,
        rows={name:len(rs) for name,rs in data.items()},source_sha256=sha(__file__)))


def freeze(reviewed):
    m=load()
    if not reviewed:raise RuntimeError('actually inspect latest four PNGs before freeze')
    for name in ('initialization','isolation','training','probes','evaluation','update-reconstruction','replay','analysis','exports','figures'):
        if read_json(TARGET/'qa'/(name+'.json'))['status']!='PASSED':raise RuntimeError('missing QA '+name)
    suites=list(ET.parse(TARGET/'qa/pytest-final.xml').getroot().iter('testsuite'))
    if any(int(s.get(k,0)) for s in suites for k in ('failures','errors','skipped')):raise RuntimeError('tests failed')
    count=sum(int(s.get('tests',0)) for s in suites)
    if count!=7:raise RuntimeError('actual seven no-MILP unit tests required')
    b=read_json(TARGET/'qa/budget.json')
    if b['calls']!=978 or b['seconds']>600:raise RuntimeError('budget mismatch')
    if sum(p.stat().st_size for p in TARGET.rglob('*') if p.is_file())>512*2**20:raise RuntimeError('storage cap')
    verify_hashes(TARGET,read_json(TARGET/'qa/exports.json')['raw_hashes']);verify_hashes(Path('/'),m['history_hashes'])
    verify_hashes(Path('/'),m['source']['parent']['files'])
    verify_hashes(Path('/'),read_json(MECHANISM/'final_manifest.json')['delivery_sources'])
    verify_hashes(Path('/'),read_json(TARGET/'qa/figures.json')['hashes'])
    if read_json(TARGET/'qa/replay.json')['source_sha256']!=sha(PACKAGE/'qa.py'):raise RuntimeError('QA source changed')
    write_json(TARGET/'qa/tests.json',dict(status='PASSED',tests=count,deselected=2,solver_tests_excluded=True,
        xml_sha256=sha(TARGET/'qa/pytest-final.xml'),scope='four variant tests + three original KL/capture tests, all without MILP'))
    write_json(TARGET/'qa/visual.json',dict(status='PASSED',actual_latest_png_count=4,at=now(),SVG_same_source_XML_checked=True))
    hashes={p.relative_to(TARGET).as_posix():sha(p) for p in TARGET.rglob('*') if p.is_file() and p not in (TARGET/'manifest.json',TARGET/'final_manifest.json')}
    m.update(status='COMPLETED',completed_at=now(),artifact_hashes=hashes,
        delivery_sources={str(PACKAGE/n):sha(PACKAGE/n) for n in ('qa.py','delivery.py')})
    write_json(TARGET/'manifest.json',m);write_json(TARGET/'final_manifest.json',m);verify_hashes(TARGET,hashes)
    print('COMPLETED',len(hashes),sha(TARGET/'final_manifest.json'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('plot','verify','freeze'));p.add_argument('--reviewed',action='store_true');a=p.parse_args()
    if a.stage=='freeze':freeze(a.reviewed)
    else:globals()[a.stage]()

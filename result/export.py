"""Publication assets built only from persisted measurements."""
from pathlib import Path
import json
import shutil
import numpy as np
from result.storage import read_csv, write_csv, atomic_json, digest
from result.statistics import validate
from result.recording import records


def export(root,data,c,specs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);validate(root,data,c,specs)
    sm=json.loads((root/'statistics_manifest.json').read_text())
    for name,sha in {**sm['inputs'],**sm['outputs']}.items():
        if digest(root/name)!=sha:raise ValueError('Statistical input/output changed')
    target=root/'paper_assets';target.mkdir(parents=True,exist_ok=True);made=[]
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    micro=c['purpose']=='engineering-smoke'
    def save(fig,name):
        if micro:fig.suptitle('Engineering smoke only — not performance evidence',fontsize=9)
        fig.tight_layout()
        for ext in ('pdf','svg','png'):
            p=target/f'{name}.{ext}';fig.savefig(p,dpi=300);made.append(p)
        plt.close(fig)
    def esc(x):return str(x).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
    def table(name,headers,rows):
        p=target/f'{name}.tex'
        lines=['% Generated from recorded data; '+('SMOKE ONLY' if micro else 'fixed parameter grid'),
               r'\begin{tabular}{'+'l'*len(headers)+'}',r'\toprule',' & '.join(map(esc,headers))+r'\\',r'\midrule']
        lines.extend(' & '.join(esc(v) for v in r)+r'\\' for r in rows)
        lines += [r'\bottomrule',r'\end{tabular}'];p.write_text('\n'.join(lines),encoding='utf-8');made.append(p)
    for p in root.glob('*.csv'):
        out=target/p.name;shutil.copy2(p,out);made.append(out)
    summaries=read_csv(root/'summary.csv');main=[r for r in summaries if r['split']=='main'];methods=sorted({r['variant'] for r in main})
    lookup={(r['group'],r['variant']):r for r in main}
    table('comparison',['Method','All (%)','Frequent (%)','Infrequent (%)'],
        [[m,*[f"{100*float(lookup[g,m]['eta']):.2f}" if (g,m) in lookup else '--' for g in ('all','frequent','infrequent')]] for m in methods])
    selected=[r for r in read_csv(root/'effects.csv') if r['split']=='main' and r['group']=='all' and r['contrast'].split()[-1] in ('no_graph','no_scenario','no_demo','impact_only')]
    table('components',['Conditional contrast','Difference (pp)','Lower','Upper'],
          [[r['contrast'],*[f'{100*float(r[k]):.3f}' if r[k] else '--' for k in ('difference','ci_low','ci_high')]] for r in selected])
    for name,splits in [('sensitivity',('sensitivity',)),('generalization',('long_stream','large_shop')),('initialization',('initialization',)),('validation_best',('main_best',))]:
        rr=[r for r in summaries if r['split'] in splits and r['group']=='all']
        table(name,['Dataset / method','Fulfillment (%)','Seed SD (pp)'],
              [[r['split']+' / '+r['variant'],f"{100*float(r['eta']):.2f}",f"{100*float(r['seed_sd']):.2f}" if r['seed_sd'] else '--'] for r in rr])
    rr=[lookup['frequent',m] for m in methods if ('frequent',m) in lookup]
    fig,ax=plt.subplots(figsize=(11,4));ax.bar([r['variant'] for r in rr],[100*float(r['eta']) for r in rr],
        yerr=[100*float(r['seed_sd']) if r['seed_sd'] else 0 for r in rr]);ax.set_ylabel('On-time fulfillment (%)');ax.tick_params(axis='x',rotation=65);save(fig,'fulfillment')
    fig,ax=plt.subplots(figsize=(8,4));ax.barh([r['contrast'] for r in selected],[100*float(r['difference']) for r in selected]);ax.axvline(0,color='black',lw=.8);ax.set_xlabel('Conditional mean difference (percentage points)');save(fig,'components')
    pc=read_csv(root/'per_case.csv');fig,axes=plt.subplots(1,2,figsize=(10,4))
    for method in ('full','SPT','hgnn','dual_attention','dqn','ddqn','a2c','ppo'):
        rr=[r for r in pc if r['split']=='main' and r['variant']==method]
        for ax,factor in zip(axes,('iota','orders')):
            xs=sorted({float(r[factor]) for r in rr});ys=[100*np.mean([float(r['eta']) for r in rr if float(r[factor])==x]) for x in xs]
            ax.plot(xs,ys,marker='o',label=method);ax.set_xlabel(factor);ax.set_ylabel('Fixed-grid fulfillment (%)')
    axes[0].legend(fontsize=7,ncol=2);save(fig,'parameter_effects')
    curves=read_csv(root/'learning_curves.csv');fig,ax=plt.subplots(figsize=(8,4))
    for method in ('full','hgnn','dual_attention','dqn','ddqn','a2c','ppo'):
        rr=[r for r in curves if r['method']==method and not r['run'].startswith('sensitivity')]
        for run in sorted({r['run'] for r in rr}):
            line=[r for r in rr if r['run']==run];ax.plot([int(r['steps']) for r in line],[float(r['eta']) for r in line],alpha=.3,lw=.7)
        points=sorted({int(r['steps']) for r in rr if int(r['steps']) in c['training']['milestones']})
        if points:ax.plot(points,[np.mean([float(r['eta']) for r in rr if int(r['steps'])==p]) for p in points],label=method,marker='o')
    ax.set_xlabel('Charged training interactions');ax.set_ylabel('Validation fulfillment');ax.legend(fontsize=7);save(fig,'learning')
    # Main-test curves use the same fixed cells at every charged-budget milestone.
    checkpoints=[root/'main.csv',*(root/f'main_{p}.csv' for p in c['training']['milestones'] if (root/f'main_{p}.csv').exists())]
    fixed_rows=[r for path in checkpoints for r in read_csv(path) if r['seed']!='0'];curve_rows=[]
    for method in sorted({r['variant'] for r in fixed_rows}):
        for seed in sorted({r['seed'] for r in fixed_rows if r['variant']==method}):
            selected=[r for r in fixed_rows if r['variant']==method and r['seed']==seed]
            for point in sorted({int(r['budget']) for r in selected}):
                rr=[r for r in selected if int(r['budget'])==point]
                curve_rows.append(dict(variant=method,seed=seed,budget=point,eta=float(np.mean([float(r['eta']) for r in rr]))))
    write_csv(target/'fixed_budget_learning.csv',curve_rows);made.append(target/'fixed_budget_learning.csv')
    fig,ax=plt.subplots(figsize=(8,4))
    for method in ('full','hgnn','dual_attention','dqn','ddqn','a2c','ppo'):
        rr=[r for r in curve_rows if r['variant']==method];points=sorted({r['budget'] for r in rr})
        ax.plot(points,[np.mean([r['eta'] for r in rr if r['budget']==p]) for p in points],marker='o',label=method)
    ax.set(xlabel='Charged training interactions',ylabel='Fixed main-grid fulfillment');ax.legend(fontsize=7);save(fig,'fixed_budget_learning')
    sensitive=[r for r in summaries if r['split']=='sensitivity' and r['group']=='all'];fig,ax=plt.subplots(figsize=(9,4));ax.bar([r['variant'] for r in sensitive],[100*float(r['eta']) for r in sensitive]);ax.tick_params(axis='x',rotation=45);ax.set_ylabel('Fulfillment (%)');save(fig,'sensitivity')
    general=[r for r in summaries if r['split'] in ('long_stream','large_shop') and r['group']=='all'];fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,split in zip(axes,('long_stream','large_shop')):
        rr=[r for r in general if r['split']==split];ax.bar([r['variant'] for r in rr],[100*float(r['eta']) for r in rr]);ax.tick_params(axis='x',rotation=80);ax.set_title(split);ax.set_ylabel('Fulfillment (%)')
    save(fig,'generalization')
    costs=read_csv(root/'costs.csv');names=sorted({r['run'].rsplit('_s',1)[0] for r in costs});costrows=[]
    for method in names:
        rr=[r for r in costs if r['run'].rsplit('_s',1)[0]==method]
        costrows.append([method,*[f"{np.mean([float(r[k]) for r in rr]):.1f}" for k in ('real_steps','scenario_steps','demonstration_steps','elapsed_seconds')]])
    table('costs',['Configuration','Real','Scenario','Demonstration','Job wall seconds'],costrows)
    fig,ax=plt.subplots(figsize=(10,4));ax.bar([r[0] for r in costrows],[float(r[-1]) for r in costrows]);ax.tick_params(axis='x',rotation=75);ax.set_ylabel('Mean job wall seconds (not matrix wall time)');save(fig,'costs')
    latency=read_csv(root/'latency.csv');fig,ax=plt.subplots(figsize=(8,4))
    for method in ('full','hgnn','dual_attention','dqn','ddqn','a2c','ppo','SPT'):
        vals=sorted(1000*float(r['inference_seconds']) for r in latency if r['run']==method or r['run'].rsplit('_s',1)[0]==method)
        if vals:ax.plot(vals,np.arange(1,len(vals)+1)/len(vals),label=method)
    ax.set_xscale('log');ax.set_xlabel('Shared-state inference time (ms)');ax.set_ylabel('Empirical cumulative fraction');ax.legend(fontsize=7);save(fig,'latency')
    raw=read_csv(root/'main.csv');fig,axes=plt.subplots(1,2,figsize=(10,4))
    for method in ('full','SPT','hgnn','dual_attention'):
        rr=[r for r in raw if r['variant']==method]
        axes[0].scatter([int(r['orders']) for r in rr],[float(r['wall_seconds']) for r in rr],label=method,alpha=.7)
        axes[1].scatter([float(r['iota']) for r in rr],[float(r['decision_p50_ms']) for r in rr],label=method,alpha=.7)
    axes[0].set(xlabel='Orders',ylabel='Whole schedule wall seconds');axes[1].set(xlabel='Arrival frequency',ylabel='Median full decision time (ms)');axes[0].legend();save(fig,'runtime_parameters')
    behavior=read_csv(root/'behavior.csv');behaviorrows=[]
    for method in methods:
        for phase in ('startup','arrivals','drain'):
            rr=[r for r in behavior if r['variant']==method];steps=sum(int(r[f'{phase}_steps']) for r in rr);capacity=sum(float(r[f'{phase}_capacity_time']) for r in rr)
            behaviorrows.append([method,phase,sum(int(r[f'{phase}_waits']) for r in rr)/max(steps,1),sum(float(r[f'{phase}_held_machine_time']) for r in rr)/max(capacity,1e-9)])
    table('behavior',['Method','Phase','Wait-action share','Held-capacity share'],[[m,p,f'{w:.4f}',f'{h:.4f}'] for m,p,w,h in behaviorrows])
    fig,ax=plt.subplots(figsize=(10,4))
    for phase in ('startup','arrivals','drain'):
        rr=[r for r in behaviorrows if r[1]==phase];ax.plot([r[0] for r in rr],[r[2] for r in rr],marker='o',label=phase)
    ax.tick_params(axis='x',rotation=70);ax.set_ylabel('Wait-action share');ax.legend();save(fig,'behavior')
    gaps=read_csv(root/'offline_gaps.csv');fig,ax=plt.subplots(figsize=(6,4));valid=[r for r in gaps if r['reference']]
    ax.scatter([float(r['reference']) for r in valid],[float(r['eta']) for r in valid],s=12,alpha=.5);ax.plot([0,1],[0,1],color='black',lw=.7);ax.set(xlabel='Offline feasible incumbent',ylabel='Online fulfillment');save(fig,'offline')
    table('offline',['Instance','Status','Incumbent','Upper bound'],[[r['instance_id'],r['status'],r['eta'] or '--',r['upper'] or '--'] for r in read_csv(root/'exact.csv')])
    # Deterministic example selection by cell ID, never by performance.
    chosen=min(r['instance_id'] for r in raw);fig,axes=plt.subplots(2,1,figsize=(11,6));mechanism=[]
    for ax,method in zip(axes,('full','SPT')):
        rr=next(r for r in raw if r['variant']==method and r['instance_id']==chosen and int(r['seed']) in (0,1))
        cache=root/'evaluations'/rr['evaluation_key'];entry=json.loads((cache/'complete.json').read_text());rrroot=cache/entry['raw_directory']
        for event in records(rrroot,'events'):
            if event['kind']=='operation_start':ax.broken_barh([(event['time'],event['duration'])],(event['machine']-.35,.7),facecolors=plt.cm.tab20(event['order']%20))
        ax.set(title=method,xlabel='Simulation time',ylabel='Machine')
    save(fig,'gantt')
    for r in raw:
        if r['variant']!='full':continue
        cache=root/'evaluations'/r['evaluation_key'];entry=json.loads((cache/'complete.json').read_text())
        for step in records(cache/entry['raw_directory'],'decisions'):
            model=step.get('model')
            if model and model['gates'] is not None:
                mechanism.append(dict(instance_id=r['instance_id'],seed=r['seed'],phase=step['phase'],decision=step['step'],
                     mean_gate=float(np.mean(model['gates'])),mean_absolute_correction=float(np.mean(np.abs(model['correction'])))))
    write_csv(target/'mechanism.csv',mechanism,['instance_id','seed','phase','decision','mean_gate','mean_absolute_correction']);made.append(target/'mechanism.csv')
    fig,ax=plt.subplots(figsize=(6,4));ax.scatter([r['mean_gate'] for r in mechanism],[r['mean_absolute_correction'] for r in mechanism],s=5,alpha=.3);ax.set(xlabel='Mean residual gate',ylabel='Mean absolute score correction');save(fig,'mechanism')
    report=target/'REPORT.md';report.write_text('# '+('工程冒烟：不是正式性能证据' if micro else '固定参数基准实验报告')+'\n\n'
        '每个参数组合仅一个固定算例；区间只反映训练种子波动。全部对照与负结果保留。\n\n'
        'CSV 为表图原始数据，TeX 为表格，PDF/SVG 为矢量图。来源见 manifest.json；完整轨迹位于上级 evaluations 与 runs。\n',encoding='utf-8');made.append(report)
    atomic_json(target/'manifest.json',dict(purpose=c['purpose'],statistics=digest(root/'statistics_manifest.json'),
        data=digest(Path(data)/'manifest.json'),assets={p.name:digest(p) for p in made},
        source_records='../evaluation_manifest.json',uncertainty='training seeds conditional on the fixed parameter grid'))
    return target

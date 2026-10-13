# Build bfcl_criterion_19.csv from BFCL-Result score CSVs (downloaded 2026-10-02 into score/).
import csv, json
RAW='https://raw.githubusercontent.com/HuanzhiMao/BFCL-Result/main/{snap}/score/{f}.csv'
# (short name, kind, exact display name on the board, BFCL registry key at f7cf735, result dir in BFCL-Result, model id sent by handler)
M=[
('Granite-4.0-350m','open','Granite-4.0-350m (FC)','ibm-granite/granite-4.0-350m','ibm-granite_granite-4.0-350m','ibm-granite/granite-4.0-350m'),
('Granite-3.1-8B-Instruct','open','Granite-3.1-8B-Instruct (FC)','ibm-granite/granite-3.1-8b-instruct','ibm-granite_granite-3.1-8b-instruct','ibm-granite/granite-3.1-8b-instruct'),
('Llama-3.2-1B-Instruct','open','Llama-3.2-1B-Instruct (FC)','meta-llama/Llama-3.2-1B-Instruct-FC','meta-llama_Llama-3.2-1B-Instruct-FC','meta-llama/Llama-3.2-1B-Instruct'),
('Llama-3.2-3B-Instruct','open','Llama-3.2-3B-Instruct (FC)','meta-llama/Llama-3.2-3B-Instruct-FC','meta-llama_Llama-3.2-3B-Instruct-FC','meta-llama/Llama-3.2-3B-Instruct'),
('Llama-3.3-70B-Instruct','open','Llama-3.3-70B-Instruct (FC)','meta-llama/Llama-3.3-70B-Instruct-FC','meta-llama_Llama-3.3-70B-Instruct-FC','meta-llama/Llama-3.3-70B-Instruct'),
('Falcon3-1B-Instruct','open','Falcon3-1B-Instruct (FC)','tiiuae/Falcon3-1B-Instruct-FC','tiiuae_Falcon3-1B-Instruct-FC','tiiuae/Falcon3-1B-Instruct'),
('Falcon3-3B-Instruct','open','Falcon3-3B-Instruct (FC)','tiiuae/Falcon3-3B-Instruct-FC','tiiuae_Falcon3-3B-Instruct-FC','tiiuae/Falcon3-3B-Instruct'),
('Falcon3-7B-Instruct','open','Falcon3-7B-Instruct (FC)','tiiuae/Falcon3-7B-Instruct-FC','tiiuae_Falcon3-7B-Instruct-FC','tiiuae/Falcon3-7B-Instruct'),
('Falcon3-10B-Instruct','open','Falcon3-10B-Instruct (FC)','tiiuae/Falcon3-10B-Instruct-FC','tiiuae_Falcon3-10B-Instruct-FC','tiiuae/Falcon3-10B-Instruct'),
('Qwen3-4B-Instruct-2507','open','Qwen3-4B-Instruct-2507 (FC)','Qwen/Qwen3-4B-Instruct-2507-FC','Qwen_Qwen3-4B-Instruct-2507-FC','Qwen/Qwen3-4B-Instruct-2507'),
('MiniCPM3-4B','open','MiniCPM3-4B-FC (FC)','openbmb/MiniCPM3-4B-FC','openbmb_MiniCPM3-4B-FC','openbmb/MiniCPM3-4B'),
('GPT-5.2','api','GPT-5.2-2025-12-11 (FC)','gpt-5.2-2025-12-11-FC','gpt-5.2-2025-12-11-FC','gpt-5.2-2025-12-11'),
('GPT-4.1','api','GPT-4.1-2025-04-14 (FC)','gpt-4.1-2025-04-14-FC','gpt-4.1-2025-04-14-FC','gpt-4.1-2025-04-14'),
('GPT-4.1-mini','api','GPT-4.1-mini-2025-04-14 (FC)','gpt-4.1-mini-2025-04-14-FC','gpt-4.1-mini-2025-04-14-FC','gpt-4.1-mini-2025-04-14'),
('Claude Haiku 4.5','api','Claude-Haiku-4-5-20251001 (FC)','claude-haiku-4-5-20251001-FC','claude-haiku-4-5-20251001-FC','claude-haiku-4-5-20251001'),
('Claude Sonnet 4.5','api','Claude-Sonnet-4-5-20250929 (FC)','claude-sonnet-4-5-20250929-FC','claude-sonnet-4-5-20250929-FC','claude-sonnet-4-5-20250929'),
('Claude Opus 4.5','api','Claude-Opus-4-5-20251101 (FC)','claude-opus-4-5-20251101-FC','claude-opus-4-5-20251101-FC','claude-opus-4-5-20251101'),
('Command A','api','Command A (FC)','command-a-03-2025-FC','command-a-03-2025-FC','command-a-03-2025'),
('Command R7B','api','Command R7B (FC)','command-r7b-12-2024-FC','command-r7b-12-2024-FC','command-r7b-12-2024'),
]
def load(snap):
    ov={r['Model']:r for r in csv.DictReader(open(f'score/{snap}/data_overall.csv'))}
    ag={r['Model']:r for r in csv.DictReader(open(f'score/{snap}/data_agentic.csv'))}
    mt={r['Model']:r for r in csv.DictReader(open(f'score/{snap}/data_multi_turn.csv'))}
    return ov,ag,mt
p=lambda s: float(s.strip('%'))
snaps={'1216':'2025-12-16','1103':'2025-11-03','0826':'2025-08-26'}
D={k:load(v) for k,v in snaps.items()}
n_rows={k:len(D[k][0]) for k in D}
rows=[]
for short,kind,disp,key,rdir,mid in M:
    r=dict(model=short,kind=kind,bfcl_entry=disp,bfcl_registry_key_f7cf735=key,bfcl_result_dir=rdir,model_id_sent_by_handler=mid,mode='FC')
    for tag in ['1216','1103','0826']:
        ov,ag,mt=D[tag]
        if disp in ov:
            o=ov[disp]; a=ag[disp]; m=mt[disp]
            assert abs(p(o['Multi Turn Acc'])-p(m['Multi Turn Overall Acc']))<1e-9, (disp,tag)
            assert abs(p(o['Web Search Acc'])-p(a['Web Search Summary']))<1e-9 and abs(p(o['Memory Acc'])-p(a['Memory Summary']))<1e-9
            vals=dict(rank=int(o['Rank']),overall=p(o['Overall Acc']),non_live=p(o['Non-Live AST Acc']),live=p(o['Live Acc']),irrelevance=p(o['Irrelevance Detection']),relevance=p(o['Relevance Detection']),
                      multi_turn=p(o['Multi Turn Acc']),mt_base=p(o['Multi Turn Base']),mt_miss_func=p(o['Multi Turn Miss Func']),mt_miss_param=p(o['Multi Turn Miss Param']),mt_long_context=p(o['Multi Turn Long Context']),
                      agentic=p(a['Agentic Overall Acc']),web_search=p(o['Web Search Acc']),ws_base=p(a['Web Search Base']),ws_no_snippet=p(a['Web Search No Snippet']),memory=p(o['Memory Acc']),mem_kv=p(a['Memory KV']),mem_vector=p(a['Memory Vector']),mem_rec_sum=p(a['Memory Recursive Summarization']))
        else:
            vals={k:'' for k in ['rank','overall','non_live','live','irrelevance','relevance','multi_turn','mt_base','mt_miss_func','mt_miss_param','mt_long_context','agentic','web_search','ws_base','ws_no_snippet','memory','mem_kv','mem_vector','mem_rec_sum']}
        if tag!='1216':
            for k in ['mt_base','mt_miss_func','mt_miss_param','mt_long_context','relevance','ws_base','ws_no_snippet','mem_kv','mem_vector','mem_rec_sum']: vals.pop(k)
        for k,v in vals.items(): r[f'{k}_{tag}']=v
    r['n_rows_board_1216']=n_rows['1216']
    r['src_overall_1216']=RAW.format(snap='2025-12-16',f='data_overall')
    r['src_agentic_1216']=RAW.format(snap='2025-12-16',f='data_agentic')
    r['src_multi_turn_1216']=RAW.format(snap='2025-12-16',f='data_multi_turn')
    r['src_overall_1103']=RAW.format(snap='2025-11-03',f='data_overall')
    r['src_overall_0826']=RAW.format(snap='2025-08-26',f='data_overall')
    r['src_agentic_0826']=RAW.format(snap='2025-08-26',f='data_agentic')
    r['src_raw_results_1216']=f'https://github.com/HuanzhiMao/BFCL-Result/tree/main/2025-12-16/result/{rdir}'
    r['src_live_board']='https://gorilla.cs.berkeley.edu/data_overall.csv (byte-identical to 2025-12-16 on 2026-10-02)'
    r['src_model_config']='https://github.com/ShishirPatil/gorilla/blob/f7cf7359b7ac615a0b294831c5ba2bc95ee4a000/berkeley-function-call-leaderboard/bfcl_eval/constants/model_config.py'
    rows.append(r)
w=csv.DictWriter(open('bfcl_criterion_19.csv','w',newline=''),fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(len(rows),'rows;',len(rows[0]),'cols')
print(f"{'model':24s} rank  overall nonlive live  irrel  MT    agent  WS    mem  | 0826: overall MT agent | 1103: overall MT agent")
for r in rows:
    print(f"{r['model']:24s} {r['rank_1216']:4d} {r['overall_1216']:6.2f} {r['non_live_1216']:6.2f} {r['live_1216']:6.2f} {r['irrelevance_1216']:6.2f} {r['multi_turn_1216']:6.2f} {r['agentic_1216']:6.2f} {r['web_search_1216']:6.2f} {r['memory_1216']:6.2f} | {r['overall_0826']} {r['multi_turn_0826']} {r['agentic_0826']} | {r['overall_1103']} {r['multi_turn_1103']} {r['agentic_1103']}")
# floors and summary stats
import itertools
def tau(x,y):
    c=d=0
    for (a,b),(e,f) in itertools.combinations(zip(x,y),2):
        s=(a-e)*(b-f)
        if s>0: c+=1
        elif s<0: d+=1
    n=len(x); 
    import math
    tx=sum(1 for a,b in itertools.combinations(x,2) if a==b); ty=sum(1 for a,b in itertools.combinations(y,2) if a==b)
    n0=n*(n-1)/2
    return (c-d)/math.sqrt((n0-tx)*(n0-ty))
for k in ['overall','multi_turn','agentic','web_search','memory']:
    v=[r[f'{k}_1216'] for r in rows]
    print(k,'min',min(v),'max',max(v),'<5:',sum(x<5 for x in v),'/19', [r['model'] for r in rows if r[f'{k}_1216']<5])
both=[r for r in rows if r['overall_0826']!='']
print('in 08-26:',len(both),[r['model'] for r in rows if r['overall_0826']==''])
for k in ['overall','multi_turn','agentic']:
    print('tau 0826 vs 1216',k,round(tau([r[f'{k}_0826'] for r in both],[r[f'{k}_1216'] for r in both]),3),'max|delta|',max(abs(r[f'{k}_0826']-r[f'{k}_1216']) for r in both))
both=[r for r in rows if r['overall_1103']!='']
print('in 11-03:',len(both),[r['model'] for r in rows if r['overall_1103']==''])
for k in ['overall','multi_turn','agentic']:
    print('tau 1103 vs 1216',k,round(tau([r[f'{k}_1103'] for r in both],[r[f'{k}_1216'] for r in both]),3),'max|delta|',round(max(abs(r[f'{k}_1103']-r[f'{k}_1216']) for r in both),2), [(r['model'],round(r[f'{k}_1216']-r[f'{k}_1103'],2)) for r in both if abs(r[f'{k}_1103']-r[f'{k}_1216'])>=5])
print('tau(overall,MT)',round(tau([r['overall_1216'] for r in rows],[r['multi_turn_1216'] for r in rows]),3),'tau(overall,agentic)',round(tau([r['overall_1216'] for r in rows],[r['agentic_1216'] for r in rows]),3),'tau(MT,agentic)',round(tau([r['multi_turn_1216'] for r in rows],[r['agentic_1216'] for r in rows]),3),'tau(non_live,MT)',round(tau([r['non_live_1216'] for r in rows],[r['multi_turn_1216'] for r in rows]),3))

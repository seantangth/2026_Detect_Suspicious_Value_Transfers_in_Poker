"""Generalised partner-presence contrast (v032): more per-(hand, player) behavioural metrics, same own-baseline machinery.
Metrics: entropy-adjusted NLL by street (pre/flop/turn/river) and by action type (fold/call/agg max, raw NLL), size NLL,
vpip / pfr / n_agg / n_call / n_fold (from actions), net_frac / won_share (from seats). Global Welch d/t + time-quintile d/t,
two-direction mean/min. Output presence_contrast_v2_{phase}.parquet (prefix pc2_)."""
import polars as pl, numpy as np, time
R=(str(__import__('pathlib').Path(__file__).resolve().parents[2]) + '/'); raw=R+'1_data/raw/detect-suspicious-value-transfers-in-poker/'; OUT=R+'5_outputs/eqx_0913/'; SQ=R+'5_outputs/seqnll_0912/'
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
t0=time.time()
# per-(hand, player) metrics from actions (18.6M rows) + gbnll_action (entropy-adjusted by street)
act=pl.scan_parquet(raw+'actions.parquet').select('hand_id','player_id','street','action','amount')
act=act.with_columns(pl.col('hand_id').cast(pl.Utf8), pl.col('player_id').cast(pl.Utf8), pl.col('action').cast(pl.Utf8), pl.col('street').cast(pl.Utf8))
is_agg=pl.col('action').is_in(['bet','raise','all_in']); is_call=pl.col('action')=='call'; is_fold=pl.col('action')=='fold'; pre=pl.col('street').str.to_lowercase().str.starts_with('pre')
a1=act.group_by('hand_id','player_id').agg((is_agg).sum().alias('n_agg'), is_call.sum().alias('n_call'), is_fold.sum().alias('n_fold'), ((is_agg|is_call)&pre).any().cast(pl.Float64).alias('vpip'), (is_agg&pre).any().cast(pl.Float64).alias('pfr')).collect(engine='streaming')
log(f'actions agg rows {a1.height:,}')
ga=pl.scan_parquet(SQ+'gbnll_action.parquet').select('hand_id','player_id','sidx','nll_type_N','entropy_N','nll_size_N').with_columns(pl.col('hand_id').cast(pl.Utf8), pl.col('player_id').cast(pl.Utf8), (pl.col('nll_type_N')-pl.col('entropy_N')).alias('x'))
a2=ga.group_by('hand_id','player_id').agg(*[pl.col('x').filter(pl.col('sidx')==k).sum().alias(f'nllx_s{k}') for k in range(4)], pl.col('nll_size_N').fill_null(0.0).sum().alias('nllsz_sum')).collect(engine='streaming')
log(f'gbnll_action agg rows {a2.height:,}')
gp=pl.read_parquet(SQ+'gbnll_player.parquet', columns=['hand_id','player_id','nll_N_fold_max','nll_N_call_max','nll_N_agg_max']).with_columns(pl.col('hand_id').cast(pl.Utf8), pl.col('player_id').cast(pl.Utf8))
h=pl.read_parquet(raw+'hands.parquet', columns=['hand_id','table_id','phase','started_at']).with_columns(((pl.col('started_at').rank('ordinal').over('table_id','phase')-1)*5//pl.len().over('table_id','phase')).cast(pl.Int8).alias('q5')).drop('started_at')
s=pl.read_parquet(raw+'seats.parquet', columns=['hand_id','player_id','starting_stack','net_chips','won_share']).join(h,on='hand_id').join(a1,on=['hand_id','player_id'],how='left').join(a2,on=['hand_id','player_id'],how='left').join(gp,on=['hand_id','player_id'],how='left')
MET=['nllx_s0','nllx_s1','nllx_s2','nllx_s3','nllsz_sum','nll_N_fold_max','nll_N_call_max','nll_N_agg_max','vpip','pfr','n_agg','n_call','n_fold','net_frac','won_share']
s=s.with_columns((pl.col('net_chips')/pl.col('starting_stack').clip(1)).alias('net_frac')).select('hand_id','player_id','table_id','phase','q5', *[pl.col(m).cast(pl.Float64).fill_null(0.0).fill_nan(0.0) for m in MET])
log(f'seat rows {s.height:,}')
def stats(df, keys, sfx): return df.group_by(keys).agg([pl.len().alias('n'+sfx)]+[pl.col(m).sum().alias(f'{m}_s{sfx}') for m in MET]+[(pl.col(m)**2).sum().alias(f'{m}_q{sfx}') for m in MET])
def welch(d, sp, sa, tag):
    ex=[]
    for m in MET:
        np_, na=pl.col('n'+sp), pl.col('n'+sa); mp=pl.col(f'{m}_s{sp}')/np_; ma=pl.col(f'{m}_s{sa}')/na.clip(1)
        vp=(pl.col(f'{m}_q{sp}')/np_-mp**2).clip(0); va=(pl.col(f'{m}_q{sa}')/na.clip(1)-ma**2).clip(0); diff=pl.when(na>0).then(mp-ma).otherwise(0.0); se2=vp/np_+va/na.clip(1)+1e-6
        ex+=[diff.alias(f'{m}_d{tag}'), (diff/se2.sqrt()).alias(f'{m}_t{tag}'), se2.alias(f'{m}_v{tag}')]
    return d.with_columns(ex)
res={'development':[], 'evaluation':[]}
for tb in s['table_id'].unique().to_list():
    st=s.filter(pl.col('table_id')==tb)
    for ph in ('development','evaluation'):
        x=st.filter(pl.col('phase')==ph).drop('table_id','phase'); tot=stats(x,['player_id'],'_all'); totq=stats(x,['player_id','q5'],'_allq')
        pr=x.join(x.select('hand_id', pl.col('player_id').alias('other')), on='hand_id').filter(pl.col('player_id')!=pl.col('other'))
        p=stats(pr,['player_id','other'],'_p').join(tot,on='player_id')
        for m in MET: p=p.with_columns((pl.col(f'{m}_s_all')-pl.col(f'{m}_s_p')).alias(f'{m}_s_a'), (pl.col(f'{m}_q_all')-pl.col(f'{m}_q_p')).alias(f'{m}_q_a'))
        p=welch(p.with_columns((pl.col('n_all')-pl.col('n_p')).alias('n_a')),'_p','_a','g')
        pq=stats(pr,['player_id','other','q5'],'_pq').join(totq,on=['player_id','q5'])
        for m in MET: pq=pq.with_columns((pl.col(f'{m}_s_allq')-pl.col(f'{m}_s_pq')).alias(f'{m}_s_aq'), (pl.col(f'{m}_q_allq')-pl.col(f'{m}_q_pq')).alias(f'{m}_q_aq'))
        pq=welch(pq.with_columns((pl.col('n_allq')-pl.col('n_pq')).alias('n_aq')),'_pq','_aq','q'); wsum=pl.col('n_pq').sum()
        lq=pq.group_by('player_id','other').agg([((pl.col(f'{m}_dq')*pl.col('n_pq')).sum()/wsum).alias(f'{m}_dl') for m in MET]+[(((pl.col(f'{m}_dq')*pl.col('n_pq')).sum()/wsum)/(((pl.col('n_pq')/wsum)**2*pl.col(f'{m}_vq')).sum()+1e-12).sqrt()).alias(f'{m}_tl') for m in MET])
        p=p.join(lq,on=['player_id','other'],how='left')
        res[ph].append(p.select(['player_id','other']+[c for c in p.columns if c[-3:] in ('_dg','_tg','_dl','_tl')]))
log('per-table done')
for ph in ('development','evaluation'):
    d=pl.concat(res[ph]); cp=pl.read_parquet(R+f'1_data/processed/cand_pairs_{ph}.parquet').select('pair_id', pl.col('a').cast(pl.Utf8), pl.col('b').cast(pl.Utf8))
    fc=[c for c in d.columns if c not in ('player_id','other')]
    ab=d.rename({'player_id':'a','other':'b'}).rename({c:c+'_ab' for c in fc}); ba=d.rename({'player_id':'b','other':'a'}).rename({c:c+'_ba' for c in fc})
    j=cp.join(ab,on=['a','b'],how='left').join(ba,on=['a','b'],how='left'); ex=[]
    for c in fc: ex+=[((pl.col(c+'_ab')+pl.col(c+'_ba'))/2).alias(f'pc2_{c}_mean'), pl.min_horizontal(c+'_ab',c+'_ba').alias(f'pc2_{c}_min')]
    o=j.select(['pair_id']+ex).fill_null(0.0).fill_nan(0.0); o.write_parquet(OUT+f'presence_contrast_v2_{ph}.parquet'); log(f'{ph}: pairs {o.height:,} cols {o.width-1} unmatched {int(j[fc[0]+"_ab"].is_null().sum())} finite {bool(np.isfinite(o.drop("pair_id").to_numpy()).all())}')
log(f'ALL DONE {time.time()-t0:.0f}s')

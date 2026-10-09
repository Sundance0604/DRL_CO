"""Read-only CFS adapter adapted from Intercity's catalog/processor/store.

Reference: agent/paper-solution-approaches at a0076597899950722b114a8e5127f36992f498a8.
Preserves indexed OD queries, both-direction minimum counts, positive survey weights,
log(U)/weight weighted reservoir without replacement, and observed/generated separation.
No Intercity business solver, load clipping, or horizon-clipped deadline is copied.
"""
import heapq
import math
import random
import sqlite3
from pathlib import Path
from .network import geometric_network

OBSERVED_FIELDS = ['SHIPMT_ID','ORIG_CFS_AREA','DEST_CFS_AREA','MODE','SCTG',
                   'SHIPMT_VALUE','SHIPMT_WGHT','SHIPMT_DIST_GC','TEMP_CNTL_YN',
                   'EXPORT_YN','HAZMAT','WGT_FACTOR']


def normalize_area(value):
    parts = str(value).strip().split('-')
    if len(parts) != 2 or not all(p.isdigit() for p in parts):
        raise ValueError('invalid CFS area code')
    return parts[0].zfill(2)+'-'+parts[1].zfill(3)


def read_pools(source, area_codes, minimum_each_direction=10, maximum_weight=40000):
    source=Path(source).resolve(strict=True)
    conn=sqlite3.connect(source.as_uri()+'?mode=ro',uri=True)
    conn.row_factory=sqlite3.Row
    try:
        columns={r['name'] for r in conn.execute('PRAGMA table_info(shipments)')}
        if not set(OBSERVED_FIELDS)<=columns:
            raise ValueError('CFS SQLite missing required observed columns')
        metadata=dict(conn.execute('SELECT key,value FROM metadata').fetchall())
        if metadata.get('schema_version')!='1':
            raise ValueError('unsupported CFS SQLite schema version')
        pools,counts={},{}
        codes=sorted(normalize_area(c) for c in area_codes)
        if any(c.split('-')[1] in {'000','00000','99999'} for c in codes):
            raise ValueError('suppressed/remainder metro code not allowed')
        for i,a in enumerate(codes):
            for b in codes[i+1:]:
                directions=[]
                for x,y in [(a,b),(b,a)]:
                    rows=[dict(r) for r in conn.execute(
                        'SELECT '+','.join(OBSERVED_FIELDS)+' FROM shipments INDEXED BY idx_shipments_od_mode '
                        "WHERE ORIG_CFS_AREA=? AND DEST_CFS_AREA=? AND MODE IN ('111','112') "
                        "AND EXPORT_YN='N' AND SHIPMT_WGHT>0 AND SHIPMT_WGHT<=? "
                        'AND SHIPMT_DIST_GC>=50 AND WGT_FACTOR>0 ORDER BY SHIPMT_ID', (x,y,maximum_weight))]
                    directions.append(rows)
                counts[a+'__'+b]=[len(r) for r in directions]
                if min(counts[a+'__'+b])>=minimum_each_direction:
                    pools[a+'__'+b]=directions
        if not pools:
            raise ValueError('no supported bidirectional CFS pair; no synthetic fallback')
        supported={c for pair in pools for c in pair.split('__')}
        if supported!=set(codes):
            raise ValueError('a configured metro has no supported bidirectional pair')
        return pools,counts,metadata
    finally:
        conn.close()


def weighted_reservoir(rows, size, rng):
    if size < 0:
        raise ValueError('sample size cannot be negative')
    if size == 0:
        return []
    if len(rows)<size:
        raise ValueError('insufficient eligible shipments for balanced sampling')
    heap=[]
    for sequence,row in enumerate(rows):
        if not math.isfinite(float(row['WGT_FACTOR'])) or float(row['WGT_FACTOR']) <= 0:
            raise ValueError('survey weight must be finite and positive')
        priority=math.log(max(rng.random(),1e-15))/float(row['WGT_FACTOR'])
        item=(priority,sequence,row)
        if len(heap)<size:
            heapq.heappush(heap,item)
        elif item[:2]>heap[0][:2]:
            heapq.heapreplace(heap,item)
    return [item[2] for item in sorted(heap,reverse=True)]


def make_scenario(region, pools, seed, config):
    coordinates=region['coordinates']
    graph,xy=geometric_network(coordinates,config['knn'],config['stretch'])
    from .engine.mt_prototype import Net
    net=Net(graph,seed,config['speed'])
    rng=random.Random(seed)
    pairs=sorted(pools)
    # Stratified pair/direction allocation, survey-weighted sampling within strata.
    # This balances experimental spatial coverage; it is not a US population estimate.
    allocation={p:[0,0] for p in pairs}
    for i in range(config['orders']):
        allocation[pairs[(i//2)%len(pairs)]][i%2]+=1
    records=[]
    for pair in pairs:
        for d in (0,1):
            records+=weighted_reservoir(pools[pair][d],allocation[pair][d],rng)
    rng.shuffle(records)
    temporal=random.Random(seed+1)
    orders=[]
    for i,row in enumerate(records):
        origin,dest=normalize_area(row['ORIG_CFS_AREA']),normalize_area(row['DEST_CFS_AREA'])
        book=temporal.randrange(config['horizon'])
        ready=book+temporal.randint(0,config['ready_delay_max'])
        deadline=ready+net.tau[origin,dest]+temporal.randint(*config['deadline_slack'])
        orders.append(dict(id='cfs-'+row['SHIPMT_ID'],departure=origin,destination=dest,
                           passenger=math.ceil(row['SHIPMT_WGHT']/config['pounds_per_unit']),
                           book_time=book,start_time=ready,end_time=deadline,
                           revenue=config['fare_per_network_mile']*net.d[origin,dest]+config['fare_per_pound']*row['SHIPMT_WGHT'],
                           penalty=config['order_backlog_penalty']))
    fleet_rng=random.Random(seed+2)
    hubs=list(sorted(graph))
    fleet=hubs*config['vehicles_per_hub']
    fleet_rng.shuffle(fleet)
    scenario=dict(id=region['id']+'-'+str(seed),seed=seed,split='test',horizon=config['horizon'],
                  nodes=hubs,edges=[[a,b,w] for a,b,w in sorted(graph.edges(data='weight'))],
                  hubs=fleet,orders=orders,radii={n:0 for n in hubs},coordinates=coordinates,xy_miles=xy,
                  network_kind='geometry-informed metro network')
    provenance=dict(seed=seed,temporal_seed=seed+1,fleet_seed=seed+2,observed_fields=OBSERVED_FIELDS,
                    observed_shipments=records,generated_orders=orders,
                    generated_fields=['booking','ready','deadline','load-unit conversion','fare','penalty','fleet','network'],
                    interpretation='model-generated temporal fields, not observed CFS timestamps; assumed tariff, not goods value',
                    allocation=allocation,sampling='balanced OD strata; log(U)/WGT_FACTOR weighted reservoir without replacement')
    return scenario,provenance

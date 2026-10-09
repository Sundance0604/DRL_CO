"""State-aware scalar lookup with explicit support-based fallback provenance."""
import time
from model_families.single.value.providers import MCTableValueProvider

def corners(model,t,h,q):
    """Interpolation support follows the existing time/q linear convention."""
    times=model['base']['values'][h][str(q)]['times']
    left=max((a for a in times if a<=t),default=times[0]);right=min((a for a in times if a>=t),default=times[-1])
    qs=model['measured_q'];ql=max((a for a in qs if a<=q),default=qs[0]);qr=min((a for a in qs if a>=q),default=qs[-1])
    return sorted({(int(a),int(b)) for a in [left,right] for b in [ql,qr]})

class BucketValueProvider:
    def __init__(self,model):
        self.model=model;self.base=MCTableValueProvider(model['base'])
        self.bucket_providers={str(b):MCTableValueProvider(model['buckets'][str(b)]) for b in range(3)}
        self.calls=0;self.fallback_calls=0;self.lookup_seconds=0.;self.query_log=[]
    def marginal_release_value(self,t,h,q,s):
        started=time.perf_counter();self.calls+=1
        b=s.get('bucket',{}).get(h);source=[]
        if t>=s['T']:
            value=0.;fallback=False;reason='terminal_exact_zero'
        else:
            assert b in [0,1,2], 'Missing current-state bucket; no silent fallback'
            support=corners(self.model,t,h,q)
            source=[dict(time=tt,q=qq,count=self.model['support'][h][str(b)][str(qq)].get(str(tt),0)) for tt,qq in support if tt<s['T']]
            fallback=q>self.model['observed_max_q'] or any(r['count']<10 for r in source)
            reason='unsupported_q_endpoint' if q>self.model['observed_max_q'] else 'sample_count_below_10' if fallback else 'bucket_supported'
            value=self.base.marginal_release_value(t,h,q,s) if fallback else self.bucket_providers[str(b)].marginal_release_value(t,h,q,s)
        self.fallback_calls+=int(fallback)
        self.query_log.append(dict(observation_period=s['period'],release_time=t,hub=h,q=q,bucket=b,
            R=s.get('pressure',{}).get(h,{}).get('R'),value_USD=value,fallback=fallback,fallback_reason=reason,
            source_corners=source,current_bucket_not_future=True))
        self.lookup_seconds+=time.perf_counter()-started
        return value

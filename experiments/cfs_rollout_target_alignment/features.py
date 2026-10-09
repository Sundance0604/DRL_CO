"""Identical common transforms; only the allowed information set differs."""
import copy
import numpy as np
from scipy.interpolate import BSpline
from model_families.single.value.providers import feature_row,MCTableValueProvider
from model_families.single.local_value.providers import features as action_features

OLD=['o_remaining','o_other_idle','o_incoming','o_backlog','o_expected','o_q']
ADDED=['a_duration','a_deadhead','a_loaded','a_compatible_load','a_expiring_load',
       'a_release_idle','a_origin_idle','a_wait','a_same_hub']
ACTION=OLD+ADDED
ALIASES={'a_remaining':'o_remaining','a_expected':'o_expected','a_incoming':'o_incoming + 1 (constant; absorbed by intercept)'}

def fields(ctx,online,q):
    old=feature_row(ctx['release_time'],ctx['release_hub'],q,online)
    action=action_features(ctx,online)
    return {f'o_{name}':old[name] for name in ['remaining','other_idle','incoming','backlog','expected','q']} | {
        f'a_{name}':action[name] for name in ['remaining','duration','deadhead','loaded','compatible_load',
            'expiring_load','incoming','release_idle','origin_idle','expected','wait','same_hub']} | dict(hub=ctx['release_hub'],q=q)

def basis(x,b):
    x=np.asarray(x,float)
    if b['kind']=='linear':return ((x-b['center'])/b['scale'])[:,None]
    if b['kind']=='chebyshev':return np.polynomial.chebyshev.chebvander(np.clip(2*(x-b['lower'])/b['scale']-1,-1,1),b['degree'])[:,1:]
    kn=np.array(b['knots']);x=np.clip(x,b['lower'],b['upper'])
    return BSpline(kn,np.eye(len(kn)-4),3)(x)@np.array(b['transform'])-np.array(b['center'])

def design(rows,meta,deployed=False):
    parts=[np.ones((len(rows),1)),np.array([[float(r['hub']==h) for h in meta['hubs']] for r in rows])]
    if deployed:parts.append(np.tile(meta['q_weights'],(len(rows),1)))
    else:parts.append(np.array([[float(int(r['q'])==q) for q in meta['qs']] for r in rows]))
    for name in meta['features']:
        if deployed and name=='o_q':
            a=np.asarray(meta['q_weights'])@basis(meta['qs'],meta['bases'][name])
            parts.append(np.tile(a,(len(rows),1)))
        else:parts.append(basis([r[name] for r in rows],meta['bases'][name]))
    return np.column_stack(parts)

class AlignedValueProvider:
    """Original MC prefix slots + route correction, unchanged q-average integration.

    The correction predicts system action advantage, not an exact additive joint
    value. The original post-T release gate is retained and audited separately.
    """
    def __init__(self,model):
        self.model=copy.deepcopy(model);self.slot=MCTableValueProvider(model['slot_model'])
        self.beta=np.array(model['coefficients'])
    def corrections(self,contexts,online):
        if not contexts:return np.zeros(0)
        rows=[fields(c,online,self.model['qs'][0]) for c in contexts]
        out=design(rows,self.model,True)@self.beta
        out=np.where([c['release_time']>=online['T'] for c in contexts],0.,out)
        assert np.isfinite(out).all();return out
    def marginal_release_value(self,time,hub,supply_slot,online_state_summary,action_context=None):
        base=self.slot.marginal_release_value(time,hub,supply_slot,online_state_summary)
        return float(base if action_context is None else base+self.corrections([action_context],online_state_summary)[0])

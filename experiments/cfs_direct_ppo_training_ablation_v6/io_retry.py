"""Attempt-local, bounded PermissionError retries; serialization/RNG unchanged."""
import json
import os
import time
from pathlib import Path

DELAYS=(.05,.1,.2,.4,.8,1.,1.)
RETRY_EVENTS=[]


def write_json(path,value):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+'.tmp')
    payload=json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)
    for attempt in range(len(DELAYS)+1):
        try:
            temporary.write_text(payload,encoding='utf-8')
            os.replace(temporary,path)
            return
        except PermissionError:
            RETRY_EVENTS.append(dict(path=str(path),attempt=attempt+1,at=time.time()))
            if attempt==len(DELAYS):raise
            time.sleep(DELAYS[attempt])

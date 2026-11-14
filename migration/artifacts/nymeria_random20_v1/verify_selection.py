"""Verify the frozen physical-window draw using only Python's standard library."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json
R=Path(__file__).resolve().parent
manifest=json.loads((R/'manifest.json').read_text());parent=json.loads((R/'eligible_windows.json').read_text());rows=manifest['rows'];keys=[tuple(k) for k in parent['keys']]
hash_keys=hashlib.sha256(json.dumps(sorted(keys),separators=(',',':')).encode()).hexdigest()
assert hash_keys==manifest['parent_portable_keys_sha256']==parent['parent_portable_keys_sha256']
assert len(keys)==len(set(keys))==11677 and len(rows)==1420 and len({r['id'] for r in rows})==1420
assert set(Counter(r['uuid'] for r in rows).values())=={20}
groups=defaultdict(list)
for uid,start in keys:groups[uid].append(start)
for uid,starts in groups.items():
 starts=sorted(starts);scores=sorted(starts,key=lambda start:hashlib.sha256(f'nymeria-random20-v1|{manifest["selection_seed"]}|{uid}|{start}'.encode()).digest());expected=scores[:20];actual=[r for r in rows if r['uuid']==uid]
 assert {r['start'] for r in actual}==set(expected)
 for row in actual:
  assert starts[row['eligible_index_in_recording']]==row['start'] and expected[row['random_selection_rank']]==row['start']
  assert row['split']=='test' and row['frames']==97 and row['fps']==20 and row['end_exclusive']==row['start']+97
  identity=json.dumps([uid,row['start'],97,20],separators=(',',':'));assert row['id']=='nymeria_'+hashlib.sha256(identity.encode()).hexdigest()[:24]
  expected_seed=int.from_bytes(hashlib.sha256(('nymeria-random20-sampling-v1|'+identity).encode()).digest()[:4],'big')%(2**31);assert row['seed']==expected_seed
print(json.dumps({'passed':True,'windows':1420,'recordings':71,'windows_per_recording':20,'selection_seed':manifest['selection_seed'],'manifest_sha256':hashlib.sha256((R/'manifest.json').read_bytes()).hexdigest()},indent=2))

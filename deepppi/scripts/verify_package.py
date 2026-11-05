"""Check every packaged file and the frozen D protein-disjoint split (stdlib)."""
from pathlib import Path
import csv,gzip,hashlib,json,sys
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def rows(p):
    with gzip.open(p,'rt',encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def verify(root=ROOT):
    root=Path(root).resolve()
    manifest=json.loads((root/'MANIFEST.json').read_text(encoding='utf-8'))
    for rel,meta in manifest['files'].items():
        p=(root/rel).resolve()
        if not p.is_relative_to(root) or p.is_symlink() or not p.is_file():raise ValueError('Unsafe/missing file: '+rel)
        if p.stat().st_size!=meta['bytes'] or digest(p)!=meta['sha256']:raise ValueError('Integrity mismatch: '+rel)
    d=root/'data/D';meta=json.loads((d/'manifest.json').read_text(encoding='utf-8'))
    for rel,h in meta['output_sha256'].items():
        if digest(d/rel)!=h:raise ValueError('Dataset hash mismatch: '+rel)
    aa='ACDEFGHIKLMNPQRSTVWY'
    table=bytes({ord(c):i+1 for i,c in enumerate(aa)}.get(n,0) for n in range(256))
    proteins={}
    for r in rows(d/'proteins.csv.gz'):
        pid=r['protein_id'];tokens=r['sequence'].upper().encode('ascii',errors='replace').translate(table)[:512].ljust(512,b'\0')
        if pid in proteins or not pid.startswith('180675.') or hashlib.sha256(tokens).hexdigest()!=r['model_input_sha256']:raise ValueError('Invalid protein identity: '+pid)
        proteins[pid]=r['model_input_sha256']
    sets={};summary={}
    for split in ['train','validation','test']:
        rs=rows(d/'protein_disjoint'/(split+'.csv.gz'))
        exp=meta['protocols']['protein_disjoint']['splits'][split]
        if len(rs)!=exp['rows'] or dict(Counter(r['label'] for r in rs))!=exp['labels']:raise ValueError('Count mismatch: '+split)
        pairset=set();modelset=set();pids=set();groups=set()
        for r in rs:
            a,b=r['protein_a'],r['protein_b']
            pair=hashlib.sha256('|'.join(sorted([a,b])).encode()).hexdigest()
            modelpair=hashlib.sha256('|'.join(sorted([proteins[a],proteins[b]])).encode()).hexdigest()
            if pair!=r['pair_id'] or modelpair!=r['model_pair_id'] or pair in pairset or modelpair in modelset:raise ValueError('Invalid/duplicate pair in '+split)
            pairset.add(pair);modelset.add(modelpair);pids.update([a,b]);groups.update([proteins[a],proteins[b]])
        sets[split]=(pairset,modelset,pids,groups)
        summary[split]={'rows':len(rs),'labels':dict(Counter(r['label'] for r in rs)),'protein_ids':len(pids),'input_groups':len(groups)}
    overlaps=[]
    names=list(sets)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            counts=[len(x&y) for x,y in zip(sets[a],sets[b])]
            if any(counts):raise ValueError('Cross-split leakage: '+a+'/'+b)
            overlaps.append({'left':a,'right':b,'pair_overlap':counts[0],'model_pair_overlap':counts[1],'protein_id_overlap':counts[2],'input_group_overlap':counts[3]})
    for seed in [42,43,44]:
        w=root/'weights/D'/('seed_'+str(seed))
        m=json.loads((w/'run_metadata.json').read_text(encoding='utf-8'))
        if m['status']!='complete' or digest(w/'best.weights.h5')!=m['weights_sha256']:raise ValueError('Checkpoint mismatch')
    return {'status':'verified','manifest_files_checked':len(manifest['files']),'dataset':'D/protein_disjoint','counts':summary,'overlap_checks':overlaps,'checkpoints':3}
if __name__=='__main__':print(json.dumps(verify(),ensure_ascii=False,indent=2))

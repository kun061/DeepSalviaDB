"""Check the frozen D protein-disjoint dataset included in this repository."""
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
    return {'status':'verified','dataset_files_checked':len(meta['output_sha256']),'dataset':'D/protein_disjoint','counts':summary,'overlap_checks':overlaps}
if __name__=='__main__':print(json.dumps(verify(),ensure_ascii=False,indent=2))

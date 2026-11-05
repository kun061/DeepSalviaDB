#!/usr/bin/env python3
import argparse, csv, json, os, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / 'Pretrained_Models/plant-dnagpt-BPE'
DEFAULT_DATA = ROOT / 'DSS3/promoter_1kb/chromosome_holdout_v4_gc_matched'
DEFAULT_OUT = ROOT / 'finetune/dss3_bpe_1000bp_head_only_v4_gc_seed42'

class SeqDataset(Dataset):
    def __init__(self, frame, tokenizer):
        self.labels = frame.label.astype(int).to_numpy()
        self.ids = [tokenizer(s, add_special_tokens=True, truncation=False)['input_ids'] for s in frame.sequence]
    def __len__(self): return len(self.labels)
    def __getitem__(self, i): return self.ids[i], int(self.labels[i])

def collate(batch, pad_id):
    n = max(len(x) for x, _ in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), n), dtype=torch.long)
    labels = torch.tensor([y for _, y in batch], dtype=torch.long)
    for i, (x, _) in enumerate(batch):
        ids[i, :len(x)] = torch.tensor(x, dtype=torch.long); mask[i, :len(x)] = 1
    return ids, mask, labels

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); set_seed(seed)

def read_split(data_dir, name):
    df = pd.read_csv(Path(data_dir) / f'{name}.csv')
    if not {'sequence','label'}.issubset(df.columns): raise ValueError(f'{name}.csv must contain sequence,label')
    if not df.sequence.map(lambda x: isinstance(x, str) and len(x) == 1000).all():
        raise ValueError(f'{name}.csv contains non-1000-bp sequences')
    return df

def load_components(model_path, output_dir=None):
    tok = AutoTokenizer.from_pretrained(model_path, model_max_length=512, use_fast=True)
    if tok.pad_token_id is None:
        if tok.eos_token_id is None: raise ValueError('Tokenizer has neither pad_token nor eos_token')
        tok.pad_token = tok.eos_token
    info = {}
    loaded = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=2, ignore_mismatched_sizes=True, output_loading_info=True)
    model, info = loaded
    model.config.pad_token_id = tok.pad_token_id
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        json.dump({'missing_keys': info.get('missing_keys', []), 'unexpected_keys': info.get('unexpected_keys', []),
                   'mismatched_keys': [list(x) for x in info.get('mismatched_keys', [])],
                   'allowed_new_head': [k for k in info.get('missing_keys', []) if k.startswith(('score.', 'classifier.'))]},
                  open(Path(output_dir)/'model_loading_report.json','w'), indent=2)
    return tok, model

def audit(model_path, data_dir, output_dir):
    tok, model = load_components(model_path, output_dir)
    result = {'model_path': str(model_path), 'data_dir': str(data_dir), 'expected_model_max_length': 512,
              'tokenizer_model_max_length': tok.model_max_length, 'tokenizer_vocab_size': len(tok),
              'pad_token_id': tok.pad_token_id, 'model_type': model.config.model_type,
              'max_position_embeddings': getattr(model.config, 'n_positions', getattr(model.config, 'max_position_embeddings', None))}
    for split in ('train','dev','test'):
        df = read_split(data_dir, split); lengths = [len(tok(s, add_special_tokens=True, truncation=False)['input_ids']) for s in df.sequence]
        result[split] = {'count': len(df), 'min_tokens': min(lengths), 'max_tokens': max(lengths), 'mean_tokens': float(np.mean(lengths)),
                         'over_512': int(sum(x > 512 for x in lengths)), 'over_position_capacity': int(sum(x > result['max_position_embeddings'] for x in lengths))}
    result['safe_without_truncation'] = all(v['over_position_capacity'] == 0 for k,v in result.items() if k in ('train','dev','test'))
    Path(output_dir).mkdir(parents=True, exist_ok=True); json.dump(result, open(Path(output_dir)/'input_length_audit.json','w'), indent=2)
    print(json.dumps(result, indent=2)); return result

def configure_head(model):
    backbone = getattr(model, 'transformer', None) or getattr(model, 'base_model', None)
    for p in model.parameters(): p.requires_grad = False
    for p in model.score.parameters(): p.requires_grad = True
    if backbone is None: raise ValueError('Could not identify GPT-2 backbone')
    names = [n for n,p in model.named_parameters() if p.requires_grad]
    if not names or not all(n.startswith('score.') for n in names): raise AssertionError(f'Unexpected trainable params: {names}')
    return backbone, names

def metrics(y, pred):
    return {'accuracy': accuracy_score(y,pred), 'precision': precision_score(y,pred,zero_division=0), 'recall': recall_score(y,pred,zero_division=0), 'f1': f1_score(y,pred,zero_division=0)}

def run(args):
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True); seed_all(args.seed)
    audit(args.model_path, args.data_dir, out)
    tok, model = load_components(args.model_path, out); backbone, names = configure_head(model)
    train, dev = read_split(args.data_dir,'train'), read_split(args.data_dir,'dev')
    tr, dv = SeqDataset(train,tok), SeqDataset(dev,tok)
    with open(out/'trainable_parameters.txt','w') as f: f.write('\n'.join(names)+f'\ncount={sum(p.numel() for p in model.parameters() if p.requires_grad)}\n')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model.to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=0.01)
    dl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate(b,tok.pad_token_id))
    ev = DataLoader(dv, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate(b,tok.pad_token_id))
    history=[]; best=-1.0; start=time.time()
    for epoch in range(args.epochs):
        model.train(); backbone.eval(); opt.zero_grad()
        for j,(ids,mask,y) in enumerate(dl):
            o=model(input_ids=ids.to(device), attention_mask=mask.to(device), labels=y.to(device))
            if o.logits.shape[-1] != 2: raise AssertionError(f'Expected logits [batch,2], got {o.logits.shape}')
            group_start=(j//args.grad_accum)*args.grad_accum; group_size=min(args.grad_accum,len(dl)-group_start)
            (o.loss/group_size).backward()
            if (j+1)%args.grad_accum==0 or (j+1)==len(dl): opt.step(); opt.zero_grad()
        model.eval(); ys=[]; ps=[]; total=0.0
        with torch.inference_mode():
            for ids,mask,y in ev:
                o=model(input_ids=ids.to(device), attention_mask=mask.to(device), labels=y.to(device)); total += o.loss.item()*len(y)
                ys.extend(y.numpy()); ps.extend(o.logits.argmax(1).cpu().numpy())
        m={'epoch':epoch+1,'eval_loss':total/len(dv),**metrics(ys,ps)}; history.append(m); print(m, flush=True)
        if m['f1'] > best:
            best=m['f1']; model.save_pretrained(out/'best_model'); tok.save_pretrained(out/'best_model'); json.dump(m,open(out/'best_dev.json','w'),indent=2)
    pd.DataFrame(history).to_csv(out/'training_history.csv',index=False); pd.DataFrame(history).to_csv(out/'validation_metrics.csv',index=False)
    json.dump({'model_name':str(args.model_path),'seed':args.seed,'sequence_length_bp':1000,'tokenization':'raw DNA BPE','model_max_length':512,'epochs':args.epochs,'batch_size':args.batch_size,'gradient_accumulation':args.grad_accum,'learning_rate':args.learning_rate,'optimizer':'AdamW','weight_decay':0.01,'scheduler':None,'trainable_parameters':names,'trainable_count':sum(p.numel() for p in model.parameters() if p.requires_grad),'total_count':sum(p.numel() for p in model.parameters()),'training_seconds':time.time()-start},open(out/'training_config.json','w'),indent=2)

def main():
    p=argparse.ArgumentParser(description='Plant DNAGPT-BPE 1000-bp head-only fine-tuning (explicit modes).')
    p.add_argument('--audit',action='store_true'); p.add_argument('--smoke-test',action='store_true'); p.add_argument('--train',action='store_true')
    p.add_argument('--epochs',type=int,default=5); p.add_argument('--batch-size',type=int,default=4); p.add_argument('--grad-accum',type=int,default=4); p.add_argument('--learning-rate',type=float,default=1e-3); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--model-path',type=Path,default=DEFAULT_MODEL); p.add_argument('--data-dir',type=Path,default=DEFAULT_DATA); p.add_argument('--output-dir',type=Path,default=DEFAULT_OUT)
    a=p.parse_args(); modes=sum((a.audit,a.smoke_test,a.train))
    if modes==0: p.print_help(); return
    if modes>1: p.error('choose exactly one of --audit, --smoke-test, --train')
    if a.audit: audit(a.model_path,a.data_dir,a.output_dir); return
    if a.smoke_test:
        audit(a.model_path,a.data_dir,a.output_dir); tok,model=load_components(a.model_path,a.output_dir); backbone,names=configure_head(model); df=read_split(a.data_dir,'train').head(2); ds=SeqDataset(df,tok); ids,mask,y=collate([ds[i] for i in range(len(ds))],tok.pad_token_id); o=model(input_ids=ids,attention_mask=mask,labels=y); o.loss.backward(); assert o.logits.shape==(len(ds),2); print('SMOKE_TEST_OK',o.logits.shape,'trainable=',names); return
    run(a)
if __name__=='__main__': main()

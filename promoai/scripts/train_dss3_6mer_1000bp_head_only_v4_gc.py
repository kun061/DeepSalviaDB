import os, json, random, time, hashlib, argparse
import numpy as np, pandas as pd, torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, set_seed
from torch.utils.data import DataLoader, Dataset

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL=os.path.join(ROOT,'Pretrained_Models/plant-dnagpt-6mer')
DATA=os.path.join(ROOT,'DSS3/promoter_1kb/chromosome_holdout_v4_gc_matched')
OUT=os.path.join(ROOT,'finetune/dss3_6mer_1000bp_head_only_v4_gc_seed42')
SEED=42
def kmers(s): return ' '.join(s[i:i+6].upper() for i in range(len(s)-5))
def load_csv(name):
 d=pd.read_csv(os.path.join(DATA,name)); assert d.sequence.map(len).eq(1000).all(); return d
class DS(Dataset):
 def __init__(self,d,tok): self.x=[tok(kmers(s),truncation=False,add_special_tokens=True)['input_ids'] for s in d.sequence]; self.y=d.label.to_numpy()
 def __len__(self): return len(self.y)
 def __getitem__(self,i): return self.x[i],int(self.y[i])
def collate(b,pad):
 n=max(len(x[0]) for x in b); ids=torch.full((len(b),n),pad,dtype=torch.long); mask=torch.zeros((len(b),n),dtype=torch.long)
 for i,(x,y) in enumerate(b): ids[i,:len(x)]=torch.tensor(x); mask[i,:len(x)]=1
 return ids,mask,torch.tensor([x[1] for x in b])
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=float,default=5); ap.add_argument('--batch',type=int,default=4); ap.add_argument('--grad_accum',type=int,default=4); args=ap.parse_args()
 os.makedirs(OUT,exist_ok=True); set_seed(SEED); random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
 tok=AutoTokenizer.from_pretrained(MODEL,model_max_length=1024,use_fast=False)
 train,dev=load_csv('train.csv'),load_csv('dev.csv')
 tr, dv=DS(train,tok),DS(dev,tok); assert max(map(len,tr.x))==997 and min(map(len,tr.x))==997
 model=AutoModelForSequenceClassification.from_pretrained(MODEL,num_labels=2,ignore_mismatched_sizes=True); model.config.pad_token_id=tok.pad_token_id
 loaded=[n for n,p in model.named_parameters() if p.requires_grad]
 for n,p in model.transformer.named_parameters(): p.requires_grad=False
 trainable=[n for n,p in model.named_parameters() if p.requires_grad]
 assert trainable and all(n.startswith('score.') for n in trainable)
 with open(os.path.join(OUT,'trainable_parameters.txt'),'w') as f: f.write('\n'.join(trainable)+f'\ncount={sum(p.numel() for p in model.parameters() if p.requires_grad)}\n')
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model.to(device); opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3,weight_decay=.01); dl=DataLoader(tr,batch_size=args.batch,shuffle=True,collate_fn=lambda b:collate(b,tok.pad_token_id)); ev=DataLoader(dv,batch_size=args.batch,shuffle=False,collate_fn=lambda b:collate(b,tok.pad_token_id)); best=-1; history=[]; t=time.time()
 for epoch in range(int(args.epochs)):
  model.train(); model.transformer.eval(); opt.zero_grad()
  for j, (ids, mask, y) in enumerate(dl):
      group_start = (j // args.grad_accum) * args.grad_accum
      group_size = min(args.grad_accum, len(dl) - group_start)
      o = model(input_ids=ids.to(device), attention_mask=mask.to(device), labels=y.to(device))
      (o.loss / group_size).backward()
      if (j+1) % args.grad_accum == 0 or (j+1) == len(dl):
          opt.step()
          opt.zero_grad()
      if (j+1) % 200 == 0 or (j+1) == len(dl):
          print(f"epoch={epoch+1}/{int(args.epochs)} batch={j+1}/{len(dl)} loss={o.loss.item():.4f}", flush=True)
  model.eval(); ys=[]; ps=[]; loss=0
  with torch.inference_mode():
   for ids,mask,y in ev:
    o=model(input_ids=ids.to(device),attention_mask=mask.to(device),labels=y.to(device)); loss+=o.loss.item()*len(y); ys.extend(y.numpy()); ps.extend(torch.argmax(o.logits,1).cpu().numpy())
  met={'epoch':epoch+1,'eval_loss':loss/len(dv),'accuracy':accuracy_score(ys,ps),'precision':precision_score(ys,ps,zero_division=0),'recall':recall_score(ys,ps,zero_division=0),'f1':f1_score(ys,ps,zero_division=0)}; history.append(met); print(met)
  if met['f1']>best: best=met['f1']; model.save_pretrained(os.path.join(OUT,'best_model')); tok.save_pretrained(os.path.join(OUT,'best_model')); json.dump(met,open(os.path.join(OUT,'best_dev.json'),'w'),indent=2)
 elapsed=time.time()-t; pd.DataFrame(history).to_csv(os.path.join(OUT,'training_history.csv'),index=False); pd.DataFrame(history).to_csv(os.path.join(OUT,'validation_metrics.csv'),index=False)
 json.dump({'model_name':MODEL,'seed':SEED,'sequence_length_bp':1000,'kmer':6,'model_max_length':1024,'learning_rate':1e-3,'epochs':args.epochs,'batch':args.batch,'gradient_accumulation':args.grad_accum,'trainable_count':sum(p.numel() for p in model.parameters() if p.requires_grad),'total_count':sum(p.numel() for p in model.parameters()),'training_seconds':elapsed},open(os.path.join(OUT,'training_config.json'),'w'),indent=2)
 json.dump({'train':len(train),'dev':len(dev),'train_labels':train.label.value_counts().to_dict(),'dev_labels':dev.label.value_counts().to_dict()},open(os.path.join(OUT,'data_manifest.json'),'w'),indent=2)
 print('TRAIN_DONE', elapsed, os.path.join(OUT, 'best_model'), best)
if __name__=='__main__': main()

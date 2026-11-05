#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
from transformers import AutoModelForSequenceClassification, AutoTokenizer
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description='Independent Test evaluation for Plant DNAGPT-BPE.'); p.add_argument('--test',action='store_true'); p.add_argument('--checkpoint',type=Path,default=ROOT/'finetune/dss3_bpe_1000bp_head_only_v4_gc_seed42/best_model'); p.add_argument('--test-csv',type=Path,default=ROOT/'DSS3/promoter_1kb/chromosome_holdout_v4_gc_matched/test.csv'); p.add_argument('--test-meta-csv',type=Path,default=ROOT/'DSS3/promoter_1kb/chromosome_holdout_v4_gc_matched/test.meta.csv'); p.add_argument('--output-dir',type=Path,default=ROOT/'finetune/dss3_bpe_1000bp_head_only_v4_gc_seed42/test_results'); a=p.parse_args()
 if not a.test: p.print_help(); return
 if not a.checkpoint.is_dir(): raise FileNotFoundError(f'Best checkpoint does not exist: {a.checkpoint}. Run training explicitly first.')
 tok=AutoTokenizer.from_pretrained(a.checkpoint,model_max_length=512); model=AutoModelForSequenceClassification.from_pretrained(a.checkpoint); model.eval(); model.config.pad_token_id=tok.pad_token_id
 d=pd.read_csv(a.test_csv); meta=pd.read_csv(a.test_meta_csv) if a.test_meta_csv.exists() else pd.DataFrame({'sample_id':np.arange(len(d))}); assert len(d)==len(meta) and d.sequence.map(lambda x:isinstance(x,str) and len(x)==1000).all()
 dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model.to(dev); probs=[]
 with torch.inference_mode():
  for i in range(0,len(d),8):
   z=tok(d.sequence.iloc[i:i+8].tolist(),padding=True,truncation=False,return_tensors='pt'); probs.extend(torch.softmax(model(**{k:v.to(dev) for k,v in z.items()}).logits,1)[:,1].cpu().numpy())
 y=d.label.to_numpy(); prob=np.asarray(probs); pred=(prob>=.5).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel(); a.output_dir.mkdir(parents=True,exist_ok=True)
 res={'accuracy':accuracy_score(y,pred),'precision':precision_score(y,pred,zero_division=0),'recall':recall_score(y,pred,zero_division=0),'f1':f1_score(y,pred,zero_division=0),'roc_auc':roc_auc_score(y,prob),'pr_auc_ap':average_precision_score(y,prob),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),'test_size':len(y),'threshold':.5,'checkpoint_path':str(a.checkpoint)}; json.dump(res,open(a.output_dir/'test_metrics.json','w'),indent=2)
 out=pd.DataFrame({'sample_id':meta.iloc[:,0],'true_label':y,'predicted_label':pred,'probability_background':1-prob,'probability_promoter':prob}); out.to_csv(a.output_dir/'test_predictions.csv',index=False); pd.DataFrame([[tn,fp],[fn,tp]],index=['true_0','true_1'],columns=['pred_0','pred_1']).to_csv(a.output_dir/'confusion_matrix.csv')
 import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
 fpr,tpr,_=roc_curve(y,prob); plt.plot(fpr,tpr); plt.savefig(a.output_dir/'roc_curve.png'); plt.close(); pr,re,_=precision_recall_curve(y,prob); plt.plot(re,pr); plt.savefig(a.output_dir/'pr_curve.png'); plt.close(); print(json.dumps(res,indent=2))
if __name__=='__main__': main()

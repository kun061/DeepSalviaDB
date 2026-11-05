import os,json,hashlib,argparse
import numpy as np,pandas as pd,torch
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,confusion_matrix,classification_report,roc_curve,precision_recall_curve
from transformers import AutoTokenizer,AutoModelForSequenceClassification
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); DATA=os.path.join(ROOT,'DSS3/promoter_1kb/chromosome_holdout_v4_gc_matched'); OUT=os.path.join(ROOT,'finetune/dss3_6mer_1000bp_head_only_v4_gc_seed42'); CK=os.path.join(OUT,'best_model')
def kmers(s): return ' '.join(s[i:i+6].upper() for i in range(len(s)-5))
def main():
 tok=AutoTokenizer.from_pretrained(CK,model_max_length=1024); model=AutoModelForSequenceClassification.from_pretrained(CK); model.config.pad_token_id=tok.pad_token_id; model.eval(); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model.to(dev)
 d=pd.read_csv(os.path.join(DATA,'test.csv')); m=pd.read_csv(os.path.join(DATA,'test.meta.csv')); assert len(d)==len(m) and d.sequence.map(len).eq(1000).all()
 probs=[]
 with torch.inference_mode():
  for i in range(0,len(d),8):
   z=tok([kmers(x) for x in d.sequence.iloc[i:i+8]],padding=True,truncation=False,return_tensors='pt'); z={k:v.to(dev) for k,v in z.items()}; probs.extend(torch.softmax(model(**z).logits,1)[:,1].cpu().numpy())
 y=d.label.to_numpy(); p=np.array(probs); pred=(p>=.5).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel(); td=os.path.join(OUT,'test_results'); os.makedirs(td,exist_ok=True)
 res={'accuracy':accuracy_score(y,pred),'precision':precision_score(y,pred,zero_division=0),'recall':recall_score(y,pred,zero_division=0),'f1':f1_score(y,pred,zero_division=0),'roc_auc':roc_auc_score(y,p),'pr_auc_ap':average_precision_score(y,p),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),'test_size':len(y),'positive_count':int(y.sum()),'negative_count':int((y==0).sum()),'threshold':.5,'checkpoint_path':CK,'seed':42,'model_name':'Plant DNAGPT-6mer','sequence_length_bp':1000,'kmer':6,'actual_token_length':997,'model_max_length':1024}
 json.dump(res,open(os.path.join(td,'test_metrics.json'),'w'),indent=2); pd.DataFrame({'sample_id':m.sample_id,'chromosome':m.chromosome,'true_label':y,'predicted_label':pred,'probability_background':1-p,'probability_promoter':p}).to_csv(os.path.join(td,'test_predictions.csv'),index=False); pd.DataFrame([[tn,fp],[fn,tp]],index=['true_0','true_1'],columns=['pred_0','pred_1']).to_csv(os.path.join(td,'confusion_matrix.csv')); open(os.path.join(td,'classification_report.txt'),'w').write(classification_report(y,pred,digits=6))
 import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
 fpr,tpr,_=roc_curve(y,p); plt.plot(fpr,tpr); plt.savefig(os.path.join(td,'roc_curve.png')); plt.close(); pr,re,_=precision_recall_curve(y,p); plt.plot(re,pr); plt.savefig(os.path.join(td,'pr_curve.png')); plt.close(); plt.figure(); plt.imshow([[tn,fp],[fn,tp]]); plt.colorbar(); plt.savefig(os.path.join(td,'confusion_matrix.png')); plt.close(); print(json.dumps(res,indent=2))
if __name__=='__main__': main()

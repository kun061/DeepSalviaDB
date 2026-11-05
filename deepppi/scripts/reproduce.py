"""Re-evaluate the packaged D checkpoints without raw STRING downloads."""
from pathlib import Path
import argparse,importlib.metadata,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'revision'))
from train_review import load_prepared,arrays,build_model,setup_tensorflow,read_rows,write_rows,save_json,sha_file
from review_metrics import binary_metrics
from verify_package import verify

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'runs/evaluation_D')
    parser.add_argument('--weights-dir',type=Path,default=ROOT/'weights/D')
    args=parser.parse_args()
    checked=verify()
    if args.output_dir.exists():raise ValueError('Keep existing output; choose a new --output-dir.')
    meta,proteins,groups,splits=load_prepared(ROOT/'data/D',['protein_disjoint'])
    tf=setup_tensorflow(42)
    args.output_dir.mkdir(parents=True)
    x,y=arrays(splits['protein_disjoint']['test'],proteins)
    records=[];comparisons=[]
    original=args.weights_dir.resolve()==(ROOT/'weights/D').resolve()
    keys=['roc_auc','pr_auc_trapezoid','average_precision','precision','recall','f1','accuracy','specificity','mcc']
    for seed in [42,43,44]:
        run=args.weights_dir/('seed_'+str(seed));weights=run/'best.weights.h5'
        source=json.loads((run/'run_metadata.json').read_text(encoding='utf-8'))
        if sha_file(weights)!=source['weights_sha256']:raise ValueError('Checkpoint hash mismatch')
        tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(seed)
        model=build_model(tf);model.load_weights(str(weights))
        if model.count_params()!=214977:raise ValueError('Unexpected model')
        prob=model.predict(x,batch_size=128,verbose=0).ravel()
        swap=model.predict(x[::-1],batch_size=128,verbose=0).ravel()
        symmetry=float(np.max(np.abs(prob-swap)))
        if symmetry>1e-5:raise ValueError('Swap symmetry check failed')
        metrics,curves=binary_metrics(y,prob)
        out=args.output_dir/('seed_'+str(seed));out.mkdir()
        rs=splits['protein_disjoint']['test']
        write_rows(out/'predictions.csv.gz',[dict(r,probability=float(v),predicted_label=int(v>=.5)) for r,v in zip(rs,prob)])
        write_rows(out/'curves.csv.gz',[dict(zip(curves,vs)) for vs in zip(*curves.values())])
        comparison=None
        if original:
            reference=ROOT/'results/D_test'/('seed_'+str(seed))
            expected=json.loads((reference/'metrics.json').read_text(encoding='utf-8'))['metrics']
            previous=read_rows(reference/'predictions.csv.gz')
            if [r['pair_id'] for r in previous]!=[r['pair_id'] for r in rs]:raise ValueError('Test reference identity mismatch')
            err=float(np.max(np.abs(prob-np.array([float(r['probability']) for r in previous]))))
            metricerr=max(abs(metrics[k]-expected[k]) for k in keys)
            if err>1e-6 or metricerr>1e-6:raise ValueError('Reproduction differs from recorded reference; inspect platform/environment.')
            comparison={'seed':seed,'max_probability_difference':err,'max_metric_difference':metricerr,'tolerance':1e-6}
            comparisons.append(comparison)
        save_json(out/'metrics.json',{'seed':seed,'metrics':metrics,'max_swap_difference':symmetry,'reference_comparison':comparison})
        records.append(dict(seed=seed,**metrics))
        print('Evaluated seed '+str(seed)+': '+str(metrics),flush=True)
    write_rows(args.output_dir/'all_metrics.csv',records)
    aggregate={}
    for k in keys:
        vals=np.array([r[k] for r in records]);aggregate[k+'_mean']=float(vals.mean());aggregate[k+'_sd']=float(vals.std(ddof=1))
    save_json(args.output_dir/'aggregate.json',aggregate)
    save_json(args.output_dir/'verification.json',{'status':'complete','data_and_manifest':checked,'reference_comparisons':comparisons,'original_checkpoint_reference_checked':original,'packages':{n:importlib.metadata.version(n) for n in ['tensorflow','keras','numpy','h5py','matplotlib']},'tensorflow_build':tf.sysconfig.get_build_info(),'gpus':[g.name for g in tf.config.list_physical_devices('GPU')],'python':sys.version})
    print('Complete: '+str(args.output_dir))
if __name__=='__main__':main()

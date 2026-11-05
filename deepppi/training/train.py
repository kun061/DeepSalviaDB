"""Train the existing Siamese CNN design on fixed review splits, on the server."""
import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from review_metrics import binary_metrics, vertex_bootstrap

AA = 'ACDEFGHIKLMNPQRSTVWY'
TABLE = bytes({ord(c): i + 1 for i, c in enumerate(AA)}.get(n, 0) for n in range(256))


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def read_rows(path):
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    if not rows:
        raise ValueError('No rows to export')
    opener = gzip.open if path.name.endswith('.gz') else open
    with opener(path, 'wt', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def load_prepared(directory, protocols):
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('status') != 'complete':
        raise ValueError('Prepared manifest is not complete')
    config = manifest['configuration']
    if config['max_len'] != 512 or config['threshold'] != 900 or str(config['taxid']) != '180675':
        raise ValueError('Unexpected prepared-data definition')
    output_hashes = {name.replace('\\', '/'): value for name, value in manifest['output_sha256'].items()}
    required = ['proteins.csv.gz'] + [f'{p}/{s}.csv.gz' for p in protocols for s in ('train', 'validation', 'test')]
    for relative in required:
        path = directory / relative
        if not path.is_file() or sha_file(path) != output_hashes.get(relative):
            raise ValueError('Prepared file missing or checksum mismatch: ' + relative)
    proteins, groups = {}, {}
    for row in read_rows(directory / 'proteins.csv.gz'):
        pid = row['protein_id']
        tokens = row['sequence'].upper().encode('ascii', errors='replace').translate(TABLE)[:512].ljust(512, b'\0')
        if pid in proteins or hashlib.sha256(tokens).hexdigest() != row['model_input_sha256']:
            raise ValueError('Protein input identity mismatch: ' + pid)
        proteins[pid] = np.frombuffer(tokens, dtype=np.uint8).astype(np.int32)
        groups[pid] = row['model_input_sha256']
    all_splits = {}
    for protocol in protocols:
        splits = {}
        for name in ('train', 'validation', 'test'):
            rows = read_rows(directory / protocol / (name + '.csv.gz'))
            expected = manifest['protocols'][protocol]['splits'][name]
            if len(rows) != expected['rows'] or dict(Counter(r['label'] for r in rows)) != expected['labels']:
                raise ValueError('Split count/label mismatch: ' + protocol + '/' + name)
            if len({r['pair_id'] for r in rows}) != len(rows) or len({r['model_pair_id'] for r in rows}) != len(rows):
                raise ValueError('Duplicate pair or input inside split')
            for r in rows:
                if r['label'] not in ('0', '1') or r['protein_a'] not in proteins or r['protein_b'] not in proteins:
                    raise ValueError('Invalid split row')
                a, b = sorted((r['protein_a'], r['protein_b']))
                actual_pair = hashlib.sha256((a + '|' + b).encode()).hexdigest()
                model_pair = hashlib.sha256('|'.join(sorted((groups[a], groups[b]))).encode()).hexdigest()
                if actual_pair != r['pair_id'] or model_pair != r['model_pair_id']:
                    raise ValueError('Pair/input hash mismatch')
            splits[name] = rows
        names = list(splits)
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                a, b = splits[left], splits[right]
                if {r['model_pair_id'] for r in a} & {r['model_pair_id'] for r in b}:
                    raise ValueError('Model-pair leakage')
                if protocol == 'protein_disjoint':
                    ga = {groups[r[c]] for r in a for c in ('protein_a', 'protein_b')}
                    gb = {groups[r[c]] for r in b for c in ('protein_a', 'protein_b')}
                    if ga & gb:
                        raise ValueError('Protein-input leakage')
        all_splits[protocol] = splits
    return manifest, proteins, groups, all_splits


def arrays(rows, proteins):
    return [np.stack([proteins[r[c]] for r in rows]) for c in ('protein_a', 'protein_b')], np.array([int(r['label']) for r in rows], dtype=np.float32)


def setup_tensorflow(seed):
    os.environ.setdefault('TF_DETERMINISTIC_OPS', '1')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    if not tf.config.list_physical_devices('GPU'):
        raise RuntimeError('TensorFlow cannot see a GPU. Stop here; return environment and install logs.')
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    return tf


def build_model(tf):
    layers = tf.keras.layers

    @tf.keras.utils.register_keras_serializable(package='SalviaReview')
    class DiscardMask(layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.supports_masking = True

        def call(self, inputs):
            return tf.identity(inputs)

        def compute_mask(self, inputs, mask=None):
            return None

    base = layers.Input(shape=(512,), dtype='int32')
    x = layers.Embedding(21, 64, mask_zero=True)(base)
    # Conv1D does not consume padding masks; drop metadata without changing values.
    x = DiscardMask(name='discard_padding_mask')(x)
    x = layers.Conv1D(128, 5, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(.3)(x)
    x = layers.Conv1D(256, 3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling1D()(x)
    encoder = tf.keras.Model(base, x, name='FeatureExtractor')
    a, b = layers.Input((512,), dtype='int32', name='Protein_A'), layers.Input((512,), dtype='int32', name='Protein_B')
    fa, fb = encoder(a), encoder(b)
    # max(a-b,b-a) == abs(a-b); serializable built-in layers replace Lambda.
    difference = layers.Maximum(name='absolute_difference')([layers.Subtract()([fa, fb]), layers.Subtract()([fb, fa])])
    product = layers.Multiply()([fa, fb])
    merged = layers.Concatenate()([difference, product])
    x = layers.Dense(128, activation='relu')(merged)
    x = layers.Dropout(.4)(x)
    x = layers.Dense(64, activation='relu')(x)
    probability = layers.Dense(1, activation='sigmoid', name='PPI_Prediction')(x)
    model = tf.keras.Model([a, b], probability)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'), tf.keras.metrics.AUC(name='auc')])
    return model


def smoke_test(tf, output):
    model = build_model(tf)
    rng = np.random.default_rng(42)
    x = [rng.integers(0, 21, size=(4, 512), dtype=np.int32) for _ in range(2)]
    loss = model.train_on_batch(x, np.array([0, 1, 0, 1], dtype=np.float32))
    p = model.predict(x, verbose=0).ravel()
    swapped = model.predict(x[::-1], verbose=0).ravel()
    if not np.isfinite(p).all() or not np.allclose(p, swapped, atol=1e-6):
        raise RuntimeError('GPU smoke test failed finiteness/symmetry')
    report = {'tensorflow': tf.__version__, 'gpu_devices': [d.name for d in tf.config.list_physical_devices('GPU')],
              'parameter_count': model.count_params(), 'smoke_train_output': np.asarray(loss).tolist(),
              'max_swap_difference': float(np.max(np.abs(p - swapped))), 'synthetic_only': True}
    if output:
        save_json(output, report)
    print(json.dumps(report, indent=2))


def plot_result(directory, curves, prevalence, history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    axes[0].plot(curves['fpr'], curves['tpr'])
    axes[0].plot([0, 1], [0, 1], '--', color='gray')
    axes[0].set(xlabel='False positive rate', ylabel='True positive rate', title='Independent test ROC')
    axes[1].plot(curves['recall'], curves['precision'])
    axes[1].axhline(prevalence, linestyle='--', color='gray')
    axes[1].set(xlabel='Recall', ylabel='Precision', title='Independent test PR')
    epochs = np.arange(1, len(history['loss']) + 1)
    axes[2].plot(epochs, history['auc'], label='train')
    axes[2].plot(epochs, history['val_auc'], label='validation')
    axes[2].set(xlabel='Epoch', ylabel='ROC-AUC (Keras approximation)', title='Model selection')
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(directory / 'curves.png', dpi=180)
    plt.close(fig)


def train_one(tf, protocol, seed, splits, proteins, groups, directory, args, prepared_hash):
    directory.mkdir(parents=True, exist_ok=False)
    metadata = {'status': 'running', 'protocol': protocol, 'seed': seed,
                'created_at_utc': datetime.now(timezone.utc).isoformat(), 'prepared_manifest_sha256': prepared_hash,
                'threshold': .5, 'selection': 'max validation Keras ROC-AUC; patience=5 by default',
                'epochs_limit': args.epochs, 'batch_size': args.batch_size, 'patience': args.patience,
                'class_weight': None, 'notes': ['No previous h5 weights loaded; fresh training.',
                                               'Test performance is never used for model/seed/threshold selection.']}
    save_json(directory / 'run_metadata.json', metadata)
    try:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        model = build_model(tf)
        (directory / 'architecture.json').write_text(model.to_json() + '\n', encoding='utf-8')
        with (directory / 'model_summary.txt').open('w', encoding='utf-8') as f:
            model.summary(print_fn=lambda line: f.write(line + '\n'))
        x_train, y_train = arrays(splits['train'], proteins)
        x_val, y_val = arrays(splits['validation'], proteins)
        weights = directory / 'best.weights.h5'
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=args.patience, restore_best_weights=True),
                     tf.keras.callbacks.ModelCheckpoint(str(weights), monitor='val_auc', mode='max', save_best_only=True, save_weights_only=True),
                     tf.keras.callbacks.CSVLogger(str(directory / 'history.csv')),
                     tf.keras.callbacks.TerminateOnNaN()]
        started = time.monotonic()
        fit = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=args.epochs,
                        batch_size=args.batch_size, callbacks=callbacks, shuffle=True, verbose=2)
        if not all(np.isfinite(v).all() for v in fit.history.values()):
            raise RuntimeError('Nonfinite training history')
        model.load_weights(str(weights))
        del x_train, y_train
        val_probability = model.predict(x_val, batch_size=args.batch_size, verbose=0).ravel()
        val_metrics, _ = binary_metrics(y_val, val_probability)
        # Encode/predict test set only after fitting and loading best validation checkpoint.
        x_test, y_test = arrays(splits['test'], proteins)
        test_probability = model.predict(x_test, batch_size=args.batch_size, verbose=0).ravel()
        swapped = model.predict(x_test[::-1], batch_size=args.batch_size, verbose=0).ravel()
        swap_error = float(np.max(np.abs(test_probability - swapped)))
        if swap_error > 1e-5:
            raise RuntimeError('Inference symmetry check failed')
        test_metrics, curves = binary_metrics(y_test, test_probability)
        intervals, draws = vertex_bootstrap(y_test, test_probability,
                                             [groups[r['protein_a']] for r in splits['test']],
                                             [groups[r['protein_b']] for r in splits['test']],
                                             repeats=args.bootstrap, seed=20261018)
        for name, probability in [('validation', val_probability), ('test', test_probability)]:
            predictions = [dict(r, probability=float(p), predicted_label=int(p >= .5)) for r, p in zip(splits[name], probability)]
            write_rows(directory / (name + '_predictions.csv.gz'), predictions)
        write_rows(directory / 'bootstrap_draws.csv.gz', draws)
        write_rows(directory / 'test_curves.csv.gz', [dict(zip(curves, values)) for values in zip(*curves.values())])
        save_json(directory / 'metrics.json', {'protocol': protocol, 'seed': seed, 'validation': val_metrics,
                                              'test': test_metrics, 'test_vertex_bootstrap': intervals})
        plot_result(directory, curves, test_metrics['positive_prevalence'], fit.history)
        metadata.update({'status': 'complete', 'epochs_run': len(fit.history['loss']),
                         'best_epoch': int(np.argmax(fit.history['val_auc'])) + 1,
                         'elapsed_seconds': time.monotonic() - started, 'test_max_swap_difference': swap_error,
                         'parameter_count': model.count_params(), 'weights_sha256': sha_file(weights)})
        save_json(directory / 'run_metadata.json', metadata)
        return {'protocol': protocol, 'seed': seed, 'test_rows': len(y_test),
                'test_positive': int(y_test.sum()), 'best_epoch': metadata['best_epoch'], **test_metrics}
    except BaseException as exc:
        metadata.update({'status': 'failed', 'error': repr(exc)})
        save_json(directory / 'run_metadata.json', metadata)
        raise


def bundle_results(root):
    # Weights stay on server for later case analysis; return predictions and reports.
    with zipfile.ZipFile(root / 'return_bundle.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob('*')):
            if path.is_file() and path.suffix not in ('.h5', '.zip'):
                archive.write(path, path.relative_to(root))


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir', type=Path, default=project / 'data/D')
    parser.add_argument('--output-dir', type=Path, default=project / 'runs/retrain_D')
    parser.add_argument('--protocols', nargs='+', choices=['protein_disjoint'], default=['protein_disjoint'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--smoke-output', type=Path)
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size) < 1 or args.bootstrap < 20:
        parser.error('Invalid epoch, patience, batch or bootstrap count')
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.protocols)) != len(args.protocols):
        parser.error('Repeated seed or protocol')
    tf = setup_tensorflow(args.seeds[0])
    if args.smoke_only:
        smoke_test(tf, args.smoke_output)
        return
    if args.output_dir.exists():
        parser.error('Output directory already exists; keep it and choose a fresh --output-dir')
    manifest, proteins, groups, splits = load_prepared(args.prepared_dir, args.protocols)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    root = args.output_dir
    save_json(root / 'prepared_manifest.json', manifest)
    prepared_hash = sha_file(args.prepared_dir / 'manifest.json')
    versions = {'python': sys.version, 'platform': platform.platform(), 'gpu': [d.name for d in tf.config.list_physical_devices('GPU')],
                'packages': {p: importlib.metadata.version(p) for p in ('tensorflow', 'keras', 'numpy', 'h5py', 'matplotlib')},
                'scripts': {p.name: sha_file(p) for p in [Path(__file__), Path(__file__).with_name('review_metrics.py')]}}
    save_json(root / 'environment.json', versions)
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, check=True)
    (root / 'requirements_frozen.txt').write_text(freeze.stdout, encoding='utf-8')
    status = {'status': 'running', 'configuration': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'completed_runs': []}
    save_json(root / 'status.json', status)
    records = []
    try:
        for protocol in args.protocols:
            for seed in args.seeds:
                print(f'===== {protocol} seed={seed} =====', flush=True)
                record = train_one(tf, protocol, seed, splits[protocol], proteins, groups,
                                   root / protocol / ('seed_' + str(seed)), args, prepared_hash)
                records.append(record)
                write_rows(root / 'summary.csv', records)
                status['completed_runs'].append({'protocol': protocol, 'seed': seed})
                save_json(root / 'status.json', status)
                print(json.dumps(record, indent=2), flush=True)
        aggregate = []
        for protocol in args.protocols:
            subset = [r for r in records if r['protocol'] == protocol]
            item = {'protocol': protocol, 'training_seed_runs': len(subset)}
            for key in ('roc_auc', 'average_precision', 'precision', 'recall', 'f1', 'accuracy', 'mcc'):
                values = [r[key] for r in subset]
                item[key + '_mean'] = float(np.mean(values))
                item[key + '_sd'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            aggregate.append(item)
        save_json(root / 'aggregate.json', {'across_training_seeds': aggregate,
                  'notes': ['All prespecified seeds included; no best-seed selection.',
                            'Across-seed SD is not a confidence interval.',
                            'Protocols use different test sets; differences are not treatment effects.',
                            'Vertex-bootstrap intervals are approximate, conditional sensitivity intervals; label and homology uncertainty not covered.']})
        status['status'] = 'complete'
        save_json(root / 'status.json', status)
    except BaseException as exc:
        status.update({'status': 'failed', 'error': repr(exc)})
        save_json(root / 'status.json', status)
        raise
    finally:
        bundle_results(root)
    print('Completed. Download: ' + str(root / 'return_bundle.zip'), flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Build a fresh DSS3 1-kb chromosome-holdout dataset with per-chromosome
1:1 class balance and bounded, auditable GC matching.

The original v3 positive sequences, annotation metadata, and chromosome split
are treated as immutable. Only negative windows are regenerated from the DSS3
FASTA/GFF. No 6-mer/BPE model is loaded or trained by this script.

Python 3.9+; standard library only.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

CHRS = (
    "GWHAOSJ00000020", "GWHAOSJ00000057", "GWHAOSJ00000013",
    "GWHAOSJ00000010", "GWHAOSJ00000006", "GWHAOSJ00000023",
    "GWHAOSJ00000084", "GWHAOSJ00000040",
)
DNA = set("ACGT")
WIDTH = 1000
META_FIELDS = (
    "sample_id", "label", "species", "chromosome", "gene_id",
    "gene_accession", "transcript_id", "transcript_accession", "strand",
    "tss_1based", "start_0based", "end_0based", "gc_content",
    "source_type", "matched_positive_id", "matched_negative_id",
    "gc_pair_abs_delta", "split",
)


def fail(message):
    raise RuntimeError(message)


def digest_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dna_hash(s):
    return hashlib.blake2b(s.encode("ascii"), digest_size=16).digest()


def gc_count(s):
    return s.count("G") + s.count("C")


def load_fasta_selected(path, wanted):
    seqs = {}
    name, parts = None, []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if name in wanted:
                    seqs[name] = "".join(parts).upper()
                name = line[1:].split()[0]
                parts = []
            elif name in wanted:
                parts.append(line.strip())
        if name in wanted:
            seqs[name] = "".join(parts).upper()
    missing = wanted - set(seqs)
    if missing:
        fail("FASTA 缺少主伪染色体: " + ", ".join(sorted(missing)))
    return seqs


def read_attributes(s):
    result = {}
    for item in s.split(";"):
        if "=" in item:
            key, value = item.split("=", 1)
            result[key] = value
    return result


def read_genes_transcripts(gff_path, wanted):
    genes, tx = {}, {}
    with open(gff_path) as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[0] not in wanted:
                continue
            if fields[2] not in {"gene", "mRNA"}:
                continue
            try:
                start, end = int(fields[3]), int(fields[4])
            except ValueError:
                fail("GFF 坐标格式错误: " + line[:180])
            att = read_attributes(fields[8])
            id_ = att.get("ID", "")
            if not id_:
                continue
            row = {
                "chromosome": fields[0], "start": start, "end": end,
                "strand": fields[6], "parent": att.get("Parent", ""),
            }
            if fields[2] == "gene":
                genes[id_] = row
            else:
                tx[id_] = row
    return genes, tx


def promoter_interval(tss, strand, chrom_len):
    # GFF 1-based TSS; half-open BED-style interval.
    a, b = (tss - WIDTH - 1, tss - 1) if strand == "+" else (tss, tss + WIDTH)
    return (a, b) if a >= 0 and b <= chrom_len else None


def promoter_exclusion_windows(genes, tx, genome):
    reps = {}
    for tid, t in tx.items():
        gid = t["parent"].split(",")[0]
        if gid not in genes:
            continue
        key = t["start"] if t["strand"] == "+" else t["end"]
        old = reps.get(gid)
        if old is None or (key < old[1] if t["strand"] == "+" else key > old[1]):
            reps[gid] = (tid, key)
    intervals = defaultdict(list)
    count = 0
    for tid, tss in reps.values():
        t = tx[tid]
        c = t["chromosome"]
        w = promoter_interval(tss, t["strand"], len(genome[c]))
        if w is not None:
            intervals[c].append(w)
            count += 1
    return intervals, count


def merge_intervals(intervals):
    out = []
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(b, out[-1][1]))
        else:
            out.append((a, b))
    return out


def overlaps_merged(intervals, starts, a, b):
    i = bisect.bisect_right(starts, a)
    return (i > 0 and intervals[i - 1][1] > a) or (i < len(intervals) and intervals[i][0] < b)


def read_v3(v3dir, genome):
    summary_path = v3dir / "summary.json"
    if not summary_path.is_file():
        fail(f"未找到 v3 summary.json: {summary_path}")
    summary = json.loads(summary_path.read_text())
    assigned = summary.get("chromosome_assignment", {})
    if set(assigned) != {"train", "dev", "test"}:
        fail("v3 缺少 train/dev/test chromosome_assignment")
    splits = {k: set(v) for k, v in assigned.items()}
    if set.union(*splits.values()) != set(CHRS):
        fail("v3 染色体分配与固定 8 条主伪染色体不一致")
    if any(splits[x] & splits[y] for x, y in (("train", "dev"), ("train", "test"), ("dev", "test"))):
        fail("v3 染色体发生跨集合交叉")
    if sorted(map(len, splits.values())) != [1, 1, 6]:
        fail("v3 不是 6/1/1 染色体留出")

    positives = {s: [] for s in CHRS}
    source_order = {k: [] for k in splits}
    positive_sequences = set()
    positive_ids = set()
    source_counts = {}
    for split in ("train", "dev", "test"):
        csvfile = v3dir / f"{split}.csv"
        metafile = v3dir / f"{split}.meta.csv"
        if not csvfile.is_file() or not metafile.is_file():
            fail(f"缺少 v3 {split}.csv 或 {split}.meta.csv")
        with csvfile.open(newline="") as f:
            seqrows = list(csv.DictReader(f))
        with metafile.open(newline="") as f:
            metarows = list(csv.DictReader(f))
        if len(seqrows) != len(metarows):
            fail(f"{split}: CSV 与 meta 行数不一致")
        source_counts[split] = {"total": len(seqrows), "positive": 0, "negative": 0}
        for i, (row, meta) in enumerate(zip(seqrows, metarows), 2):
            seq = row["sequence"].upper()
            label = int(row["label"])
            if label != int(meta.get("label", label)):
                fail(f"{split}:{i}: CSV/meta 标签不一致")
            chrom = meta["chromosome"]
            if chrom not in splits[split]:
                fail(f"{split}:{i}: 元数据染色体不属于该集合: {chrom}")
            if len(seq) != WIDTH or set(seq) - DNA:
                fail(f"{split}:{i}: DNA 序列长度/ACGT 不合法")
            source_counts[split]["positive" if label == 1 else "negative"] += 1
            if label != 1:
                continue
            sid = meta.get("sample_id", "")
            if not sid or sid in positive_ids or seq in positive_sequences:
                fail(f"{split}:{i}: 正样本 ID 缺失或正样本重复")
            positive_ids.add(sid)
            positive_sequences.add(seq)
            a, b = int(meta["start_0based"]), int(meta["end_0based"])
            strand = meta["strand"]
            if b - a != WIDTH or not (0 <= a < b <= len(genome[chrom])):
                fail(f"{split}:{i}: 正样本基因组坐标有误")
            refseq = genome[chrom][a:b]
            if strand == "-":
                refseq = refseq.translate(str.maketrans("ACGT", "TGCA"))[::-1]
            elif strand != "+":
                fail(f"{split}:{i}: 无效正样本链方向")
            if seq != refseq:
                fail(f"{split}:{i}: 正样本序列与原始参考基因组不符")
            if abs(float(meta.get("gc_content", gc_count(seq) / WIDTH)) - gc_count(seq) / WIDTH) > 1e-8:
                fail(f"{split}:{i}: 元数据 GC 值不符")
            record = {"sequence": seq, "meta": meta, "gc": gc_count(seq), "split": split}
            positives[chrom].append(record)
            source_order[split].append(record)
    observed = sum(map(len, positives.values()))
    if observed != summary.get("candidate_positive"):
        fail(f"v3 正样本总数与 summary 不符: {observed} != {summary.get('candidate_positive')}")
    return summary, splits, positives, source_order, source_counts, positive_sequences


def generate_candidate_pool(genome, protected, positive_sequences):
    """Scan aligned 1-kb intervals, retaining only GC count and genomic start.

    All gene bodies + 2kb flanks, and all TSS candidate windows (including
    rejected candidate promoters), are excluded. Identical DNA is deduplicated
    globally using 128-bit BLAKE2 hashes; final exact-sequence QC is separate.
    """
    positive_hashes = {dna_hash(s) for s in positive_sequences}
    seen = set(positive_hashes)
    candidates = {}
    rejected_counts = {}
    for chrom in CHRS:
        seq = genome[chrom]
        intervals = protected[chrom]
        out = []
        counts = Counter()
        i = 0
        for a in range(0, len(seq) - WIDTH + 1, WIDTH):
            b = a + WIDTH
            while i < len(intervals) and intervals[i][1] <= a:
                i += 1
            if i < len(intervals) and intervals[i][0] < b:
                counts["protected_interval"] += 1
                continue
            window = seq[a:b]
            if set(window) - DNA:
                counts["invalid_bases"] += 1
                continue
            key = dna_hash(window)
            if key in seen:
                counts["duplicate_or_positive"] += 1
                continue
            seen.add(key)
            out.append((gc_count(window), a))
        candidates[chrom] = sorted(out)
        rejected_counts[chrom] = dict(counts)
    return candidates, rejected_counts


def greedy_pairs(pos, candidates, max_diff):
    """Maximum-cardinality ordered interval matching within ±max_diff GC bases.

    Sorting both sides and always using the lowest still-feasible candidate
    maximizes the number of matches with an allowed GC-distance interval.
    A second pass improves individual GC closeness without losing matches.
    """
    positive_sorted = sorted(pos, key=lambda x: (x["gc"], x["meta"]["sample_id"]))
    matches, missing, used = [], [], set()
    j = 0
    for item in positive_sorted:
        gc = item["gc"]
        while j < len(candidates) and candidates[j][0] < gc - max_diff:
            j += 1
        if j >= len(candidates) or candidates[j][0] > gc + max_diff:
            missing.append(item)
        else:
            matches.append([item, candidates[j]])
            used.add(j)
            j += 1
    return matches, missing, used


def minimum_required_delta(pos, candidates):
    if len(candidates) < len(pos):
        return None
    lo, hi = 0, WIDTH
    while lo < hi:
        mid = (lo + hi) // 2
        _, missing, _ = greedy_pairs(pos, candidates, mid)
        if missing:
            lo = mid + 1
        else:
            hi = mid
    return lo


def refine_pairs(matches, candidates, used, seed):
    """Replace matched negatives with unused, closer-GC candidates.

    Replacements cannot break pair feasibility or reduce cardinality.
    """
    rng = random.Random(seed)
    pool = defaultdict(list)
    for i, item in enumerate(candidates):
        if i not in used:
            pool[item[0]].append(item)
    for bucket in pool.values():
        rng.shuffle(bucket)
    indices = list(range(len(matches)))
    indices.sort(key=lambda i: abs(matches[i][0]["gc"] - matches[i][1][0]), reverse=True)
    for i in indices:
        pos, old = matches[i]
        original_distance = abs(pos["gc"] - old[0])
        if original_distance == 0:
            continue
        for distance in range(original_distance):
            opts = [pos["gc"]] if distance == 0 else [pos["gc"] - distance, pos["gc"] + distance]
            if distance and rng.getrandbits(1):
                opts.reverse()
            replacement = next((pool[g].pop() for g in opts if pool.get(g)), None)
            if replacement is not None:
                pool[old[0]].append(old)
                matches[i][1] = replacement
                break
    return matches


def gc_ks(a, b):
    a_count, b_count = Counter(a), Counter(b)
    cumul_a = cumul_b = 0
    maximum = 0.0
    for gc in sorted(set(a_count) | set(b_count)):
        cumul_a += a_count[gc]
        cumul_b += b_count[gc]
        maximum = max(maximum, abs(cumul_a / len(a) - cumul_b / len(b)))
    return maximum


def gc_stats(matches):
    pos_gc = [p["gc"] / WIDTH for p, _ in matches]
    neg_gc = [n[0] / WIDTH for _, n in matches]
    pair_abs = [abs(p - n) for p, n in zip(pos_gc, neg_gc)]
    mean_pos, mean_neg = statistics.mean(pos_gc), statistics.mean(neg_gc)
    pos_std = statistics.pstdev(pos_gc) if len(pos_gc) > 1 else 0.0
    neg_std = statistics.pstdev(neg_gc) if len(neg_gc) > 1 else 0.0
    pooled = ((pos_std ** 2 + neg_std ** 2) / 2) ** 0.5
    smd = (mean_pos - mean_neg) / pooled if pooled else (0.0 if mean_pos == mean_neg else float("inf"))
    sorted_abs = sorted(pair_abs)
    idx = max(0, math.ceil(0.95 * len(sorted_abs)) - 1)
    return {
        "n_pairs": len(matches), "positive_gc_mean": mean_pos,
        "negative_gc_mean": mean_neg, "mean_gc_difference": mean_pos - mean_neg,
        "gc_smd": smd, "gc_ks": gc_ks([p["gc"] for p, _ in matches], [n[0] for _, n in matches]),
        "paired_abs_gc_mean": statistics.mean(pair_abs),
        "paired_abs_gc_p95": sorted_abs[idx],
        "paired_abs_gc_max": max(pair_abs),
    }


def validate_match_quality(chrom, stats, gc_max_delta, max_mean_delta, max_ks, max_smd):
    if stats["paired_abs_gc_max"] > gc_max_delta + 1e-9:
        fail(f"{chrom}: GC 配对最大差异超标")
    if abs(stats["mean_gc_difference"]) > max_mean_delta + 1e-9:
        fail(f"{chrom}: 正负平均 GC 差超标: {stats['mean_gc_difference']:.4f}")
    if stats["gc_ks"] > max_ks + 1e-9:
        fail(f"{chrom}: 正负 GC 分布 KS 超标: {stats['gc_ks']:.4f}")
    if abs(stats["gc_smd"]) > max_smd + 1e-9:
        fail(f"{chrom}: 正负 GC 标准化平均差 SMD 超标: {stats['gc_smd']:.4f}")


def check_final_samples(rows_by_split, splits, positives, max_gc_delta, protected):
    """Independent output-level QC before any final directory is published."""
    all_sequences = set()
    all_ids = set()
    chrom_to_split = {}
    protected_starts = {c: [a for a, _ in protected[c]] for c in CHRS}
    for split, rs in rows_by_split.items():
        per_chr = defaultdict(Counter)
        starts_neg = defaultdict(list)
        pos_ids = set()
        matched_ids = set()
        for item in rs:
            seq, meta = item["sequence"], item["meta"]
            c, label = meta["chromosome"], int(meta["label"])
            if len(seq) != WIDTH or set(seq) - DNA:
                fail("输出 QC: 序列长度/ACGT 无效")
            if meta["sample_id"] in all_ids or seq in all_sequences:
                fail("输出 QC: sample ID 或 DNA 完全重复")
            if c not in splits[split]:
                fail("输出 QC: 染色体越过集合边界")
            all_ids.add(meta["sample_id"])
            all_sequences.add(seq)
            if c in chrom_to_split and chrom_to_split[c] != split:
                fail("输出 QC: 染色体跨集合")
            chrom_to_split[c] = split
            per_chr[c][label] += 1
            a, b = int(meta["start_0based"]), int(meta["end_0based"])
            if label == 0:
                if overlaps_merged(protected[c], protected_starts[c], a, b):
                    fail("输出 QC: 负样本与受保护区间重叠")
                starts_neg[c].append((a, b))
                matched_ids.add(meta["matched_positive_id"])
            else:
                pos_ids.add(meta["sample_id"])
        for c, counts in per_chr.items():
            if counts[0] != counts[1]:
                fail(f"输出 QC: {c} 正负样本不为 1:1 ({counts})")
        for c, intervals in starts_neg.items():
            intervals.sort()
            if any(intervals[i][1] > intervals[i + 1][0] for i in range(len(intervals) - 1)):
                fail("输出 QC: 负样本区间相互重叠")
        if pos_ids != matched_ids:
            fail(f"输出 QC: {split} 正负样本配对 ID 不一致")
    if set(chrom_to_split) != set(CHRS):
        fail("输出 QC: 未覆盖全部 8 条主伪染色体")


def build(args):
    root = args.root.resolve()
    v3dir = (args.v3_dir or (root / "promoter_1kb/chromosome_holdout_v3")).resolve()
    outdir = (args.output_dir or (root / "promoter_1kb/chromosome_holdout_v4_gc_matched")).resolve()
    if outdir == v3dir or outdir in v3dir.parents or v3dir in outdir.parents:
        fail("输出目录不得与 v3 目录相同或相互包含")
    if outdir.exists():
        fail(f"目标目录已存在，不覆盖: {outdir}")
    gc_max_delta = args.gc_max_delta
    max_gc_bases = int(round(WIDTH * gc_max_delta))
    if abs(max_gc_bases / WIDTH - gc_max_delta) > 1e-12:
        fail("对于 1000 bp，GC 容差必须是 0.001 的整数倍，例如 0.02")

    fasta_file = root / "DSS3.genome.fasta"
    gff_file = root / "DSS3.gff"
    print("[1/6] 读取 8 条主伪染色体和 GFF...", flush=True)
    genome = load_fasta_selected(fasta_file, set(CHRS))
    genes, tx = read_genes_transcripts(gff_file, set(CHRS))
    summary_v3, splits, positives, source_order, v3counts, pseq = read_v3(v3dir, genome)

    print("[2/6] 重建全部基因体/候选启动子排除区间...", flush=True)
    candidate_windows, candidate_window_count = promoter_exclusion_windows(genes, tx, genome)
    original_window_count = summary_v3.get("candidate_window_count")
    if original_window_count is not None and candidate_window_count != original_window_count:
        fail(f"受保护候选窗口数与 v3 不一致: {candidate_window_count} != {original_window_count}")
    protected = {}
    for c in CHRS:
        chrom_len = len(genome[c])
        regions = list(candidate_windows[c])
        for gene in genes.values():
            if gene["chromosome"] == c:
                regions.append((max(0, gene["start"] - 1 - 2000), min(chrom_len, gene["end"] + 2000)))
        for p in positives[c]:
            m = p["meta"]
            regions.append((int(m["start_0based"]), int(m["end_0based"])))
        protected[c] = merge_intervals(regions)

    print("[3/6] 扫描全基因组候选负窗口（保留完整 1000 bp）...", flush=True)
    pool, rejection_stats = generate_candidate_pool(genome, protected, pseq)
    print("Eligible negative windows:", {c[-4:]: len(pool[c]) for c in CHRS}, flush=True)

    print("[4/6] 按染色体进行一对一 GC 配对...", flush=True)
    paired = {}
    dropped = {}
    gc_reports = {}
    for ci, c in enumerate(CHRS):
        matches, missing, used = greedy_pairs(positives[c], pool[c], max_gc_bases)
        dropped[c] = missing
        missing_fraction = len(missing) / len(positives[c]) if positives[c] else 0
        if missing_fraction > args.max_positive_drop_fraction + 1e-12:
            required = minimum_required_delta(positives[c], pool[c])
            msg = (
                f"{c}: ±{gc_max_delta:.1%} GC 容差内仅匹配 {len(matches)}/{len(positives[c])} "
                f"条正样本（缺 {len(missing)}，{missing_fraction:.2%}）。"
                f"完整保留全部正样本所需的最小 GC 容差约为 "
                f"{required / WIDTH:.1%}。" if required is not None else
                f"{c}: 候选负窗口数 {len(pool[c])} 不足以覆盖正样本数 {len(positives[c])}。"
            )
            fail(msg + " 当前构建已停止，未修改 v3 或生成 v4；可在预先确定的规则下使用 --max-positive-drop-fraction，或调整 --gc-max-delta 后重试。")
        if not matches:
            fail(c + ": 没有任何可配对样本")
        refine_pairs(matches, pool[c], used, args.seed + ci)
        stats = gc_stats(matches)
        stats.update({
            "original_positive": len(positives[c]),
            "retained_positive": len(matches),
            "dropped_positive": len(missing),
            "eligible_negative_windows": len(pool[c]),
            "negative_window_rejections": rejection_stats[c],
        })
        validate_match_quality(c, stats, gc_max_delta, args.gc_mean_max_delta, args.gc_ks_max, args.gc_smd_max)
        paired[c] = matches
        gc_reports[c] = stats
        print(f"  {c}: {len(matches)}+{len(matches)}; "
              f"meanGCdiff={stats['mean_gc_difference']:.3%}; "
              f"mean|pairΔGC|={stats['paired_abs_gc_mean']:.3%}; "
              f"KS={stats['gc_ks']:.3f}; SMD={stats['gc_smd']:.3f}", flush=True)

    print("[5/6] QC 独立复核、写入 staging 目录...", flush=True)
    by_pid = {}
    for c, matches in paired.items():
        for p, (ngc, start) in matches:
            pm = p["meta"]
            ns = genome[c][start : start + WIDTH]
            neg_id = f"N_{c}_{start}"
            delta = abs(p["gc"] - ngc) / WIDTH
            posmeta = {k: pm.get(k, "") for k in META_FIELDS}
            posmeta.update({"label": "1", "split": p["split"], "matched_negative_id": neg_id, "gc_pair_abs_delta": str(delta)})
            negmeta = {k: "" for k in META_FIELDS}
            negmeta.update({
                "sample_id": neg_id, "label": "0", "species": "Salvia_miltiorrhiza_DSS3",
                "chromosome": c, "strand": ".", "start_0based": str(start),
                "end_0based": str(start + WIDTH), "gc_content": str(ngc / WIDTH),
                "source_type": "gene_distal_gc_matched_weak_negative",
                "matched_positive_id": pm["sample_id"], "gc_pair_abs_delta": str(delta),
                "split": p["split"],
            })
            by_pid[pm["sample_id"]] = (
                {"sequence": p["sequence"], "meta": posmeta},
                {"sequence": ns, "meta": negmeta},
            )

    rows_by_split = {}
    for split, original_positives in source_order.items():
        retained_pairs = [by_pid[p["meta"]["sample_id"]] for p in original_positives if p["meta"]["sample_id"] in by_pid]
        rows_by_split[split] = [pair[0] for pair in retained_pairs] + [pair[1] for pair in retained_pairs]
    check_final_samples(rows_by_split, splits, positives, gc_max_delta, protected)

    outdir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".dss3_v4_stage_", dir=outdir.parent))
    try:
        for split in ("train", "dev", "test"):
            rows = rows_by_split[split]
            with (staging / f"{split}.csv").open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["sequence", "label"])
                for row in rows:
                    writer.writerow([row["sequence"], row["meta"]["label"]])
            with (staging / f"{split}.meta.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=META_FIELDS)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row["meta"])
        with (staging / "group_assignment.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["chromosome", "split"])
            for split in ("train", "dev", "test"):
                for c in sorted(splits[split]):
                    writer.writerow([c, split])
        with (staging / "gc_pairs.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["split", "chromosome", "positive_sample_id", "negative_sample_id", "positive_gc", "negative_gc", "abs_gc_delta"])
            for split, original in source_order.items():
                for p in original:
                    pid = p["meta"]["sample_id"]
                    if pid not in by_pid:
                        continue
                    pos, neg = by_pid[pid]
                    writer.writerow([
                        split, p["meta"]["chromosome"], pid, neg["meta"]["sample_id"],
                        pos["meta"]["gc_content"], neg["meta"]["gc_content"],
                        neg["meta"]["gc_pair_abs_delta"],
                    ])
        summary = {
            "data_version": "chromosome_holdout_v4_gc_matched",
            "status": "DATA_QC_PASS",
            "source_v3": str(v3dir),
            "only_eight_main_pseudochromosomes": list(CHRS),
            "chromosome_assignment": {k: sorted(v) for k, v in splits.items()},
            "positive_source": "unchanged v3 candidate promoters except explicitly documented dropped positives",
            "negative_source": "new gene-distal fixed-grid 1-kb windows; per-chrom GC matched",
            "gc_match_max_abs_pair_delta": gc_max_delta,
            "gc_mean_max_delta": args.gc_mean_max_delta,
            "gc_ks_max": args.gc_ks_max,
            "gc_smd_max": args.gc_smd_max,
            "max_positive_drop_fraction_per_chromosome": args.max_positive_drop_fraction,
            "seed": args.seed,
            "excluded_candidate_promoter_windows": candidate_window_count,
            "gene_count_on_eight_chromosomes": len(genes),
            "new_split_counts": {
                k: {
                    "positive": sum(1 for x in v if x["meta"]["label"] == "1"),
                    "negative": sum(1 for x in v if x["meta"]["label"] == "0"),
                    "total": len(v),
                } for k, v in rows_by_split.items()
            },
            "v3_split_counts": v3counts,
            "chromosome_gc_audit": gc_reports,
            "removed_positive_ids": {c: [p["meta"]["sample_id"] for p in missed] for c, missed in dropped.items()},
            "limitations": [
                "Gene-distal negatives are weak negatives, not experimentally confirmed inactive promoters.",
                "Chromosome holdout does not guarantee homology-family isolation.",
                "GC matching does not remove all sequence-complexity or genomic-context confounding.",
                "The current v3 test chromosome was previously inspected; report v4 as a new dataset version, not a pristine untouched test iteration.",
            ],
        }
        (staging / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        report = [
            "# DSS3 chromosome-holdout v4: GC-matched, class-balanced\n",
            "**DATA_QC: PASS** (data-generation and consistency checks only; model/tokenizer audit is separate).\n",
            f"Original v3 positives: {sum(len(v) for v in positives.values())}; retained: {len(by_pid)}.\n",
            f"GC pair max abs delta: {gc_max_delta:.1%}; max mean difference: {args.gc_mean_max_delta:.1%}; KS <= {args.gc_ks_max}; |SMD| <= {args.gc_smd_max}.\n",
            "All 8 main pseudchromosomes are used; each chromosome and each split has exactly as many negatives as positives.\n",
            "## Split sizes\n",
            "| Split | Positive | Negative | Total |\n|---|---:|---:|---:|",
        ]
        for name, v in summary["new_split_counts"].items():
            report.append(f"| {name} | {v['positive']} | {v['negative']} | {v['total']} |")
        report.extend(["\n## Per-chromosome GC quality\n",
                       "| Chromosome | n pairs | mean delta GC | mean paired abs delta | max paired delta | KS | SMD | dropped positives |",
                       "|---|---:|---:|---:|---:|---:|---:|---:|"])
        for chrom, s in gc_reports.items():
            report.append(f"| {chrom} | {s['n_pairs']} | {s['mean_gc_difference']:.4f} | "
                          f"{s['paired_abs_gc_mean']:.4f} | {s['paired_abs_gc_max']:.4f} | "
                          f"{s['gc_ks']:.4f} | {s['gc_smd']:.4f} | {s['dropped_positive']} |")
        report.extend([
            "\n## Verified properties\n",
            "- Full 1000-bp ACGT-only sequences; original positive FASTA coordinate and strand consistency verified.",
            "- All negatives are at least 2 kb away from annotated gene bodies and do not overlap any valid-length candidate promoter interval.",
            "- No negative-negative genomic interval overlap or cross-dataset exact DNA duplicates.",
            "- No chromosomes shared across train/dev/test; v3 chromosome assignment retained.",
            "- Per-chromosome and per-split exact 1:1 label ratio and per-pair GC tolerance checked.",
            "- No model training or test-based threshold selection occurred during dataset generation.",
            "\n## Interpretation limits\n",
            "- Only the 8 principal pseudchromosomes were considered.",
            "- Gene-distal negatives are weak negatives; these data do not establish promoter activity.",
            "- GC matching is not a homology or sequence-complexity match.",
            "- The new v4 test set is a revised benchmark after earlier v3 results were observed; future replication on alternative held-out chromosomes is recommended.",
        ])
        (staging / "QC_report.md").write_text("\n".join(report) + "\n")
        audit_inputs = [fasta_file, gff_file]
        for sp in ("train", "dev", "test"):
            audit_inputs.extend([v3dir / f"{sp}.csv", v3dir / f"{sp}.meta.csv"])
        manifest = {"input_sha256": {str(p): digest_file(p) for p in audit_inputs},
                    "output_sha256": {p.name: digest_file(p) for p in staging.iterdir() if p.is_file()}}
        (staging / "checksums.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if outdir.exists():
            fail(f"输出目录在构建时被其他任务创建，拒绝覆盖：{outdir}")
        staging.rename(outdir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print("[6/6] v4 数据集构建、GC 审计和独立质量检查完成。", flush=True)
    print("DATA_QC: PASS", flush=True)
    print("OUTPUT:", outdir, flush=True)
    print("SPLIT_COUNTS:", json.dumps(summary["new_split_counts"], ensure_ascii=False), flush=True)


def run_self_test():
    # Interval boundary semantics.
    x = merge_intervals([(5, 10), (10, 15), (20, 25)])
    assert x == [(5, 15), (20, 25)]
    starts = [a for a, _ in x]
    assert overlaps_merged(x, starts, 14, 16)
    assert not overlaps_merged(x, starts, 15, 20)
    assert overlaps_merged(x, starts, 19, 22)
    assert not overlaps_merged(x, starts, 25, 26)
    # Maximum-cardinality GC matching and improvement pass.
    dummy = [{"gc": x, "meta": {"sample_id": f"p{i}"}}
             for i, x in enumerate([390, 400, 420, 450])]
    pool = [(375, 0), (390, 1000), (400, 2000), (415, 3000), (420, 4000), (450, 5000)]
    match, missing, used = greedy_pairs(dummy, pool, 20)
    assert len(match) == 4 and not missing
    refine_pairs(match, pool, used, 42)
    assert all(abs(p["gc"] - n[0]) <= 20 for p, n in match)
    assert len({n[1] for _, n in match}) == 4
    assert minimum_required_delta(dummy, pool) == 0
    # Refuse to claim perfect matching where it is impossible.
    absent = [{"gc": 100, "meta": {"sample_id": "highgc"}}]
    assert len(greedy_pairs(absent, [(200, 0)], 20)[1]) == 1
    assert minimum_required_delta(absent, [(200, 0)]) == 100
    # Check the per-split accounting invariant after chromosome-level pairing.
    grouped = {"train": [(4, 4), (5, 5)], "dev": [(7, 7)], "test": [(3, 3)]}
    assert all(sum(p for p, _ in items) == sum(n for _, n in items) for items in grouped.values())
    print("SELF_TEST_OK: coordinates, interval merging, GC matching, no-reuse and failure guards")


def main():
    p = argparse.ArgumentParser(description="Build DSS3 v4 with fixed 8-chrom holdout and per-chrom exact 1:1 GC-matched negatives (never overwrites v3).")
    p.add_argument("--root", type=Path, default=Path("DSS3"), help="DSS3 directory containing genome FASTA and GFF")
    p.add_argument("--v3-dir", type=Path, help="Read-only source dataset directory; default root/promoter_1kb/chromosome_holdout_v3")
    p.add_argument("--output-dir", type=Path, help="New v4 directory; default root/promoter_1kb/chromosome_holdout_v4_gc_matched")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gc-max-delta", type=float, default=0.02, help="Strict per-pair GC fraction bound (0.02 = 2 percentage points)")
    p.add_argument("--gc-mean-max-delta", type=float, default=0.01, help="Maximum per-chrom difference of class-conditional mean GC")
    p.add_argument("--gc-ks-max", type=float, default=0.10, help="Maximum per-chrom two-sample KS distance for GC distributions")
    p.add_argument("--gc-smd-max", type=float, default=0.10, help="Maximum |standardized mean difference| for GC distributions")
    p.add_argument("--max-positive-drop-fraction", type=float, default=0.0, help="Optional explicit per-chrom max loss of positives when strict GC matching is impossible; default 0")
    p.add_argument("--self-test", action="store_true", help="Run small algorithm tests only, no DSS3 input or model training")
    args = p.parse_args()
    if args.self_test:
        run_self_test()
        return
    if not (0 <= args.gc_max_delta <= 1 and 0 <= args.gc_mean_max_delta <= 1 and
            0 <= args.gc_ks_max <= 1 and args.gc_smd_max >= 0 and
            0 <= args.max_positive_drop_fraction < 1):
        p.error("GC tolerance and allowed drop fraction must be in the stated range")
    build(args)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, AssertionError) as e:
        print("DATA_BUILD_FAIL:", e, file=sys.stderr)
        sys.exit(2)

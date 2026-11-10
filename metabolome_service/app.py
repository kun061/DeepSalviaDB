import logging
import os
from pathlib import Path
import pandas as pd
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from functools import lru_cache
from typing import List, Dict, Any
import io  
import matplotlib  
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from venn import venn  
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False # 顺便解决某些图表负号显示为方块的问题

# 配置日志
logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR.parent / ".env", override=False)
DATA_DIR = Path(os.getenv("METABOLOME_DATA_DIR", BASE_DIR / "data" / "pred_gene"))
ANNOTATION_PATH = Path(
    os.getenv("METABOLOME_ANNOTATION_PATH", BASE_DIR / "data" / "dss_final_annotation.csv")
)

# === 1. 初始化独立 FastAPI worker ===
app = FastAPI(title="DeepSalviaDB Metabolome Service", version="1.0.0")
SUPPORTED_STAGES = ["DAG30", "DAG60", "DAG90", "DAG240", "DAG270", "DAG390"]
SUPPORTED_MODELS = ["RandomForest", "LASSO", "ElasticNet", "GradientBoosting"]

# === 2. 数据模型定义 (新增 model_type 参数) ===
class PredictionRequest(BaseModel):
    metabolite: str
    stage: str
    control_stage: str = 'DAG30'
    model_types: List[str] = Field(default_factory=lambda: ['RandomForest'])
    top_n: int = 20

# 注释匹配只用于对模型预测结果做独立验证，不参与模型训练或候选基因排序。
PATHWAY_ANNOTATION_KEYWORDS = {
    "Tanshinone": [
        "tanshinone", "miltiradiene synthase", "ferruginol synthase",
        "11-hydroxysugiol 20-monooxygenase", "copalyl diphosphate synthase",
        "diterpene synthase", "kaurene synthase", "cyp76ah", "cyp76ak", "cyp71d373",
        "cyp71d375", "cyp71d411"
    ],
    "Salvianolic Acid": [
        "salvianolic acid", "rosmarinic acid", "lithospermic acid",
        "phenyllactic acid", "rosmarinic acid synthase",
        "hydroxyphenylpyruvate reductase", "tyrosine aminotransferase",
        "phenylalanine ammonia-lyase", "cinnamate 4-hydroxylase",
        "4-coumarate--coa ligase", "cytochrome p450 98a"
    ]
}

# === 3. 全局数据加载 ===
logger.info("Initializing Metabolome AI Model Data...")
data_load_error = None
count_df = fpkm_df = metab_df = fpkm_mean = metab_mean = None
try:
    count_df = pd.read_csv(DATA_DIR / 'count_updated.txt', sep='\t', index_col='Geneid')
    fpkm_df = pd.read_csv(DATA_DIR / 'result_fpkm_updated.txt', sep='\t')
    metab_df = pd.read_csv(DATA_DIR / 'pathway_metabolites_intensity.csv')
    stages = SUPPORTED_STAGES

    fpkm_mean = pd.DataFrame({'Geneid': fpkm_df['Geneid']})
    for stage in stages:
        cols = [c for c in fpkm_df.columns if stage in c]
        fpkm_mean[stage] = pd.to_numeric(fpkm_df[cols].mean(axis=1), errors='coerce')
    fpkm_mean = fpkm_mean.set_index('Geneid').dropna()

    metab_mean = pd.DataFrame({'Name': metab_df['Name']})
    for stage in stages:
        cols = [c for c in metab_df.columns if stage in c]
        metab_mean[stage] = pd.to_numeric(metab_df[cols].mean(axis=1), errors='coerce')
    metab_mean = metab_mean.set_index('Name').dropna()
    
    logger.info("Metabolome AI Model Data Loaded Successfully.")
except Exception as e:
    data_load_error = str(e)
    logger.error(f"Failed to load omics data for Metabolome Service: {str(e)}")
    count_df, fpkm_mean, metab_mean = None, None, None

# 基因功能注释单独加载；注释缺失不应让模型服务整体不可用。
annotation_lookup: Dict[str, Dict[str, str]] = {}
metabolite_pathway_lookup: Dict[str, str] = {}
annotation_load_error = None
try:
    annotation_df = pd.read_csv(ANNOTATION_PATH, dtype=str).fillna('')
    required_columns = {'Local_GeneID', 'Symbol', 'description', 'Aliases'}
    missing_columns = required_columns - set(annotation_df.columns)
    if missing_columns:
        raise ValueError(f"dss_final_annotation.csv 缺少字段: {sorted(missing_columns)}")

    # 同一 Local_GeneID 可能有重复注释；合并而不是任意丢弃。
    for gene_id, group in annotation_df.groupby('Local_GeneID', sort=False):
        descriptions = list(dict.fromkeys(v.strip() for v in group['description'] if v.strip()))
        symbols = list(dict.fromkeys(v.strip() for v in group['Symbol'] if v.strip() and v.strip() != '-'))
        aliases = list(dict.fromkeys(v.strip() for v in group['Aliases'] if v.strip() and v.strip() != '-'))
        annotation_lookup[str(gene_id).strip()] = {
            'Symbol': '; '.join(symbols),
            'Description': '; '.join(descriptions),
            'Aliases': '; '.join(aliases)
        }
    logger.info(f"Loaded functional annotations for {len(annotation_lookup)} genes.")
except Exception as e:
    annotation_load_error = str(e)
    logger.warning(f"Functional annotation validation is unavailable: {e}")

if metab_df is not None and {'Name', 'Pathway'}.issubset(metab_df.columns):
    metabolite_pathway_lookup = (
        metab_df[['Name', 'Pathway']]
        .dropna()
        .drop_duplicates('Name')
        .set_index('Name')['Pathway']
        .astype(str)
        .to_dict()
    )


def get_annotation_evidence(gene_id: str, metabolite: str) -> Dict[str, Any]:
    """Return transparent annotation evidence for a predicted gene/metabolite pair."""
    annotation = annotation_lookup.get(str(gene_id), {})
    description = annotation.get('Description', '')
    description_lower = description.casefold()
    pathway = metabolite_pathway_lookup.get(metabolite, '')

    exact_keywords = [metabolite.casefold()]
    pathway_keywords = [k.casefold() for k in PATHWAY_ANNOTATION_KEYWORDS.get(pathway, [])]
    matched_exact = [k for k in exact_keywords if k and k in description_lower]
    matched_pathway = [k for k in pathway_keywords if k and k in description_lower]
    matched_keywords = list(dict.fromkeys(matched_exact + matched_pathway))

    if matched_exact:
        match_type = 'exact_metabolite'
    elif matched_pathway:
        match_type = 'biosynthetic_pathway'
    else:
        match_type = 'none'

    return {
        'Symbol': annotation.get('Symbol', ''),
        'Description': description,
        'Aliases': annotation.get('Aliases', ''),
        'Annotation_Supported': bool(matched_keywords),
        'Annotation_Match_Type': match_type,
        'Matched_Keywords': matched_keywords,
        'Related_Metabolite': metabolite if matched_keywords else '',
        'Relationship_Label': f"Related to {metabolite}" if matched_keywords else '',
        'Metabolite_Pathway': pathway
    }


def get_all_intersections(sets_dict, include_empty: bool = False):
    """
    计算 N 个集合之间所有可能的交集区域（唯一元素）
    """
    model_names = list(sets_dict.keys())
    n = len(model_names)
    intersections = {}
    
    # 遍历从 1 到 2^n - 1 的所有组合
    for i in range(1, 1 << n):
        included_models = [model_names[j] for j in range(n) if (i >> j) & 1]
        excluded_models = [model_names[j] for j in range(n) if not ((i >> j) & 1)]
        
        # 取包含模型的交集
        res_set = set.intersection(*(set(sets_dict[m]) for m in included_models))
        # 剔除掉属于排除模型的元素（保证唯一性）
        for m in excluded_models:
            res_set -= set(sets_dict[m])
            
        if res_set or include_empty:
            # Key 用 & 连接，方便前端识别模型组合。
            key = " & ".join(included_models)
            intersections[key] = sorted(res_set)
            
    return intersections


def generate_venn_svg(venn_data_dict):
    """
    将模型预测的字典转化为纯正的科研 SVG 韦恩图字符串
    【究极融合版：完美兼容 3 集合与 4 集合的最佳坐标】
    """
    sets_dict = {k: set(v) for k, v in venn_data_dict.items() if len(v) > 0}
    n = len(sets_dict)
    
    if n < 2:
        return ""
        
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 使用 Cell/Nature 经典高对比度色卡
    hex_colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F']
    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(hex_colors[:n])
    
    # 绘图
    venn(sets_dict, ax=ax, cmap=custom_cmap, fmt="{size}", fontsize=14, legend_loc=None)
    
    model_names = list(sets_dict.keys())
    
    # 🌟 坐标字典：各取所长，互不干扰
    if n == 2:
        coords = [(0.10, 0.5), (0.90, 0.5)]
        aligns_h = ['right', 'left']
        aligns_v = ['center', 'center']
    elif n == 3:
        # 🌟 恢复到上个版本的完美 3 集合坐标 (最下方的名字死死贴在底边外部)
        coords = [(0.15, 0.85), (0.85, 0.85), (0.50, 0.02)]
        aligns_h = ['right', 'left', 'center']
        aligns_v = ['bottom', 'bottom', 'top']
    elif n == 4:
        # 🌟 保留这个版本完美的 4 集合“皇冠”坐标
        coords = [(0.18, 0.68), (0.35, 0.88), (0.65, 0.88), (0.82, 0.68)]
        aligns_h = ['right', 'right', 'left', 'left']
        aligns_v = ['center', 'bottom', 'bottom', 'center']
    else:
        coords = [(0.5, 0.95)] * n
        aligns_h = ['center'] * n
        aligns_v = ['bottom'] * n

    for i in range(min(n, len(coords))):
        ax.text(coords[i][0], coords[i][1], model_names[i], 
                transform=ax.transAxes, 
                fontsize=16, 
                fontweight='bold',
                color=hex_colors[i], 
                ha=aligns_h[i], 
                va=aligns_v[i])

    buf = io.BytesIO()
    plt.savefig(buf, format='svg', transparent=True, bbox_inches='tight')
    plt.close(fig)
    
    return buf.getvalue().decode('utf-8')
@lru_cache(maxsize=128)
def run_ml_pipeline(metabolite: str, target_stage: str, control_stage: str, model_types: tuple, top_n: int):
    if count_df is None:
        raise RuntimeError("底层多组学数据未正确加载。")

    # [1. DESeq2 降维] 保持不变
    selected_cols = [c for c in count_df.columns if target_stage in c or control_stage in c]
    count_matrix = count_df[selected_cols].T.round().astype(int) 
    metadata = pd.DataFrame(
        {'condition': [target_stage if target_stage in c else control_stage for c in count_matrix.index]}, 
        index=count_matrix.index
    )
    
    dds = DeseqDataSet(counts=count_matrix, metadata=metadata, design_factors="condition", n_cpus=4)
    dds.deseq2() 
    stat_res = DeseqStats(dds, contrast=["condition", target_stage, control_stage], n_cpus=4)
    stat_res.summary()
    res_df = stat_res.results_df
    sig_degs = res_df[(res_df['padj'] < 0.05) & (res_df['log2FoldChange'].abs() > 1)].index.tolist()
    
    valid_degs = [gene for gene in sig_degs if gene in fpkm_mean.index]
    if not valid_degs:
        return {
            "table_data": [],
            "interactive_data": {},
            "base_sets": {},
            "annotation_validation": {
                "found": False,
                "matched_gene_count": 0,
                "best_hit": None,
                "hits_by_combination": {},
                "message": "No significant DEGs were available for annotation validation."
            }
        }
        
    X = fpkm_mean.loc[valid_degs].T
    y = metab_mean.loc[metabolite].values
    
    model_gene_scores = []
    
    # [2. 运行机器学习模型] 保持不变
    for m_type in model_types:
        if m_type == 'LASSO':
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            ml_model = Lasso(alpha=0.05, max_iter=10000, random_state=42)
            ml_model.fit(X_scaled, y)
            importances = np.abs(ml_model.coef_)
        elif m_type == 'ElasticNet':
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            ml_model = ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=10000, random_state=42)
            ml_model.fit(X_scaled, y)
            importances = np.abs(ml_model.coef_)
        elif m_type == 'GradientBoosting':
            ml_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            ml_model.fit(X, y)
            importances = ml_model.feature_importances_
        else: 
            ml_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
            ml_model.fit(X, y)
            importances = ml_model.feature_importances_
            
        max_score = importances.max()
        norm_importances = (importances / max_score) * 100 if max_score > 0 else importances
            
        gene_dict = {}
        for i, gene in enumerate(X.columns):
            if norm_importances[i] > 0:
                gene_dict[gene] = norm_importances[i]
                
        model_gene_scores.append(gene_dict)

    # 3. 【核心修正】：严格让韦恩图和数据池尊重 Top N
    base_sets = {}  # 🌟 给 venn 库画图用（只有选中的 2-4 个模型，如 'RF', 'LASSO'）
    union_top_genes = set() 
    gene_sources = {}
    
    for idx, m_type in enumerate(model_types):
        sorted_genes = sorted(model_gene_scores[idx].items(), key=lambda x: x[1], reverse=True)
        top_genes = [g[0] for g in sorted_genes[:top_n]]
        
        base_sets[m_type] = top_genes
        union_top_genes.update(top_genes)
        for gene in top_genes:
            if gene not in gene_sources: gene_sources[gene] = []
            gene_sources[gene].append(m_type)

    # 计算 15 种交集用于前端交互
    interactive_data = get_all_intersections(base_sets)

    # 4. 组装最终结果
    final_results = []
    all_stages = SUPPORTED_STAGES
    
    for gene in union_top_genes:
        scores = [model_gene_scores[idx].get(gene, 0) for idx, m_type in enumerate(model_types)]
        avg_score = sum(scores) / len(model_types)
        models_hit = gene_sources.get(gene, [])
        hit_count = len(models_hit) 
        
        target_fpkm = fpkm_mean.loc[gene, target_stage]
        control_fpkm = fpkm_mean.loc[gene, control_stage]
        log2fc = res_df.loc[gene, 'log2FoldChange']
        fpkm_profile = {s: round(fpkm_mean.loc[gene, s], 2) for s in all_stages}
        
        result_row = {
            'Geneid': gene,
            'Models': models_hit,
            'Hit_Count': hit_count,
            'Impact_Score': round(avg_score, 2),
            'Control_FPKM': round(control_fpkm, 2),
            'Target_FPKM': round(target_fpkm, 2),
            'log2FC': round(log2fc, 2),
            'FPKM_Profile': fpkm_profile
        }
        result_row.update(get_annotation_evidence(gene, metabolite))
        final_results.append(result_row)

    # 注释命中优先显示；其后仍按模型共识数和预测影响分数排序。
    final_df = pd.DataFrame(final_results).sort_values(
        by=['Annotation_Supported', 'Hit_Count', 'Impact_Score'],
        ascending=[False, False, False]
    )
    table_records = final_df.to_dict(orient="records")
    records_by_gene = {row['Geneid']: row for row in table_records}

    # 覆盖全部 2^N-1 个独占模型组合，包括当前没有候选基因的组合。
    all_combination_regions = get_all_intersections(base_sets, include_empty=True)
    hits_by_combination = {}
    for combination, genes in all_combination_regions.items():
        hits_by_combination[combination] = [
            {
                'Geneid': gene,
                'Symbol': records_by_gene[gene]['Symbol'],
                'Description': records_by_gene[gene]['Description'],
                'Match_Type': records_by_gene[gene]['Annotation_Match_Type'],
                'Matched_Keywords': records_by_gene[gene]['Matched_Keywords'],
                'Impact_Score': records_by_gene[gene]['Impact_Score']
            }
            for gene in genes
            if gene in records_by_gene and records_by_gene[gene]['Annotation_Supported']
        ]

    validated_records = [row for row in table_records if row['Annotation_Supported']]

    # 再检查每个模型的完整非零重要性排名。这样即使已知相关基因落在 Top N
    # 之外，也能报告其真实模型排名，而不把它强行塞回 Top N 或改动训练结果。
    full_rankings = {}
    for idx, model_name in enumerate(model_types):
        full_rankings[model_name] = sorted(
            model_gene_scores[idx].items(), key=lambda item: item[1], reverse=True
        )
    rank_lookup = {
        model_name: {gene: rank for rank, (gene, _score) in enumerate(ranking, start=1)}
        for model_name, ranking in full_rankings.items()
    }
    score_lookup = {
        model_name: dict(ranking) for model_name, ranking in full_rankings.items()
    }

    all_scored_genes = set().union(*(set(scores) for scores in model_gene_scores))
    ranked_annotation_hits = []
    for gene in all_scored_genes:
        evidence = get_annotation_evidence(gene, metabolite)
        if not evidence['Annotation_Supported']:
            continue
        gene_ranks = {
            model_name: rank_lookup[model_name][gene]
            for model_name in model_types if gene in rank_lookup[model_name]
        }
        gene_scores = {
            model_name: round(float(score_lookup[model_name][gene]), 2)
            for model_name in model_types if gene in score_lookup[model_name]
        }
        models_with_hit = [m for m in model_types if m in gene_ranks]
        ranked_annotation_hits.append({
            'Geneid': gene,
            'Symbol': evidence['Symbol'],
            'Description': evidence['Description'],
            'Aliases': evidence['Aliases'],
            'Match_Type': evidence['Annotation_Match_Type'],
            'Matched_Keywords': evidence['Matched_Keywords'],
            'Models': models_with_hit,
            'Model_Combination': ' & '.join(models_with_hit),
            'Model_Ranks': gene_ranks,
            'Model_Scores': gene_scores,
            'Best_Rank': min(gene_ranks.values()),
            'Within_Displayed_Top_N': gene in union_top_genes
        })

    ranked_annotation_hits.sort(
        key=lambda row: (
            0 if row['Match_Type'] == 'exact_metabolite' else 1,
            -len(row['Models']),
            row['Best_Rank']
        )
    )

    # 严格验证原始 Top-N；不补入、不替换、不重排任何模型候选基因。
    best_hit = validated_records[0] if validated_records else None
    annotation_validation = {
        'found': best_hit is not None,
        'found_in_displayed_top_n': bool(validated_records),
        'included_in_result': bool(validated_records),
        'supplemented_from_full_ranking': False,
        'supplemented_from_deseq_evidence': False,
        'top_n_matched_gene_count': len(validated_records),
        'result_matched_gene_count': len(validated_records),
        'ranked_candidate_matched_gene_count': len(ranked_annotation_hits),
        'best_hit': best_hit,
        'hits_by_combination': hits_by_combination,
        'ranked_annotation_hits': ranked_annotation_hits,
        'message': (
            f"Found {len(validated_records)} annotation-supported candidate gene(s) in the original Top N."
            if validated_records else
            "No annotation-supported gene was found in the original Top N; no gene was added or reordered."
        )
    }
    return {
        "table_data": table_records,
        "interactive_data": interactive_data,
        "base_sets": base_sets,
        "annotation_validation": annotation_validation
    }

@app.get("/health")
def health():
    ready = count_df is not None and fpkm_mean is not None and metab_mean is not None
    return {
        "status": "ok" if ready else "unavailable",
        "service": "metabolome",
        "pipeline": "on-demand DESeq2 screening and machine-learning ranking",
        "pretrained_weights": False,
        "data_ready": ready,
        "annotation_ready": bool(annotation_lookup),
        "data_error": data_load_error,
        "annotation_error": annotation_load_error,
    }


@app.get("/metadata")
def metadata():
    return {
        "stages": SUPPORTED_STAGES,
        "models": SUPPORTED_MODELS,
        "metabolites": sorted(metab_mean.index.astype(str).tolist()) if metab_mean is not None else [],
        "pretrained_weights": False,
    }


# === 2. 修改 API 接口 (传入 req.top_n) ===
@app.post("/predict_multi_gene")
def predict_metabolome_genes(req: PredictionRequest):
    if metab_mean is None:
        raise HTTPException(status_code=500, detail="服务数据未就绪")
    if req.metabolite not in metab_mean.index:
        raise HTTPException(status_code=404, detail=f"未找到该代谢物: {req.metabolite}")
    if not req.model_types:
        raise HTTPException(status_code=400, detail="请至少选择一个 AI 算法")
    if req.stage not in SUPPORTED_STAGES or req.control_stage not in SUPPORTED_STAGES:
        raise HTTPException(status_code=400, detail=f"时期必须属于: {SUPPORTED_STAGES}")
    if req.stage == req.control_stage:
        raise HTTPException(status_code=400, detail="目标时期和对照时期不能相同")
    unknown_models = sorted(set(req.model_types) - set(SUPPORTED_MODELS))
    if unknown_models:
        raise HTTPException(status_code=400, detail=f"不支持的算法: {unknown_models}")
    if req.top_n < 1 or req.top_n > 500:
        raise HTTPException(status_code=400, detail="top_n 必须在 1 到 500 之间")
        
    try:
        model_tuple = tuple(req.model_types)
        # 获取计算结果
        results = run_ml_pipeline(req.metabolite, req.stage, req.control_stage, model_tuple, req.top_n)
        
        svg_content = ""
        # 🌟 修复点：这里只传 base_sets (只有2~4个Key) 进去画图，再也不会报错了！
        if len(req.model_types) > 1 and results.get("base_sets"):
            svg_content = generate_venn_svg(results["base_sets"])
        
        return {
            "status": "success",
            "models_used": req.model_types,
            "data": results.get("table_data", []), 
            "venn_data": results.get("interactive_data", {}), # 依然把裂变后的15个组合传给前端
            "venn_svg": svg_content,
            "annotation_validation": results.get("annotation_validation", {})
        }
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Prediction Error: {error_details}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8002, workers=1)

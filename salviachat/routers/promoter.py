# Promoter prediction API router.
import time
import os
import threading
from pathlib import Path
import re
import torch
import torch.nn.functional as F
from fastapi import APIRouter, HTTPException # 改用 APIRouter
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("PROMOAI_MODEL_PATH", BASE_DIR.parent / "promoai" / "finetune" / "dss3_bpe_1000bp_head_only_v4_gc_seed42" / "best_model"))
PREDICTION_LOCK = threading.Lock()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PLANT_MOTIFS = {
    "TATA-box": {"pattern": "TATA[AT]A[AT]", "color": "#FF4B4B", "desc": "Core promoter element around -30 of transcription start"},
    "CAAT-box": {"pattern": "CAA[AT]", "color": "#FF9F43", "desc": "Common cis-acting element in promoter and enhancer regions"},
    "AT~TATA-box": {"pattern": "TATATA", "color": "#EA2027", "desc": "AT-rich TATA-box variant"},
    "G-box":    {"pattern": "CACGTG", "color": "#FDCB6E", "desc": "Light responsive element"},
    "GT1-motif":{"pattern": "GGTTAA", "color": "#E1B12C", "desc": "Light responsive element"},
    "I-box":    {"pattern": "GATAAG", "color": "#D35400", "desc": "Part of a light responsive element"},
    "Box 4":    {"pattern": "ATTAAT", "color": "#F39C12", "desc": "Part of a conserved DNA module involved in light responsiveness"},
    "MRE":      {"pattern": "AACCTAA", "color": "#F1C40F", "desc": "MYB binding site involved in light responsiveness"},
    "Sp1":      {"pattern": "CC[GA]CCC", "color": "#E67E22", "desc": "Light responsive element"},
    "AE-box":   {"pattern": "AGAAACAT", "color": "#FAD390", "desc": "Part of a module for light response"},
    "AT1-motif":  {"pattern": "AATTATTTTTTATT", "color": "#F8C291", "desc": "Part of a light responsive module"},
    "ATCT-motif": {"pattern": "AATCTAATCC", "color": "#E58E26", "desc": "Part of a conserved DNA module involved in light responsiveness"},
    "GATA-motif": {"pattern": "GATAGGG", "color": "#F6B93B", "desc": "Part of a light responsive element"},
    "TCCC-motif": {"pattern": "TCTCCCT", "color": "#FA983A", "desc": "Part of a light responsive element"},
    "GA-motif":   {"pattern": "ATAGATAA", "color": "#F9CA24", "desc": "Part of a light responsive element"},
    "ABRE":        {"pattern": "ACGTG", "color": "#2ECC71", "desc": "ABA (Abscisic acid) responsive element"},
    "ABRE3a":      {"pattern": "TACGTG", "color": "#05C46B", "desc": "ABA responsive element variant"},
    "CGTCA-motif": {"pattern": "CGTCA", "color": "#26DE81", "desc": "MeJA (Methyl jasmonate) responsive element"},
    "TGACG-motif": {"pattern": "TGACG", "color": "#20BF6B", "desc": "MeJA responsive element"},
    "TCA-element": {"pattern": "CCATCTTTTT|TCATTTTC", "color": "#1ABC9C", "desc": "SA (Salicylic acid) responsive element"},
    "AuxRR-core":  {"pattern": "GGTCCAT", "color": "#16A085", "desc": "Auxin (IAA) responsive element"},
    "TGA-element": {"pattern": "AACGAC", "color": "#009432", "desc": "Auxin responsive element"},
    "GARE-motif":  {"pattern": "TCTGTTG", "color": "#A3CB38", "desc": "Gibberellin (GA) responsive element"},
    "P-box":       {"pattern": "CCTTTTG", "color": "#C4E538", "desc": "Gibberellin responsive element"},
    "TATC-box":    {"pattern": "TATCCCA", "color": "#008B74", "desc": "Gibberellin responsive element"},
    "ERE":         {"pattern": "A[AT]TTTAAA", "color": "#B8E994", "desc": "Ethylene responsive element"},
    "MBS":             {"pattern": "CAACTG", "color": "#3498DB", "desc": "MYB binding site involved in drought-inducibility"},
    "LTR":             {"pattern": "CCGAAA", "color": "#2980B9", "desc": "Low-temperature (cold) responsive element"},
    "TC-rich repeats": {"pattern": "GTTTTCTTAC|ATTCTCTAAC", "color": "#9B59B6", "desc": "Defense and stress responsive element"},
    "ARE":             {"pattern": "AAACCA", "color": "#8E44AD", "desc": "Anaerobic induction essential element"},
    "WUN-motif":       {"pattern": "AAATTTCCT", "color": "#5758BB", "desc": "Wound-responsive element"},
    "DRE":             {"pattern": "TACCGACAT", "color": "#4834D4", "desc": "Dehydration, low-temp, salt stress responsive"},
    "W-box":           {"pattern": "TTGAC[CT]", "color": "#54A0FF", "desc": "WRKY binding site involved in defense"},
    "STRE":            {"pattern": "AGGGG", "color": "#12CBC4", "desc": "Stress responsive element"},
    "CAT-box":     {"pattern": "GCCACT", "color": "#E84393", "desc": "Meristem expression related element"},
    "O2-site":     {"pattern": "GATGATGT", "color": "#FD79A8", "desc": "Zein metabolism regulation element"},
    "RY-element":  {"pattern": "CATGCATG", "color": "#D63031", "desc": "Seed-specific regulation element"},
    "skn-1_motif": {"pattern": "GTCAT", "color": "#FF7675", "desc": "Endosperm expression element"},
    "GCN4_motif":  {"pattern": "TGTGTCA", "color": "#FAB1A0", "desc": "Endosperm expression related element"},
    "circadian":   {"pattern": "CAA[ACTG]{4}ATC", "color": "#833471", "desc": "Circadian control element"},
    "MYB":         {"pattern": "C[AC]GT[AT]G", "color": "#0ABDE3", "desc": "MYB transcription factor binding site"},
    "MYC":         {"pattern": "CA[ACTG]{2}TG", "color": "#48DBFB", "desc": "MYC transcription factor binding site"},
    "MYB-like sequence": {"pattern": "TAACCA", "color": "#1289A7", "desc": "MYB-like binding site"},
    "Myb-binding site":  {"pattern": "CAACAG", "color": "#4BCFFA", "desc": "MYB transcription factor binding site"},
    "AAGAA-motif":       {"pattern": "[Gg]?GTAAAGAAA|GAAAGAA", "color": "#D980FA", "desc": "AAGAA motif"},
    "AT-rich element":   {"pattern": "ATAGAAATCAA", "color": "#9980FA", "desc": "Binding site of AT-rich DNA binding protein (ATBP-1)"}
}

def seq2kmer(seq, k=6):
    kmer_seq = ""
    for i in range(len(seq) - k + 1):
        kmer_seq += seq[i : i + k] + " "
    return kmer_seq.strip()

def reverse_complement(seq):
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return "".join(complement.get(base, base) for base in reversed(seq))

print(f"正在加载 ProMoAI BPE 模型至 {DEVICE}: {MODEL_PATH} ...")
tokenizer = None
model = None
model_load_error = None
try:
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH), local_files_only=True)
    model.to(DEVICE)
    model.eval()
    print("✅ 预测 AI 模型加载成功！")
except Exception as e:
    model_load_error = str(e)
    print(f"❌ ProMoAI 模型加载失败: {e}")

class DnaRequest(BaseModel):
    sequence: str
    threshold: float = 0.5
    strand: str = "forward"

@router.get("/health")
def promoter_health():
    return {
        "status": "ok" if model is not None else "unhealthy",
        "loaded": model is not None,
        "model_path": str(MODEL_PATH),
        "device": str(DEVICE),
        "error": model_load_error,
        "tokenization": "raw DNA BPE",
    }

@router.post("/predict")
def predict_promoter(request: DnaRequest):
    start_time = time.time()
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail=f"ProMoAI model unavailable: {model_load_error}")
    
    raw_seq = request.sequence.upper().replace("\n", "").replace(" ", "").strip()
    if not raw_seq or len(raw_seq) < 10:
        raise HTTPException(status_code=400, detail="Sequence too short")
    if re.search(r"[^ACGTN]", raw_seq):
        raise HTTPException(status_code=422, detail="Sequence may only contain A, C, G, T, or N")
    if len(raw_seq) > 1000:
        raise HTTPException(status_code=422, detail="ProMoAI was trained on 1000 bp sequences; maximum length is 1000")

    if request.strand not in {"forward", "reverse"}:
        raise HTTPException(status_code=422, detail="strand must be forward or reverse")

    target_seq = raw_seq
    if request.strand == "reverse":
        target_seq = reverse_complement(raw_seq)

    g_count = target_seq.count('G')
    c_count = target_seq.count('C')
    gc_content = ((g_count + c_count) / len(target_seq)) * 100

    inputs = tokenizer(target_seq, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    with PREDICTION_LOCK, torch.inference_mode():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
    
    score = probs[0][1].item()
    is_promoter = score >= request.threshold

    features_data = []
    motif_stats = {}
    
    if is_promoter:
        for name, info in PLANT_MOTIFS.items():
            pattern = info["pattern"]
            for match in re.finditer(pattern, target_seq):
                start, end = match.span()
                matched_seq = match.group()
                features_data.append({
                    "name": name, "start": start, "end": end, "length": end - start,
                    "color": info["color"], "desc": info.get("desc", ""),
                    "seq": matched_seq, "type": info.get("type", "other")
                })
                if name not in motif_stats:
                    motif_stats[name] = {"count": 0, "color": info["color"]}
                motif_stats[name]["count"] += 1
        features_data.sort(key=lambda x: x["start"])

    motif_list = [{"name": k, "count": v["count"], "color": v["color"]} for k, v in motif_stats.items()]
    time_taken = time.time() - start_time

    return {
        "is_promoter": is_promoter,
        "score": round(score, 4),
        "confidence": f"{score*100:.2f}%",
        "gc_content": round(gc_content, 2),
        "time_taken": round(time_taken, 4),
        "seq_length": len(target_seq),
        "features": features_data,
        "motifs": motif_list
    }

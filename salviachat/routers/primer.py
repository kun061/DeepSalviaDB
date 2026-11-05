from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import primer3

# 定义路由
router = APIRouter()

# 定义前端传过来的请求体结构
class PrimerDesignRequest(BaseModel):
    seq: str
    lenMin: int = 18
    lenOpt: int = 20
    lenMax: int = 22
    prodMin: int = 150
    prodMax: int = 300
# 定义分析请求的结构体
class AnalyzeRequest(BaseModel):
    primerF: str = ""
    primerR: str = ""
@router.post("/design")
async def design_primers(request: PrimerDesignRequest):
    try:
        # 1. 组装目标序列参数
        seq_args = {
            'SEQUENCE_ID': 'Target',
            'SEQUENCE_TEMPLATE': request.seq
        }

        # 2. 组装全局热力学和长度参数
        global_args = {
            'PRIMER_TASK': 'generic',
            'PRIMER_PICK_LEFT_VARIABLE': 1,
            'PRIMER_PICK_RIGHT_VARIABLE': 1,
            'PRIMER_OPT_SIZE': request.lenOpt,
            'PRIMER_MIN_SIZE': request.lenMin,
            'PRIMER_MAX_SIZE': request.lenMax,
            # 注意：primer3-py 要求 size range 是一个嵌套列表
            'PRIMER_PRODUCT_SIZE_RANGE': [[request.prodMin, request.prodMax]],
            
            # 你可以在这里加入更严苛的生信参数，比如盐浓度、最大多聚体等
            'PRIMER_MAX_POLY_X': 5,
            'PRIMER_SALT_MONOVALENT': 50.0,
        }

        # 3. 内存级调用原生 C 引擎 (极速、无路径问题)
        primer_results = primer3.bindings.design_primers(seq_args, global_args)

        # 4. 解析结果为 Vue 前端所需格式
        pairs = []
        pair_count = primer_results.get('PRIMER_PAIR_NUM_RETURNED', 0)

        for i in range(pair_count):
            penalty = primer_results.get(f'PRIMER_PAIR_{i}_PENALTY', 0)
            score = round(100 - penalty, 1)  # 转换为前端的百分制评分
            
            # Primer3 的位置是从 0 开始的 (起始位点, 长度)
            left_pos, left_len = primer_results.get(f'PRIMER_LEFT_{i}')
            right_pos, right_len = primer_results.get(f'PRIMER_RIGHT_{i}')

            pair = {
                "productSize": primer_results.get(f'PRIMER_PAIR_{i}_PRODUCT_SIZE'),
                "tmDiff": round(abs(primer_results.get(f'PRIMER_LEFT_{i}_TM') - primer_results.get(f'PRIMER_RIGHT_{i}_TM')), 1),
                "score": max(0, score),
                "f": {
                    "start": left_pos + 1,
                    "len": left_len,
                    "seq": primer_results.get(f'PRIMER_LEFT_{i}_SEQUENCE'),
                    "tm": round(primer_results.get(f'PRIMER_LEFT_{i}_TM'), 1),
                    "gc": round(primer_results.get(f'PRIMER_LEFT_{i}_GC_PERCENT'), 1)
                },
                "r": {
                    "end": right_pos + 1,
                    "len": right_len,
                    "seq": primer_results.get(f'PRIMER_RIGHT_{i}_SEQUENCE'),
                    "tm": round(primer_results.get(f'PRIMER_RIGHT_{i}_TM'), 1),
                    "gc": round(primer_results.get(f'PRIMER_RIGHT_{i}_GC_PERCENT'), 1)
                }
            }
            pairs.append(pair)

        return {"status": "success", "data": pairs}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Primer3 引擎计算异常: {str(e)}")
@router.post("/analyze")
async def analyze_primers(request: AnalyzeRequest):
    try:
        response_data = {
            "resultF": None,
            "resultR": None,
            "pairAssessment": None
        }

        # 内部评估单条引物的函数
        def evaluate_single(seq: str):
            if not seq: return None
            
            seq = seq.upper()
            length = len(seq)
            
            # 使用 Primer3 计算基础属性
            tm = round(primer3.calc_tm(seq), 1)
            gc = round((seq.count('G') + seq.count('C')) / length * 100, 1)
            
            # 计算发夹结构 (Hairpin)
            hairpin = primer3.calc_hairpin(seq)
            hairpin_tm = hairpin.tm if hairpin.structure_found else -99
            
            # 计算同源二聚体 (Homodimer - 自己和自己结合)
            homodimer = primer3.calc_homodimer(seq)
            homodimer_tm = homodimer.tm if homodimer.structure_found else -99

            # 基础打分逻辑 (你可以根据需求微调)
            score = 100
            
            # 长度评估
            len_status = "success"
            if length < 18 or length > 25:
                len_status = "warning"
                score -= 10
            
            # Tm 评估
            tm_status = "success"
            if tm < 52 or tm > 65:
                tm_status = "warning"
                score -= 10
                
            # GC 评估
            gc_status = "success"
            if gc < 40 or gc > 60:
                gc_status = "warning"
                score -= 10

            # 发夹结构评估 (如果发夹 Tm 大于 30°C，说明室温下很稳定，极其危险)
            hp_status = "success"
            hp_msg = "None/Weak"
            if hairpin_tm > 35:
                hp_status = "danger"
                hp_msg = f"High Risk (Tm {round(hairpin_tm,1)}°C)"
                score -= 20
            elif hairpin_tm > 25:
                hp_status = "warning"
                hp_msg = f"Moderate (Tm {round(hairpin_tm,1)}°C)"
                score -= 10

            # GC Clamp 评估 (3'端最后5个碱基中G/C的数量)
            clamp_status = "success"
            clamp_msg = "Good"
            if length >= 5:
                gc_count = seq[-5:].count('G') + seq[-5:].count('C')
                if gc_count == 0:
                    clamp_status = "warning"
                    clamp_msg = "Too weak (0 G/C)"
                    score -= 5
                elif gc_count > 3:
                    clamp_status = "danger"
                    clamp_msg = "Too tight (>3 G/C)"
                    score -= 10

            return {
                "length": length,
                "lengthStatus": len_status,
                "gc": gc,
                "gcStatus": gc_status,
                "tm": tm,
                "tmStatus": tm_status,
                "clampMsg": clamp_msg,
                "clampStatus": clamp_status,
                "hairpinMsg": hp_msg,
                "hairpinStatus": hp_status,
                "homodimerTm": round(homodimer_tm, 1) if homodimer_tm > -50 else "None",
                "overallScore": max(0, score),
                "overallStatus": "danger" if score < 60 else "warning" if score < 80 else "success"
            }

        # 1. 分别评估 F 和 R
        if request.primerF:
            response_data["resultF"] = evaluate_single(request.primerF)
        if request.primerR:
            response_data["resultR"] = evaluate_single(request.primerR)

        # 2. 如果两者都有，评估双引物间的关系 (Tm 差值 & 异源二聚体)
        if request.primerF and request.primerR:
            f_tm = response_data["resultF"]["tm"]
            r_tm = response_data["resultR"]["tm"]
            tm_diff = round(abs(f_tm - r_tm), 1)
            
            # 计算异源二聚体 (Primer-Dimer)
            heterodimer = primer3.calc_heterodimer(request.primerF.upper(), request.primerR.upper())
            dimer_tm = heterodimer.tm if heterodimer.structure_found else -99
            
            dimer_status = "success"
            dimer_msg = "Safe"
            if dimer_tm > 35:
                dimer_status = "danger"
                dimer_msg = f"High Dimer Risk (Tm {round(dimer_tm,1)}°C)"
            elif dimer_tm > 25:
                dimer_status = "warning"
                dimer_msg = "Moderate Dimer Risk"

            response_data["pairAssessment"] = {
                "tmDiff": tm_diff,
                "tmStatus": "success" if tm_diff <= 5 else "warning",
                "dimerStatus": dimer_status,
                "dimerMsg": dimer_msg
            }

        return {"status": "success", "data": response_data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分析异常: {str(e)}")
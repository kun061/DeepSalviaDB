from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
import logging
from typing import Dict, Any
import json
import os
import math
import re
from langchain_core.runnables import RunnableConfig
import mysql.connector
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![A-Za-z0-9_])"
)
NO_RESULT_RE = re.compile(
    r"\b(?:no (?:matching )?(?:data|record|result)|not (?:available|found)|"
    r"could not find|cannot find)\b",
    re.IGNORECASE,
)


def _numbers_in(value: Any) -> list[float]:
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, str):
        numbers = []
        for match in NUMBER_RE.finditer(value.replace(",", "")):
            try:
                numbers.append(float(match.group()))
            except ValueError:
                pass
        return numbers
    if isinstance(value, dict):
        return [
            number
            for key, child in value.items()
            for number in (_numbers_in(key) + _numbers_in(child))
        ]
    if isinstance(value, (list, tuple)):
        return [number for child in value for number in _numbers_in(child)]
    return []


def _unsupported_answer_numbers(question: str, evidence: Any, answer: str) -> list[float]:
    allowed = _numbers_in(question) + _numbers_in(evidence)
    return [
        number
        for number in _numbers_in(answer)
        if not any(math.isclose(number, item, rel_tol=5e-4, abs_tol=5e-3) for item in allowed)
    ]



def _missing_requested_replicate_values(
    question: str, evidence: Any, answer: str
) -> list[dict[str, Any]]:
    """Find omitted hormone replicate values explicitly requested by the user."""
    if not re.search(r"\breplicate(?:s)?\b", question, re.IGNORECASE):
        return []
    stages = set(re.findall(r"\bDAG\s*(30|60|90|240|270|390)\b", question, re.IGNORECASE))
    if not stages or not isinstance(evidence, list):
        return []
    answer_numbers = _numbers_in(answer)
    missing: list[dict[str, Any]] = []
    for row in evidence:
        if not isinstance(row, dict):
            continue
        for key, value in row.items():
            normalized_key = re.sub(r"[^a-z0-9]+", "", str(key).casefold())
            if not any(
                f"dag{stage}" in normalized_key or f"{stage}dag" in normalized_key
                for stage in stages
            ) or "rep" not in normalized_key:
                continue
            if isinstance(value, (int, float)) and not any(
                math.isclose(float(value), number, rel_tol=5e-4, abs_tol=5e-3)
                for number in answer_numbers
            ):
                missing.append({"field": str(key), "value": value})
    return missing

def _missing_requested_hgvs_values(question: str, evidence: Any, answer: str) -> list[dict[str, Any]]:
    """Return HGVS fields omitted from an answer that explicitly requests HGVS."""
    if not re.search(r"\bhgvs\b", question, re.IGNORECASE) or not isinstance(evidence, list):
        return []
    normalized_answer = re.sub(r"\s+", "", answer).casefold()
    missing: list[dict[str, Any]] = []
    for row in evidence:
        if not isinstance(row, dict):
            continue
        for key, value in row.items():
            if "hgvs" not in str(key).casefold() or value in (None, "", "-"):
                continue
            if re.sub(r"\s+", "", str(value)).casefold() not in normalized_answer:
                missing.append({"field": str(key), "value": value})
    return missing


def _limit_evidence(value: Any, max_items: int, max_string: int) -> Any:
    if isinstance(value, str):
        if len(value) <= max_string:
            return value
        return value[:max_string] + f"... [truncated; original length={len(value)}]"
    if isinstance(value, dict):
        return {
            key: _limit_evidence(child, max_items, max_string)
            for key, child in value.items()
        }
    if isinstance(value, list):
        limited = [
            _limit_evidence(child, max_items, max_string)
            for child in value[:max_items]
        ]
        if len(value) > max_items:
            limited.append({"_notice": f"{len(value) - max_items} additional items omitted"})
        return limited
    return value


def _bounded_evidence_json(value: Any, max_chars: int = 24000) -> str:
    serialized = json.dumps(value, ensure_ascii=False, default=str)
    if len(serialized) <= max_chars:
        return serialized
    for max_items, max_string in ((100, 4000), (20, 1000), (5, 500)):
        limited = _limit_evidence(value, max_items, max_string)
        serialized = json.dumps(limited, ensure_ascii=False, default=str)
        if len(serialized) <= max_chars:
            return serialized
    return json.dumps(
        {"_notice": "Evidence exceeded the model context and was omitted."},
        ensure_ascii=False,
    )


class Generator:
    def __init__(self):
        self.llm = ChatOllama(
            model=os.getenv("SALVIACHAT_GENERATOR_MODEL", os.getenv("SALVIACHAT_MODEL", "qwen2.5:14b")),
            base_url=os.getenv("SALVIACHAT_OLLAMA_BASE_URL", "http://localhost:11434"),
            #api_key="ollama",
            num_ctx=int(os.getenv("SALVIACHAT_NUM_CTX", "16000")),
            temperature=float(os.getenv("SALVIACHAT_TEMPERATURE", "0")),
            num_predict=int(os.getenv("SALVIACHAT_GENERATOR_NUM_PREDICT", "1024")),
            keep_alive=os.getenv("SALVIACHAT_KEEP_ALIVE", "24h"),
        )
        # self.llm = ChatOpenAI(
        #     model="deepseek-chat", # 👈 模型版本写在这里 (DeepSeek V3)
        #     api_key="YOUR_API_KEY", 
        #     base_url="https://api.deepseek.com",
        #     temperature=0,
        #     max_tokens=2000
        # )
        self.abbreviation_map = {
            "Mapping Dictionary": {
                "T0": "0h Start (Baseline)",
                "T1": "3h Duration",
                "T2": "27h Duration",
                "T3": "51h Duration",
                "C": "Control Group - This is NOT Cold stress",
                "H": "HEAT Shock Group (38℃)",
                "R": "recovered at normal temp  (26℃) after heat shock",
                "CK_N": "grown under normal nitrogen conditions",
                "N0": "Nitrogen Deficiency",
                "L_N": "Low Nitrogen",
                "H_N": "High Nitrogen"
            }
        }
        self.db_config = {
            "host": os.getenv("SALVIACHAT_DB_HOST", "127.0.0.1"),
            "port": int(os.getenv("SALVIACHAT_DB_PORT", "3306")),
            "user": os.getenv("SALVIACHAT_DB_USER", "salviachat"),
            "password": os.getenv("SALVIACHAT_DB_PASSWORD", ""),
            "database": os.getenv("SALVIACHAT_DB_NAME", "salvia")
        }
    def _search_homology_aliases(self, strict_term: str, fuzzy_term: str) -> list:
        """从 Homology CSV 文件中检索 aliases"""
        import csv
        import os
        import re
        matches = []
        
        # ⚠️ 请务必确保这里的路径和文件名后缀是正确的 (.csv)
        file_path = os.getenv("SALVIACHAT_HOMOLOGY_CSV", str(Path(__file__).resolve().parents[2] / "data" / "Homology.csv")) 
        
        try:
            if not os.path.exists(file_path):
                logger.warning("Homology CSV file not found.")
                return []
            
            # 核心技巧：将 SQL 的 f%3%1 转换为 Python 正则的 f.*3.*1
            pattern = fuzzy_term.replace('%', '.*')
            regex = re.compile(pattern, re.IGNORECASE)
            
            # utf-8-sig 能自动处理 Windows Excel 导出的带 BOM 的 CSV 文件
            with open(file_path, mode='r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                
                # 自动寻找列名中包含 alias/aliases 的列
                alias_col_name = None
                if reader.fieldnames:
                    for field in reader.fieldnames:
                        if "alias" in field.lower():
                            alias_col_name = field
                            break
                            
                if not alias_col_name:
                    logger.warning("No 'aliases' column found in the CSV.")
                    return []
                    
                for row in reader:
                    aliases_raw = row.get(alias_col_name, "")
                    if not aliases_raw:
                        continue
                        
                    # 很多 CSV 里的别名是用逗号或分号隔开的，比如 "F3H1, SmF3H"
                    alias_list = re.split(r'[,;]\s*', aliases_raw)
                    
                    for alias in alias_list:
                        alias = alias.strip()
                        # 重点：同时支持精准部分匹配和漏打字母的正则匹配
                        if strict_term in alias.lower() or regex.search(alias):
                            matches.append(f"{alias} (Gene Alias)")
                            if len(matches) >= 3: 
                                return matches
                                
        except Exception as e:
            logger.error(f"Error reading Homology CSV file: {e}")
            
        return matches
    async def _extract_entity(self, question: str, config: RunnableConfig) -> str:
        """用 LLM 提取用户问题中的核心查询词"""
        extract_prompt = f"""
        Extract the core biological entity (like gene name, chromosome, species, or compound) from this user question.
        Return ONLY the entity name, nothing else.
        Question: "{question}"
        """
        response = await self.llm.ainvoke([HumanMessage(content=extract_prompt)])
        return response.content.strip()

    def _fetch_database_matches(self, target_entity: str) -> list:
        """执行混合降级查询：跨越 MySQL 和本地 Homology 文件"""
        import re
        if not target_entity or len(target_entity) < 2:
            return []
            
        try:
            # 🚀 核心修复：剔除大模型经常附带的“生物学通用废话”
            # 把 "chromosome Chr85" 强制剥离成 "Chr85"
            stop_words = r'(?i)(chromosome|chrom|gene|protein|rna|cds|species|compound|metabolite|alias)\s*'
            target_entity = re.sub(stop_words, '', target_entity).strip()
            
            # 清洗并拆分为模糊匹配模式
            clean_term = re.sub(r'[^a-zA-Z0-9_\-\u4e00-\u9fa5]', '', target_entity)
            if not clean_term: 
                return []
            
            strict_term = clean_term[:6].lower()
            # 截取前 4 个字符构造极致模糊匹配 (Chr85 -> c%h%r%8)
            fuzzy_term = "%".join(list(clean_term[:4].lower())) 
            
            # 1. 捞取 Homology CSV 文件中的别名
            file_matches = self._search_homology_aliases(strict_term, fuzzy_term)
            
            # 2. 捞取 MySQL 中的其他实体
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor(dictionary=True)
            
            fallback_sql = f"""
            (SELECT DISTINCT chromosome AS matched_name, 'Chromosome' AS type 
             FROM genes 
             WHERE LOWER(chromosome) LIKE '%{strict_term}%' OR LOWER(chromosome) LIKE '%{fuzzy_term}%' LIMIT 3)
            UNION
            (SELECT DISTINCT gene_id AS matched_name, 'Gene ID' AS type 
             FROM genes 
             WHERE LOWER(gene_id) LIKE '%{strict_term}%' OR LOWER(gene_id) LIKE '%{fuzzy_term}%' LIMIT 3)
            UNION
            (SELECT DISTINCT name AS matched_name, 'Metabolite' AS type 
             FROM salvia_metabolite_info 
             WHERE LOWER(name) LIKE '%{strict_term}%' OR chinese_name LIKE '%{strict_term}%' LIMIT 3)
            UNION
            (SELECT DISTINCT substance_name AS matched_name, 'Hormone' AS type 
             FROM salvia_hormone_profilling 
             WHERE LOWER(substance_name) LIKE '%{strict_term}%' OR LOWER(compounds) LIKE '%{strict_term}%' LIMIT 3)
            UNION
            (SELECT DISTINCT taxon AS matched_name, 'Species' AS type 
             FROM taxon_metadata 
             WHERE LOWER(taxon) LIKE '%{strict_term}%' OR LOWER(acronym) LIKE '%{strict_term}%' LIMIT 3)
            LIMIT 5;
            """
            
            cursor.execute(fallback_sql)
            sql_results = cursor.fetchall()
            conn.close()
            
            sql_matches = [f"{row['matched_name']} ({row['type']})" for row in sql_results if row['matched_name']]
            
            # 3. 将文件结果和数据库结果合并返回
            combined_matches = file_matches + sql_matches
            return combined_matches[:8]
            
        except Exception as e:
            logger.error(f"Fallback DB search failed: {e}")
            return []
    # ==========================================
    # [修改点 1]：将 def 变成 async def，在参数里加上 config: RunnableConfig
    # ==========================================
    async def generate(self, state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
        # Keep metadata
        metadata = state.get("metadata", {})
        messages = state["messages"]
        
        # Check if the request is not related to bioinformatics
        if metadata.get("routing_reason") == "IRRELEVANT REQUEST.":
            irrelevant_prompt = """
You are a strict bio-information assistant. The user has asked a question that is not related to bioinformatics.
Please generate a polite response that:
1. Explains you are a bio-information assistant
2. Clarifies the types of questions you can answer (genes, proteins, diseases, sequences, etc.)
3. Politely declines to answer the current question
4. Keep the response concise and professional
"""
           # ==========================================
            # [修改点 2]：将 invoke 改为 await ainvoke,关键：把 config 传进去
            # ==========================================
            response =await self.llm.ainvoke([SystemMessage(content=irrelevant_prompt)],config=config)
            
            return {
                "messages": [
                    AIMessage(
                        content=response.content,
                        additional_kwargs={"type": "final_answer"}
                    )
                ],
                "next": "END",
                "metadata": metadata,
                "thinking_content": "User question is not related to bioinformatics, generate a polite response to decline."
            }
        last_human_idx = -1
        for i, msg in enumerate(messages):
            if isinstance(msg, HumanMessage):
                last_human_idx = i
        
        if last_human_idx == -1:
             return {"status": "error", "error": "No user question found"}
        
        original_question = messages[last_human_idx].content
        current_turn_messages = messages[last_human_idx+1:]
        
        db_results = None
        for msg in reversed(current_turn_messages):
            if isinstance(msg, AIMessage) and msg.additional_kwargs.get("type") == "salvia_db_response":
                db_results = msg.additional_kwargs.get("results")
                break
        tool_results = [
            {"tool": getattr(msg, "name", None), "content": msg.content}
            for msg in current_turn_messages
            if isinstance(msg, ToolMessage)
        ]
        authoritative_evidence = {
            "database_results": db_results,
            "tool_results": tool_results,
        }

        eval_result = metadata.get("eval_result", {})
        if eval_result.get("next_step") == "suggest":
            # Empty evidence is deterministic. Fuzzy search here can turn an
            # honest no-result response into unsupported biological claims.
            no_result_answer = (
                "No matching data were found in the database for this query. "
                "Please verify the identifier, species name, and requested field."
            )
            return {
                "messages": [
                    AIMessage(
                        content=no_result_answer,
                        additional_kwargs={
                            "type": "final_answer",
                            "table_data": None,
                            "chart_data": None,
                        },
                    )
                ],
                "metadata": metadata,
                "thinking_content": "Returned a deterministic evidence-safe no-result answer.",
            }
        # ============================================================
        # 步骤 3：图表数据构建逻辑 (Chart Logic) [多组学全能升级版]
        # ============================================================
        chart_payload = None
        
        # 1. 意图检测 (Intent Detection)
        q_lower = original_question.lower()
        
        # A. 显式画图需求：加入英文 compare, distribute, trend
        explicit_viz_keywords = ["visualiz", "chart", "plot", "graph", "diagram", "画图", "绘图", "图表", "可视化", "柱状图", "条形图", "折线图", "曲线图", "散点图", "饼图", "饼状图", "趋势", "对比", "比较", "分布", "占比", "compare", "distribut", "trend"]
        
        # B. 隐式数据需求
        implicit_data_keywords = [
            "fpkm", "rpkm", "count", "tpm", "expression", "表达量", "丰度",
            "phenotype", "phenotypic", "trait", "height", "length", "weight", "content", "activity", 
            "表型", "性状", "株高", "根长", "重量", "含量", "活性", "生物量",
            "abundance", "metabolite", "compound", "代谢组", "hormone", "激素", "iaa", "aba",
            "gc content", "reads", "genome size", "size"
        ]
        
        explicit_viz_requested = any(k in q_lower for k in explicit_viz_keywords)
        user_wants_viz = explicit_viz_requested or \
                         any(k in q_lower for k in implicit_data_keywords)
        
        # 2. 数据处理：构建不同类型的 ECharts 数据
        if db_results and len(db_results) > 0:
            first_row = db_results[0]
            
            # 【关键修复 1】：只限制 Y 轴的数值，不限制 X 轴！
            y_exclude_keys = [
                'gene_id', 'species_name', 'chromosome', 'start_bp', 'end_bp', 'strand', 
                'compound_id', 'sample_id', 'id', 'compound_name', 'name',
                'm/z', 'mz', 'rt', 'retention_time', 'formula', 'adduct'
            ]
            # --------------------------------------------------
            # 🌟 场景 0: 基因密度热力图 (Gene Density Heatmap)
            # 特征：用户明确问 density，且返回结果包含大量的 start_bp
            # --------------------------------------------------
            if ("density" in q_lower or "密度" in q_lower) and "start_bp" in first_row:
                bin_size = 100000
                
                # 1. 遍历所有查询结果，按染色体分组整理位置数据
                chrom_data = {}
                # species_label = first_row.get("species_name", "Species")
                
                for row in db_results:
                    chrom = row.get("chromosome", "Unknown")
                    pos = row.get("start_bp")
                    if pos is not None:
                        if chrom not in chrom_data:
                            chrom_data[chrom] = []
                        chrom_data[chrom].append(int(pos))
                
                # 2. 如果成功分出了染色体数据
                if chrom_data:
                    chromosomes_payload = []
                    
                    for chrom, positions in chrom_data.items():
                        max_pos = max(positions)
                        # 初始化 Bins 数组
                        num_bins = (max_pos // bin_size) + 1
                        bins = [0] * num_bins
                        
                        # 基因填入 Bins
                        for pos in positions:
                            bin_idx = pos // bin_size
                            if bin_idx < num_bins:
                                bins[bin_idx] += 1
                                
                        chromosomes_payload.append({
                            "name": chrom,
                            "length": max_pos, # 加缓冲
                            "bins": bins,
                            "total_genes": len(positions) # 记录总基因数用于可能的排序
                        })
                    
                    # 3. 核心逻辑：按染色体长度排序（从大到小），并只截取前 8 条
                    # (如果你想按基因最密集排序，可以改成 key=lambda x: x["total_genes"]/x["length"])
                    chromosomes_payload = sorted(chromosomes_payload, key=lambda x: x["length"], reverse=True)[:8]

                    # 4. 构建完美适配前端的 Chart Payload
                    chart_payload = {
                        "type": "heatmap_custom",
                        "title": f"Gene Density",
                        "binSize": bin_size,
                        "chromosomes": chromosomes_payload
                    }
                    first_row = {}
            # --------------------------------------------------
            # 场景 1: 重测序质量控制 (QC) -> 散点图 Scatter Plot
            # --------------------------------------------------
            if "gc_content" in first_row and ("number_of_reads" in first_row or "clean_reads" in first_row):
                y_key = "number_of_reads" if "number_of_reads" in first_row else "clean_reads"
                scatter_data = []
                for row in db_results:
                    if row.get("gc_content") is not None and row.get(y_key) is not None:
                        lib_name = str(row.get("library_name") or row.get("taxon") or row.get("cultivar") or "Unknown")
                        scatter_data.append({
                            "name": lib_name,
                            "value": [float(row["gc_content"]), float(row[y_key])]
                        })
                if scatter_data:
                    chart_payload = {
                        "type": "scatter",
                        "title": "Sequencing QC (GC Content vs Reads)",
                        "xAxisName": "GC Content (%)",
                        "yAxisName": "Reads Count",
                        "data": scatter_data
                    }

            # --------------------------------------------------
            # 🌟 场景 2: 饼图逻辑 (绝对互斥版：支持突变双饼图 & 通用单饼图)
            # --------------------------------------------------
            first_keys_str = " ".join(first_row.keys()).lower()
            is_summary_pie = any(k in first_keys_str for k in ["impact_", "snp_", "indel_", "missense_"])
            is_explicit_pie = any(k in q_lower for k in ["pie", "饼图", "饼状图", "占比", "distribution"])
            
            # 🚀 绝对互斥路由起点（只要进了这个 if，底下的 elif 绝对不执行）
            if is_summary_pie or (is_explicit_pie and len(db_results) > 1):
                
                if is_summary_pie:
                    # 1. 提取突变类型数据 (SNP vs INDEL)
                    type_data = []
                    if first_row.get("snp_count") is not None and float(first_row["snp_count"]) > 0: 
                        type_data.append({"name": "SNP", "value": float(first_row["snp_count"])})
                    if first_row.get("indel_count") is not None and float(first_row["indel_count"]) > 0: 
                        type_data.append({"name": "INDEL", "value": float(first_row["indel_count"])})

                    # 2. 提取突变影响数据 (HIGH/MODERATE/LOW/MODIFIER)
                    impact_keys = {
                        "impact_high": "HIGH Impact", "impact_moderate": "MODERATE Impact", 
                        "impact_low": "LOW Impact", "impact_modifier": "MODIFIER",
                        "missense_variant": "Missense", "synonymous_variant": "Synonymous"
                    }
                    impact_data = []
                    for db_key, display_name in impact_keys.items():
                        # 兼容大模型可能返回的大写或小写字段名
                        real_key = next((k for k in first_row.keys() if db_key in k.lower()), None)
                        if real_key and first_row.get(real_key) is not None and float(first_row[real_key]) > 0: 
                            impact_data.append({"name": display_name, "value": float(first_row[real_key])})
                    
                    if type_data or impact_data:
                        chart_payload = {
                            "type": "variant_pie",
                            "title": f"Mutation Summary ({first_row.get('gene_id', 'Gene')})",
                            "typeData": type_data,
                            "impactData": impact_data
                        }
                        
                elif is_explicit_pie and len(db_results) > 1:
                    pie_data = []
                    name_col, val_col = None, None
                    for k, v in first_row.items():
                        if not isinstance(v, (int, float)) and k.lower() != "unit": name_col = k
                        elif isinstance(v, (int, float)) and k not in y_exclude_keys: val_col = k
                    
                    if name_col and val_col:
                        for row in db_results:
                            if row.get(val_col) is not None and float(row[val_col]) > 0:
                                pie_data.append({"name": str(row[name_col]), "value": float(row[val_col])})

                    if pie_data:
                        chart_payload = {
                            "type": "pie",
                            "title": "Distribution Analysis",
                            "data": pie_data
                        }

            # --------------------------------------------------
            # 场景 3: 单行宽表 (处理表达量/单行激素等) -> 柱状图/折线图
            # --------------------------------------------------
            # 🚀 极其重要的 elif：一旦被饼图认领，这里就会彻底锁死，防止柱状图抢戏！
            elif len(db_results) == 1 and user_wants_viz:
                row = db_results[0]
                # 🛡️ 量纲冲突拦截器
                keys_str = " ".join(row.keys()).lower()
                has_length = "length" in keys_str or "diameter" in keys_str
                has_weight = "weight" in keys_str
                has_count = "number" in keys_str or "count" in keys_str
                
                if not ((has_length and has_weight) or (has_length and has_count) or (has_weight and has_count)):
                    data_points = []
                    for k, v in row.items():
                        if k not in y_exclude_keys and isinstance(v, (int, float)):
                            data_points.append({"label": k, "value": float(v)})
                    
                    if len(data_points) > 1 or (explicit_viz_requested and data_points):
                        detected_unit = None
                        if "fpkm" in q_lower: detected_unit = "FPKM"
                        elif "rpkm" in q_lower: detected_unit = "RPKM"
                        elif "tpm" in q_lower: detected_unit = "TPM"
                        elif "count" in q_lower: detected_unit = "Counts"
                        elif any(k in q_lower for k in ["ng/g", "hormone", "激素", "iaa"]): detected_unit = "Value"
                        
                        if not detected_unit:
                            import re
                            unit_match = re.search(r'\((.*?)\)', keys_str)
                            if unit_match: detected_unit = unit_match.group(1)
                            else: detected_unit = "Value"
                            
                        entity_name = row.get('compounds') or row.get('name') or row.get('gene_id') or row.get('substance_name') or 'Analyte'
                        
                        if isinstance(entity_name, str) and entity_name != 'Analyte':
                            entity_name = entity_name.replace('_', ' ').upper()

                        chart_payload = {
                            "type": "bar",
                            "title": f"Data Profile: {entity_name}",
                            "labels": [dp["label"] for dp in data_points],
                            "values": [dp["value"] for dp in data_points],
                            "unit": detected_unit
                        }

            # --------------------------------------------------
            # 场景 4: 多行长表 (栽培种丰度对比、物种大小对比等) -> 柱状图
            # --------------------------------------------------
            elif len(db_results) > 1 and user_wants_viz:
                x_col = None
                y_col = None
                
                # 智能寻找 X轴(文本) 和 Y轴(数值)
                for k, v in first_row.items():
                    if isinstance(v, str):
                        if any(hint in k.lower() for hint in ['gene', 'taxon', 'name', 'species', 'sample', 'library', 'cultivar', 'compound', 'chromosome', 'stage', 'category']):
                            x_col = k
                            break
                if not x_col: 
                    for k, v in first_row.items():
                        if v is not None and k.lower() != "unit": x_col = k; break
                        
                possible_y_cols = [k for k, v in first_row.items() if isinstance(v, (int, float)) and k not in y_exclude_keys]
                
                if possible_y_cols:
                    import re
                    for col in possible_y_cols:
                        col_clean = re.sub(r'\(.*?\)', '', col).replace("_", " ").strip().lower()
                        if col_clean in q_lower:
                            y_col = col
                            break
                    
                    if not y_col:
                        for col in possible_y_cols:
                            if not any(meta in col.lower() for meta in ['replicate', 'stage']):
                                y_col = col
                                break
                        if not y_col: y_col = possible_y_cols[0]
                        
                if x_col and y_col:
                    data_points = []
                    for row in db_results:
                        if row.get(y_col) is not None:
                            data_points.append({
                                "label": str(row.get(x_col, "Unknown")), 
                                "value": float(row[y_col])
                            })

                    if len(data_points) > 1:
                        detected_unit = "Value"
                        title = "Comparative Analysis"
                        if "genome" in y_col: detected_unit = "Gb"; title = "Genome Size Comparison"
                        elif "abundance" in y_col: detected_unit = "Relative Abundance"; title = "Metabolite Abundance"
                        elif "weight" in y_col: detected_unit = "g"
                        elif "length" in y_col: detected_unit = "cm"
                        elif "area" in y_col: detected_unit = "cm²"
                        
                        compound_name = str(first_row.get('compound_name') or first_row.get('name') or "")
                        if compound_name and "abundance" in y_col: title = f"Abundance of {compound_name}"

                        chart_payload = {
                            "type": "bar",
                            "title": title,
                            "labels": [dp["label"] for dp in data_points],
                            "values": [dp["value"] for dp in data_points],
                            "unit": detected_unit
                        }

        # ============================================================
        # 步骤 5：表格数据展示逻辑 (Smart Table Logic)
        # ============================================================
        frontend_table_data = None
        
        # 🚀 核心修改：拦截器
        # 如果上方已经成功构建了基因密度热力图，或者问题中包含 density/密度 关键字，直接保持 frontend_table_data 为 None
        if (chart_payload and chart_payload.get("type") == "heatmap_custom") or "density" in q_lower or "密度" in q_lower:
            logger.info("Gene density query detected. Table display is intercepted and disabled.")
            frontend_table_data = None  # 确保不传入任何表格数据
            
        # 只有在非密度查询时，才走原有的表格展示规则
        elif db_results and isinstance(db_results, list) and len(db_results) > 0:
            row_count = len(db_results)
            col_count = len(db_results[0].keys()) if row_count > 0 else 0
            
            # 规则 A: 多行数据 -> 展示表格
            if row_count > 1:
                frontend_table_data = db_results[:100] # 限制最大行数
            # 规则 B: 单行数据且字段多 -> 展示表格
            elif row_count == 1 and col_count > 5:
                frontend_table_data = db_results
            # 其他情况 -> 不展示表格

        # ============================================================
        # 步骤 5：构建 Prompt
        # ============================================================
        authoritative_results = _bounded_evidence_json(authoritative_evidence)
        
        combined_prompt = f"""
USER QUESTION:
{original_question}
--------------------------------
AUTHORITATIVE RETRIEVED EVIDENCE:
{authoritative_results}
--------------------------------
You are an expert bioinformatics assistant.
You have structured evidence retrieved from databases or approved tools.
Answer the user's question based on the provided context.
### EVIDENCE GROUNDING RULES (HIGHEST PRIORITY)
1. Use only facts that appear in AUTHORITATIVE RETRIEVED EVIDENCE. SQL text,
   agent reasoning, earlier attempts, model memory, examples, and plausible
   biological assumptions are not evidence.
2. Copy every identifier, chromosome, coordinate, strand, sequence, category,
   numeric value, and unit exactly from the retrieved evidence.
   Do not round database numbers unless the user explicitly requests rounding.
   In multi-omics answers, never transfer a unit from one field to another; hormone replicate values have no unit unless their own evidence supplies one.
3. If a requested field is absent, NULL, or empty, say that it is not available
   in the database. Never invent or complete a missing value.
   Conversely, when database_results is non-empty, never claim that no matching
   data were found. Report every explicitly requested field, including fields
   whose value is zero.
4. Do not infer significance, regulation, causality, absence of expression, or
   biological meaning unless the evidence explicitly contains that conclusion.
5. Extra commentary must not introduce new factual claims. Prefer a short,
   direct answer containing only the fields requested by the user.
### STEP 0: UNIT & COLUMN DECODING (CRITICAL)
1. **Detect Unit**: Look for "RPKM", "FPKM", or "Count" in the User Question or Data.
   - Check for phenotype units: "cm", "g", "mg/g", "unit/g", etc.
   - If the data is a phenotype measurement (e.g., Plant Height), use the specific unit found in the data.
   - If user asks for **RPKM**, output **RPKM**.
   - If user asks for **Count**, output **Counts**.
   - Never guess a unit. If no unit is present in the question, column name, or evidence, omit it.
   - Metabolite DAG fields represent relative abundance unless the evidence explicitly supplies another unit; do not label them as Counts.

2. **Decode Columns**: 
   - Look at the column names in "AUTHORITATIVE RETRIEVED EVIDENCE".
   - Using the "DATA ABBREVIATION DICTIONARY" above, translate the cryptic headers into human-readable text.
   - *Example*: If you see column `C_L_12h` with value 50, say: "In the **Control Leaf** at **12 hours**, the expression was 50..." (NOT "In C L 12h...").

### STEP 1: INTENT ANALYSIS (Choose Your Style)

**TYPE A: FACTUAL Q&A (Location, Length, Name)**
- Trigger: Questions like "Where is it located?", "How long is it?", "What is the CDS?"
- **Style**: Direct and clearly structured.
- If AUTHORITATIVE RETRIEVED EVIDENCE contains multiple requested records, include each requested record.
- Do not use demonstration coordinates or shorten chromosome accessions.

**TYPE B: DATA ANALYSIS (Expression, Treatment, Abundance)**
- Trigger: Questions about "Expression", "IAA", "Stress", "Count", "FPKM".
- **Style**: A concise factual answer containing only the requested values.

### STEP 2: NARRATIVE GENERATION RULES (For Type B)

1. Report trends, maxima, minima, rankings, differences, ratios, or averages only
   when that derived value or ordering is explicitly present in the evidence.
2. Never calculate, estimate, round, or mentally average values. Derived values
   must be computed by SQL or deterministic application code before generation.
3. Do not add biological interpretation, background knowledge, significance,
   regulation, induction, causality, or mechanistic explanations.
4. Avoid introductory and concluding filler. Each factual clause must answer a
   field explicitly requested by the user.

### STEP 3: DATA CLEANING (Internal)
- **Intersection Filter**: If user asks for "IAA in Leaf", IGNORE 'Root' or 'Heat' columns.
- **Sorting**: Mention time points in chronological order (0h -> 1h -> 24h).

### 🛑 FORMATTING RULES (STRICT)
1. **ABSOLUTELY NO LATEX OR MATH FORMULAS**:
2. **SEQUENCE FORMATTING (CRITICAL)**:
   - If you output a DNA/RNA/Protein sequence that is longer than 20 characters, **YOU MUST** wrap it in a Markdown code block labeled 'text'.
   - **Do NOT** output raw sequence strings directly in the paragraph.
   - *Bad*: The sequence is ATCG...
   - *Good*:
     The sequence is:
     ```text
     ATCGGCTAGCTAGCTAGCTAG...
     ```
3. **NO MARKDOWN IMAGES**: 
   - **DO NOT** generate image links like `![Chart](...)` or `![Image](...)`.
   - The system will automatically generate the visualization charts based on the data.
   - Just describe the data in text.
4. **FINAL EVIDENCE CHECK**:
   - Compare every factual clause against AUTHORITATIVE RETRIEVED EVIDENCE before answering.
   - Remove any number or categorical claim that is absent from the evidence.
   - Preserve full database identifiers such as chromosome accessions.
   - Answer only the fields requested by the user.
--------------------------------
**Generate a concise, evidence-grounded response now.**
"""
        # If eval_count, record log
        if metadata.get("eval_count"):
            logger.info(f"Generate final answer (after {metadata['eval_count']} evaluations)")
        else:
            logger.info("Generate final answer...")
        # ==========================================
        # [修改点 3]：将 invoke 改为 await ainvoke
        # ==========================================
        response = await self.llm.ainvoke(
            [SystemMessage(content=combined_prompt)], config=config
        )
        if db_results and NO_RESULT_RE.search(str(response.content)):
            correction_prompt = f"""The draft incorrectly says that no data were found,
but the database evidence below is non-empty. Rewrite the answer and report every
field explicitly requested by the user. Preserve exact strings and numbers,
including zero values. Do not add facts outside the evidence.

USER QUESTION:
{original_question}

DATABASE EVIDENCE:
{_bounded_evidence_json(db_results)}

DRAFT ANSWER:
{response.content}

Return only the corrected concise answer."""
            response = await self.llm.ainvoke(
                [SystemMessage(content=correction_prompt)], config=config
            )
            if NO_RESULT_RE.search(str(response.content)):
                # Accuracy-first fallback: a structured row is preferable to a
                # fluent but false no-result claim.
                response = AIMessage(
                    content="Database results: " + _bounded_evidence_json(db_results)
                )

        missing_replicates = _missing_requested_replicate_values(
            original_question, db_results, str(response.content)
        )
        if missing_replicates:
            correction_prompt = f"""The draft omitted explicitly requested hormone
replicate values: {json.dumps(missing_replicates, ensure_ascii=False)}.
Rewrite the answer and include every listed replicate with its exact value.
Do not calculate a mean and do not omit replicate 2 or replicate 3.

USER QUESTION:
{original_question}

DATABASE EVIDENCE:
{_bounded_evidence_json(db_results)}

DRAFT ANSWER:
{response.content}

Return only the corrected concise answer."""
            response = await self.llm.ainvoke(
                [SystemMessage(content=correction_prompt)], config=config
            )
            if _missing_requested_replicate_values(
                original_question, db_results, str(response.content)
            ):
                response = AIMessage(
                    content="Database results: " + _bounded_evidence_json(db_results)
                )


        missing_hgvs = _missing_requested_hgvs_values(
            original_question, db_results, str(response.content)
        )
        if missing_hgvs:
            correction_prompt = f"""The draft omitted explicitly requested HGVS values:
{json.dumps(missing_hgvs, ensure_ascii=False)}. Rewrite the answer and include every
listed HGVS field with its exact value. Do not omit HGVS.c or HGVS.p values.

USER QUESTION:
{original_question}

DATABASE EVIDENCE:
{_bounded_evidence_json(db_results)}

DRAFT ANSWER:
{response.content}

Return only the corrected concise answer."""
            response = await self.llm.ainvoke(
                [SystemMessage(content=correction_prompt)], config=config
            )
            if _missing_requested_hgvs_values(
                original_question, db_results, str(response.content)
            ):
                response = AIMessage(
                    content="Database results: " + _bounded_evidence_json(db_results)
                )


        unsupported_numbers = _unsupported_answer_numbers(
            original_question, authoritative_evidence, str(response.content)
        )
        if unsupported_numbers:
            correction_prompt = f"""The draft answer contains numeric values that are absent
from the user question and authoritative evidence: {unsupported_numbers}.
Rewrite the answer using only exact values in AUTHORITATIVE RETRIEVED EVIDENCE.
Do not add numbered-list indices, estimates, calculations, or commentary.

USER QUESTION:
{original_question}

AUTHORITATIVE RETRIEVED EVIDENCE:
{authoritative_results}

DRAFT ANSWER:
{response.content}

Return only the corrected concise answer."""
            response = await self.llm.ainvoke(
                [SystemMessage(content=correction_prompt)], config=config
            )

        logger.info("Answer generated")
        
        return {
            "messages": [
                AIMessage(
                    content=response.content,
                    additional_kwargs={
                        "type": "final_answer",
                        "table_data": frontend_table_data,
                        "chart_data": chart_payload
                    }
                )
            ],
            "metadata": {
                **metadata,
                "final_table_data": frontend_table_data,
                "final_chart_data": chart_payload
            }
        }

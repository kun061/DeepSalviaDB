import os
import re
import json
import logging
from pathlib import Path
import pandas as pd # 必须引入 pandas
from langchain_core.messages import SystemMessage

logger = logging.getLogger(__name__)

GLOBAL_GENE_MAPPING = {}
MAPPING_LOADED = False
DEFAULT_HOMOLOGY_PATH = Path(__file__).resolve().parents[2] / "data" / "Homology.csv"
def load_gene_mapping_once(csv_path=None):
    csv_path = Path(csv_path) if csv_path else DEFAULT_HOMOLOGY_PATH

    global GLOBAL_GENE_MAPPING, MAPPING_LOADED
    if MAPPING_LOADED:
        return # 已经加载过了，直接跳过
        
    logger.info(f"开始加载本地基因映射表: {csv_path}")
    try:
        df = pd.read_csv(csv_path, encoding='utf-8', low_memory=False)
        mapping = {}
        for index, row in df.iterrows():
            aliases_str = str(row.get('aliases', '')).strip()
            symbol = str(row.get('symbol', '')).strip()
            
            # 把 symbol 和 aliases 合并为所有可能的查询词
            possible_names = []
            if symbol and symbol not in ('-', 'nan'):
                possible_names.append(symbol.lower())
            if aliases_str and aliases_str not in ('-', 'nan'):
                aliases_list = [a.strip().lower() for a in aliases_str.split(',')]
                possible_names.extend(aliases_list)
                
            # 🎯 构建精确 ID 字典 (注意：左侧名字将作为下游 target_species 返回)
            precise_ids = {
            "Salvia_bowleyana": row.get('Salvia_bowleyana', None),
            "Salvia divinorum": row.get('Salvia divinorum', None),
            "Salvia miltiorrhiza--DSS3": row.get('Salvia miltiorrhiza--DSS3', None),
            "Salvia miltiorrhiza--shh": row.get('Salvia miltiorrhiza--shh', None),
            "Salvia miltiorrhiza--Bunge": row.get('Salvia miltiorrhiza--Bunge', None),
            "Salvia_miltiorrhiza_sami-il01": row.get('Salvia_miltiorrhiza_sami-il01', None),
            "Salvia hispanica": row.get('Salvia hispanica', None),
            "Salvia splendens": row.get('Salvia splendens', None)
            # ... 继续添加其他你需要查询的列
            }
            # 清理空值
            precise_ids = {k: v for k, v in precise_ids.items() if pd.notna(v) and str(v).strip() != '-'}
            
            # 注入全局字典
            for name in possible_names:
                if name:
                    if name not in mapping:
                        mapping[name] = []
                    mapping[name].append(precise_ids)
        
        GLOBAL_GENE_MAPPING = mapping
        MAPPING_LOADED = True
        logger.info(f"映射表加载成功！共录入 {len(GLOBAL_GENE_MAPPING)} 个唯一别名。")
    except Exception as e:
        logger.error(f"加载 csv 映射表失败: {e}")


class EntityLinker:
    def __init__(self, csv_path=None):
        load_gene_mapping_once(csv_path)
        self.mapping = GLOBAL_GENE_MAPPING

    def _extract_entities(self, query, llm):
        """利用 Few-Shot 提取俗名，并根据意图进行双轨物种映射"""
        prompt = f"""
        你是一个多组学生物信息学实体提取器。请提取：基因别名(alias)、查询意图(query_type)和物种名(species)。
        
        【第一步：意图分类 (query_type)】
        - 表达量/转录组: "expression" (关键词: 表达量, FPKM, Count, root, leaf, stress, RNA-Seq)
        - 基因组/序列: "genome" (关键词: 序列, CDS, 蛋白, 染色体, 位置, length)
        - 默认: "genome"

        【第二步：物种提取规则 —— 泛基因组模式】
        * 如果用户提问中明确指定了物种或品种（如：丹参、奇亚籽、hispanica、Bunge），请提取对应的标准名称。
        * 如果用户提问中【没有提到】任何具体的物种或品种（例如：“返回ABCC1的 CDS 序列”），请务必将 species 的值填为 "all"，绝对不要自作主张分配默认物种！

        【第三步：基因别名(alias)提取规则 —— 极其重要！】
        - 别名可能是标准的基因缩写（如 ABCC1, SmCPS1, MYB1）。
        - 别名可能是包含数字的 LOC 编号（如 LOC130984914）。
        - 别名可能包含连字符或下划线。
        - ⛔ 负向约束：绝对不要把“基因”、“序列”、“表达量”、“FPKM”、“未知”等通用中文或英文词汇作为别名提取！如果没有找到明确的基因代号，请填 "null"。

        【Few-Shot 学习示例】
        用户: "返回 ABCC1 的 CDS 序列"
        输出: {{"alias": "ABCC1", "query_type": "genome", "species": "all"}}

        用户: "奇亚籽里的 SmCPS1 表达量是多少"
        输出: {{"alias": "SmCPS1", "query_type": "expression", "species": "hispanica"}}

        用户: "丹参中哪个基因控制开花？"
        输出: {{"alias": "null", "query_type": "genome", "species": "Salvia miltiorrhiza--Bunge"}}

        用户提问：{query}
        
        请严格输出 JSON 格式。
        """
        response = None
        try:
            response = llm.invoke([SystemMessage(content=prompt)])
            content = response.content
            
            # 🚀 强大的正则提取，专治大模型废话
            json_match = re.search(r'(\{.*?\})', content, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            else:
                # 备用方案：如果正则没抓到，尝试直接暴力解析
                clean_content = content.replace("```json", "").replace("```", "").strip()
                return json.loads(clean_content)
                
        except Exception as e:
            # 💡 极度关键：把大模型到底回复了什么鬼东西打印出来！
            raw_response = getattr(response, "content", "<no response>")
            logger.warning(f"实体提取失败: {e} | 大模型原始回复: {raw_response}")
            return {"alias": None, "species": None, "query_type": "genome"}

    def _search_local_dict(self, alias, species, query_type):
        """去内存字典查询，区分表达量(DSS3单轨)和基因组(多轨)"""
        if not self.mapping:
            logger.error("🚨 基因字典为空！请检查服务器上的 CSV 路径和文件是否正确！")
            return []
            
        alias_lower = str(alias).lower().strip()
        matched_records = self.mapping.get(alias_lower, [])
        
        if not matched_records:
            return []

        results = []
        for record in matched_records:
            # ==========================================
            # 🚀 轨道 B：表达量/转录组查询 (强行降维到 DSS3)
            # ==========================================
            if query_type == "expression":
                # 1. 不管用户问什么物种，找基因ID时统统去拿 'Salvia miltiorrhiza--DSS3' 这一列的值
                dss3_id = record.get("Salvia miltiorrhiza--DSS3")
                
                if dss3_id:
                    # 2. 伪装结果：ID用DSS3的，但物种名原封不动保留用户问的（比如 Salvia aerea）
                    # 这样 Agent 下一步去查 SQL 表达量表时，才知道找哪个样本的数据
                    target_species = species if species not in ["null", "all", None] else "Salvia miltiorrhiza--DSS3"
                    results.append({
                        "species": target_species,
                        "standard_id": dss3_id
                    })
            
            # ==========================================
            # 🧬 轨道 A：基因组查询 (严格物种对应)
            # ==========================================
            else:
                for sp_name, exact_id in record.items():
                    if species and species not in ["null", "all"]:
                        # 只有当包含目标物种时才保留
                        if species.lower() not in sp_name.lower():
                            continue
                            
                    results.append({
                        "species": sp_name,
                        "standard_id": exact_id
                    })
                    
        # 去重并返回
        return [dict(t) for t in {tuple(d.items()) for d in results}]

    def _extract_entities_fast(self, query):
        """Resolve exact local aliases without an extra LLM round trip."""
        query_lower = query.casefold()
        tokens = set(re.findall(r"[A-Za-z][A-Za-z0-9_-]{1,}", query))
        candidates = sorted(
            (token for token in tokens if token.casefold() in self.mapping),
            key=len,
            reverse=True,
        )
        alias = candidates[0] if candidates else None

        species_patterns = (
            (r"\b(?:dss3|danshen|丹参)\b", "Salvia miltiorrhiza--DSS3"),
            (r"\bbunge\b", "Salvia miltiorrhiza--Bunge"),
            (r"\b(?:hispanica|chia)\b|奇亚籽", "Salvia hispanica"),
            (r"\bsplendens\b|一串红", "Salvia splendens"),
            (r"\bdivinorum\b", "Salvia divinorum"),
            (r"\bbowleyana\b", "Salvia_bowleyana"),
            (r"\bsami[-_]?il01\b", "Salvia_miltiorrhiza_sami-il01"),
        )
        species = "all"
        for pattern, standard_name in species_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                species = standard_name
                break

        query_type = (
            "expression"
            if re.search(
                r"\b(?:expression|fpkm|rpkm|count|rna-?seq|transcript)\b|表达量|转录组",
                query_lower,
                re.IGNORECASE,
            )
            else "genome"
        )
        return {"alias": alias or "null", "species": species, "query_type": query_type}

    def process_query(self, query, llm):
        fast_mode = os.getenv("SALVIACHAT_FAST_MODE", "true").lower() in {"1", "true", "yes", "on"}
        entities = self._extract_entities_fast(query) if fast_mode else self._extract_entities(query, llm)
        
        # 🛡️ 防御 1：如果大模型彻底抽风返回了 None，手动给一个兜底字典
        if not entities:
            logger.warning("大模型未返回有效实体，启用兜底机制。")
            entities = {"alias": "null", "species": "all", "query_type": "genome"}
            
        alias = entities.get("alias")
        species = entities.get("species")
        query_type = entities.get("query_type", "genome")
        
        if not alias or alias == "null":
            return {"status": "no_alias", "query": query}

        # 去本地查字典
        matched_results = self._search_local_dict(alias, species, query_type)

        if not matched_results:
            return {"status": "not_found", "query": query}

        # ==========================================
        # 下面的代码【完全不改】！
        # 因为 matched_results 和你之前的 neo4j_results 结构完全一样！
        # ==========================================
        if species == "all":
            id_list = [f"'{record['standard_id']}'" for record in matched_results]
            ids_string = ", ".join(id_list)
            
            action_hint = f"""
            [组学意图]: {query_type} (跨物种/泛基因组查询)
            [目标物种]: 涵盖所有包含该同源基因的物种
            [强制查询ID集合]: {ids_string} 
            [SQL构建指示]: 这是一个跨物种查询。请务必使用 WHERE gene_id IN ({ids_string}) 进行过滤！如果查序列，请记得 JOIN genes 表以显示 species_name。
            """
            rewritten_query = f"{query}\n\n=== 实体消歧结果与行动指令 ===\n{action_hint}"
            return {"status": "resolved", "query": rewritten_query}

        if len(matched_results) == 1:
            standard_id = matched_results[0]['standard_id']
            target_species = matched_results[0]['species']
            
            if query_type == "expression":
                action_hint = f"""
                [组学意图]: 转录组表达量分析 (Expression)
                [目标数据表物种]: {target_species} (请使用该物种相关的表达量表)
                [强制查询ID]: '{standard_id}' (直接用于 SQL 的 WHERE gene_id = '{standard_id}')
                """
            else:
                action_hint = f"""
                [组学意图]: 基因组序列或位置 (Genome)
                [目标物种]: {target_species}
                [专属基因ID]: '{standard_id}'
                """
            
            rewritten_query = f"{query}\n\n=== 实体消歧结果与行动指令 ===\n{action_hint}"
            return {"status": "resolved", "query": rewritten_query}
            
        elif len(matched_results) > 1:
            species_list = [r['species'] for r in matched_results]
            msg_injection = f"\n\n[SYSTEM WARNING]: The user asked about '{alias}', which corresponds to homologous genes in multiple species ({', '.join(species_list)}). DO NOT query the database yet. Reply to the user directly, asking them to clarify which species they mean."
            ambiguous_query = query + msg_injection
            return {"status": "ambiguous", "query": ambiguous_query}
            
        else:
            return {"status": "not_found", "query": query}

    def close(self):
        # 不再需要关闭数据库连接
        pass
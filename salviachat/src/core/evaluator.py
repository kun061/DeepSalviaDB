from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
import logging
from typing import Dict, Any
import json
import os
import re

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


QUESTION_FIELD_RULES = {
    "description": (r"\b(annotation|functional description|description)\b", {"description"}),
    "symbol": (r"\bsymbol\b", {"symbol"}),
    "aliases": (r"\balias(?:es)?\b", {"aliases"}),
    "chromosome": (
        r"\b(?:which|what) chromosome\b|\bchromosomal location\b|"
        r"\b(?:give|return|show|provide)(?: me)? the chromosome\b",
        {"chromosome", "chrom"},
    ),
    "start_bp": (r"\b(start coordinate|start position|start bp)\b", {"start_bp", "start_pos"}),
    "end_bp": (r"\b(end coordinate|end position|end bp)\b", {"end_bp", "end_pos"}),
    "strand": (r"\bstrand\b", {"strand"}),
    "formula": (r"\b(formula|molecular formula)\b", {"formula"}),
    "mz": (r"\b(m/z|mass.to.charge)\b", {"mz"}),
    "pathway": (r"\bpathway\b", {"pathway"}),
    "snp_count": (r"\bsnps?\b", {"snp_count"}),
    "indel_count": (r"\bindels?\b", {"indel_count"}),
    "impact_high": (r"\b(?:high impact|impact high)\b", {"impact_high"}),
    "impact_moderate": (r"\b(?:moderate impact|impact moderate)\b", {"impact_moderate"}),
    "impact_low": (r"\b(?:low impact|impact low)\b", {"impact_low"}),
}


def _normalize_column(name: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).casefold())


def _missing_requested_fields(question: str, results: Any) -> list[str]:
    if not isinstance(results, list) or not results or not isinstance(results[0], dict):
        return []
    available = {_normalize_column(key) for row in results if isinstance(row, dict) for key in row}
    missing = []
    for label, (pattern, aliases) in QUESTION_FIELD_RULES.items():
        if re.search(pattern, question, re.IGNORECASE):
            normalized_aliases = {_normalize_column(alias) for alias in aliases}
            if not any(
                alias == column or alias in column or column in alias
                for alias in normalized_aliases for column in available
            ):
                missing.append(label)
    return missing


def _result_shape_issue(question: str, results: Any) -> str | None:
    """Detect common cases where executable SQL still cannot answer the question."""
    if not isinstance(results, list) or not results or not isinstance(results[0], dict):
        return None
    columns = {
        _normalize_column(key)
        for row in results if isinstance(row, dict)
        for key in row
    }
    question_lower = question.casefold()
    asks_mean = bool(re.search(
        r"\b(?:calculate|give|return|show|which)\b.*\b(?:mean|average)\b",
        question_lower,
    ))
    if asks_mean and not any(
        column.startswith(("mean", "avg", "average")) for column in columns
    ):
        return "The result does not contain an explicitly computed mean_value."

    asks_difference = bool(re.search(
        r"\b(?:difference|subtract|minus|latter minus|change between)\b",
        question_lower,
    ))
    if asks_difference and not any(
        token in column for column in columns
        for token in ("difference", "diff", "change")
    ):
        return "The result does not contain an explicitly computed difference."

    asks_single_rank = bool(
        re.search(r"\bwhich\b.*\b(?:highest|lowest|maximum|minimum)\b", question_lower)
    )
    if asks_single_rank and len(results) != 1:
        return (
            "The ranking question requires exactly one winning row; use the "
            "appropriate aggregate, ORDER BY, and LIMIT 1."
        )
    return None


def _sql_error_repair_hint(error: str) -> str:
    error_lower = error.casefold()
    if "unknown column 'sample_id'" in error_lower:
        return (
            " This is a wide expression matrix: the requested sample is a "
            "backtick-quoted SELECT column, not a sample_id value. Filter only "
            "by gene_id."
        )
    if "unknown column 's.compound_id'" in error_lower:
        return (
            " In the cultivated-metabolomics join, compound_id belongs to "
            "compound_abundance (a) and salvia_compounds (c), not salvia_species (s)."
        )
    if "unknown column 's.abundance'" in error_lower:
        return " abundance belongs to compound_abundance (a), not salvia_species (s)."
    if "unknown column 's.cultivar'" in error_lower:
        return (
            " salvia_species has sample_id, chinese_name, and scientific_name; "
            "it has no cultivar column."
        )
    return ""


def _empty_result_repair_hint(question: str) -> str | None:
    """Return one schema-aware retry hint for high-risk empty-result queries."""
    q = question.casefold()
    if ("fpkm" in q or "rpkm" in q) and (" sample " in f" {q} " or " in salvia_" in q):
        return (
            "The empty result may come from treating a wide-matrix sample as a "
            "row value. Select the exact backtick-quoted sample column and "
            "filter only by gene_id."
        )
    if "hormone" in q or re.search(r"\brecord\s+\d+\b", q):
        return (
            "Retry the hormone lookup with the correct identity mapping: "
            "English full name is compounds, Chinese name is substance_name, "
            "and ABA/IAA-like abbreviation is class_name."
        )
    if "abundance" in q and ("compound" in q or "salvia" in q):
        return (
            "Retry using the cultivated-metabolomics three-table join: "
            "compound_abundance a JOIN salvia_compounds c on compound_id and "
            "JOIN salvia_species s on sample_id; filter c.name and "
            "s.scientific_name."
        )
    if "&" in question and re.search(r"\b(?:snp|indel|variant|impact)\b", q):
        return (
            "Retry with the complete ampersand-containing gene identifier as "
            "one exact gene_id literal; do not split it into an IN list."
        )
    return None

class Evaluator:
    def __init__(self):
        self.llm = ChatOllama(
            model=os.getenv("SALVIACHAT_EVALUATOR_MODEL", os.getenv("SALVIACHAT_MODEL", "qwen2.5-coder:7b")),
            base_url=os.getenv("SALVIACHAT_OLLAMA_BASE_URL", "http://localhost:11434"),
            #api_key="ollama",
            num_ctx=int(os.getenv("SALVIACHAT_NUM_CTX", "16000")),
            temperature=float(os.getenv("SALVIACHAT_TEMPERATURE", "0")),
            num_predict=int(os.getenv("SALVIACHAT_EVALUATOR_NUM_PREDICT", "128")),
            keep_alive=os.getenv("SALVIACHAT_KEEP_ALIVE", "24h"),
            format="json",
        )
    
    def evaluate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Get or initialize evaluator counter
        if "metadata" not in state:
            state["metadata"] = {}
        
        if "eval_count" not in state["metadata"]:
            state["metadata"]["eval_count"] = 0
        
        # Increment evaluator counter
        state["metadata"]["eval_count"] += 1
        eval_count = state["metadata"]["eval_count"]
        
        # Check if evaluation limit is reached
        if eval_count >= 5:
            logger.info(f"Evaluation limit reached ({eval_count}/5), force to generate")
            return {
                "next": "generate",
                "metadata": {
                    **state["metadata"],
                    "thinking_content": "Evaluation count limit reached, force to generate"
                }
            }

        messages = state["messages"]
        # Get all user messages
        user_question = [msg for msg in messages if isinstance(msg, HumanMessage) and msg.additional_kwargs.get("type") == "user_question"]

        if not user_question:
            return {
                "status": "error",
                "error": "No user question found",
                "metadata": {
                    **state["metadata"],
                    "thinking_content": "No user question found"
                }
            }
        
        # Get first user question
        original_question = user_question[0].content

        # A SQL execution error is not an empty biological result. Route it back
        # to SQL generation with the exact database diagnostic until a later
        # successful database response supersedes it.
        latest_error = next(
            (
                (index, msg.content) for index, msg in reversed(list(enumerate(messages)))
                if isinstance(msg, AIMessage)
                and msg.additional_kwargs.get("type") == "salvia_error"
            ),
            None,
        )
        latest_success_index = max(
            (
                index for index, msg in enumerate(messages)
                if isinstance(msg, AIMessage)
                and msg.additional_kwargs.get("type") == "salvia_db_response"
            ),
            default=-1,
        )
        if latest_error and latest_error[0] > latest_success_index:
            reason = (
                f"The previous SQL failed: {latest_error[1]} "
                "Generate a corrected SQL query and do not repeat the failed statement."
                + _sql_error_repair_hint(latest_error[1])
            )
            eval_result = {"next_step": "router", "reason": reason}
            return {
                "next": "router",
                "metadata": {
                    **state["metadata"],
                    "eval_result": eval_result,
                    "thinking_content": reason,
                },
            }

        # Deterministic gate: a non-empty result is not sufficient when it
        # omits fields explicitly requested by the user.
        db_responses = [
            msg.additional_kwargs.get("results")
            for msg in messages
            if isinstance(msg, AIMessage)
            and msg.additional_kwargs.get("type") == "salvia_db_response"
        ]
        latest_results = db_responses[-1] if db_responses else None
        latest_response = next((
            msg for msg in reversed(messages)
            if isinstance(msg, AIMessage)
            and msg.additional_kwargs.get("type") == "salvia_db_response"
        ), None)
        latest_batches = (
            latest_response.additional_kwargs.get("result_batches")
            if latest_response is not None else None
        )
        if (
            isinstance(latest_batches, list)
            and len(latest_batches) > 1
            and any(isinstance(batch, list) and not batch for batch in latest_batches)
        ):
            reason = (
                "A multi-query request returned an empty result for at least one "
                "requested section. Retry the missing section using the exact schema; "
                "do not generate a partial answer."
            )
            return {
                "next": "router",
                "metadata": {
                    **state["metadata"],
                    "eval_result": {"next_step": "router", "reason": reason},
                    "thinking_content": reason,
                },
            }
        # One schema-aware retry prevents a syntactically valid but semantically
        # wrong query from being mistaken for evidence that the data do not exist.
        empty_hint = (
            _empty_result_repair_hint(original_question)
            if latest_results == [] and eval_count == 1
            else None
        )
        if empty_hint:
            eval_result = {"next_step": "router", "reason": empty_hint}
            return {
                "next": "router",
                "metadata": {
                    **state["metadata"],
                    "eval_result": eval_result,
                    "thinking_content": empty_hint,
                },
            }
        missing_fields = _missing_requested_fields(original_question, latest_results)
        if missing_fields:
            reason = (
                "The SQL result omitted explicitly requested fields: "
                + ", ".join(missing_fields)
                + ". Generate a corrected SQL query selecting these exact fields."
            )
            eval_result = {"next_step": "router", "reason": reason}
            logger.info(reason)
            return {
                "next": "router",
                "metadata": {
                    **state["metadata"],
                    "eval_result": eval_result,
                    "thinking_content": reason,
                },
            }

        shape_issue = _result_shape_issue(original_question, latest_results)
        if shape_issue:
            reason = shape_issue + " Generate a corrected SQL query."
            eval_result = {"next_step": "router", "reason": reason}
            logger.info(reason)
            return {
                "next": "router",
                "metadata": {
                    **state["metadata"],
                    "eval_result": eval_result,
                    "thinking_content": reason,
                },
            }

        # Successful structured retrieval has already passed deterministic
        # field/shape/error checks above, so fast mode can skip an evaluator LLM call.
        fast_mode = os.getenv("SALVIACHAT_FAST_MODE", "true").lower() in {"1", "true", "yes", "on"}
        if fast_mode:
            if isinstance(latest_results, list) and latest_results:
                reason = "Structured database evidence passed deterministic validation."
                return {
                    "next": "generate",
                    "metadata": {
                        **state["metadata"],
                        "eval_result": {"next_step": "generate", "reason": reason},
                        "thinking_content": reason,
                    },
                }
            if latest_results == []:
                reason = "The database query completed and returned no rows."
                return {
                    "next": "suggest",
                    "metadata": {
                        **state["metadata"],
                        "eval_result": {"next_step": "suggest", "reason": reason},
                        "thinking_content": reason,
                    },
                }
            if any(isinstance(msg, ToolMessage) for msg in messages):
                reason = "Tool execution completed; continue to final generation."
                return {
                    "next": "generate",
                    "metadata": {
                        **state["metadata"],
                        "eval_result": {"next_step": "generate", "reason": reason},
                        "thinking_content": reason,
                    },
                }

        # Get related history messages
        agent_history = [
            msg for msg in messages
            if (
                (isinstance(msg, AIMessage) and msg.additional_kwargs.get("type") in [
                    "blast_progress", "blast_response",
                    "salvia_sql_progress", "salvia_db_response", "salvia_error",
                    "kg_cypher_progress", "kg_db_response", "kg_error",
                    "kg_rag_response" ,
                ])
                or isinstance(msg, ToolMessage)
            )
        ]

        # Format history as text
        history_text = ""
        for msg in agent_history:
            msg_type = msg.additional_kwargs.get("type", "unknown")
            history_text += f"\n--- {msg_type} ---\n{msg.content}\n"

        
        # Use LLM to evaluate if information is sufficient
        eval_prompt = [
        SystemMessage(content=f"""\
You are a strict answer evaluator and traffic router.
You must make a decision about whether the conversation history can answer the question, and where to route the workflow next.

【缺失数据宽容原则 (Missing Data Tolerance) - 极其重要！】
1. 生物学数据库是不完美的。某些基因可能在数据库中确实没有注释出 CDS、RNA 或 Protein 序列（字段为 NULL 或空）。
2. 如果 Agent 已经成功执行了查询，但部分基因的序列字段为空，这属于“客观的数据库现状”，绝不是 Agent 检索失败！
3. ⛔ 绝对不要因为“部分基因缺失所求序列”而判定信息不充分或打回重做！
4. 遇到这种情况，必须立刻判定为信息已充足（进入后续流程），让下游诚实地告诉用户数据库现状。

【无数据终止原则 (No Data Termination) - 新增！】
1. 如果历史记录显示数据库已经执行了检索（例如执行了 SQL 查询或 Cypher 查询），但明确返回了空结果（Empty / No match / 未找到）。
2. 不要要求系统继续反复尝试检索。必须立即终止流程，引导系统进入数据推荐/说明环节。

You need to return a JSON object with the following structure:
{{
    "next_step": "generate", "router", or "suggest",
    "reason": "Brief explanation of your decision"
}}

Examples:
1. When info is sufficient and NO chart is needed:
{{
    "next_step": "generate",
    "reason": "Found the official gene symbol and sequence, no visualization required."
}}

2. When info is insufficient and we need to fetch more data:
{{
    "next_step": "router",
    "reason": "Only have gene IDs, still need to fetch detailed metabolic pathway data."
}}

3. When query was executed but explicitly returned empty/no results:
{{
    "next_step": "suggest",
    "reason": "The database explicitly returned no results for this specific query. Further routing will not help."
}}

--------------------------------
USER QUESTION: {original_question}
--------------------------------
CONVERSATION HISTORY:
{history_text if history_text else "No previous conversation history."}
--------------------------------
Please return your decision as a valid JSON object.
""")
        ]
        
        try:
            response = self.llm.invoke(eval_prompt)
            
            # Parse JSON response
            try:
                eval_result = json.loads(response.content)
            except json.JSONDecodeError:
                json_match = re.search(r'({.*?})', response.content.replace('\n', ''), re.DOTALL)
                if not json_match:
                    logger.error(f"Cannot extract valid JSON from evaluator response: {response.content}")
                    return {
                        "next": "router",
                        "metadata": {
                            **state["metadata"],
                            "eval_error": "Invalid JSON response from evaluator",
                            "thinking_content": "Invalid JSON response from evaluator"
                        }
                    }
                eval_result = json.loads(json_match.group(1))
            
            logger.info(f"Evaluator decision: {eval_result} (evaluation count: {eval_count}/5)")
            
            next_step = eval_result.get("next_step", "").lower()

            # Non-empty rows are authoritative evidence, including 0, NULL, empty strings, and "-".
            # Missing requested columns were already handled by deterministic gates above.
            if next_step == "suggest" and isinstance(latest_results, list) and latest_results:
                eval_result = {
                    "next_step": "generate",
                    "reason": (
                        "Database retrieval returned one or more rows. Values such as "
                        "0, NULL, empty strings, and '-' are valid database evidence; "
                        "report them faithfully instead of claiming no match."
                    ),
                }
                next_step = "generate"
            
            if next_step == "router":
                logger.info(f"Information insufficient, return router to continue querying. Reason: {eval_result.get('reason')}")
                return {
                    "next": "router",
                    "metadata": {
                        **state["metadata"],
                        "eval_result": eval_result,
                        "thinking_content": f"Information is insufficient, return router to continue querying. {eval_result.get('reason')}"
                    }
                }
            elif next_step == "generate":
                logger.info(f"Information sufficient & No chart needed, enter generate stage. Reason: {eval_result.get('reason')}")
                return {
                    "next": "generate",
                    "metadata": {
                        **state["metadata"],
                        "eval_result": eval_result,
                        "thinking_content": f"Information is sufficient, enter generate stage. {eval_result.get('reason')}"
                    }
                }
            elif next_step == "suggest":
                logger.info(f"Data completely missing, enter suggest stage. Reason: {eval_result.get('reason')}")
                return {
                    "next": "suggest",
                    "metadata": {
                        **state["metadata"],
                        "eval_result": eval_result,
                        "thinking_content": f"Requested data is not available in the database, switching to suggestion mode. {eval_result.get('reason')}"
                    }
                }
            else:
                logger.warning(f"Invalid decision '{next_step}', fallback to router")
                return {
                    "next": "router",
                    "metadata": {
                        **state["metadata"],
                        "eval_error": f"Invalid decision '{next_step}' from evaluator",
                        "thinking_content": "Invalid decision from evaluator, return router to continue querying"
                    }
                }
            
        except Exception as e:
            logger.error(f"Evaluation error: {str(e)}")
            return {
                "next": "router",
                "metadata": {
                    **state["metadata"],
                    "eval_error": f"Evaluation error: {str(e)}",
                    "thinking_content": f"Evaluation error: {str(e)}"
                }
            }
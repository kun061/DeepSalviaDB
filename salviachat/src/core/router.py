from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
import os
import logging
import json
import re
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.tools.entity_linker import EntityLinker
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

class Router:
    def __init__(self):
        # 本地模型调用
        self.llm = ChatOllama(
            model=os.getenv("SALVIACHAT_ROUTER_MODEL", os.getenv("SALVIACHAT_MODEL", "qwen2.5-coder:7b")),
            base_url=os.getenv("SALVIACHAT_OLLAMA_BASE_URL", "http://localhost:11434"),
            #api_key="ollama",
            num_ctx=int(os.getenv("SALVIACHAT_NUM_CTX", "16000")),
            temperature=float(os.getenv("SALVIACHAT_TEMPERATURE", "0")),
            num_predict=int(os.getenv("SALVIACHAT_ROUTER_NUM_PREDICT", "128")),
            keep_alive=os.getenv("SALVIACHAT_KEEP_ALIVE", "24h"),
            format="json",
        )
        # self.llm = ChatOpenAI(
        #     model="deepseek-chat", # (DeepSeek V3)
        #     api_key="YOUR_API_KEY", 
        #     base_url="https://api.deepseek.com",
        #     temperature=0,      
        #     max_tokens=2000
        # )
    def route(self, state):
        # Keep metadata
        metadata = state.get("metadata", {})
        messages = state["messages"]
        
        # Get user question
        user_question = [msg for msg in messages if isinstance(msg, HumanMessage)]
        
        # Get history
        agent_history = [
            msg for msg in messages
            if isinstance(msg, AIMessage) and
            (msg.additional_kwargs.get("type") in ["salvia_db_response", "salvia_sql_progress" , "blast_response", "blast_progress" ])
        ]
        
        if not user_question:
            return {
                "status": "error",
                "error": "No user question found",
                "metadata": {
                    **metadata,
                    "thinking_content": "No user question found"
                }
            }
        
        # Extract user question text
        original_question = user_question[0].content
        # 前置图谱消歧与重写
        rewritten_message = None
        try:
            linker = EntityLinker()
            link_result = linker.process_query(original_question, self.llm)
            linker.close()
    
            if link_result["status"] != "no_alias":
                original_question = link_result["query"]
                
                # 构造提示消息，并使用明确的标签包裹，让下游正则无脑提取！
                rewritten_message = SystemMessage(
                    content=f"[ENTITY_LINKER_SUCCESS]: The user's query contains a biological alias. It has been successfully resolved to the official database ID. Please strictly use this ID for your SQL query. \n\n<RESOLVED_ID>{original_question}</RESOLVED_ID>",
                    additional_kwargs={"type": "entity_linking_resolved"}
                )
                logger.info(f"Entity Linking [{link_result['status']}]: Rewritten")
        except Exception as e:
            logger.error(f"Entity linking tool failed: {e}")
            
        # Fast deterministic routing avoids a full LLM call for clear requests.
        fast_mode = os.getenv("SALVIACHAT_FAST_MODE", "true").lower() in {"1", "true", "yes", "on"}
        if fast_mode:
            tool_pattern = re.compile(
                r"\b(?:blast|alignment|align|primer|esmfold|protein structure|"
                r"domain prediction|enrichment|go enrichment|kegg enrichment)\b|"
                r"比对|引物|蛋白结构|富集分析|结构预测",
                re.IGNORECASE,
            )
            database_pattern = re.compile(
                r"\b(?:gene|chromosome|expression|fpkm|rpkm|phenotype|hormone|"
                r"metabolite|compound|variant|snp|indel|exon|intron|cds|sequence|"
                r"salvia|dag\s*\d+|gwh[a-z0-9_.:&-]+)\b|"
                r"基因|染色体|表达量|表型|激素|代谢物|化合物|变异|序列|丹参|鼠尾草",
                re.IGNORECASE,
            )
            if tool_pattern.search(original_question):
                fast_agent = "tool_agent"
            elif metadata.get("eval_result") or database_pattern.search(original_question):
                fast_agent = "salvia_agent"
            else:
                fast_agent = None
            if fast_agent:
                result = {
                    "next": fast_agent,
                    "metadata": {
                        **metadata,
                        "routing_reason": f"Fast deterministic route: {fast_agent}",
                        "thinking_content": f"{fast_agent} selected by deterministic routing.",
                    },
                }
                if rewritten_message:
                    result["messages"] = [rewritten_message]
                return result

        # Format history text
        history_text = ""
        for msg in agent_history:
            msg_type = msg.additional_kwargs.get("type", "unknown")
            history_text += f"\n--- {msg_type} ---\n{msg.content}\n"
        
        # Get evaluator's opinion
        eval_result = metadata.get("eval_result", {})
        eval_reason = eval_result.get("reason", "No evaluation reason provided")
        
        combined_prompt = f"""
You are a strict router that uses JSON to make routing decisions. You should analyze the conversation history above to make an informed decision about which agent to use.

ROUTER OPTIONS:
    - salvia_agent: THE PRIMARY DATABASE TOOL. Use this for ANY gene lookup, expression data, quantitative data, **phenotype/trait data (e.g., root length, fresh weight, DAG stage)**, **hormone profiling (e.g., IAA, ABA, dag30), cultivated species metabolomics (abundance, compounds)**,**resequencing data (reads, GC content), population info (cultivar, origin), and taxon metadata (genome size, family)**,**genetic variants (SNPs, Indels, AF), mutation impacts, and gene structural features (Exons, CDS, Introns)**,or specific biological database queries.
    - tool_agent : Used for performing dynamic computing tasks in bioinformatics! The trigger keywords include: alignment, BLAST, enrichment analysis, GO/KEGG, prediction of protein structure, ESMFold, design of primers, and prediction of domains.
    - irrelevant_questions: the question is not related to bioinformatics or plant science.

RULES:
1. If the user asks for a specific gene ID, exact expression value, specific numerical phenotype data, or the content/concentration/trend of a specific chemical compound (e.g., Tanshinone), **hormone,cultivated species**,**sequencing stats/population origin/taxon info**,**genetic variants/SNPs, or gene structure (exon/intron counts)**,choose 'salvia_agent'.
2. If the user asks "What GO terms..." or another supported computation task, choose 'tool_agent'.
3. General literature questions without structured database evidence are outside this deployment; choose 'irrelevant_questions'.
4. [CRITICAL]: If the USER QUESTION contains a "[SYSTEM WARNING]", immediately route to 'salvia_agent'.

EXAMPLE ROUTING (Few-Shot):
User: "Show me the expression of his_Gene123" -> {{"agent": "salvia_agent", "reason": "Querying exact gene expression data."}}
User: "What is the content change trend of Tanshinone I?" -> {{"agent": "salvia_agent", "reason": "Querying quantitative trend of a specific compound."}}
User: "查询化合物 Tanshinone I 的含量变化趋势。" -> {{"agent": "salvia_agent", "reason": "Querying quantitative trend of a specific compound in Chinese."}}
User: "Summarize the medicinal history of Salvia." -> {{"agent": "irrelevant_questions", "reason": "This deployment has no literature/RAG agent."}}
You should consider the following:
    - What is the question?
    - What information we have gathered so far?
    - Which tool would be most appropriate for the next step?
    - The evaluator's opinion.

--------------------------------
USER QUESTION: 
{original_question}
--------------------------------
CONVERSATION HISTORY:
{history_text if history_text else "No previous interaction history."}
--------------------------------
PREVIOUS EVALUATOR'S OPINION:
{eval_reason}
--------------------------------

You MUST output your decision in the following JSON format:
{{
    "agent": "salvia_agent" or "tool_agent" or "irrelevant_questions",
    "reason": "Brief explanation of why this agent was chosen for the next step"
}}
                                     
Do not include any other text or formatting. ONLY return the JSON object.
"""
        
        router_prompt = [SystemMessage(content=combined_prompt)]
        
        # Add retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            response = self.llm.invoke(router_prompt)
            try:
                response_json = json.loads(response.content)
                if "agent" in response_json and "reason" in response_json:
                    agent = response_json["agent"]
                    reason = response_json["reason"]
                    
                    if agent in ["salvia_agent", "tool_agent", "irrelevant_questions"]:
                        logger.info(f"Routing decision: {agent}, reason: {reason}")
                        
                        # 构建返回的字典
                        return_dict = {
                            "next": agent,
                            "metadata": {
                                **metadata,
                                "routing_reason": "IRRELEVANT REQUEST." if agent == "irrelevant_questions" else reason,
                                "thinking_content": f"{agent} was chosen for the next step. Reason: {reason}"
                            }
                        }
                        
                        # 🚀 极其重要：如果存在重写消息，把它塞进 messages 列表里传下去！
                        if rewritten_message:
                            return_dict["messages"] = [rewritten_message]
                            
                        return return_dict
                    else:
                        logger.warning(f"Unknown agent: {agent}")
                else:
                    logger.warning(f"JSON response format is incorrect: {response_json}")
            except json.JSONDecodeError:
                logger.warning(f"Try {attempt + 1}/{max_retries}: cannot parse JSON: {response.content}")
        
        # Default to salvia_agent
        logger.warning("Multiple attempts failed, default to salvia_agent")
        return {
            "next": "salvia_agent",
            "metadata": {
                **metadata,
                "thinking_content": "Failed to route, default to salvia_agent"
            }
        }
        
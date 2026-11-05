import os
import logging
from typing import Dict, Any
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
#from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

class KnowledgeComponent:
    def __init__(self):
        self.llm = ChatOllama(
            model="qwen2.5:7b", # 建议用稍微大一点的模型做知识总结
            base_url="http://localhost:11434",
            temperature=0.3, # 稍微给点温度，让语言更自然，但不要太高以免产生幻觉
            num_ctx=8000
        )
        # self.llm = ChatOpenAI(
        #     model="deepseek-reasoner", # 👈 模型版本写在这里！调用 R1 深度思考模型
        #     api_key="YOUR_API_KEY",
        #     base_url="https://api.deepseek.com",
        #     temperature=0.3    # 稍微给一点温度，让润色更自然
        # )
        try:
            # 必须使用与建库时完全一致的 Embedding 模型
            self.embeddings = HuggingFaceEmbeddings(model_name=os.getenv("SALVIACHAT_EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5"))
            # 加载本地的 Chroma 数据库
            self.vector_db = Chroma(
                persist_directory=os.getenv("SALVIACHAT_VECTOR_DB", "data/salvia_abstract_vectordb"), 
                embedding_function=self.embeddings
            )
            logger.info("ChromaDB 本地知识库加载成功！")
        except Exception as e:
            logger.error(f"ChromaDB 加载失败，请检查路径: {e}")
            self.vector_db = None

    def answer_general_question(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Step 1: 检索文献并生成回答"""
        metadata = state.get("metadata", {})
        messages = state["messages"]
        
        user_question = [msg for msg in messages if isinstance(msg, HumanMessage)][-1].content
        
        retrieved_context = ""
        
        # 🚀 核心动作：去数据库里搜最相关的 4 个片段
        if self.vector_db:
            try:
                # k=4 表示召回相关度最高的 4 个块
                docs = self.vector_db.similarity_search(user_question, k=4)
                
                # 将搜出来的文档块拼成一段长文本，供大模型阅读
                formatted_docs = []
                for i, doc in enumerate(docs):
                    # 如果你在建库时加了 metadata，比如文件名，这里也可以提出来： doc.metadata.get('source')
                    source = doc.metadata.get('source', '未知文献')
                    formatted_docs.append(f"[资料 {i+1}] (来源: {source}):\n{doc.page_content}")
                
                retrieved_context = "\n\n".join(formatted_docs)
                logger.info(f"成功检索到 {len(docs)} 段相关文献。")
            except Exception as e:
                logger.error(f"向量检索过程出错: {e}")

        # 构建给大模型的 Prompt
        system_prompt = f"""
你是一位顶尖的植物学专家和生物信息学研究员，特别精通鼠尾草属（Salvia）植物的研究。
用户正在向你请教一个科普或学术问题。

请务必基于以下我为你检索到的【本地文献资料】来回答。
要求：
1. 回答要严谨、专业，层次分明（可以适当使用列表）。
2. 如果提供的资料能回答问题，请在事实陈述后自然地融入资料内容。
3. 如果提供的资料与问题完全无关，请直白地告诉用户“目前的本地文献库中没有找到明确记载”，并仅依靠你的专业常识给出简要推测。绝不能捏造文献中没有的数据。

【检索到的本地文献资料】：
{retrieved_context if retrieved_context else "（未检索到相关本地资料）"}
"""
        try:
            # 召唤 Qwen 结合文献生成最终回答
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_question)
            ])
            
            return {
                "messages": [
                    AIMessage(
                        content=response.content,
                        additional_kwargs={"type": "kg_rag_response"}
                    )
                ],
                "next": "end", # 知识问答完毕，直接终结子图
                "metadata": {
                    **metadata, 
                    "thinking_content": "Vector RAG retrieved documents and LLM generated the answer."
                }
            }
        except Exception as e:
            logger.error(f"知识总结生成失败: {e}")
            return {"messages": [AIMessage(content=f"知识总结生成失败: {str(e)}")]}
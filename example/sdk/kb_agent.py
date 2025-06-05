"""示例脚本：构建一个能跨知识库检索并生成报告的简单智能体。

本脚本展示如何使用 RAGFlow SDK 与 OpenAI API 协同工作，
按照以下步骤执行：
1. 从问题中提取关键词；
2. 在知识库1中循环检索相关文档；
3. 下载文档后使用 `MarkItDown` 转换为 Markdown，
   让 LLM 结合问题进行分析；
4. 汇总所有分析结果生成 Markdown 报告，
   并将报告回传到知识库2 用于存档。

运行前请设置环境变量 ``RAGFLOW_API_KEY``、``KB1_ID``、``KB2_ID``、
``OPENAI_API_KEY``。如需自定义模型服务地址和名称，可通过 ``OPENAI_BASE_URL``
和 ``OPENAI_MODEL`` 指定，默认为硅基流动的 DeepSeek-R1。若需要在分析
长文档时切换到千问的 ``qwen-long-latest``，可通过 ``OPENAI_LONG_MODEL`` 指定。
如果长文本模型与默认模型由不同厂商提供，可另外设置 ``OPENAI_LONG_API_KEY``
和 ``OPENAI_LONG_BASE_URL`` 以使用不同的鉴权信息和服务地址。
回复长度分别受 ``OPENAI_MAX_TOKENS`` 与 ``OPENAI_LONG_MAX_TOKENS`` 控制。
"""

import os
import time
import re
import logging
import asyncio
import json
from typing import List, Tuple, Dict

from openai import AsyncOpenAI
from ragflow_sdk import RAGFlow
from markitdown import MarkItDown
import io
import tiktoken

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# RAGFlow 服务地址，默认指向本地
RAGFLOW_HOST = os.environ.get("RAGFLOW_HOST", "http://127.0.0.1:9380")

# 获取鉴权信息和知识库 ID
RAGFLOW_API_KEY = os.environ.get("RAGFLOW_API_KEY")
KB1_ID = os.environ.get("KB1_ID")
KB2_ID = os.environ.get("KB2_ID")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")
# 当长文本模型来自不同厂商时，可通过以下变量指定其专用的 KEY 和地址
OPENAI_LONG_API_KEY = os.environ.get("OPENAI_LONG_API_KEY", OPENAI_API_KEY)
OPENAI_LONG_BASE_URL = os.environ.get("OPENAI_LONG_BASE_URL", OPENAI_BASE_URL)
# 默认模型使用 DeepSeek-R1（由硅基流动提供，支持 96K 上下文）
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "Pro/deepseek-ai/DeepSeek-R1")
# 长文本分析模型使用千问的 qwen-long-latest，最大上下文约 10K，最大输出 8192
OPENAI_LONG_MODEL = os.environ.get("OPENAI_LONG_MODEL", "Qwen/qwen-long-latest")
# 回复长度限制，可通过环境变量自定义
OPENAI_MAX_TOKENS = int(os.environ.get("OPENAI_MAX_TOKENS", "2048"))
OPENAI_LONG_MAX_TOKENS = int(os.environ.get("OPENAI_LONG_MAX_TOKENS", "8192"))

# 配置 OpenAI 客户端（异步）
client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
client_long = AsyncOpenAI(api_key=OPENAI_LONG_API_KEY, base_url=OPENAI_LONG_BASE_URL)

encoding = tiktoken.get_encoding("cl100k_base")


class RateLimiter:
    """控制请求启动间隔，使并发请求也能均匀发送"""

    def __init__(self, interval: float):
        self.interval = interval
        self.lock = asyncio.Lock()
        self.next_time = 0.0

    async def wait(self):
        async with self.lock:
            now = time.time()
            if now < self.next_time:
                await asyncio.sleep(self.next_time - now)
            self.next_time = time.time() + self.interval


rate_limiter = RateLimiter(1.5)
rate_limiter_long = RateLimiter(1.5)


def count_tokens(text: str) -> int:
    """统计文本的 token 数量，用于判断是否超出模型上下文"""
    return len(encoding.encode(text))


def parse_json_from_text(text: str) -> Dict[str, str]:
    """从 LLM 回复中提取 JSON 对象并解析为字典"""
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except Exception as exc:
        logging.error("JSON 解析失败: %s", exc)
        return {}


# ---------- 工具函数 ----------


async def extract_keywords(question: str, limit: int = 5) -> List[str]:
    """使用长文本模型从问题中提取关键词"""

    logging.info("[LLM] 正在从问题中提取关键词: %s", question)
    prompt = f"你是一个需求分析助理，从下面问题中精准提取不超过{limit}个核心关键词，这些关键词应聚焦在“业务场景”、“需求目标”和“功能特性”。关键词间用逗号分隔。\n问题：" + question
    await rate_limiter_long.wait()
    resp = await client_long.chat.completions.create(
        model=OPENAI_LONG_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.choices[0].message.content
    logging.info("[LLM] 关键词提取结果: %s", text)
    keywords = re.split(r"[,\s]+", text.strip())
    keywords = [k for k in keywords if k][:limit]
    logging.info("[LLM] 解析后的关键词: %s", keywords)
    return keywords


async def extract_keywords_from_insights(
    insights: List[Dict[str, str]],
    question: str,
    base_keywords: List[str],
    limit: int = 5,
) -> List[str]:
    """根据文档分析结果提取额外关键词"""

    if not insights or limit <= 0:
        return []
    joined = "\n".join(json.dumps(i, ensure_ascii=False) for i in insights if i)
    if not joined:
        return []
    prompt = (
        f"你是需求分析助理，已提取的关键词有：{','.join(base_keywords)}。"
        f"根据下面的文档分析结论和问题'{question}'，补充不超过{limit}个新的关键词，"
        "按重要性排序，用逗号分隔给出。\n文档分析结论：\n" + joined
    )
    await rate_limiter_long.wait()
    resp = await client_long.chat.completions.create(
        model=OPENAI_LONG_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.choices[0].message.content
    logging.info("[LLM] 追加关键词提取结果: %s", text)
    kws = [k.strip() for k in re.split(r"[,\s]+", text) if k.strip()]
    result = []
    for k in kws:
        if k not in base_keywords and k not in result:
            result.append(k)
        if len(result) >= limit:
            break
    return result


def retrieve_docs(rag: RAGFlow, dataset_id: str, question: str, threshold: float = 0.2) -> Tuple[List[str], List[str]]:
    """检索知识库并返回相关文档的 ID 与名称"""

    logging.info("在知识库 %s 中检索，查询: %s", dataset_id, question)
    chunks = rag.retrieve(
        dataset_ids=[dataset_id],
        question=question,
        similarity_threshold=threshold,
        top_k=20,
    )
    doc_ids, doc_names = [], []
    for c in chunks:
        if c.document_id not in doc_ids:
            doc_ids.append(c.document_id)
            doc_names.append(c.document_name)
    logging.info("共检索到 %d 个文档", len(doc_ids))
    return doc_ids, doc_names


def download_and_convert(rag: RAGFlow, dataset_id: str, doc_id: str, fallback_name: str) -> tuple[str, str]:
    """下载文档并返回 ``(markdown, 文件名)``"""
    try:
        dataset = rag.list_datasets(id=dataset_id)[0]
        document = dataset.list_documents(id=doc_id)[0]
        name = document.name or fallback_name
        logging.info("正在从知识库 %s 下载文档 %s", dataset_id, name)
        content = document.download()
        logging.info("正在将文档 %s 转换为 Markdown", name)
        md = MarkItDown()
        result = md.convert_stream(io.BytesIO(content))
        logging.info("转换后的 Markdown 长度: %d", len(result.markdown))
        return result.markdown, name
    except Exception as exc:
        logging.error("下载或转换文档 %s 失败: %s", fallback_name, exc)
        return "", fallback_name


async def analyze_document(question: str, md_text: str) -> Dict[str, str]:
    """分析单个 Markdown 文档并以结构化 JSON 返回结果"""

    logging.info("[LLM] 正在分析文档，长度 %d", len(md_text))
    example = {
        "需求背景": "",
        "需求目标": "",
        "需求方案": {
            "触发方式": "",
            "参与角色": "",
            "处理流程": "",
            "系统规则": "",
            "字段与界面": "",
            "通知与输出": "",
        },
        "测试要点": "",
    }
    prompt = (
        "你是一名资深需求分析师，专注于提取需求背景、需求目标、需求方案、测试要点这几个方面的关键内容。"
        "请仔细分析下面的需求分析文档，针对问题'" + question + "'，精炼并逐项列出相关内容，"
        "若文档未提及某项，请将对应字段设为空字符串。\n"
        "请按照以下 JSON 结构回复：\n" + json.dumps(example, ensure_ascii=False) + "\n\n"
        "文档内容：\n" + md_text
    )
    tokens = count_tokens(prompt)
    # 根据输入 token 数量决定使用常规模型还是长上下文模型
    model = OPENAI_LONG_MODEL if tokens > 95000 else OPENAI_MODEL
    use_long = model == OPENAI_LONG_MODEL
    max_tokens = OPENAI_LONG_MAX_TOKENS if use_long else OPENAI_MAX_TOKENS
    logging.info("[LLM] 使用模型 %s，输入 %d tokens，回复上限 %d", model, tokens, max_tokens)
    cli = client_long if use_long else client
    limiter = rate_limiter_long if use_long else rate_limiter
    await limiter.wait()
    resp = await cli.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    result = resp.choices[0].message.content
    logging.info("[LLM] 分析结果: %s", result)
    return parse_json_from_text(result)


async def compose_report(
    question: str,
    insights: List[Dict[str, str]],
    references: List[Tuple[str, str]],
) -> tuple[str, str]:
    """综合所有分析结果并生成 Markdown 报告"""

    context_lines: List[str] = []
    doc_list: List[Tuple[int, str]] = []
    idx = 1
    for (doc_id, name), insight in zip(references, insights):
        if not insight:
            continue

        # 判断该分析是否包含有效内容
        def has_value(obj: Dict) -> bool:
            for v in obj.values():
                if isinstance(v, dict):
                    if has_value(v):
                        return True
                elif str(v).strip():
                    return True
            return False

        if not has_value(insight):
            continue

        context_lines.append(f"{idx}. {name}: " + json.dumps(insight, ensure_ascii=False))
        doc_list.append((idx, name))
        idx += 1

    context = "\n".join(context_lines)
    doc_list_str = "\n".join(f"{i}. {name}" for i, name in doc_list)
    prompt = (
        f"你是需求分析领域的专家，请基于以下文档内容，针对问题“{question}”提供清晰、结构化的回答。"
        "请按照以下格式输出：\n【需求背景】\n【需求目标】\n【需求方案】\n- **触发方式**：...\n- **参与角色**：...\n- **处理流程**：...\n- **系统规则**：...\n- **字段与界面**：...\n- **通知与输出**：...\n【测试要点】\n如无信息请说明“文档未提及”。\n\n"
        "引用文档时请使用 [^编号] 标注，编号对应文档清单。\n\n"
        f"文档内容：\n{context}\n\n文档清单：\n{doc_list_str}\n\n不要提供未提及内容或一般概念解释，不做任何补充性建议。"
    )

    tokens = count_tokens(prompt)
    model = OPENAI_LONG_MODEL if tokens > 95000 else OPENAI_MODEL
    use_long = model == OPENAI_LONG_MODEL
    max_tokens = OPENAI_LONG_MAX_TOKENS if use_long else OPENAI_MAX_TOKENS
    logging.info(
        "[LLM] 汇总报告使用模型 %s，输入 %d tokens，回复上限 %d",
        model,
        tokens,
        max_tokens,
    )
    cli = client_long if use_long else client
    limiter = rate_limiter_long if use_long else rate_limiter
    await limiter.wait()
    resp = await cli.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    body = resp.choices[0].message.content.strip()

    title_prompt = f"请根据以下问题，生成一个简洁明确的中文标题，不超过20个字，切勿添加额外说明或标注。\n问题：“{question}”\n文档：“{body}”"
    await limiter.wait()
    resp = await cli.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": title_prompt}],
        max_tokens=64,
    )
    title = resp.choices[0].message.content.strip()

    now = time.strftime("%Y-%m-%d %H:%M:%S")
    doc_lines = [f"[^{i}]: {name}" for i, name in doc_list]
    report = f"标题：{title}\n时间：{now}\n\n内容：\n{body}\n\n参考文档：\n" + "\n".join(doc_lines)
    logging.info("生成最终报告，包含 %d 个引用", len(doc_list))
    return report, title


# ---------- 主流程 ----------


async def main(question: str):
    """根据输入问题生成调研报告"""
    if not (RAGFLOW_API_KEY and KB1_ID and KB2_ID and OPENAI_API_KEY):
        raise RuntimeError("Required environment variables: RAGFLOW_API_KEY, KB1_ID, KB2_ID, OPENAI_API_KEY")

    logging.info("收到的问题: %s", question)
    # 初始化 RAGFlow 客户端并提取初始关键词
    rag = RAGFlow(api_key=RAGFLOW_API_KEY, base_url=RAGFLOW_HOST)
    keywords = await extract_keywords(question)

    # Step2：在知识库1中循环检索
    logging.info("开始在知识库1中检索")
    insights: List[Dict[str, str]] = []
    references: List[Tuple[str, str]] = []
    tried = set()
    for _ in range(5):
        q = ",".join(keywords)
        ids, names = retrieve_docs(rag, KB1_ID, q)
        logging.info("检索关键词: %s -> 找到 %d 个文档", q, len(ids))
        new_refs = [(i, n) for i, n in zip(ids, names) if i not in tried]
        if not new_refs:
            break
        documents = []
        for doc_id, doc_name in new_refs:
            tried.add(doc_id)
            logging.info("分析文件 %s", doc_name)
            md, real_name = download_and_convert(rag, KB1_ID, doc_id, doc_name)
            documents.append((doc_id, real_name, md))

        sem = asyncio.Semaphore(20)

        async def sem_analyze(md: str):
            async with sem:
                return await analyze_document(question, md)

        tasks = [sem_analyze(md) for _, _, md in documents]
        results = await asyncio.gather(*tasks)
        insights.extend(results)
        references.extend([(doc_id, name) for doc_id, name, _ in documents])

        extra = []
        if len(keywords) < 10:
            extra = await extract_keywords_from_insights(results, question, keywords, 10 - len(keywords))
            if extra:
                keywords.extend(extra)
                logging.info("扩展后的关键词: %s", keywords)
        if not extra or len(keywords) >= 10:
            break

    report, title = await compose_report(question, insights, references)
    logging.info("报告生成完毕，正在上传到知识库2")

    # 将生成的报告上传回知识库2
    ts = time.strftime("%Y%m%d%H%M")
    safe_title = re.sub(r"[\\/:*?\"<>|]", "", title)
    filename = f"{ts}{safe_title}.md"
    dataset = rag.list_datasets(id=KB2_ID)[0]
    dataset.upload_documents([{"display_name": filename, "blob": report.encode("utf-8")}])
    logging.info("已上传报告 %s", filename)

    # 控制台输出报告内容
    print(report)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python kb_agent.py 'your question'")
        sys.exit(1)
    # 从命令行读取问题并执行主流程
    asyncio.run(main(sys.argv[1]))

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
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

# Rerank 服务相关配置
RERANK_MODEL = os.environ.get("RERANK_MODEL", "")
RERANK_BASE_URL = os.environ.get("RERANK_BASE_URL", "")

# 检索结果返回的最大文档数
TOP_K = int(os.environ.get("TOP_K", "100"))

# 本地保存报告的目录
REPORT_BASE_DIR = os.environ.get("REPORT_BASE_DIR", "reports")

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

# 全局统计信息：累计 tokens、使用的模型及开始时间和费用
TOKENS_IN = 0
TOKENS_OUT = 0
TOTAL_COST = 0.0
MODELS_USED: set[str] = set()
START_TIME = time.time()

# 各模型的费用配置，格式示例：
# {"model_name": {"prompt": 0.001, "completion": 0.002}}
DEFAULT_MODEL_PRICES = {
    "Pro/deepseek-ai/DeepSeek-R1": {"prompt": 0.004, "completion": 0.016},
    "qwen-long-latest": {"prompt": 0.0005, "completion": 0.002},
    "Qwen/qwen-long-latest": {"prompt": 0.0005, "completion": 0.002},
    "qwen3-235b-a22b": {"prompt": 0.002, "completion": 0.020},
    "Qwen/Qwen3-235B-A22B": {"prompt": 0.0025, "completion": 0.010},
}

MODEL_PRICES_ENV = os.environ.get("MODEL_PRICES")
if MODEL_PRICES_ENV:
    try:
        MODEL_PRICES = json.loads(MODEL_PRICES_ENV)
    except Exception as exc:
        logging.warning("Failed to parse MODEL_PRICES: %s", exc)
        MODEL_PRICES = DEFAULT_MODEL_PRICES
else:
    MODEL_PRICES = DEFAULT_MODEL_PRICES


async def call_chat(
    *,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int | None = None,
    use_long: bool = False,
) -> str:
    """统一的 LLM 调用，负责限流及统计"""

    cli = client_long if use_long else client
    limiter = rate_limiter_long if use_long else rate_limiter
    await limiter.wait()
    est_prompt_tokens = sum(count_tokens(m.get("content", "")) for m in messages)
    resp = await cli.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
    )
    usage = getattr(resp, "usage", None)
    if usage:
        tokens_prompt = usage.prompt_tokens or est_prompt_tokens
        tokens_resp = usage.completion_tokens or count_tokens(resp.choices[0].message.content)
    else:
        tokens_prompt = est_prompt_tokens
        tokens_resp = count_tokens(resp.choices[0].message.content)
    global TOKENS_IN, TOKENS_OUT, TOTAL_COST
    TOKENS_IN += tokens_prompt
    TOKENS_OUT += tokens_resp
    price = MODEL_PRICES.get(model, {})
    TOTAL_COST += tokens_prompt / 1000 * float(price.get("prompt", 0))
    TOTAL_COST += tokens_resp / 1000 * float(price.get("completion", 0))
    MODELS_USED.add(model)
    return resp.choices[0].message.content.strip()


async def call_chat_checked(
    *,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int | None = None,
    use_long: bool = False,
    patterns: List[str] | None = None,
    retries: int = 2,
) -> str:
    """Call the model and retry if output doesn't match patterns."""

    for attempt in range(retries + 1):
        text = await call_chat(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            use_long=use_long,
        )
        if not patterns or all(re.search(p, text) for p in patterns):
            return text
        logging.warning("LLM output format mismatch, retrying %d/%d", attempt + 1, retries)
    return text


def count_tokens(text: str) -> int:
    """统计文本的 token 数量，用于判断是否超出模型上下文"""
    return len(encoding.encode(text))


def parse_json_from_text(text: str) -> Any:
    """从文本中提取 JSON 对象或数组并解析。

    清除 ```json 包裹及 Markdown 链接，定位首个 JSON 片段并尝试解析，
    如失败则返回 ``None``。
    """

    text = text.strip()
    # Remove fenced code block markers such as ```json ... ```
    text = re.sub(r"^```(?:json)?\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```\s*$", "", text).strip()
    # Strip markdown style links which may break JSON
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(text)
        return obj
    except Exception:
        match = re.search(r"\{.*\}|\[.*\]", text, flags=re.S)
        if match:
            cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", match.group(0))
            try:
                obj, _ = decoder.raw_decode(cleaned)
                return obj
            except Exception:
                pass
        logging.error("JSON 解析失败: %s", text)
        return None


def _canonical_doc_key(name: str) -> tuple[str, str]:
    """Return (date, base_name) extracted from document display name."""
    m = re.match(r"^(\d{8})发布_[^_]+_(.+)$", name)
    if m:
        date, tail = m.groups()
    else:
        date, tail = "00000000", name
    tail = re.sub(r"[\s_]*\(\d+\)|[\s_]*（\d+）", "", tail)
    tail = re.sub(r"\s*-?副本", "", tail)
    return date, tail


def deduplicate_references(
    refs: List[Tuple[str, str]],
    insights: List[Any],
) -> tuple[List[Tuple[str, str]], List[Any]]:
    """Remove duplicate docs keeping the latest by date."""
    mapping: Dict[str, Tuple[str, str, Any, str]] = {}
    order: List[str] = []
    for (doc_id, name), insight in zip(refs, insights):
        date, key = _canonical_doc_key(name)
        cur = mapping.get(key)
        if cur is None or date > cur[3]:
            mapping[key] = (doc_id, name, insight, date)
            if cur is None:
                order.append(key)
    dedup_refs: List[Tuple[str, str]] = []
    dedup_insights: List[Any] = []
    for key in order:
        doc_id, name, insight, _ = mapping[key]
        dedup_refs.append((doc_id, name))
        dedup_insights.append(insight)
    return dedup_refs, dedup_insights


# ---------- 工具函数 ----------


def doc_has_content(insight: List[Dict[str, str]] | Any) -> bool:
    """判断文档分析结果是否包含有价值的内容"""

    if not isinstance(insight, list):
        return False
    for item in insight:
        if not isinstance(item, dict):
            continue
        viewpoint = str(item.get("观点", "")).strip()
        if viewpoint:
            return True
    return False


def sanitize_doc_name(name: str) -> str:
    """Escape brackets in document names to avoid Markdown links."""

    return name.replace("[", "\\[").replace("]", "\\]")


def wrap_details(label: str, content: str) -> str:
    """Wrap content in a collapsible HTML details block."""

    return f"<details><summary>{label}</summary>\n\n{content}\n</details>\n\n"


def strip_code_fences(text: str) -> str:
    """Remove leading and trailing fenced code blocks."""

    text = text.strip()
    text = re.sub(r"^```(?:\w+)?\n?", "", text)
    text = re.sub(r"```\s*$", "", text)
    return text.strip()


def fold_snippet_section(text: str) -> str:
    """Always collapse the snippet section using HTML details."""

    m = re.search(r"(#### 3\. 典型原文摘录\n)(.+)", text, flags=re.S)
    if not m:
        return text
    header, rest = m.groups()
    folded = wrap_details("典型原文摘录", rest.strip())
    return text[: m.start()] + header + folded


async def cluster_insight_batch(items: List[Tuple[int, Dict[str, str]]]) -> List[Dict[str, Any]]:
    """Group a batch of insights by theme using the LLM."""

    records = [
        {
            "文档编号": i,
            "发布时间": d.get("发布时间", ""),
            "观点": d.get("观点", ""),
            "说明": d.get("说明", ""),
            "原文摘录": d.get("原文摘录", ""),
        }
        for i, d in items
    ]
    prompt = (
        "请对这些 insight 进行主题聚类和归纳总结：\n\n"
        "对每个主题，输出以下字段：\n"
        "- 主题：一句话描述聚合逻辑点。\n"
        "- 观点摘要：简要列出该主题下的主要观点（合并重复）。\n"
        "- 共识（可选）：若多个 insight 在条件和时间上完全一致，可列为共识。\n"
        "- 差异说明：若观点不同但适用于不同时间或业务，请说明为并存逻辑。\n"
        "- 代表原文：列出1~3条原文摘录，标注文档编号。\n\n"
        "⚠️ 特别说明：不同发布时间的 insight 属于时间演进，不是冲突；"
        "不同业务条线或条件下的规则可能并存，不能作为分歧；"
        "只有当多个 insight 对相同业务背景且同一时间点逻辑完全相反时，才视为冲突。\n"
        "仅以 JSON 数组返回，不要添加其他说明。\n"
        "若原文中含有 Markdown 链接等特殊格式，请转为纯文本或转义，确保 JSON 可正常解析。\n"
        "数据：\n" + json.dumps(records, ensure_ascii=False)
    )
    tokens = count_tokens(prompt)
    model = OPENAI_LONG_MODEL if tokens > 95000 else OPENAI_MODEL
    use_long = model == OPENAI_LONG_MODEL
    max_tokens = OPENAI_LONG_MAX_TOKENS if use_long else OPENAI_MAX_TOKENS
    text = await call_chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        use_long=use_long,
    )
    data = parse_json_from_text(text)
    if not isinstance(data, list):
        logging.error("主题聚类返回格式异常: %s", text)
        return []
    return data


def merge_by_function(insights: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
    """按功能模块聚合 insight，保留不同条件与发布时间的多条记录"""

    from collections import defaultdict

    merged = defaultdict(list)
    for item in insights:
        module = str(item.get("功能模块", "")).strip()
        if module:
            merged[module].append(item)
    return dict(merged)


async def extract_keywords(question: str, limit: int = 5) -> List[str]:
    """使用长文本模型从问题中提取关键词"""

    logging.info("[LLM] 正在从问题中提取关键词: %s", question)
    prompt = (
        f"你是一位业务信息抽取专家，请从下面的问题中提取不超过 {limit} 个核心关键词或短语。\n\n"
        "关键词应满足以下要求：\n"
        "1. 覆盖任务的**对象**（如机构、系统、业务）、**关键动作**（如调研、升级、整改）、**时间要素**（如2025年）；\n"
        "2. 对于带有“规则”“流程”“现状”等修饰词的表达，应简化为核心业务词，例如“投保规则”应提取为“投保”；\n"
        "3. 忽略泛化词汇，如“调研”“分析”“研究”“情况”等泛泛动作或语气词，除非是问题真正的业务动词；\n"
        "4. 保持精简，去除无用描述；仅以 JSON 数组返回关键词结果，不要包含任何解释说明。\n\n"
        "示例输入1：“调研投保规则”\n"
        '示例输出1:["投保"]\n\n'
        "示例输入2：“调研一下建行2025年相关的改造”\n"
        '示例输出2:["建行", "2025年", "改造"]\n\n'
        f"问题：{question}"
    )
    text = await call_chat(
        model=OPENAI_LONG_MODEL,
        messages=[{"role": "user", "content": prompt}],
        use_long=True,
    )
    logging.info("[LLM] 关键词提取结果: %s", text)
    try:
        keywords = json.loads(text)
        if not isinstance(keywords, list):
            raise ValueError
    except Exception:
        keywords = [k for k in re.split(r"[,\s]+", text) if k]
    keywords = keywords[:limit]
    logging.info("[LLM] 解析后的关键词: %s", keywords)
    return keywords


async def extract_keywords_from_insights(
    insights: List[Dict[str, str]],
    question: str,
    base_keywords: List[str],
    limit: int = 5,
) -> List[str]:
    """根据文档分析结果提取额外关键词，使用思考模型"""

    if not insights or limit <= 0:
        return []
    joined = "\n".join(json.dumps(i, ensure_ascii=False) for i in insights if i)
    if not joined:
        return []
    prompt = (
        f"你是一位业务信息抽取专家，已提取的关键词有：{','.join(base_keywords)}。\n"
        f"根据下面的文档分析结论和问题'{question}'，补充不超过 {limit} 个新的与问题解答可能相关的关键词或短语。\n\n"
        "关键词应满足以下要求：\n"
        "1. 覆盖任务的**对象**（如机构、系统、业务）、**关键动作**（如调研、升级、整改）、**时间要素**（如2025年）；\n"
        "2. 对于带有“规则”“流程”“现状”等修饰词的表达，应简化为核心业务词，例如“投保规则”应提取为“投保”；\n"
        "3. 忽略泛化词汇，如“调研”“分析”“研究”“情况”等泛泛动作或语气词，除非是问题真正的业务动词；\n"
        "4. 保持精简，去除无用描述；仅以 JSON 数组返回关键词结果，不要包含任何解释说明。\n\n"
        "文档分析结论:\n" + joined
    )
    text = await call_chat(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    logging.info("[LLM] 追加关键词提取结果: %s", text)
    try:
        kws = json.loads(text)
        if not isinstance(kws, list):
            raise ValueError
    except Exception:
        kws = [k.strip() for k in re.split(r"[,\s]+", text) if k.strip()]
    result = []
    for k in kws:
        if k not in base_keywords and k not in result:
            result.append(k)
        if len(result) >= limit:
            break
    return result


async def select_relevant_keywords(question: str, keywords: List[str]) -> List[str]:
    """让 LLM 判断哪些关键词与问题直接相关"""

    if not keywords:
        return []
    prompt = f"请从以下关键词中选择与你的问题最直接相关的词，按重要性排序，仅以 JSON 数组返回，不要添加解释。\n问题：{question}\n关键词列表：{','.join(keywords)}"
    text = await call_chat(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    logging.info("[LLM] 相关关键词评估结果: %s", text)
    try:
        kws = json.loads(text)
        if not isinstance(kws, list):
            raise ValueError
    except Exception:
        kws = [k.strip() for k in re.split(r"[,\s]+", text) if k.strip()]
    result = [k for k in kws if k in keywords]
    logging.info("[LLM] 直接相关的关键词: %s", result)
    return result


def retrieve_docs(rag: RAGFlow, dataset_id: str, question: str, threshold: float = 0.3) -> Tuple[List[str], List[str]]:
    """检索知识库并返回相关文档的 ID 与名称"""

    logging.info("在知识库 %s 中检索，查询: %s", dataset_id, question)
    chunks = rag.retrieve(
        dataset_ids=[dataset_id],
        question=question,
        similarity_threshold=threshold,
        top_k=TOP_K,
        rerank_id=RERANK_MODEL,
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


async def analyze_document(
    question: str,
    md_text: str,
    filename: str,
) -> List[Dict[str, str]]:
    """分析单个 Markdown 文档并返回 insight 列表"""

    logging.info("[LLM] 正在分析文档，长度 %d", len(md_text))
    if not md_text:
        logging.error("文档 %s 内容为空，跳过分析", filename)
        return []

    example = [
        {
            "功能模块": "退保回退",
            "发布时间": "20200101",
            "适用条件": "原任务为保全生效，未存在后续保全",
            "系统行为": "系统自动展示可回退任务，阻断未生效任务回退",
            "例外或限制": "如有规则适用于特定渠道/地区/时间段，或不适用某场景，请明确列出",
            "原文摘录": "回退保全任务的任务状态必须为'保全生效'...",
        }
    ]
    prompt = (
        f"你是一位资深系统分析师，负责从历史需求文档中提取“当前系统行为（As-Is）”。\n\n"
        f"请结合提问：“{question}”，严格从以下文档中提取与问题**直接相关**的系统行为信息，并输出结构化结果。\n\n"
        "⚠️ 提取要求如下：\n"
        "- 仅提取系统已明确实现的功能，不包含规划中、建议类内容。\n"
        "- 信息必须与问题存在直接关联，无关内容禁止返回。\n"
        "- 每条信息必须提供对应原文摘录，确保可验证与问题相关。\n"
        "- 禁止联想或补全未明确写明的内容。\n"
        "- 严禁输出 JSON 之外的任何字符。\n\n"
        "返回格式：JSON 数组，每个元素包含以下字段：\n"
        "- 功能模块：功能或子系统的名称，如“退保申请阻断”\n"
        "- 发布时间：该规则对应的文档发布时间（格式：YYYYMMDD，可用文件名提取）\n"
        "- 适用条件：该行为适用的业务前提或触发条件\n"
        "- 系统行为：系统当前明确执行的动作或限制逻辑\n"
        "- 例外或限制：如有规则适用于特定渠道/地区/时间段，或不适用某场景，请明确列出\n"
        "- 原文摘录：原文中支持该逻辑的句子或段落，避免章节标题、编号或流程图说明等无意义内容。\n\n"
        f"示例：\n{json.dumps(example, ensure_ascii=False, indent=2)}\n\n"
        "若无与问题直接相关的内容，请返回空数组 [].\n\n"
        "文档内容:\n" + md_text
    )
    tokens = count_tokens(prompt)
    model = OPENAI_LONG_MODEL if tokens > 95000 else OPENAI_MODEL
    use_long = model == OPENAI_LONG_MODEL
    max_tokens = OPENAI_LONG_MAX_TOKENS if use_long else OPENAI_MAX_TOKENS
    logging.info("[LLM] 使用模型 %s，输入 %d tokens，回复上限 %d", model, tokens, max_tokens)
    result = await call_chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        use_long=use_long,
    )
    data = parse_json_from_text(result)
    if data is None:
        logging.error("LLM 返回无法解析的 JSON: %s", result)
        return []
    if not isinstance(data, list):
        logging.error("LLM 返回格式异常: %s", result)
        return []
    pub_match = re.search(r"(\d{8})", filename)
    pub = pub_match.group(1) if pub_match else ""
    for item in data:
        if "发布时间" not in item or not item["发布时间"]:
            item["发布时间"] = pub
    logging.info("[LLM] 分析结果: %s", data)
    return data


async def compose_report(
    question: str,
    insights: List[Dict[str, str]],
    references: List[Tuple[str, str]],
    keywords: List[str],
    retrieved: int,
) -> tuple[str, str]:
    """综合所有分析结果并生成 Markdown 报告"""

    def has_value(obj: Dict) -> bool:
        for v in obj.values():
            if isinstance(v, dict):
                if has_value(v):
                    return True
            elif str(v).strip():
                return True
        return False

    global TOKENS_IN, TOKENS_OUT, MODELS_USED

    docs: List[Tuple[int, str, str, List[Dict[str, str]]]] = []
    doc_idx_map: Dict[Tuple[str, str], int] = {}
    idx = 1
    for (doc_id, name), insight in zip(references, insights):
        if not insight:
            continue
        if not doc_has_content(insight):
            logging.info("文档 %s 无有效内容，跳过", name)
            continue
        pub = insight[0].get("发布时间", "") if isinstance(insight, list) and insight else ""
        doc_idx_map[(doc_id, name)] = idx
        docs.append((idx, name, pub, insight))
        idx += 1

    doc_list_full: List[Tuple[int, str, str]] = []
    insight_items: List[Tuple[int, Dict[str, str]]] = []

    for i, name, pub, insight in docs:
        doc_list_full.append((i, name, pub))
        if not isinstance(insight, list):
            continue
        for item in insight:
            insight_items.append((i, item))

    flat: List[Dict[str, Any]] = []
    for doc_idx, item in insight_items:
        record = dict(item)
        record["文档编号"] = doc_idx
        flat.append(record)

    merged = merge_by_function(flat)

    sections: List[str] = []
    for module, items in merged.items():
        items.sort(key=lambda x: x.get("发布时间", ""))
        lines = [f"### {module}", "", "| 发布时间 | 适用条件 | 系统行为 | 例外或限制 |", "|----------|----------|----------|----------|"]
        for it in items:
            lines.append(f"| {it.get('发布时间', '')} | {it.get('适用条件', '')} | {it.get('系统行为', '')} | {it.get('例外或限制', '')} |")
        quotes = [f"原文（{it.get('发布时间', '')}）：“{it.get('原文摘录', '').strip()}”[{it.get('文档编号')}]" for it in items if str(it.get("原文摘录", "")).strip()]
        if quotes:
            lines.append("")
            lines.extend(f"> {q}" for q in quotes)
        sections.append("\n".join(lines))

    theme_text = "\n\n".join(sections)
    if not theme_text.strip():
        theme_text = "未查到相关内容"

    summary_prompt = (
        "请根据提问整理本次报告的背景与目标，强调报告聚焦于需求文档中已落地的系统现状，"
        "以便后续差异分析和需求澄清。请生成报告标题。"
        "仅以 JSON 格式回复，如："
        '{"背景与目标": "...", "标题": "关于XXX现状分析报告"}'
        "标题格式必须为：关于{{主题}}现状分析报告，主题不超过20个字。"
        "不得添加其他说明。\n问题：" + question + "\n内容:\n" + theme_text
    )
    summary_text = await call_chat_checked(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": summary_prompt}],
        patterns=[r"背景与目标", r"标题"],
    )
    summary_data = parse_json_from_text(summary_text) or {}
    bg_goal = summary_data.get("背景与目标", "").strip()
    title = summary_data.get("标题", "").strip().splitlines()[0]

    body_lines = [
        "## 一、背景与目标",
        bg_goal,
        "",
        "## 二、系统现状",
        theme_text,
    ]
    body = "\n".join(body_lines)

    title_pattern = r"^关于.{1,20}现状分析报告$"
    if not re.match(title_pattern, title):
        logging.warning("标题格式不符: %s", title)

    doc_lines = [f"{i}. {sanitize_doc_name(name)}" for i, name, _ in doc_list_full]
    end_time_str = time.strftime("%Y-%m-%d %H:%M")
    duration = int(time.time() - START_TIME)
    mins, secs = divmod(duration, 60)
    meta = (
        f"**关键词**：{', '.join(keywords)}  \n"
        f"**累计检索**：{retrieved}篇  \n"
        f"**调查时间**：{end_time_str}  \n"
        f"**耗时**：{mins}分{secs}秒  \n"
        f"**tokens**: in:{TOKENS_IN} out:{TOKENS_OUT}  \n"
        f"**费用**：{TOTAL_COST:.4f}  \n"
        f"**模型**：{','.join(MODELS_USED)}\n---"
    )
    report = f"# {title}\n\n{meta}\n\n[TOC]\n\n{body}"
    if doc_lines:
        report += "\n\n## 三、引用文档\n" + "\n\n".join(doc_lines) + "\n"
    logging.info("生成最终报告，包含 %d 个引用", len(doc_list_full))
    return report, title


# ---------- 主流程 ----------


async def main(question: str):
    """根据输入问题生成现状分析报告"""
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
    all_doc_ids: set[str] = set()
    for _ in range(5):
        q = ",".join(keywords)
        ids, names = retrieve_docs(rag, KB1_ID, q)
        logging.info("检索关键词: %s -> 找到 %d 个文档", q, len(ids))

        # 让 LLM 评估最相关的关键词并分别检索
        relevant = await select_relevant_keywords(question, keywords)
        if relevant:
            sem_r = asyncio.Semaphore(5)

            async def sem_retrieve(kw: str):
                async with sem_r:
                    return await asyncio.to_thread(retrieve_docs, rag, KB1_ID, kw)

            tasks_r = [sem_retrieve(k) for k in relevant]
            results_r = await asyncio.gather(*tasks_r)
            for ids_kw, names_kw in results_r:
                for doc_id, doc_name in zip(ids_kw, names_kw):
                    if doc_id not in ids:
                        ids.append(doc_id)
                        names.append(doc_name)

        all_doc_ids.update(ids)
        logging.info("累计检索到 %d 个文档", len(all_doc_ids))

        new_refs = [(i, n) for i, n in zip(ids, names) if i not in tried]
        if not new_refs:
            break
        if len(references) + len(new_refs) > TOP_K:
            new_refs = new_refs[: TOP_K - len(references)]
        documents = []
        for doc_id, doc_name in new_refs:
            tried.add(doc_id)
            logging.info("分析文件 %s", doc_name)
            md, real_name = download_and_convert(rag, KB1_ID, doc_id, doc_name)
            if not md:
                logging.error("文件 %s 下载或转换失败，已从待分析列表移除", real_name)
                continue
            documents.append((doc_id, real_name, md))

        sem = asyncio.Semaphore(20)

        async def sem_analyze(md: str, name: str):
            async with sem:
                return await analyze_document(question, md, name)

        tasks = [sem_analyze(md, name) for _, name, md in documents]
        results = await asyncio.gather(*tasks)
        insights.extend(results)
        references.extend([(doc_id, name) for doc_id, name, _ in documents])

        extra = []
        if len(keywords) < 10:
            extra = await extract_keywords_from_insights(results, question, keywords, 10 - len(keywords))
            if extra:
                keywords.extend(extra)
                logging.info("扩展后的关键词: %s", keywords)
        if not extra:
            break

    references, insights = deduplicate_references(references, insights)
    report, title = await compose_report(question, insights, references, keywords, len(all_doc_ids))
    logging.info("报告生成完毕，正在上传到知识库2")

    # 将生成的报告上传回知识库2
    ts = time.strftime("%Y%m%d%H%M")
    safe_title = re.sub(r"[\\/:*?\"<>|\s]", "", title)
    filename = f"{ts}{safe_title}.md"
    # 本地目录：以时间和标题区分
    report_dir = os.path.join(REPORT_BASE_DIR, f"{ts}{safe_title}")
    os.makedirs(report_dir, exist_ok=True)
    dataset = rag.list_datasets(id=KB2_ID)[0]
    dataset.upload_documents([{"display_name": filename, "blob": report.encode("utf-8")}])
    # 保存最终报告
    await asyncio.to_thread(
        Path(os.path.join(report_dir, filename)).write_text,
        report,
        encoding="utf-8",
    )
    logging.info("已上传报告 %s", filename)

    # 使用 pandoc 转为 HTML 文档并立即打开
    html_name = filename.rsplit(".", 1)[0] + ".html"
    html_path = os.path.join(report_dir, html_name)
    try:
        await asyncio.to_thread(
            subprocess.run,
            ["pandoc", os.path.join(report_dir, filename), "-o", html_path],
            check=True,
        )
        if sys.platform.startswith("darwin"):
            await asyncio.to_thread(subprocess.run, ["open", html_path])
        elif os.name == "nt":
            os.startfile(html_path)  # type: ignore[attr-defined]
        else:
            await asyncio.to_thread(subprocess.run, ["xdg-open", html_path])
    except Exception as exc:
        logging.error("HTML 生成或打开失败: %s", exc)

    # 控制台输出报告内容
    print(report)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python kb_agent.py 'your question'")
        sys.exit(1)
    # 从命令行读取问题并执行主流程
    asyncio.run(main(sys.argv[1]))

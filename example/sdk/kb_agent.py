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
from typing import Dict, List, Tuple

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

# 默认需要归纳的业务要素
DEFAULT_ELEMENT_KEYS = [
    "触发方式",
    "处理流程",
    "系统规则",
    "字段与界面",
    "通知与输出",
]


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


def doc_has_content(insight: Dict[str, str]) -> bool:
    """判断文档分析结果是否包含有价值的内容"""

    if not isinstance(insight, dict):
        return False
    biz = str(insight.get("业务问题", "")).strip()
    snippet = str(insight.get("与问题相关的原文摘录", "")).strip()
    plan = insight.get("需求方案", {})
    plan_values = plan.values() if isinstance(plan, dict) else []
    return bool(biz or snippet or any(str(v).strip() for v in plan_values))


def wrap_details(label: str, content: str) -> str:
    """Wrap content in a collapsible HTML details block."""

    return f"<details><summary>{label}</summary>\n\n{content}\n</details>\n\n"


def fold_snippet_section(text: str) -> str:
    """Always collapse the snippet section using HTML details."""

    m = re.search(r"(#### 3\. 典型原文摘录\n)(.+)", text, flags=re.S)
    if not m:
        return text
    header, rest = m.groups()
    folded = wrap_details("典型原文摘录", rest.strip())
    return text[: m.start()] + header + folded


async def reduce_element(element: str, items: List[Tuple[int, str]]) -> str:
    """Recursively summarize a specific element across documents."""

    lines = [f"{i}. {text.strip()}" for i, text in items if text.strip()]
    if not lines:
        return ""

    async def summarize_once(block: List[str]) -> str:
        context = "\n".join(block)
        prompt = (
            f"请根据以下“{element}”字段内容，严格按照如下结构分条归纳：\n\n"
            "#### 1. 通用做法\n"
            "- 仅列本要素的主要共性做法。\n\n"
            "#### 2. 分歧与争议\n"
            "- 每条写明具体分歧/争议点，明确不同做法并标注涉及文档编号（如[^1][^4]）。\n\n"
            "#### 3. 典型原文摘录\n"
            "> 每条仅列一句关键原文，标注文档编号（如[^2]）。\n"
            "> 如内容多，可只选最具代表性的2-3条。\n\n"
            "回复格式务必严格与上方示例对齐，不要出现任何说明或多余结构。\n"
            "字段内容如下（每条已标明文档编号）：\n\n" + context
        )
        tokens = count_tokens(prompt)
        model = OPENAI_LONG_MODEL if tokens > 95000 else OPENAI_MODEL
        use_long = model == OPENAI_LONG_MODEL
        max_tokens = OPENAI_LONG_MAX_TOKENS if use_long else OPENAI_MAX_TOKENS
        logging.info("[LLM] 汇总 %s，模型 %s，输入 %d tokens", element, model, tokens)
        patterns = [
            r"#### 1\. 通用做法",
            r"#### 2\. 分歧与争议",
            r"#### 3\. 典型原文摘录",
        ]
        text = await call_chat_checked(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            use_long=use_long,
            patterns=patterns,
        )
        m = re.search(r"(#### 1\. 通用做法.*)", text, flags=re.S)
        return m.group(1).strip() if m else text.strip()

    chunk_size = 20
    current = lines
    while len(current) > 1:
        summaries: List[str] = []
        for start in range(0, len(current), chunk_size):
            chunk = current[start : start + chunk_size]
            summaries.append(await summarize_once(chunk))
        current = summaries
    return fold_snippet_section(current[0])


async def extract_keywords(question: str, limit: int = 5) -> List[str]:
    """使用长文本模型从问题中提取关键词"""

    logging.info("[LLM] 正在从问题中提取关键词: %s", question)
    prompt = (
        f"你是一个需求分析助理，请从下面的问题中提取不超过{limit}个核心关键词。"
        "关键词应聚焦于业务动作或场景，并尽量精简，不包含'规则'、'流程'等修饰词。"
        "请仅以 JSON 数组返回，不要添加任何解释。"
        '例如：["投保", "核保"]\n问题：' + question
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


async def extract_extra_elements(question: str, base_elements: List[str], limit: int = 3) -> List[str]:
    """Determine additional elements to summarize based on the question.

    This step uses the long-context model to better handle lengthy questions.
    """

    prompt = (
        "你是需求分析助理，请根据下列问题判断除了常规要素外还需要额外归纳哪些要素。\n"
        f"常规要素包括：{','.join(base_elements)}。\n"
        f"若问题中提及其他关键维度，请列出这些要素名称，不超过{limit}个。\n"
        "若无额外要素，请返回空数组。仅以 JSON 数组返回，不要添加解释。\n"
        "问题：" + question
    )
    text = await call_chat(
        model=OPENAI_LONG_MODEL,
        messages=[{"role": "user", "content": prompt}],
        use_long=True,
    )
    try:
        elems = json.loads(text)
        if not isinstance(elems, list):
            raise ValueError
    except Exception:
        elems = [x.strip() for x in re.split(r"[,\s]+", text) if x.strip()]
    extra = []
    for e in elems:
        if e not in base_elements and e not in extra:
            extra.append(e)
        if len(extra) >= limit:
            break
    logging.info("[LLM] 解析出的额外要素: %s", extra)
    return extra


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
        f"你是需求分析助理，已提取的关键词有：{','.join(base_keywords)}。"
        f"根据下面的文档分析结论和问题'{question}'，补充不超过{limit}个新的与问题解答可能相关的关键词。"
        "关键词应聚焦于业务动作或场景，并尽量精简，不包含'规则'、'流程'等修饰词，例如'投保规则'应简化为'投保'。"
        '请仅以 JSON 数组返回，不要添加解释，格式示例：["核保", "退保"]。\n文档分析结论:\n' + joined
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
    element_keys: List[str] | None = None,
) -> Dict[str, str]:
    """分析单个 Markdown 文档并以结构化 JSON 返回结果"""

    logging.info("[LLM] 正在分析文档，长度 %d", len(md_text))
    if not md_text:
        logging.error("文档 %s 内容为空，跳过分析", filename)
        return {}
    if element_keys is None:
        element_keys = DEFAULT_ELEMENT_KEYS

    plan_fields = {k: "" for k in element_keys}
    plan_fields.update(
        {
            "参与角色": "",
        }
    )
    example = {
        "文档标题": "",
        "发布时间": "",
        "业务问题": "",
        "需求方案": plan_fields,
        "与问题相关的原文摘录": "",
    }
    prompt = (
        "你是一名资深需求分析师，请专注于分析下列需求文档中与业务问题“" + question + "”最直接相关的内容，提炼关键信息。"
        "请只输出与该问题相关的业务问题与需求方案内容，其它字段如无信息可留空。"
        "如不确定与问题关联性，请先保留该内容，由后续批量归纳时判断其价值。"
        "并在'与问题相关的原文摘录'字段摘录最关键的原文或段落，便于后续引用。\n"
        f"需求方案的要素包括：{','.join(element_keys + ['参与角色'])}。\n"
        "请按照以下 JSON 结构回复：\n" + json.dumps(example, ensure_ascii=False) + "\n\n文档内容：\n" + md_text
    )
    tokens = count_tokens(prompt)
    # 根据输入 token 数量决定使用常规模型还是长上下文模型
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
    # 从文件名中解析发布时间，如 20181108发布_JK005-1234_xxx.docx
    m = re.search(r"(\d{8})", filename)
    if m:
        data["发布时间"] = m.group(1)
    else:
        data["发布时间"] = ""
    if "文档标题" not in data or not data["文档标题"]:
        data["文档标题"] = filename
    logging.info("[LLM] 分析结果: %s", data)
    return data


async def compose_report(
    question: str,
    insights: List[Dict[str, str]],
    references: List[Tuple[str, str]],
    element_keys: List[str] | None = None,
) -> tuple[str, str, Dict[str, str]]:
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

    docs: List[Tuple[int, str, str, Dict]] = []
    idx = 1
    for (doc_id, name), insight in zip(references, insights):
        if not insight:
            continue
        if not doc_has_content(insight):
            logging.info("文档 %s 无有效内容，跳过", name)
            continue
        pub = insight.get("发布时间", "") if isinstance(insight, dict) else ""
        docs.append((idx, name, pub, insight))
        idx += 1

    if element_keys is None:
        element_keys = DEFAULT_ELEMENT_KEYS

    doc_list_full: List[Tuple[int, str, str]] = []
    elements: Dict[str, List[Tuple[int, str]]] = {k: [] for k in element_keys}

    for i, name, pub, insight in docs:
        doc_list_full.append((i, name, pub))
        plan = insight.get("需求方案", {})
        if not isinstance(plan, dict):
            continue
        for key in element_keys:
            text = str(plan.get(key, "")).strip()
            if text:
                elements[key].append((i, text))

    element_summaries: Dict[str, str] = {}
    tasks = [reduce_element(key, items) for key, items in elements.items()]
    results = await asyncio.gather(*tasks)
    for key, summary in zip(elements.keys(), results):
        element_summaries[key] = summary

    context_for_overall = "\n\n".join(f"{k}:\n{v}" for k, v in element_summaries.items() if v)
    overall_prompt = (
        f"请基于以下各要素的分条归纳内容，生成“主要结论与摘要”部分，严格使用如下结构：\n"
        "#### 2.1 共性做法\n- 共性1...\n\n"
        "#### 2.2 分歧与争议\n| 主题 | 观点 |\n| ---- | ---- |\n| 分歧1 | ... |\n\n"
        "#### 2.3 规范化建议\n- 建议1...\n\n"
        "其中“2.2 分歧与争议”部分必须使用 Markdown 表格呈现。\n"
        "仅按上述结构输出，不得添加其他说明或段落。所有文档编号用脚注标注。\n"
        f"内容如下：\n{context_for_overall}"
    )
    model = OPENAI_LONG_MODEL if count_tokens(overall_prompt) > 95000 else OPENAI_MODEL
    use_long = model == OPENAI_LONG_MODEL
    max_tokens = OPENAI_LONG_MAX_TOKENS if use_long else OPENAI_MAX_TOKENS
    overall_patterns = [
        r"#### 2\.1 共性做法",
        r"#### 2\.2 分歧与争议",
        r"#### 2\.3 规范化建议",
    ]
    overall_summary = await call_chat_checked(
        model=model,
        messages=[{"role": "user", "content": overall_prompt}],
        max_tokens=max_tokens,
        use_long=use_long,
        patterns=overall_patterns,
    )
    m = re.search(r"(#### 2\.1 共性做法.*)", overall_summary, flags=re.S)
    if m:
        overall_summary = m.group(1).strip()

    summary_prompt = "请简明扼要地概括下列内容的核心观点，仅反馈观点，不要复述原文或添加解释。\n内容：\n" + overall_summary

    for _ in range(2):
        short_summary = await call_chat(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": summary_prompt}],
        )
        short_summary = short_summary.strip().splitlines()[0]
        short_summary = re.sub(r"^#+", "", short_summary).strip()
        short_summary = re.sub(r"^本报告核心观点[:：\s]*", "", short_summary)
        if short_summary not in overall_summary:
            break
        summary_prompt = "请使用不同措辞进一步提炼以下内容的核心观点，避免与原文重复：\n" + overall_summary

    body_lines = [
        "## 一、调研背景与目标",
        question,
        "",
        "## 二、主要结论与摘要",
        f"本报告核心观点：{short_summary}",
        overall_summary,
        "",
        "## 三、要素逐项归纳",
    ]
    idx_elem = 1
    for key in element_keys:
        summary = element_summaries.get(key, "")
        if summary:
            body_lines.append(f"### 3.{idx_elem} {key}")
            body_lines.append(summary)
            idx_elem += 1
    body = "\n".join(body_lines)

    title_prompt = (
        f"请根据以下问题生成标题，要求：\n1. 仅输出一行，不得包含換行或附加说明。\n2. 格式必须为：关于{{{{主题}}}}调研报告。\n3. 主题不超过20个字。\n问题：{question}\n摘要：{overall_summary}"
    )
    title_pattern = r"^关于.{1,20}调研报告$"
    title = await call_chat_checked(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": title_prompt}],
        max_tokens=64,
        patterns=[title_pattern],
    )
    title = title.strip().splitlines()[0]

    doc_lines = [f"[^{i}]: {name}" for i, name, _ in doc_list_full]
    end_time_str = time.strftime("%Y-%m-%d %H:%M")
    duration = int(time.time() - START_TIME)
    mins, secs = divmod(duration, 60)
    meta = f"**调查时间**：{end_time_str}  \n**耗时**：{mins}分{secs}秒  \n**tokens**: in:{TOKENS_IN} out:{TOKENS_OUT}  \n**费用**：{TOTAL_COST:.4f}  \n**模型**：{','.join(MODELS_USED)}\n---"
    report = f"# {title}\n\n{meta}\n\n[TOC]\n\n{body}\n\n## 四、引用文档\n" + "\n\n".join(doc_lines) + "\n"
    logging.info("生成最终报告，包含 %d 个引用", len(doc_list_full))
    return report, title, element_summaries


# ---------- 主流程 ----------


async def main(question: str):
    """根据输入问题生成调研报告"""
    if not (RAGFLOW_API_KEY and KB1_ID and KB2_ID and OPENAI_API_KEY):
        raise RuntimeError("Required environment variables: RAGFLOW_API_KEY, KB1_ID, KB2_ID, OPENAI_API_KEY")

    logging.info("收到的问题: %s", question)
    # 初始化 RAGFlow 客户端并提取初始关键词
    rag = RAGFlow(api_key=RAGFLOW_API_KEY, base_url=RAGFLOW_HOST)
    keywords = await extract_keywords(question)
    extra_elements = await extract_extra_elements(question, DEFAULT_ELEMENT_KEYS)
    element_keys = DEFAULT_ELEMENT_KEYS + [e for e in extra_elements if e not in DEFAULT_ELEMENT_KEYS]

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
                return await analyze_document(question, md, name, element_keys)

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

    report, title, element_summaries = await compose_report(question, insights, references, element_keys)
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
    # 保存各要素汇总
    for key, summary in element_summaries.items():
        safe_key = re.sub(r"[\\/:*?\"<>|\s]", "_", key)
        batch_name = f"{safe_key}.md"
        if batch_name == filename:
            batch_name = f"{safe_key}_summary.md"
        await asyncio.to_thread(
            Path(os.path.join(report_dir, batch_name)).write_text,
            summary,
            encoding="utf-8",
        )
    logging.info("已上传报告 %s", filename)

    # 使用 pandoc 转为 Word 文档并立即打开
    docx_name = filename.rsplit(".", 1)[0] + ".docx"
    docx_path = os.path.join(report_dir, docx_name)
    try:
        await asyncio.to_thread(
            subprocess.run,
            ["pandoc", os.path.join(report_dir, filename), "-o", docx_path],
            check=True,
        )
        if sys.platform.startswith("darwin"):
            await asyncio.to_thread(subprocess.run, ["open", docx_path])
        elif os.name == "nt":
            os.startfile(docx_path)  # type: ignore[attr-defined]
        else:
            await asyncio.to_thread(subprocess.run, ["xdg-open", docx_path])
    except Exception as exc:
        logging.error("Word 生成或打开失败: %s", exc)

    # 生成 PDF 版本并立即打开
    pdf_name = filename.rsplit(".", 1)[0] + ".pdf"
    pdf_path = os.path.join(report_dir, pdf_name)
    try:
        await asyncio.to_thread(
            subprocess.run,
            ["pandoc", os.path.join(report_dir, filename), "-o", pdf_path],
            check=True,
        )
        if sys.platform.startswith("darwin"):
            await asyncio.to_thread(subprocess.run, ["open", pdf_path])
        elif os.name == "nt":
            os.startfile(pdf_path)  # type: ignore[attr-defined]
        else:
            await asyncio.to_thread(subprocess.run, ["xdg-open", pdf_path])
    except Exception as exc:
        logging.error("PDF 生成或打开失败: %s", exc)

    # 控制台输出报告内容
    print(report)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python kb_agent.py 'your question'")
        sys.exit(1)
    # 从命令行读取问题并执行主流程
    asyncio.run(main(sys.argv[1]))

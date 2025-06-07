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

# Rerank 服务相关配置
RERANK_MODEL = os.environ.get("RERANK_MODEL", "")
RERANK_BASE_URL = os.environ.get("RERANK_BASE_URL", "")

# 检索结果返回的最大文档数
TOP_K = int(os.environ.get("TOP_K", "100"))

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

# 全局统计信息：累计 tokens、使用的模型及开始时间
TOKENS_IN = 0
TOKENS_OUT = 0
MODELS_USED: set[str] = set()
START_TIME = time.time()


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
    prompt = (
        f"你是一个需求分析助理，请从下面的问题中提取不超过{limit}个核心关键词。"
        "关键词应聚焦于业务动作或场景，并尽量精简，不包含'规则'、'流程'等修饰词。"
        "例如：'投保规则'应简化为'投保'。关键词间用逗号分隔。\n问题：" + question
    )
    await rate_limiter_long.wait()
    tokens_prompt = count_tokens(prompt)
    resp = await client_long.chat.completions.create(
        model=OPENAI_LONG_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    tokens_resp = count_tokens(resp.choices[0].message.content)
    global TOKENS_IN, TOKENS_OUT
    TOKENS_IN += tokens_prompt
    TOKENS_OUT += tokens_resp
    MODELS_USED.add(OPENAI_LONG_MODEL)
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
        "\n按重要性排序，用逗号分隔给出。\n文档分析结论：\n" + joined
    )
    await rate_limiter.wait()
    tokens_prompt = count_tokens(prompt)
    resp = await client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    tokens_resp = count_tokens(resp.choices[0].message.content)
    global TOKENS_IN, TOKENS_OUT
    TOKENS_IN += tokens_prompt
    TOKENS_OUT += tokens_resp
    MODELS_USED.add(OPENAI_MODEL)
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


def retrieve_docs(
    rag: RAGFlow, dataset_id: str, question: str, threshold: float = 0.3
) -> Tuple[List[str], List[str]]:
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


async def analyze_document(question: str, md_text: str, filename: str) -> Dict[str, str]:
    """分析单个 Markdown 文档并以结构化 JSON 返回结果"""

    logging.info("[LLM] 正在分析文档，长度 %d", len(md_text))
    example = {
        "文档标题": "",
        "发布时间": "",
        "业务问题": "",
        "需求方案": {
            "触发方式": "",
            "参与角色": "",
            "处理流程": "",
            "系统规则": "",
            "字段与界面": "",
            "通知与输出": "",
        },
        "与问题相关的原文摘录": "",
    }
    prompt = (
        "你是一名资深需求分析师，请专注于分析下列需求文档中与业务问题“"
        + question
        + "”最直接相关的内容，提炼关键信息。"
        "请只输出与该问题相关的业务问题与需求方案内容，其它字段如无信息可留空。"
        "并在'与问题相关的原文摘录'字段摘录最关键的原文或段落，便于后续引用。\n"
        "请按照以下 JSON 结构回复：\n"
        + json.dumps(example, ensure_ascii=False)
        + "\n\n文档内容：\n"
        + md_text
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
    tokens_resp = count_tokens(resp.choices[0].message.content)
    global TOKENS_IN, TOKENS_OUT
    TOKENS_IN += tokens
    TOKENS_OUT += tokens_resp
    MODELS_USED.add(model)
    result = resp.choices[0].message.content
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
) -> tuple[str, str]:
    """综合所有分析结果并生成 Markdown 报告，按批次处理"""

    def has_value(obj: Dict) -> bool:
        for v in obj.values():
            if isinstance(v, dict):
                if has_value(v):
                    return True
            elif str(v).strip():
                return True
        return False

    docs: List[Tuple[int, str, str, Dict]] = []
    idx = 1
    for (doc_id, name), insight in zip(references, insights):
        if not insight:
            continue
        plan = insight.get("需求方案", {}) if isinstance(insight, dict) else {}
        if isinstance(plan, dict) and all(not str(v).strip() for v in plan.values()):
            logging.info("需求方案为空，跳过文档 %s", name)
            continue
        if not has_value(insight):
            continue
        pub = insight.get("发布时间", "") if isinstance(insight, dict) else ""
        docs.append((idx, name, pub, insight))
        idx += 1

    batch_summaries: List[str] = []
    doc_list_full: List[Tuple[int, str, str]] = []
    for start in range(0, len(docs), 20):
        batch = docs[start:start + 20]
        context_lines = []
        doc_list = []
        for i, name, pub, insight in batch:
            context_lines.append(f"{i}. {name}: " + json.dumps(insight, ensure_ascii=False))
            doc_list.append((i, name, pub))
            doc_list_full.append((i, name, pub))
        context = "\n".join(context_lines)
        doc_list_str = "\n".join(f"{i}. {name}" for i, name, _ in doc_list)

        prompt = (
            f"你是需求分析领域的专家，请基于以下文档内容，针对问题‘{question}’撰写调研报告，"
            "不得添加与问题无关的说明。请严格遵循下列 Markdown 结构输出：\n\n"
            "## 一、问题分析  \n"
            "请基于用户的问题，简要描述本次调研关注的业务场景、核心背景或问题缘由。如问题本身已包含场景，可直接转述。\n\n"
            "## 二、调研目标  \n"
            "- 汇总历史文档中与“{业务问题}”相关的内容  \n"
            "- 分析差异，梳理现行规则  \n\n"
            "## 三、主要信息摘录  \n"
            "| 来源 | 发布时间 | 业务问题 | 摘要/方案要点 |  \n"
            "| --- | --- | --- | --- |  \n"
            "(请根据文档内容按编号列出，缺失信息留空)\n\n"
            "*注：如业务问题或方案要点无内容则留空。*\n\n"
            "## 四、相关内容分析  \n"
            "请优先按如下要素（如有）：触发方式、处理流程、系统规则、字段与界面、通知与输出，做结构化归纳。\n"
            "如无结构化内容，可自由梳理所有与问题直接相关的分析和原文片段。\n\n"
            "## 五、现状总结  \n"
            f"请综合上文内容，简明归纳目前关于“{question}”的系统现状、业务做法或得出的结论。如有争议点或不一致，也请注明。\n\n"
            "请在正文中使用形如[^1]的标注引用文档，编号与文档清单一致。\n\n"
            f"文档内容：\n{context}\n\n文档清单：\n{doc_list_str}\n\n"
        )

        tokens = count_tokens(prompt)
        model = OPENAI_LONG_MODEL if tokens > 95000 else OPENAI_MODEL
        use_long = model == OPENAI_LONG_MODEL
        max_tokens = OPENAI_LONG_MAX_TOKENS if use_long else OPENAI_MAX_TOKENS
        logging.info(
            "[LLM] 分批汇总使用模型 %s，输入 %d tokens，回复上限 %d",
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
        tokens_resp = count_tokens(resp.choices[0].message.content)
        global TOKENS_IN, TOKENS_OUT
        TOKENS_IN += tokens
        TOKENS_OUT += tokens_resp
        MODELS_USED.add(model)
        summary_md = resp.choices[0].message.content.strip()
        summary_md = re.sub(r"## 六、引用文档.*", "", summary_md, flags=re.S).rstrip()
        batch_summaries.append(summary_md)

    final_context = "\n\n".join(batch_summaries)
    final_prompt = (
        f"你是需求分析领域的专家，请基于下列分批汇总内容，针对问题‘{question}’生成最终调研报告，"
        "不得添加与问题无关的说明。内容以中文输出。\n\n" + final_context
    )
    tokens_final = count_tokens(final_prompt)
    model = OPENAI_LONG_MODEL if tokens_final > 95000 else OPENAI_MODEL
    use_long = model == OPENAI_LONG_MODEL
    max_tokens = OPENAI_LONG_MAX_TOKENS if use_long else OPENAI_MAX_TOKENS
    logging.info(
        "[LLM] 最终汇总使用模型 %s，输入 %d tokens，回复上限 %d",
        model,
        tokens_final,
        max_tokens,
    )
    cli = client_long if use_long else client
    limiter = rate_limiter_long if use_long else rate_limiter
    await limiter.wait()
    resp = await cli.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": final_prompt}],
        max_tokens=max_tokens,
    )
    tokens_resp = count_tokens(resp.choices[0].message.content)
    TOKENS_IN += tokens_final
    TOKENS_OUT += tokens_resp
    MODELS_USED.add(model)
    body = resp.choices[0].message.content.strip()

    title_prompt = (
        f"请根据以下问题生成标题，格式为：关于{{主题}}调研报告，不超过20个字，切勿添加额外说明或标注。\n问题：“{question}”\n文档：“{body}”"
    )
    await limiter.wait()
    tokens_prompt2 = count_tokens(title_prompt)
    resp = await cli.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": title_prompt}],
        max_tokens=64,
    )
    tokens_resp2 = count_tokens(resp.choices[0].message.content)
    TOKENS_IN += tokens_prompt2
    TOKENS_OUT += tokens_resp2
    MODELS_USED.add(model)
    title = resp.choices[0].message.content.strip()

    doc_lines = []
    for i, name, pub in doc_list_full:
        if pub:
            doc_lines.append(f"[^{i}]: {name}（{pub}）")
        else:
            doc_lines.append(f"[^{i}]: {name}")
    end_time_str = time.strftime("%Y-%m-%d %H:%M")
    duration = int(time.time() - START_TIME)
    mins, secs = divmod(duration, 60)
    meta = (
        f"调查时间：{end_time_str}\n"
        f"耗时：{mins}分{secs}秒\n"
        f"tokens: in:{TOKENS_IN} out:{TOKENS_OUT}\n"
        f"模型：{','.join(MODELS_USED)}\n"
    )
    report = (
        f"# 标题：{title}\n\n{meta}\n{body}\n\n## 六、引用文档\n" + "\n\n".join(doc_lines) + "\n"
    )
    logging.info("生成最终报告，包含 %d 个引用", len(doc_list_full))
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
        if len(references) + len(new_refs) > TOP_K:
            new_refs = new_refs[: TOP_K - len(references)]
        documents = []
        for doc_id, doc_name in new_refs:
            tried.add(doc_id)
            logging.info("分析文件 %s", doc_name)
            md, real_name = download_and_convert(rag, KB1_ID, doc_id, doc_name)
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

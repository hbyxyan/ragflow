"""示例脚本：构建一个能跨知识库检索并生成报告的简单智能体。

本脚本展示如何使用 RAGFlow SDK 与 OpenAI API 协同工作，
按照以下步骤执行：
1. 从问题中提取关键词；
2. 在知识库1中循环检索相关文档；
3. 下载文档后使用 `MarkItDown` 转换为 Markdown，
   让 LLM 结合问题进行分析；
4. 在知识库2中检索既往总结并分析；
5. 汇总所有分析结果生成 Markdown 报告，
   最后将报告回传到知识库2。

运行前请设置环境变量 ``RAGFLOW_API_KEY``、``KB1_ID``、``KB2_ID``、
``OPENAI_API_KEY``，如有需要可通过 ``OPENAI_BASE_URL`` 和 ``OPENAI_MODEL``
指定模型服务地址和名称。若需在分析长文档时切换更大上下文的模型，
可通过 ``OPENAI_LONG_MODEL`` 指定。
"""

import os
import time
import re
import logging
from typing import List, Tuple

from openai import OpenAI
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
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "Qwen/Qwen2.5-72B-Instruct")
OPENAI_LONG_MODEL = os.environ.get("OPENAI_LONG_MODEL", OPENAI_MODEL)

# 配置 OpenAI 客户端
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """统计文本的 token 数量，用于判断是否超出模型上下文"""
    return len(encoding.encode(text))


# ---------- 工具函数 ----------


def extract_keywords(question: str, limit: int = 5) -> List[str]:
    """\
    使用 OpenAI 模型从问题中提取关键词

    参数:
        question: 用户提出的问题
        limit: 最多提取的关键词数量

    返回:
        关键词列表
    """
    logging.info("[LLM] extracting keywords from question: %s", question)
    prompt = (
        f"You are an assistant that extracts the {limit} most important keywords from the question.\n"
        f"Return the keywords separated by comma.\nQuestion: {question}"
    )
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.choices[0].message.content
    logging.info("[LLM] keyword extraction result: %s", text)
    keywords = re.split(r"[,\s]+", text.strip())
    keywords = [k for k in keywords if k][:limit]
    logging.info("[LLM] parsed keywords: %s", keywords)
    return keywords


def retrieve_docs(rag: RAGFlow, dataset_id: str, question: str) -> Tuple[List[str], List[str]]:
    """\
    在指定知识库中检索问题相关的片段，
    并返回所有涉及的文档 ID 与名称
    """
    logging.info("retrieving from dataset %s with query: %s", dataset_id, question)
    chunks = rag.retrieve(dataset_ids=[dataset_id], question=question)
    doc_ids, doc_names = [], []
    for c in chunks:
        if c.document_id not in doc_ids:
            doc_ids.append(c.document_id)
            doc_names.append(c.document_name)
    logging.info("retrieved %d documents", len(doc_ids))
    return doc_ids, doc_names


def download_and_convert(rag: RAGFlow, dataset_id: str, doc_id: str) -> str:
    """\
    下载指定文档并使用 MarkItDown 转换为 Markdown
    """
    logging.info("downloading document %s from dataset %s", doc_id, dataset_id)
    dataset = rag.list_datasets(id=dataset_id)[0]
    document = dataset.list_documents(id=doc_id)[0]
    content = document.download()
    logging.info("converting document %s to markdown", doc_id)
    md = MarkItDown()
    result = md.convert_stream(io.BytesIO(content))
    logging.info("converted markdown length: %d", len(result.markdown))
    return result.markdown


def analyze_document(question: str, md_text: str) -> str:
    """\
    让 LLM 对 Markdown 文档进行分析并提取问题相关信息
    """
    logging.info("[LLM] analyzing document, length %d", len(md_text))
    prompt = (
        f"Given the question: '{question}', extract the relevant information from the following document in markdown.\n\n{md_text}\n"
    )
    tokens = count_tokens(prompt)
    model = OPENAI_LONG_MODEL if tokens > 90000 else OPENAI_MODEL
    logging.info("[LLM] use model %s with %d tokens", model, tokens)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    result = resp.choices[0].message.content
    logging.info("[LLM] analysis result: %s", result)
    return result


def compose_report(insights: List[str], kb2_insights: List[str], references: List[Tuple[str, str]]) -> str:
    """\
    整合分析结果并生成最终的 Markdown 调研报告
    """
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    ref_lines = [f"- {name} ({doc_id})" for doc_id, name in references]
    body = "\n".join(insights + kb2_insights)
    report = (
        f"标题：自动化调研报告\n时间：{now}\n\n内容:\n{body}\n\n引用:\n" + "\n".join(ref_lines)
    )
    logging.info("composed final report with %d references", len(references))
    return report


# ---------- 主流程 ----------


def main(question: str):
    """根据输入问题生成调研报告"""
    if not (RAGFLOW_API_KEY and KB1_ID and KB2_ID and OPENAI_API_KEY):
        raise RuntimeError("Required environment variables: RAGFLOW_API_KEY, KB1_ID, KB2_ID, OPENAI_API_KEY")

    logging.info("question: %s", question)
    # 初始化 RAGFlow 客户端并提取初始关键词
    rag = RAGFlow(api_key=RAGFLOW_API_KEY, base_url=RAGFLOW_HOST)
    keywords = extract_keywords(question)

    # Step2：在知识库1中循环检索
    logging.info("start retrieving from KB1")
    doc_ids, doc_names = [], []
    tried = set()
    for _ in range(5):
        q = ",".join(keywords)
        ids, names = retrieve_docs(rag, KB1_ID, q)
        logging.info("iteration query: %s -> found %d docs", q, len(ids))
        new_refs = [(i, n) for i, n in zip(ids, names) if i not in tried]
        if not new_refs:
            break
        # 记录新发现的文档
        for i, n in new_refs:
            tried.add(i)
            doc_ids.append(i)
            doc_names.append(n)
        logging.info("added %d new documents", len(new_refs))
        if len(keywords) >= 10:
            break
        # 根据文档名再提取一些额外关键词，帮助下一轮检索
        extra = [word for name in names for word in re.findall(r"[\w]+", name)]
        keywords.extend(extra[: 10 - len(keywords)])
        logging.info("expanded keywords: %s", keywords)

    insights = []
    references = []  # 记录所有引用的文档，便于生成报告

    for doc_id, doc_name in zip(doc_ids, doc_names):
        logging.info("analyzing file %s", doc_name)
        md = download_and_convert(rag, KB1_ID, doc_id)
        insight = analyze_document(question, md)
        insights.append(insight)
        references.append((doc_id, doc_name))

    # Step4：检索并分析知识库2的历史总结
    # 用同样的关键词在知识库2中检索既往总结
    logging.info("searching KB2 for historical summaries")
    kb2_doc_ids, kb2_doc_names = retrieve_docs(rag, KB2_ID, ",".join(keywords))
    kb2_insights = []
    for doc_id, doc_name in zip(kb2_doc_ids, kb2_doc_names):
        logging.info("analyzing KB2 file %s", doc_name)
        md = download_and_convert(rag, KB2_ID, doc_id)
        kb2_insight = analyze_document(question, md)
        kb2_insights.append(kb2_insight)
        references.append((doc_id, doc_name))

    report = compose_report(insights, kb2_insights, references)
    logging.info("report composed, uploading to KB2")

    # 将生成的报告上传回知识库2
    filename = f"{int(time.time())}_report.md"
    dataset = rag.list_datasets(id=KB2_ID)[0]
    dataset.upload_documents([{"display_name": filename, "blob": report.encode("utf-8")}])
    logging.info("uploaded report as %s", filename)

    # 控制台输出报告内容
    print(report)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python kb_agent.py 'your question'")
        sys.exit(1)
    # 从命令行读取问题并执行主流程
    main(sys.argv[1])

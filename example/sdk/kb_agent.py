"""示例脚本：构建一个能跨知识库检索并生成报告的简单智能体。

本脚本展示如何使用 RAGFlow SDK 与 OpenAI API 协同工作，
按照以下步骤执行：
1. 从问题中提取关键词；
2. 在知识库1中循环检索相关文档；
3. 下载并将文档内容通过 ``markitdown`` 转为 Markdown，
   让 LLM 结合问题进行分析；
4. 在知识库2中检索既往总结并分析；
5. 汇总所有分析结果生成 Markdown 报告，
   最后将报告回传到知识库2。

运行前请设置环境变量 ``RAGFLOW_API_KEY``、``KB1_ID``、``KB2_ID``
及 ``OPENAI_API_KEY``。
"""

import os
import time
import re
from typing import List, Tuple

import openai
from ragflow_sdk import RAGFlow
from markitdown import markitdown


# RAGFlow 服务地址，默认指向本地
RAGFLOW_HOST = os.environ.get("RAGFLOW_HOST", "http://127.0.0.1:9380")

# 获取鉴权信息和知识库 ID
RAGFLOW_API_KEY = os.environ.get("RAGFLOW_API_KEY")
KB1_ID = os.environ.get("KB1_ID")
KB2_ID = os.environ.get("KB2_ID")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# 配置 OpenAI API Key
openai.api_key = OPENAI_API_KEY


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
    prompt = f"""You are an assistant that extracts the {limit} most important keywords from the question.
Return the keywords separated by comma.
Question: {question}"""
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.choices[0].message.content
    keywords = re.split(r"[,\s]+", text.strip())
    return [k for k in keywords if k][:limit]


def retrieve_docs(rag: RAGFlow, dataset_id: str, question: str) -> Tuple[List[str], List[str]]:
    """\
    在指定知识库中检索问题相关的片段，
    并返回所有涉及的文档 ID 与名称
    """
    chunks = rag.retrieve(dataset_ids=[dataset_id], question=question)
    doc_ids, doc_names = [], []
    for c in chunks:
        if c.doc_id not in doc_ids:
            doc_ids.append(c.doc_id)
            doc_names.append(c.doc_name)
    return doc_ids, doc_names


def download_and_convert(rag: RAGFlow, dataset_id: str, doc_id: str) -> str:
    """\
    下载指定文档并使用 markitdown 转换为 Markdown
    """
    dataset = rag.list_datasets(id=dataset_id)[0]
    document = dataset.list_documents(id=doc_id)[0]
    content = document.download()
    md = markitdown(content)
    return md


def analyze_document(question: str, md_text: str) -> str:
    """\
    让 LLM 对 Markdown 文档进行分析并提取问题相关信息
    """
    prompt = f"""Given the question: '{question}', extract the relevant information from the following document in markdown.\n\n{md_text}\n"""
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


def compose_report(insights: List[str], kb2_insights: List[str], references: List[Tuple[str, str]]) -> str:
    """\
    整合分析结果并生成最终的 Markdown 调研报告
    """
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    ref_lines = [f"- {name} ({doc_id})" for doc_id, name in references]
    body = "\n".join(insights + kb2_insights)
    report = f"标题：自动化调研报告\n时间：{now}\n\n内容:\n{body}\n\n引用:\n" + "\n".join(ref_lines)
    return report


# ---------- 主流程 ----------


def main(question: str):
    """根据输入问题生成调研报告"""
    if not (RAGFLOW_API_KEY and KB1_ID and KB2_ID and OPENAI_API_KEY):
        raise RuntimeError("Required environment variables: RAGFLOW_API_KEY, KB1_ID, KB2_ID, OPENAI_API_KEY")

    # 初始化 RAGFlow 客户端并提取初始关键词
    rag = RAGFlow(api_key=RAGFLOW_API_KEY, base_url=RAGFLOW_HOST)
    keywords = extract_keywords(question)

    # Step2：在知识库1中循环检索
    doc_ids, doc_names = [], []
    tried = set()
    for _ in range(5):
        q = ",".join(keywords)
        ids, names = retrieve_docs(rag, KB1_ID, q)
        new_refs = [(i, n) for i, n in zip(ids, names) if i not in tried]
        if not new_refs:
            break
        # 记录新发现的文档
        for i, n in new_refs:
            tried.add(i)
            doc_ids.append(i)
            doc_names.append(n)
        if len(keywords) >= 10:
            break
        # 根据文档名再提取一些额外关键词，帮助下一轮检索
        extra = [word for name in names for word in re.findall(r"[\w]+", name)]
        keywords.extend(extra[: 10 - len(keywords)])

    insights = []
    references = []  # 记录所有引用的文档，便于生成报告

    for doc_id, doc_name in zip(doc_ids, doc_names):
        md = download_and_convert(rag, KB1_ID, doc_id)
        insight = analyze_document(question, md)
        insights.append(insight)
        references.append((doc_id, doc_name))

    # Step4：检索并分析知识库2的历史总结
    # 用同样的关键词在知识库2中检索既往总结
    kb2_doc_ids, kb2_doc_names = retrieve_docs(rag, KB2_ID, ",".join(keywords))
    kb2_insights = []
    for doc_id, doc_name in zip(kb2_doc_ids, kb2_doc_names):
        md = download_and_convert(rag, KB2_ID, doc_id)
        kb2_insight = analyze_document(question, md)
        kb2_insights.append(kb2_insight)
        references.append((doc_id, doc_name))

    report = compose_report(insights, kb2_insights, references)

    # 将生成的报告上传回知识库2
    filename = f"{int(time.time())}_report.md"
    dataset = rag.list_datasets(id=KB2_ID)[0]
    dataset.upload_documents([{"display_name": filename, "blob": report.encode("utf-8")}])

    # 控制台输出报告内容
    print(report)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python kb_agent.py 'your question'")
        sys.exit(1)
    # 从命令行读取问题并执行主流程
    main(sys.argv[1])

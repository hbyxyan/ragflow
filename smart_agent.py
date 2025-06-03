import os
# import jieba # Retained for now due to __main__ check
# import jieba.analyse # No longer needed for LLM keyword extraction
import re
import requests
import json
from datetime import datetime
import tempfile
# from ragflow import RAGFlow

# CHINESE_STOP_WORDS = [...] # No longer needed for LLM keyword extraction

# --- RAGFlow API 配置 ---
RAGFLOW_API_KEY = "YOUR_RAGFLOW_API_KEY"
RAGFLOW_API_BASE = "http://YOUR_RAGFLOW_API_HOST:8000"
KB1_NAME = "需求分析文档库"
KB2_NAME = "既往问题分析总结库"

# --- RAGFlow HTTP API Client ---
class RAGFlowHttpApiClient:
    def __init__(self, api_key, api_base_url, kb1_name=KB1_NAME, kb2_name=KB2_NAME):
        self.api_key = api_key; self.api_base_url = api_base_url.rstrip('/'); self.kb1_name = kb1_name; self.kb2_name = kb2_name
        self.default_headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        self.kb1_id = None; self.kb2_id = None

    def _request(self, method, endpoint, params=None, data=None, json_data=None, files=None, extra_headers=None):
        url = f"{self.api_base_url}{endpoint}"; current_headers = self.default_headers.copy()
        if extra_headers: current_headers.update(extra_headers)
        if files and 'Content-Type' in current_headers and current_headers['Content-Type'] == 'application/json': del current_headers['Content-Type']
        try:
            timeout_seconds = 10 if method.upper() == "GET" else 30
            response = requests.request(method.upper(), url, headers=current_headers, params=params, data=data, json=json_data, files=files, timeout=timeout_seconds)
            response.raise_for_status()
            if response.content:
                try: return response.json()
                except json.JSONDecodeError: return response.text
            return None
        except requests.exceptions.HTTPError as e:
            error_message = f"HTTP Error: {e.response.status_code}"; error_details_text = e.response.text; error_details_json = None
            try: error_details_json = e.response.json(); error_message += f" - {json.dumps(error_details_json)}"
            except json.JSONDecodeError: error_message += f" - {error_details_text}"
            print(error_message); return {"error": str(e), "status_code": e.response.status_code, "details": error_details_json if error_details_json else error_details_text}
        except requests.exceptions.RequestException as e:
            print(f"Request Exception: {e}"); return {"error": str(e), "status_code": None, "details": None}

    def _get_dataset_id_by_name(self, dataset_name):
        print(f"Attempting to find dataset ID for: {dataset_name}")
        response = self._request("GET", "/api/v1/datasets", params={"name": dataset_name, "page_size": 5})
        if response and isinstance(response, dict) and response.get("code") == 0 and response.get("data", {}).get("items"):
            for ds in response["data"]["items"]:
                if ds.get("name") == dataset_name: print(f"Found dataset: {ds.get('name')} with ID: {ds.get('id')}"); return ds.get("id")
            print(f"Warning: Dataset named '{dataset_name}' not found in 'items' list."); return None
        print(f"Error or unexpected response fetching dataset ID for '{dataset_name}'. Response: {response}"); return None

    def initialize_dataset_ids(self):
        print("Initializing dataset IDs..."); self.kb1_id = self._get_dataset_id_by_name(self.kb1_name); self.kb2_id = self._get_dataset_id_by_name(self.kb2_name)
        if not self.kb1_id: print(f"关键错误: 无法获取KB1 '{self.kb1_name}' ID。")
        else: print(f"KB1 ('{self.kb1_name}') ID: {self.kb1_id}")
        if not self.kb2_id: print(f"关键错误: 无法获取KB2 '{self.kb2_name}' ID。")
        else: print(f"KB2 ('{self.kb2_name}') ID: {self.kb2_id}")
        return self.kb1_id is not None or self.kb2_id is not None # Return true if at least one ID is found

    def get_chat_assistant_for_dataset(self, dataset_id, assistant_name_prefix="AutoGen_"):
        if not dataset_id: print("Error: Dataset ID required for chat assistant."); return None
        assistant_name = f"{assistant_name_prefix}{dataset_id}"
        print(f"  Attempting to get/create chat assistant '{assistant_name}' for dataset ID '{dataset_id}'...")
        response = self._request("GET", "/api/v1/chats", params={"name": assistant_name, "page_size": 1})
        if response and response.get("code") == 0 and response.get("data", {}).get("items"):
            assistants = response["data"]["items"]
            if assistants: # Check if list is not empty
                assistant = assistants[0] # Assuming first one is the match if page_size=1 and name is unique
                # Stricter check:
                # for assistant_item in assistants:
                # if assistant_item.get("name") == assistant_name and dataset_id in assistant_item.get("dataset_ids", []):
                if assistant.get("name") == assistant_name: # Simplified check, assumes name is unique enough
                    print(f"  Found existing assistant '{assistant_name}' with ID: {assistant['id']}")
                    return assistant["id"]

        print(f"  Assistant '{assistant_name}' not found or no exact match. Creating new one...")
        default_llm_model = os.getenv("RAGFLOW_DEFAULT_LLM_MODEL", "qwen-plus@Tongyi-Qianwen")
        create_payload = {"name": assistant_name, "dataset_ids": [dataset_id], "llm": {"model_name": default_llm_model, "temperature":0.1, "top_p":0.3}, "prompt": {"similarity_threshold":0.2,"top_n":3,"show_quote":True}}
        create_response = self._request("POST", "/api/v1/chats", json_data=create_payload)
        if create_response and create_response.get("code") == 0 and create_response.get("data", {}).get("id"):
            new_assistant_id = create_response["data"]["id"]; print(f"  Successfully created assistant '{assistant_name}' with ID: {new_assistant_id}"); return new_assistant_id
        else: print(f"  Error creating assistant '{assistant_name}': {create_response}"); return None

    def chat_with_assistant(self, chat_assistant_id, question, session_id=None, stream=False):
        if not chat_assistant_id: print("Error: Chat Assistant ID required."); return {"error": "Chat Assistant ID required"}
        payload = {"question": question, "stream": stream};
        if session_id: payload["session_id"] = session_id
        response = self._request("POST", f"/api/v1/chats/{chat_assistant_id}/completions", json_data=payload)
        if response and response.get("code") == 0 and "data" in response: # Check for "data" key presence
            return response["data"]
        else:
            error_detail = "Unknown error"
            if response and ("error" in response or "details" in response):
                 error_detail = response.get('details', response.get('error'))
            elif response and "message" in response: # Some RAGFlow errors might use "message"
                 error_detail = response.get("message")

            print(f"Error during chat: {error_detail}");
            return {"error": "Chat failed", "details": error_detail, "answer": None} # Ensure answer is None on error

    def upload_document(self, dataset_id, file_path, doc_name=None):
        if not dataset_id: print("Error: Dataset ID required."); return None; actual_doc_name = doc_name or os.path.basename(file_path)
        try:
            with open(file_path, 'rb') as f: response_data = self._request("POST", f"/api/v1/datasets/{dataset_id}/documents", files={'file': (actual_doc_name, f, 'application/octet-stream')}, extra_headers={"Authorization": f"Bearer {self.api_key}"})
            if response_data and response_data.get("code") == 0 and isinstance(response_data.get("data"), list):
                print(f"Uploaded '{actual_doc_name}'. Response: {response_data['data']}"); return response_data["data"]
            else: print(f"Failed to upload '{actual_doc_name}'. Details: {response_data.get('details', response_data.get('error')) if response_data else 'No response'}"); return None
        except FileNotFoundError: print(f"Error: File not found: {file_path}"); return {"error": "File not found"}
        except Exception as e: print(f"Upload error {file_path}: {e}"); return {"error": str(e)}

    def retrieve_chunks(self, question_text, dataset_ids, top_k=3):
        if not dataset_ids: print("Error: Dataset IDs required."); return None
        response = self._request("POST", "/api/v1/retrieval", json_data={"question": question_text, "dataset_ids": dataset_ids if isinstance(dataset_ids, list) else [dataset_ids], "top_k": top_k, "highlight": True })
        if response and response.get("code") == 0 and response.get("data", {}).get("chunks"): return response["data"]["chunks"]
        else: print(f"Error retrieving chunks: {response.get('details', response.get('error')) if response else 'API call failed'}"); return None

# --- Keyword Extraction (LLM-based) ---
def extract_keywords(client, question_text, top_k=5):
    if not client: print("错误: RAGFlow客户端未提供，无法使用LLM提取关键词。"); return []
    if not question_text: print("错误: 问题文本为空，无法提取关键词。"); return []
    print(f"\n--- 开始使用LLM提取关键词 (问题: '{question_text[:50]}...') ---")
    chat_id_for_keywords = None
    if client.kb1_id:
        chat_id_for_keywords = client.get_chat_assistant_for_dataset(client.kb1_id, assistant_name_prefix="KeywordExtractor_KB1_")
    if not chat_id_for_keywords and client.kb2_id:
        chat_id_for_keywords = client.get_chat_assistant_for_dataset(client.kb2_id, assistant_name_prefix="KeywordExtractor_KB2_")
    if not chat_id_for_keywords: print("错误: 未能获取用于关键词提取的Chat Assistant ID。返回空列表。"); return []
    prompt = (f"你是一个关键词提取助手。请从以下用户问题中提取不超过{top_k}个核心关键词。这些关键词将用于后续的知识库检索。"
              f"请确保提取的关键词简洁明了，能准确反映问题的主要意图。\n"
              f"返回格式要求：请直接返回提取的关键词，并用单个英文逗号 (,) 分隔。不要添加任何序号、解释或其他无关字符。\n"
              f"用户问题：'{question_text}'")
    print(f"  使用Chat Assistant (ID: {chat_id_for_keywords}) 和prompt提取关键词..."); # Prompt itself is too long for concise log
    response_data = client.chat_with_assistant(chat_id_for_keywords, prompt)
    if response_data and response_data.get("answer"):
        llm_answer = response_data["answer"].strip(); print(f"  LLM返回的关键词字符串: '{llm_answer}'")
        if ":" in llm_answer: llm_answer = llm_answer.split(":", 1)[-1].strip()
        if "：" in llm_answer: llm_answer = llm_answer.split("：", 1)[-1].strip()
        keywords = [kw.strip().replace("\"", "").replace("'", "").replace("“", "").replace("”", "") for kw in llm_answer.split(',') if kw.strip()]
        cleaned_keywords = [kw for kw in keywords if kw]
        if cleaned_keywords: print(f"  成功提取并解析的关键词: {cleaned_keywords}"); return cleaned_keywords[:top_k]
        else: print("  LLM返回的内容无法解析出有效关键词。"); return []
    else: print("  LLM未能成功提取关键词 (API调用可能失败或未返回答案)。"); return []

# --- Knowledge Base Search & Analysis Logic ---
def is_kb1_document_format(filename):
    if not filename or not filename.endswith(".docx"): return None
    match = re.match(r"^(?P<date>\d{8})发布_(?P<jira_key>[A-Z0-9-]+)_(?P<doc_name_part>.*)\.docx$", filename)
    return match # Return match object or None

def search_knowledge_base_1(client, keywords, max_iterations=3):
    if not client or not client.kb1_id: print("Error: KB1 ID not initialized."); return []
    if not keywords: print("KB1 Search: No keywords provided, skipping search."); return []
    print(f"\n--- Searching KB1 ({client.kb1_name}, ID: {client.kb1_id}) ---"); relevant_files_kb1 = []; processed_doc_names = set(); current_keywords = list(keywords)
    for iteration in range(max_iterations):
        if not current_keywords: print(f"Iter {iteration+1}: No keywords."); break
        query_string = " ".join(current_keywords); print(f"Iter {iteration+1}: Querying with '{query_string}'...")
        chunks = client.retrieve_chunks(query_string, [client.kb1_id], top_k=15)
        if chunks is not None:
            found_count = 0
            for chunk in chunks:
                doc_name = chunk.get("document_name")
                if doc_name and doc_name not in processed_doc_names :
                    match_info = is_kb1_document_format(doc_name) # Call is_kb1_document_format
                    if match_info: # Check if match_info is not None
                        processed_doc_names.add(doc_name); print(f"  Found valid doc: {doc_name}")
                        relevant_files_kb1.append({"filename": doc_name, "date": match_info.group('date'), "jira_key": match_info.group('jira_key'), "doc_name_part": match_info.group('doc_name_part')}); found_count += 1
            print(f"  Added {found_count} new valid docs this iteration.")
        else: print("  API call failed or returned no chunks for KB1 search.")
        if len(relevant_files_kb1) >= 5: print("Found enough files, stopping early."); break
        if iteration < max_iterations - 1:
            if len(current_keywords) > 3: current_keywords = current_keywords[:-1]
            elif len(current_keywords) > 1: current_keywords = current_keywords[:1]
            else: print("Keywords exhausted."); break
        else: print("Max iterations reached for KB1 search.")
    print(f"--- KB1 search done. Found {len(relevant_files_kb1)} valid files. ---"); return relevant_files_kb1

def analyze_kb1_documents(client, user_question, kb1_file_infos):
    if not client or not client.kb1_id: print("Error: KB1 ID not initialized."); return []
    if not kb1_file_infos: print("No KB1 files to analyze."); return []
    print(f"\n--- Analyzing {len(kb1_file_infos)} docs from KB1 ---"); kb1_analysis_results = []
    for file_info in kb1_file_infos:
        doc_name = file_info["filename"]; jira_key = file_info["jira_key"]; print(f"  Analyzing doc: {doc_name}")
        specific_query = f"{user_question} about {jira_key} (document: {doc_name})"
        chunks = client.retrieve_chunks(specific_query, [client.kb1_id], top_k=3)
        if chunks:
            doc_snippets_count = 0
            for chunk in (c for c in chunks if c.get("document_name") == doc_name):
                kb1_analysis_results.append({"source_document": doc_name, "jira_key": jira_key, "query_used": specific_query, "retrieved_snippet": chunk.get("content"), "highlight": chunk.get("highlighted_content"), "relevance_score": chunk.get("similarity"), "chunk_id": chunk.get("id", "N/A")})
                doc_snippets_count+=1
            print(f"    Extracted {doc_snippets_count} relevant snippets from {doc_name}.")
        else: print(f"    No snippets retrieved or API error for {doc_name} in KB1 deep analysis.")
    print(f"--- KB1 deep analysis done. Collected {len(kb1_analysis_results)} snippets. ---"); return kb1_analysis_results

def is_kb2_document_format(filename):
    if not filename or not filename.endswith(".md"): return None
    return re.match(r"^(?P<date>\d{8})(?P<title>.*)\.md$", filename) # Return match object

def analyze_knowledge_base_2(client, keywords, user_question):
    if not client or not client.kb2_id: print("Error: KB2 ID not initialized."); return []
    if not keywords and not user_question: print("Error: No keywords or question for KB2 query."); return []
    query_string = " ".join(keywords) if keywords else user_question
    print(f"\n--- Analyzing KB2 ({client.kb2_name}, ID: {client.kb2_id}) ---")
    print(f"  Querying KB2 with: '{query_string}'..."); chunks = client.retrieve_chunks(query_string, [client.kb2_id], top_k=10)
    kb2_analysis_results = []
    if chunks:
        for chunk in chunks:
            doc_name = chunk.get("document_name")
            if doc_name :
                match_info = is_kb2_document_format(doc_name) # Call is_kb2_document_format
                if match_info: # Check if match_info is not None
                    print(f"    Found snippet from valid KB2 doc: {doc_name}")
                    kb2_analysis_results.append({"source_document_kb2": doc_name, "date": match_info.group('date'), "title": match_info.group('title'), "retrieved_snippet": chunk.get("content"), "highlight": chunk.get("highlighted_content"), "relevance_score": chunk.get("similarity"), "chunk_id": chunk.get("id", "N/A")})
    else: print(f"  No snippets retrieved or API error for KB2 query.")
    print(f"--- KB2 analysis done. Collected {len(kb2_analysis_results)} snippets. ---"); return kb2_analysis_results

# --- Report Generation and Upload ---
def generate_llm_title(client, content_summary, user_question):
    if not client: print("Error: RAGFlow client not init for LLM title."); return f"关于“{user_question[:20]}...”的调研报告"
    chat_id_for_llm = None
    if client.kb2_id: chat_id_for_llm = client.get_chat_assistant_for_dataset(client.kb2_id, assistant_name_prefix="TitleGen_KB2_")
    if not chat_id_for_llm and client.kb1_id: chat_id_for_llm = client.get_chat_assistant_for_dataset(client.kb1_id, assistant_name_prefix="TitleGen_KB1_")
    if not chat_id_for_llm: print("Warn: No Chat Assistant ID for title gen. Using default."); return f"关于“{user_question[:20]}...”的调研报告"
    prompt = f"请为以下内容生成简洁标题（不超过30字）针对问题：'{user_question}'。内容摘要：'{content_summary[:1000]}...'"
    print(f"  Requesting title from Chat Assistant (ID: {chat_id_for_llm})..."); response = client.chat_with_assistant(chat_id_for_llm, prompt)
    if response and response.get("answer"): generated_title = response["answer"].strip().replace("\n", " ").replace("#", ""); print(f"  LLM Title: {generated_title}"); return generated_title
    else: print("  LLM failed to gen title. Using default."); return f"关于“{user_question[:20]}...”的调研报告"

def format_analysis_results_to_markdown(kb1_results, kb2_results, user_question):
    md_content = [f"## 针对用户问题：“{user_question}” 的调研分析\n"]; summary_parts = []
    if kb1_results:
        md_content.append("### 一、相关需求分析文档总结 (知识库1)\n")
        for i, res in enumerate(kb1_results):
            md_content.append(f"**{i+1}. 文档:** `{res.get('source_document', 'N/A')}` (JIRA: `{res.get('jira_key', 'N/A')}`)\n   - **相关片段:**\n     ```text\n     {(res.get('highlight') or res.get('retrieved_snippet', '')).strip()}\n     ```\n")
            summary_parts.append(f"KB1 Doc {res.get('jira_key', '')}: {res.get('retrieved_snippet', '')[:100]}...")
    else: md_content.append("### 一、相关需求分析文档总结 (知识库1)\n\n未在知识库1找到相关文档片段。\n")
    if kb2_results:
        md_content.append("### 二、相关历史问题总结 (知识库2)\n")
        for i, res in enumerate(kb2_results):
            md_content.append(f"**{i+1}. 历史总结:** `{res.get('source_document_kb2', 'N/A')}` (日期: `{res.get('date', 'N/A')}`, 标题: `{res.get('title', 'N/A')}`)\n   - **相关结论:**\n     ```text\n     {(res.get('highlight') or res.get('retrieved_snippet', '')).strip()}\n     ```\n")
            summary_parts.append(f"KB2 Doc {res.get('title', '')}: {res.get('retrieved_snippet', '')[:100]}...")
    else: md_content.append("### 二、相关历史问题总结 (知识库2)\n\n未在知识库2找到相关历史问题总结。\n")
    return "\n".join(md_content), "\n".join(summary_parts)

def generate_and_upload_report(client, user_question, kb1_results, kb2_results):
    print("\n--- Generating and Uploading Report ---")
    report_body_md, summary = format_analysis_results_to_markdown(kb1_results, kb2_results, user_question)
    report_title = generate_llm_title(client, summary, user_question)
    report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    final_report_md = f"# {report_title}\n\n**调研时间:** {report_time}\n**原始问题:** {user_question}\n\n---\n\n{report_body_md}"
    print("\n--- Generated Report ---\n" + final_report_md + "\n--- End of Report ---\n")
    upload_status = {"success": False, "message": "Upload not attempted.", "filename": None}
    if not client or not client.kb2_id: upload_status["message"] = "Error: KB2 ID not init for upload."; print(upload_status["message"]); return final_report_md, upload_status
    safe_title = re.sub(r'[^\w一-龥().-]', '_', report_title)[:50]; report_filename = f"{datetime.now().strftime('%Y%m%d')}_{safe_title}.md" # Allow parentheses and dots in filename
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".md", encoding='utf-8') as tmp_file:
            tmp_file.write(final_report_md); temp_file_path = tmp_file.name
        print(f"  Report saved to temp file: {temp_file_path}. Uploading as '{report_filename}' to KB2 (ID: {client.kb2_id})...")
        upload_response = client.upload_document(client.kb2_id, temp_file_path, doc_name=report_filename)
        if upload_response and isinstance(upload_response, list) and upload_response[0].get("id"):
            upload_status.update({"success": True, "message": f"Report '{report_filename}' uploaded to KB2.", "filename": report_filename}); print(f"  {upload_status['message']}")
        else: upload_status["message"] = f"Upload failed. API Response: {upload_response}"; print(f"  {upload_status['message']}")
    except Exception as e: upload_status["message"] = f"Exception during temp file creation or upload: {e}"; print(f"  {upload_status['message']}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path): os.remove(temp_file_path);
    return final_report_md, upload_status

# --- Client Initialization and Main Logic ---
def initialize_ragflow_client():
    global RAGFLOW_API_KEY, RAGFLOW_API_BASE; temp_api_key = os.getenv("RAGFLOW_API_KEY_INPUT", RAGFLOW_API_KEY); temp_api_base = os.getenv("RAGFLOW_API_BASE_INPUT", RAGFLOW_API_BASE)
    if temp_api_key == "YOUR_RAGFLOW_API_KEY" or not temp_api_key: print("Warn: RAGFlow API Key not set, using placeholder.")
    if "YOUR_RAGFLOW_API_HOST" in temp_api_base or not temp_api_base: print("Warn: RAGFlow API Base URL not set or is placeholder.")
    RAGFLOW_API_KEY = temp_api_key; RAGFLOW_API_BASE = temp_api_base
    print(f"Init RAGFlow HTTP Client: Base={RAGFLOW_API_BASE}, Key=...{RAGFLOW_API_KEY[-4:] if len(RAGFLOW_API_KEY) > 4 else '****'}")
    return RAGFlowHttpApiClient(api_key=RAGFLOW_API_KEY, api_base_url=RAGFLOW_API_BASE)

def main():
    print("Agent starting..."); ragflow_client = initialize_ragflow_client()
    if not ragflow_client: print("Failed to init RAGFlow client, exiting."); return
    print("RAGFlow client created. Initializing dataset IDs...");
    ragflow_client.initialize_dataset_ids() # Call it regardless, it prints errors if fails
    print("RAGFlow client ready.")
    test_question = "请问关于投保时生日输入错误的规则，以及如何处理JK005-54748这个需求？"
    print(f"\nTest question: {test_question}")

    keywords = []
    if ragflow_client: # Check if client is valid for keyword extraction
        keywords = extract_keywords(ragflow_client, test_question, top_k=5)
        print(f"LLM提取的关键词: {keywords}")
    else:
        print("RAGFlow client invalid, skipping LLM keyword extraction.")

    retrieved_kb1_files = []; kb1_deep_analysis_results = []; kb2_analysis_results = []
    if ragflow_client and ragflow_client.kb1_id:
        if keywords: retrieved_kb1_files = search_knowledge_base_1(ragflow_client, keywords, max_iterations=3)
        else: print("\nSkipping KB1 search: No keywords from LLM.")
        if retrieved_kb1_files: kb1_deep_analysis_results = analyze_kb1_documents(ragflow_client, test_question, retrieved_kb1_files)
        elif keywords : print("\nNo files retrieved from KB1 search, skipping deep analysis of KB1.") # Only print if search was attempted
    else: print("\nSkipping KB1 search & analysis: Client or KB1 ID invalid.")

    if ragflow_client and ragflow_client.kb2_id:
        if keywords: kb2_analysis_results = analyze_knowledge_base_2(ragflow_client, keywords, test_question)
        else: print("\nSkipping KB2 analysis: No keywords from LLM.")
    else: print("\nSkipping KB2 analysis: Client or KB2 ID invalid.")

    if ragflow_client:
        if not kb1_deep_analysis_results and not kb2_analysis_results: print("\nNo relevant info from KBs (or keywords failed), not generating report.")
        else:
            final_report, upload_info = generate_and_upload_report(ragflow_client, test_question, kb1_deep_analysis_results, kb2_analysis_results)
            print(f"\nReport Upload Status: {upload_info['message']}");
            if upload_info["success"]: print(f"Report uploaded as '{upload_info['filename']}'.")
    else: print("\nSkipping report generation: RAGFlow client invalid.")
    print("\nAgent execution finished.")

if __name__ == "__main__":
    try:
        import jieba # Moved import here
        jieba.lcut("Init jieba")
        print("Jieba initialized (available for potential fallback or other uses).")
    except ImportError: print(f"Jieba library not found.")
    except Exception as e: print(f"Jieba init error: {e}.")
    main()

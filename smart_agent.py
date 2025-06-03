import os
import jieba
import jieba.analyse
import re
import requests
import json
from datetime import datetime # New
import tempfile # New
# from ragflow import RAGFlow

# --- CHINESE_STOP_WORDS 和 extract_keywords 函数 ---
CHINESE_STOP_WORDS = ["的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "而", "及", "与", "或", "个", "也", "为", "之", "此", "但", "则", "其", "更", "以", "所", "从", "自", "使", "向", "例如", "比如", "如何", "什么", "怎么", "为何", "哪些", "请问", "关于", "对于"]

def extract_keywords(question_text, top_k=5):
    if not question_text: return []
    keywords = jieba.analyse.extract_tags(question_text, topK=top_k, withWeight=False, allowPOS=('n', 'nr', 'ns', 'nt', 'nz', 'v', 'vd', 'vn', 'eng'))
    final_keywords = [kw for kw in keywords if kw.lower() not in CHINESE_STOP_WORDS and (len(kw) > 1 or re.match("^[a-zA-Z]+$", kw))]
    if not final_keywords:
        keywords_fallback = jieba.analyse.extract_tags(question_text, topK=top_k, withWeight=False)
        final_keywords = [kw for kw in keywords_fallback if kw.lower() not in CHINESE_STOP_WORDS and (len(kw) > 1 or re.match("^[a-zA-Z]+$", kw))]
    return final_keywords

# --- RAGFlow API 配置 ---
RAGFLOW_API_KEY = "YOUR_RAGFLOW_API_KEY"
RAGFLOW_API_BASE = "http://YOUR_RAGFLOW_API_HOST:8000"
KB1_NAME = "需求分析文档库"
KB2_NAME = "既往问题分析总结库"

# --- RAGFlow HTTP API Client ---
class RAGFlowHttpApiClient:
    def __init__(self, api_key, api_base_url, kb1_name=KB1_NAME, kb2_name=KB2_NAME): # Default args for kb_names
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

    def _get_dataset_id_by_name(self, dataset_name): # Simplified error check
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
        return self.kb1_id is not None and self.kb2_id is not None

    def get_chat_assistant_for_dataset(self, dataset_id, assistant_name_prefix="AutoGen_"):
        if not dataset_id: print("Error: Dataset ID required for chat assistant."); return None
        assistant_name = f"{assistant_name_prefix}{dataset_id}"
        response = self._request("GET", "/api/v1/chats", params={"name": assistant_name, "page_size": 1})
        if response and response.get("code") == 0 and response.get("data", {}).get("items"):
            for assistant in response["data"]["items"]: # Check if list is not empty
                if assistant.get("name") == assistant_name and dataset_id in assistant.get("dataset_ids", []):
                    print(f"Found assistant '{assistant_name}' ID: {assistant['id']}"); return assistant["id"]
        print(f"Assistant '{assistant_name}' not found. Creating..."); default_llm_model = os.getenv("RAGFLOW_DEFAULT_LLM_MODEL", "qwen-plus@Tongyi-Qianwen")
        create_payload = {"name": assistant_name, "dataset_ids": [dataset_id], "llm": {"model_name": default_llm_model, "temperature":0.1, "top_p":0.3}, "prompt": {"similarity_threshold":0.2,"top_n":3,"show_quote":True}} # Added more default values
        create_response = self._request("POST", "/api/v1/chats", json_data=create_payload)
        if create_response and create_response.get("code") == 0 and create_response.get("data", {}).get("id"):
            new_assistant_id = create_response["data"]["id"]; print(f"Created assistant '{assistant_name}' ID: {new_assistant_id}"); return new_assistant_id
        else: print(f"Error creating assistant '{assistant_name}': {create_response}"); return None

    def chat_with_assistant(self, chat_assistant_id, question, session_id=None, stream=False):
        if not chat_assistant_id: print("Error: Chat Assistant ID required."); return {"error": "Chat Assistant ID required"}
        payload = {"question": question, "stream": stream};
        if session_id: payload["session_id"] = session_id
        response = self._request("POST", f"/api/v1/chats/{chat_assistant_id}/completions", json_data=payload)
        if response and response.get("code") == 0 and response.get("data"): return response["data"]
        else: error_detail = response.get('details', response.get('error')) if response else "API call failed"; print(f"Error during chat: {error_detail}"); return {"error": "Chat failed", "details": error_detail}

    def upload_document(self, dataset_id, file_path, doc_name=None):
        if not dataset_id: print("Error: Dataset ID required."); return None; actual_doc_name = doc_name or os.path.basename(file_path)
        try:
            with open(file_path, 'rb') as f: response_data = self._request("POST", f"/api/v1/datasets/{dataset_id}/documents", files={'file': (actual_doc_name, f, 'application/octet-stream')}, extra_headers={"Authorization": f"Bearer {self.api_key}"}) # Ensure correct content type for file
            if response_data and response_data.get("code") == 0 and isinstance(response_data.get("data"), list): # Successful upload returns a list
                print(f"Uploaded '{actual_doc_name}'. Response: {response_data['data']}"); return response_data["data"]
            else: print(f"Failed to upload '{actual_doc_name}'. Details: {response_data.get('details', response_data.get('error')) if response_data else 'No response'}"); return None
        except FileNotFoundError: print(f"Error: File not found: {file_path}"); return {"error": "File not found"}
        except Exception as e: print(f"Upload error {file_path}: {e}"); return {"error": str(e)}

    def retrieve_chunks(self, question_text, dataset_ids, top_k=3):
        if not dataset_ids: print("Error: Dataset IDs required."); return None
        response = self._request("POST", "/api/v1/retrieval", json_data={"question": question_text, "dataset_ids": dataset_ids if isinstance(dataset_ids, list) else [dataset_ids], "top_k": top_k, "highlight": True })
        if response and response.get("code") == 0 and response.get("data", {}).get("chunks"): return response["data"]["chunks"]
        else: print(f"Error retrieving chunks: {response.get('details', response.get('error')) if response else 'API call failed'}"); return None

# --- Knowledge Base Search & Analysis Logic ---
def is_kb1_document_format(filename):
    if not filename or not filename.endswith(".docx"): return None
    return re.match(r"^(?P<date>\d{8})发布_(?P<jira_key>[A-Z0-9-]+)_(?P<doc_name_part>.*)\.docx$", filename)

def search_knowledge_base_1(client, keywords, max_iterations=3):
    if not client or not client.kb1_id: print("Error: KB1 ID not initialized."); return []
    print(f"\n--- Searching KB1 ({client.kb1_name}, ID: {client.kb1_id}) ---"); relevant_files_kb1 = []; processed_doc_names = set(); current_keywords = list(keywords)
    for iteration in range(max_iterations):
        if not current_keywords: print(f"Iter {iteration+1}: No keywords."); break
        query_string = " ".join(current_keywords); print(f"Iter {iteration+1}: Querying with '{query_string}'...")
        chunks = client.retrieve_chunks(query_string, [client.kb1_id], top_k=15)
        if chunks is not None:
            found_count = 0
            for chunk in chunks:
                doc_name = chunk.get("document_name")
                if doc_name and doc_name not in processed_doc_names and (match_info := is_kb1_document_format(doc_name)): # Python 3.8+ walrus operator
                    processed_doc_names.add(doc_name); print(f"  Found valid doc: {doc_name}")
                    relevant_files_kb1.append({"filename": doc_name, "date": match_info.group('date'), "jira_key": match_info.group('jira_key'), "doc_name_part": match_info.group('doc_name_part')}); found_count += 1 # Use .group()
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
    return re.match(r"^(?P<date>\d{8})(?P<title>.*)\.md$", filename)

def analyze_knowledge_base_2(client, keywords, user_question):
    if not client or not client.kb2_id: print("Error: KB2 ID not initialized."); return []
    print(f"\n--- Analyzing KB2 ({client.kb2_name}, ID: {client.kb2_id}) ---"); kb2_analysis_results = []
    query_string = " ".join(keywords) if keywords else user_question
    if not query_string: print("Error: No query for KB2."); return []
    print(f"  Querying KB2 with: '{query_string}'..."); chunks = client.retrieve_chunks(query_string, [client.kb2_id], top_k=10)
    if chunks:
        for chunk in chunks:
            doc_name = chunk.get("document_name")
            if doc_name and (match_info := is_kb2_document_format(doc_name)): # Python 3.8+ walrus operator
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
    safe_title = re.sub(r'[^\w一-龥.-]', '_', report_title)[:50]; report_filename = f"{datetime.now().strftime('%Y%m%d')}_{safe_title}.md" # Added underscore for readability
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".md", encoding='utf-8') as tmp_file:
            tmp_file.write(final_report_md); temp_file_path = tmp_file.name
        print(f"  Report saved to temp file: {temp_file_path}. Uploading as '{report_filename}' to KB2 (ID: {client.kb2_id})...")
        upload_response = client.upload_document(client.kb2_id, temp_file_path, doc_name=report_filename)
        if upload_response and isinstance(upload_response, list) and upload_response[0].get("id"): # Assuming success returns list with doc info
            upload_status.update({"success": True, "message": f"Report '{report_filename}' uploaded to KB2.", "filename": report_filename}); print(f"  {upload_status['message']}")
        else: upload_status["message"] = f"Upload failed. API Response: {upload_response}"; print(f"  {upload_status['message']}")
    except Exception as e: upload_status["message"] = f"Exception during temp file creation or upload: {e}"; print(f"  {upload_status['message']}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path): os.remove(temp_file_path); # print(f"  Temp file {temp_file_path} deleted.")
    return final_report_md, upload_status

# --- Client Initialization and Main Logic ---
def initialize_ragflow_client():
    global RAGFLOW_API_KEY, RAGFLOW_API_BASE; temp_api_key = os.getenv("RAGFLOW_API_KEY_INPUT", RAGFLOW_API_KEY); temp_api_base = os.getenv("RAGFLOW_API_BASE_INPUT", RAGFLOW_API_BASE)
    if temp_api_key == "YOUR_RAGFLOW_API_KEY" or not temp_api_key: print("Warn: RAGFlow API Key not set, using placeholder.")
    if "YOUR_RAGFLOW_API_HOST" in temp_api_base or not temp_api_base: print("Warn: RAGFlow API Base URL not set or is placeholder.")
    RAGFLOW_API_KEY = temp_api_key; RAGFLOW_API_BASE = temp_api_base
    print(f"Init RAGFlow HTTP Client: Base={RAGFLOW_API_BASE}, Key=...{RAGFLOW_API_KEY[-4:] if len(RAGFLOW_API_KEY) > 4 else '****'}")
    return RAGFlowHttpApiClient(api_key=RAGFLOW_API_KEY, api_base_url=RAGFLOW_API_BASE) # kb_names use class defaults

def main():
    print("Agent starting..."); ragflow_client = initialize_ragflow_client()
    if not ragflow_client: print("Failed to init RAGFlow client, exiting."); return
    print("RAGFlow client created. Initializing dataset IDs...");
    if not ragflow_client.initialize_dataset_ids(): print("Dataset ID initialization failed. API calls may fail or be skipped.") # Simplified this line
    else: print(f"Dataset IDs initialized.") # Simplified
    print("RAGFlow client ready.")
    test_question = "请问关于投保时生日输入错误的规则，以及如何处理JK005-54748这个需求？"
    print(f"\nTest question: {test_question}"); keywords = extract_keywords(test_question, top_k=5)
    print(f"Keywords: {keywords}"); retrieved_kb1_files = []; kb1_deep_analysis_results = []; kb2_analysis_results = [] # Init all result lists
    if ragflow_client and ragflow_client.kb1_id:
        retrieved_kb1_files = search_knowledge_base_1(ragflow_client, keywords, max_iterations=3) # max_iterations restored
        if retrieved_kb1_files: kb1_deep_analysis_results = analyze_kb1_documents(ragflow_client, test_question, retrieved_kb1_files)
        else: print("\nNo files retrieved from KB1 search, skipping deep analysis of KB1.")
    else: print("\nSkipping KB1 search & analysis: Client or KB1 ID invalid.")
    if ragflow_client and ragflow_client.kb2_id: kb2_analysis_results = analyze_knowledge_base_2(ragflow_client, keywords, test_question)
    else: print("\nSkipping KB2 analysis: Client or KB2 ID invalid.")
    if ragflow_client:
        if not kb1_deep_analysis_results and not kb2_analysis_results: print("\nNo relevant info from KBs, not generating report.")
        else:
            final_report, upload_info = generate_and_upload_report(ragflow_client, test_question, kb1_deep_analysis_results, kb2_analysis_results)
            print(f"\nReport Upload Status: {upload_info['message']}");
            if upload_info["success"]: print(f"Report uploaded as '{upload_info['filename']}'.")
    else: print("\nSkipping report generation: RAGFlow client invalid.")
    print("\nAgent execution finished.")

if __name__ == "__main__":
    try: jieba.lcut("Init jieba"); print("Jieba initialized.")
    except Exception as e: print(f"Jieba init failed: {e}.")
    main()

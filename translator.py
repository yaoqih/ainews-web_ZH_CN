import os
import re
import requests
import frontmatter
import yaml
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 配置区域 ---
UPSTREAM_REPO = "smol-ai/ainews-web-2025"
TARGET_DIR = "src/content/issues"  # <--- 修改了这里，之前是 markdownsrc/content/issues
OUTPUT_DIR = "docs"
API_KEY = os.environ.get("LLM_API_KEY",'sk-2ZSxkHHS3V8JaqlOCBrE7PipfVtkZWcKJsHXToKixCxKeXV9')
BASE_URL = os.environ.get("LLM_BASE_URL", "https://yunwu.ai/v1")
MAX_WORKERS = 32  # 并行线程数，根据你的 API Rate Limit 调整
MAX_CHUNK_CHARS = 5000  # 每段必须 < 5000 个字符
MAX_FILE_WORKERS = int(os.environ.get("MAX_FILE_WORKERS", "16"))  # 并行处理文章数量

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

_HAS_PYFM = hasattr(frontmatter, "loads") and hasattr(frontmatter, "dumps")
_HAS_SIMPLE_FM = hasattr(frontmatter, "Frontmatter") and hasattr(frontmatter.Frontmatter, "read")

class _SimplePost:
    def __init__(self, metadata, content):
        self.metadata = metadata or {}
        self.content = content or ""

    def get(self, key, default=None):
        return self.metadata.get(key, default)

def _load_frontmatter_from_string(text):
    if _HAS_PYFM:
        return frontmatter.loads(text)
    if _HAS_SIMPLE_FM:
        data = frontmatter.Frontmatter.read(text)
        return _SimplePost(data.get("attributes") or {}, data.get("body") or "")
    return _SimplePost({}, text)

def _load_frontmatter_from_file(path):
    if _HAS_PYFM and hasattr(frontmatter, "load"):
        return frontmatter.load(path)
    with open(path, "r", encoding="utf-8") as f:
        return _load_frontmatter_from_string(f.read())

def _dump_frontmatter(post):
    if _HAS_PYFM:
        return frontmatter.dumps(post)

    metadata = getattr(post, "metadata", {}) or {}
    content = getattr(post, "content", "") or ""

    if not metadata:
        return content

    fm = yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True).strip()
    body = content.lstrip('\n')
    if body:
        return f"---\n{fm}\n---\n\n{body}"
    return f"---\n{fm}\n---\n"

def get_upstream_files():
    """获取上游仓库的文件列表"""
    url = f"https://api.github.com/repos/{UPSTREAM_REPO}/contents/{TARGET_DIR}"
    print(f"Checking upstream: {url}")
    response = requests.get(url)
    if response.status_code == 200:
        return [f for f in response.json() if f['name'].endswith('.md')]
    print(f"Error checking upstream: {response.status_code}")
    return []

FENCE_PATTERN = re.compile(r'^\s*(```|~~~)')
HEADER_PATTERN = re.compile(r'^\s{0,3}(#{1,6})\s+')

def _split_by_headers(text, levels):
    """
    按指定标题级别切分 Markdown，忽略代码块内的标题。
    使用 splitlines(keepends=True) 以尽量保持原始换行。
    """
    if not text:
        return []

    levels_set = set(levels)
    lines = text.splitlines(keepends=True)
    chunks = []
    current_chunk = []
    in_code_block = False

    for line in lines:
        line_no_nl = line.rstrip('\r\n')
        if FENCE_PATTERN.match(line_no_nl):
            in_code_block = not in_code_block

        match = HEADER_PATTERN.match(line_no_nl)
        if not in_code_block and match:
            level = len(match.group(1))
            if level in levels_set:
                if current_chunk:
                    chunks.append("".join(current_chunk))
                current_chunk = [line]
                continue

        current_chunk.append(line)

    if current_chunk:
        chunks.append("".join(current_chunk))

    return chunks

def _split_by_line_predicate(text, should_split_after_line, respect_code_blocks=True):
    if not text:
        return []

    lines = text.splitlines(keepends=True)
    chunks = []
    current_chunk = []
    in_code_block = False

    for line in lines:
        line_no_nl = line.rstrip('\r\n')

        if respect_code_blocks and FENCE_PATTERN.match(line_no_nl):
            in_code_block = not in_code_block

        current_chunk.append(line)

        if (not respect_code_blocks or not in_code_block) and should_split_after_line(line_no_nl):
            chunks.append("".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append("".join(current_chunk))

    return chunks

def _split_by_blank_lines(text):
    return _split_by_line_predicate(text, lambda line: line.strip() == '', respect_code_blocks=True)

def _split_by_lines(text):
    # 最细粒度：按行切分（允许在代码块内切分）
    return _split_by_line_predicate(text, lambda _line: True, respect_code_blocks=False)

def _hard_split(text, max_chars):
    if not text:
        return []
    # 确保每段 < max_chars
    limit = max_chars - 1
    if limit <= 0:
        return [text]
    return [text[i:i + limit] for i in range(0, len(text), limit)]

def _merge_by_size(chunks, max_chars):
    merged = []
    current = ""

    for chunk in chunks:
        if not current:
            current = chunk
            continue

        if len(current) + len(chunk) < max_chars:
            current += chunk
        else:
            merged.append(current)
            current = chunk

    if current:
        merged.append(current)

    return merged

def split_markdown_chunks(text, max_chars=MAX_CHUNK_CHARS):
    """
    多层级切分策略（无 overlap）：
    1) 先按 H1-H4 切分
    2) 超长块再按 H5 -> H6 -> 空行 -> 行 -> 硬切分 逐级细化
    3) 最后顺序合并，尽量靠近 max_chars，且保证每块 < max_chars
    """
    if not text:
        return []

    def split_with_strategies(text, step):
        if len(text) < max_chars:
            return [text]

        strategies = [
            ("headers", [1, 2, 3, 4]),
            ("headers", [5]),
            ("headers", [6]),
            ("blank_lines", None),
            ("lines", None),
            ("hard", None)
        ]

        if step >= len(strategies):
            return _hard_split(text, max_chars)

        strategy, arg = strategies[step]

        if strategy == "headers":
            parts = _split_by_headers(text, arg)
        elif strategy == "blank_lines":
            parts = _split_by_blank_lines(text)
        elif strategy == "lines":
            parts = _split_by_lines(text)
        else:
            return _hard_split(text, max_chars)

        # 如果这一层无法切分，继续向下
        if len(parts) <= 1:
            return split_with_strategies(text, step + 1)

        refined = []
        for part in parts:
            if len(part) < max_chars:
                refined.append(part)
            else:
                refined.extend(split_with_strategies(part, step + 1))

        return refined

    raw_chunks = split_with_strategies(text, 0)
    # 最终顺序合并，尽量靠近 max_chars
    merged = _merge_by_size(raw_chunks, max_chars)

    # 兜底：防止任何块 >= max_chars
    final_chunks = []
    for chunk in merged:
        if len(chunk) < max_chars:
            final_chunks.append(chunk)
        else:
            final_chunks.extend(_hard_split(chunk, max_chars))

    return final_chunks

def translate_text_chunk(text, is_metadata=False):
    """
    调用 LLM 翻译单个文本块
    """
    if not text.strip():
        return ""

    # 针对不同内容调整 Prompt
    if is_metadata:
        system_prompt = "你是一个翻译助手。请将提供的文本翻译成中文，保留原意。"
    else:
        system_prompt = """
        你是一个专业的技术翻译专家。请将以下 Markdown 片段翻译成中文。
        1. **必须保留 Markdown 格式**：不要修改标题层级、链接、加粗、代码块。
        2. **专有名词保留英文**：如 LLM, Transformer, Agent, GitHub, CUDA 等。
        3. **代码块内容不要翻译**：保留代码块内的原始代码。
        4. **只返回翻译结果**：不要包含"这是翻译"，”参考的翻译如下“等废话。
        """

    try:
        response = client.chat.completions.create(
            model="gemini-3-flash-preview", # 建议使用 mini 或 deepseek-chat 以节省成本
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Translation failed: {e}")
        return text  # 失败则返回原文

def process_single_file(file_info):
    """
    处理单个文件的完整流程：下载 -> 解析 -> 并行翻译 -> 组装 -> 保存
    """
    filename = file_info['name']
    local_path = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(local_path):
        print(f"跳过已存在: {filename}")
        return

    print(f"开始处理: {filename} ...")
    
    # 1. 下载原始内容
    raw_content = requests.get(file_info['download_url']).text
    
    # 2. 解析 Frontmatter (元数据)
    post = _load_frontmatter_from_string(raw_content)
    
    # 3. 翻译 Metadata (串行，因为内容少)
    if 'title' in post.metadata:
        post.metadata['title'] = translate_text_chunk(post.metadata['title'], is_metadata=True)
    if 'description' in post.metadata:
        post.metadata['description'] = translate_text_chunk(post.metadata['description'], is_metadata=True)
    
    # 4. 切分正文 (Body)
    body_chunks = split_markdown_chunks(post.content, max_chars=MAX_CHUNK_CHARS)
    print(f"[{filename}] 切分为 {len(body_chunks)} 个片段，准备并行翻译...")

    # 5. 并行翻译正文
    translated_chunks = [None] * len(body_chunks) # 预分配位置保证顺序
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交任务，保留索引以便后续按顺序组装
        future_to_index = {
            executor.submit(translate_text_chunk, chunk): i 
            for i, chunk in enumerate(body_chunks)
        }
        
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                translated_chunks[idx] = future.result()
            except Exception as e:
                print(f"片段 {idx} 翻译出错: {e}")
                translated_chunks[idx] = body_chunks[idx] # 出错回退到原文

    # 6. 重新组装
    post.content = "\n\n".join(translated_chunks)
    
    # 7. 保存
    with open(local_path, 'w', encoding='utf-8') as f:
        f.write(_dump_frontmatter(post))
    
    print(f"完成并保存: {filename}")

def update_index():
    """生成简单的 index.md"""
    if not os.path.exists(OUTPUT_DIR): return
    
    files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.md') and f != 'index.md'], reverse=True)
    index_content = "# AI News 中文同步版\n\n> 自动同步自 smol-ai/ainews-web-2025，由 AI 并行翻译。\n\n"
    
    for f in files:
        try:
            post = _load_frontmatter_from_file(os.path.join(OUTPUT_DIR, f))
            title = post.get('title', f.replace('.md', ''))
            date = post.get('date', '')
            index_content += f"- [{title}](./{f}) *{date}*\n"
        except:
            index_content += f"- [{f}](./{f})\n"
            
    with open(os.path.join(OUTPUT_DIR, "index.md"), "w", encoding="utf-8") as f:
        f.write(index_content)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    upstream_files = get_upstream_files()
    
    # 文件级并行 + 文件内并行
    with ThreadPoolExecutor(max_workers=MAX_FILE_WORKERS) as executor:
        future_to_name = {
            executor.submit(process_single_file, file_info): file_info.get('name', 'unknown')
            for file_info in upstream_files
        }

        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                future.result()
            except Exception as e:
                print(f"文件处理失败: {name}: {e}")

    update_index()

if __name__ == "__main__":
    main()

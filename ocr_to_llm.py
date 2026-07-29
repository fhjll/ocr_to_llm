"""
PDF 凭证 → 字段提取 → 大模型业务分类 → 输出凭证信息

流程:
  PDF 读取（文本/OCR）→ 正则提取全部字段
  → 发送给大模型判断业务类型
  → 输出: 业务类型 + 原因 + 5项凭证信息

依赖: pip install pdfplumber pypdfium2 Pillow requests openai
       （扫描件还需 paddlepaddle==2.0.2 paddleocr==2.0.7）
"""

import json
import os
import re
import sys
import multiprocessing

# =========================
# PyInstaller 资源路径兼容
# =========================

def get_base_dir() -> str:
    """获取程序运行时的基础目录（兼容 PyInstaller 打包）"""
    if getattr(sys, 'frozen', False):
        # PyInstaller 打包后，使用 exe 所在目录
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def get_resource_path(relative_path: str) -> str:
    """获取资源文件的绝对路径（兼容 PyInstaller --onedir 打包）"""
    if getattr(sys, 'frozen', False):
        # --onedir 模式: 资源在 exe 同级目录
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, relative_path)


# 延迟导入 pyautogui，避免导入失败时 GUI 无法启动
_pyautogui = None

def get_pyautogui():
    """延迟加载 pyautogui，失败时给出明确提示"""
    global _pyautogui
    if _pyautogui is None:
        try:
            import pyautogui
            _pyautogui = pyautogui
        except ImportError as e:
            raise ImportError(
                f"pyautogui 导入失败: {e}\n"
                "请确认已安装: pip install pyautogui opencv-python"
            )
    return _pyautogui
# =========================
# 配置区（请先填写实际值）
# =========================


#官方 API ---
MODEL_ENABLED = True
API_KEY = "sk-gHrc98UswpFSGCLwjjHTSqlBYF9tt7lMHy0X3mWDN4yS3X9F"
MODEL_NAME = "Qwen3-30B-A3B-Instruct-2507"
API_BASE_URL = "http://11.32.1.214:3000/v1"

# --- OCR ---
OCR_ENABLED = False       # 扫描件设为 True
OCR_USE_GPU = False
OCR_LANG = "ch"

# =========================
# 外部配置（config.json）
# =========================

CONFIG_PATH = os.path.join(get_base_dir(), "config.json")

# 默认配置: config.json 不存在时自动写入该默认值, 用户可直接编辑 json 文件
DEFAULT_CONFIG = {
    "refund": {
        "地区代码": "278",
        "报文种类代码": {
            "大额": "050621",
            "小额": "050622"
        }
    },
    "external": {
        "国库代码": {
            "咸宁": "1715000000",
            "咸安": "1715020000",
            "崇阳": "1715130000",
            "通城": "1715140000",
            "通山": "1715150000",
            "嘉鱼": "1715160000",
            "默认": "1715170000"
        },
        "出票单位": {
            "咸宁": "44212000000",
            "咸安": "44212020000",
            "崇阳": "44212230000",
            "通城": "44212220000",
            "通山": "44212240000",
            "嘉鱼": "44212210000",
            "赤壁": "44212810000"
        }
    }
}


def load_config() -> dict:
    """读取 config.json; 不存在则用默认值生成一份, 方便部署后直接修改"""
    if os.path.isfile(CONFIG_PATH):
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_CONFIG, f, ensure_ascii=False, indent=2)
    return DEFAULT_CONFIG


CONFIG = load_config()


def lookup_region_code(mapping: dict, account_name: str):
    """按地区关键字匹配账户名称, 命中返回对应代码; 未命中返回"默认"键(若有)"""
    for keyword, code in mapping.items():
        if keyword != "默认" and keyword in account_name:
            return code
    return mapping.get("默认")

# =========================
# 1. PDF 读取
# =========================

def extract_text_from_pdf(pdf_path: str) -> str:
    import pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        texts = [pg.extract_text().strip() for pg in pdf.pages if pg.extract_text()]
    return "\n".join(texts)


def ocr_pdf_to_text(pdf_path: str) -> str:
    import pypdfium2 as pdfium
    from PIL import Image
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=False, lang=OCR_LANG,
                    use_gpu=OCR_USE_GPU, show_log=False)
    doc = pdfium.PdfDocument(pdf_path)
    all_text = []
    for i in range(len(doc)):
        page = doc[i]
        bitmap = page.render(scale=300 / 72)
        pil_img = Image.frombytes("RGB", (bitmap.width, bitmap.height),
                                  bitmap.format("RGBX"))
        result = ocr.ocr(pil_img, cls=False)
        lines = []
        if result and result[0]:
            for line in result[0]:
                lines.append(line[1][0])
        all_text.extend(lines)
        print(f"  OCR 第 {i+1}/{len(doc)} 页，识别 {len(lines)} 行")
    doc.close()
    return "\n".join(all_text)


# =========================
# 2. 字段提取
# =========================

CLASSIFY_FIELDS_PATTERNS = {
    "付款人账户名称": [
        r"付款人名称\s*[:：]\s*(\S+)",
        r"付款人[：:]?\s*(\S+)",
    ],
    "收款人账户名称": [
        r"收款人名称\s*[:：]\s*(\S+)",
        r"收款人[：:]?\s*(\S+)",
    ],
    "附言": [
        r"附言\s*[:：]\s*(\S+)",
        r"摘要\s*[:：]\s*(\S+)",
    ],
}

VOUCHER_PATTERNS = {
    "受理日期": r"受理日期\s*[:：]\s*(\d{4})年(\d{1,2})月(\d{1,2})日",
    "交易流水号": r"交易流水号\s*[:：]?\s*(\d{15,})",
    "金额": r"￥\s*([\d,]+\.\d{2})",
    "报文种类": r"报文种类\s*[:：]\s*(H\w+|B\w+)",
}


def extract_fields(text: str) -> dict:
    result = {}
    for field_name, patterns in CLASSIFY_FIELDS_PATTERNS.items():
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                result[field_name] = m.group(1).strip()
                break
        if field_name not in result:
            result[field_name] = ""
    for field_name, pat in VOUCHER_PATTERNS.items():
        m = re.search(pat, text)
        if m:
            if field_name == "受理日期":
                y, mo, d = m.group(1), m.group(2), m.group(3)
                result[field_name] = f"{y}{int(mo):02d}{int(d):02d}"
            else:
                result[field_name] = m.group(1).strip()
        else:
            result[field_name] = ""
    if result.get("交易流水号"):
        result["流水号后三位"] = str(result["交易流水号"])[-3:]
    else:
        result["流水号后三位"] = ""
    msg_type = result.get("报文种类", "")
    if msg_type:
        result["报文种类_中文"] = "大额" if msg_type.startswith("H") else "小额" if msg_type.startswith("B") else msg_type
    else:
        result["报文种类_中文"] = ""
    return result


# =========================
# 3. 大模型业务分类
# =========================

SYSTEM_PROMPT = """你是一名财政业务复核员。业务类型已按关键字规则自动判定，你的职责是复核该判定是否合理。

规则原文：
- 规则1：「收款人账户名称」以「地方财政库款」结尾，或为「xxx财政局xxx资金专户」格式，否则为 illegal
- 规则2：「付款人账户名称」含「清算」「授权支付」「零余额」「待转」任意一词（条件A），且「附言」含「退款」「退回」「清算」「授权支付」任意一词（条件B），两者同时满足为 settlement_refund，否则为 external_allocation

复核要求：
- 逐条核对规则1、条件A、条件B，判断规则判定结果是否正确
- 注意 OCR 可能有个别错别字，结合语义理解账户名称和附言
- 判定正确 → agree = true；判定错误 → agree = false，并在 reason 说明你认为的正确类型和依据

输出严格遵循 JSON 格式（不要包含任何其他内容）：
{"agree": true | false, "reason": "..."}"""


def build_prompt(fields: dict, rule_result: dict) -> str:
    lines = [
        f"付款人账户名称：{fields.get('付款人账户名称', '（未识别）')}",
        f"收款人账户名称：{fields.get('收款人账户名称', '（未识别）')}",
        f"附言：{fields.get('附言', '（未识别）')}",
        f"规则判定结果：{rule_result.get('business_type')}（{rule_result.get('reason', '')}）",
    ]
    return "\n".join(lines)


def call_deepseek(prompt_text: str) -> str:
    import httpx
    from openai import OpenAI
    
    http_client = httpx.Client(proxy=None)
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL, http_client=http_client)
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text}
        ],
        response_format={"type": "json_object"},
        temperature=0
    )
    return resp.choices[0].message.content



def classify_by_rules(fields: dict) -> dict:
    """按关键字规则确定性分类，返回 {"business_type": ..., "reason": ...}"""
    payee = fields.get("收款人账户名称", "")
    payer = fields.get("付款人账户名称", "")
    remark = fields.get("附言", "")

    # 规则 1：收款人必须是财政库款/财政专户
    if not (payee.endswith("地方财政库款") or ("财政局" in payee and "资金专户" in payee)):
        return {"business_type": "illegal",
                "reason": f"收款人账户不符合财政库款/专户格式：{payee}"}

    # 规则 2：条件A、条件B 同时满足才算清算退款
    cond_a = any(k in payer for k in ("清算", "授权支付", "零余额", "待转"))
    cond_b = any(k in remark for k in ("退款", "退回", "清算", "授权支付"))
    if cond_a and cond_b:
        return {"business_type": "settlement_refund",
                "reason": "付款人含清算类关键词且附言含退款类关键词"}
    return {"business_type": "external_allocation",
            "reason": f"收款人账户：{payee}，付款人账户：{payer}，附言：{remark}"}


def review_by_llm(fields: dict, rule_result: dict) -> dict:
    """AI 复核规则分类结果，返回 {"agree": bool, "reason": ...}；复核不可用时视为通过"""
    if not MODEL_ENABLED:
        return {"agree": True, "reason": "未启用大模型，跳过复核"}
    try:
        result_text = call_deepseek(build_prompt(fields, rule_result))
    except Exception as e:
        return {"agree": True, "reason": f"复核调用失败，按规则结果执行: {e}"}
    try:
        result = json.loads(result_text)
    except json.JSONDecodeError:
        m = re.search(r"\{[^{}]*\}", result_text)
        if m:
            try:
                result = json.loads(m.group())
            except json.JSONDecodeError:
                result = None
        if result is None:
            return {"agree": True, "reason": f"复核返回非JSON，按规则结果执行: {result_text[:200]}"}
    return {"agree": bool(result.get("agree", True)), "reason": result.get("reason", "")}


# =========================
# 4. 输出格式化
# =========================

def format_output(biz_type: str, reason: str, fields: dict) -> str:
    lines = [
        f"业务类型: {biz_type}",
        f"原因: {reason}",
        "",
        "凭证信息:",
        f"  1. 凭证受理日期: {fields.get('受理日期', '未识别')}",
        f"  2. 交易流水号后三位: {fields.get('流水号后三位', '未识别')}",
        f"  3. 完整的交易流水号: {fields.get('交易流水号', '未识别')}",
        f"  4. 交易金额: {fields.get('金额', '未识别')}",
        f"  5. 报文种类: {fields.get('报文种类_中文', fields.get('报文种类', '未识别'))}",
    ]
    return "\n".join(lines)



def llm_ocr(pdf_path):
    classify_result=[]
    # if len(sys.argv) < 2:
        # print("用法: python ocr_to_llm.py <PDF文件路径>")
        # return

    # pdf_path = sys.argv[1]
    if not os.path.isfile(pdf_path):
        print(f"文件不存在: {pdf_path}")
        return
        
    print("[1/4] 读取 PDF ...")
    if OCR_ENABLED:
        raw_text = ocr_pdf_to_text(pdf_path)
    else:
        raw_text = extract_text_from_pdf(pdf_path)
    if not raw_text.strip():
        print("  ⚠ 文本提取为空，尝试 OCR ...")
        raw_text = ocr_pdf_to_text(pdf_path)
    text = raw_text.split('中国人民银行')[1:]
    for data in text:
        
        fields = extract_fields(data)
       
        rule_result = classify_by_rules(fields)
        review = review_by_llm(fields, rule_result)
        biz_type = rule_result["business_type"]
        if not review["agree"]:
            biz_type = "needs_review"
            print(f"  ⚠ AI 复核不通过，转人工: {review['reason']}")
        
        classify_result.append({"biz": biz_type, "fields": fields,
                                "reason": rule_result.get("reason", ""),
                                "review": review.get("reason", "")})
    print(classify_result)
    return classify_result
# =========================
# 5. 主流程
# =========================
## 清算退
def refund(fields):
## 清算退款
    a = get_pyautogui()
    a.sleep(1)
    find_pic(get_resource_path("imgs/tk1.png"))#1.png
    
    a.sleep(1)
    a.press('down',8)
    
    a.sleep(1)
    a.press('enter',2)
    a.sleep(1)
    
    find_pic(get_resource_path("imgs/tk2.png"))#2.png
    a.sleep(1)
        
    a.press("tab",10)
    a.sleep(0.5)
    a.write( fields["金额"])
    a.press("enter")
    a.sleep(1)
    a.press("enter")
    a.sleep(1)

    find_pic(get_resource_path("imgs/tk4.png"), confidence=0.8)#4.png
    a.sleep(1)
    find_pic(get_resource_path("imgs/tk3.png"))#3.png
    a.sleep(0.5)	
    a.press("enter")
    a.sleep(0.5)
    a.press("enter")
    a.sleep(0.5)
    a.press("tab")
    a.sleep(0.5)
            
    find_pic(get_resource_path("imgs/tk4.png"), confidence=0.8)#4.png
    a.sleep(0.5)	
    a.press("enter")
    
    a.sleep(1.5)	
    a.press("enter")
    a.sleep(0.5)
    a.write(fields["流水号后三位"])
    a.press("enter")
    a.sleep(0.5)
    a.press("enter")
    a.sleep(0.5)
    a.write(fields["金额"])
    a.sleep(0.5)
    a.press("enter")
    a.write(fields["受理日期"])
    a.press("enter")
    a.sleep(0.5)
    a.write(CONFIG["refund"]["地区代码"])
    a.press("enter")
    a.sleep(0.5)
    报文代码 = CONFIG["refund"]["报文种类代码"].get(
        fields["报文种类_中文"], CONFIG["refund"]["报文种类代码"]["小额"])
    a.write(报文代码, interval=0.1)
    a.press("enter")
    a.sleep(0.5)
    a.press("enter")
    a.sleep(0.5)

##系统外调拨
def external(fields):
    ##系统外调拨
    a = get_pyautogui()

    a.sleep(1)
    find_pic(get_resource_path("imgs/sr1.png"))#5.png
    a.sleep(0.5)
    a.press('down',2)
    a.sleep(1)
    a.press('enter',2)
    a.sleep(1)
    find_pic(get_resource_path("imgs/sr2.png"))	#6.png
    a.sleep(0.5)
    
    a.sleep(1)
    a.press("tab")
    a.sleep(1)
    a.write(fields["流水号后三位"])
    a.press("enter")
    a.sleep(1)
    a.write(fields["交易流水号"])
    a.press("enter")
    a.sleep(1)
    a.write(fields["金额"])
    
    a.press("enter")
    a.sleep(1)
    
    

    a.write(fields["流水号后三位"])
    a.press("enter")
    a.sleep(1)
    a.write(fields["受理日期"])
    
    
    
    a.sleep(0.5)
    a.press("enter")
    a.sleep(1)
    a.press('1')
    a.sleep(1) 
    a.press("tab")
    a.sleep(1) 
    #输入国库代码
    国库代码 = lookup_region_code(CONFIG["external"]["国库代码"], fields["收款人账户名称"])
    if 国库代码:
        a.write(国库代码, interval=0.1)
        
    a.sleep(2)   
    a.press("enter")      
    a.write(fields.get("收款人账号", ""))   
    
    a.sleep(0.5)
    a.press("enter")
    a.sleep(1)
    
    if fields["报文种类_中文"] == "大额":
        a.write("010221",interval=0.1)
    else:
        a.write("010222",interval=0.1)
        
    a.press("enter")
    a.sleep(1)

    #输入出票单位
    出票单位 = lookup_region_code(CONFIG["external"]["出票单位"], fields["收款人账户名称"])
    if 出票单位:
        a.write(出票单位, interval=0.1)
    
    
    a.press("enter")
    a.sleep(1)
    a.write('278')            
    a.press("enter")
    a.sleep(1)
    a.write(fields["金额"])
    a.sleep(0.5)
    a.press('enter')
    a.sleep(0.5)
    a.press('enter')
    a.sleep(0.5)
    a.write("110090199")
    a.sleep(1)
    
    a.press('enter')
    a.write(fields["金额"])
    a.sleep(0.5)
    a.press('enter')
    
    a.sleep(0.5)
    a.click(1030,990)

def main():
    if len(sys.argv) < 2:
        print("用法: python ocr_to_llm.py <PDF文件路径>")
        return

    pdf_path = sys.argv[1]
    if not os.path.isfile(pdf_path):
        print(f"文件不存在: {pdf_path}")
        return

    print("[1/4] 读取 PDF ...")
    if OCR_ENABLED:
        raw_text = ocr_pdf_to_text(pdf_path)
    else:
        raw_text = extract_text_from_pdf(pdf_path)
    if not raw_text.strip():
        print("  ⚠ 文本提取为空，尝试 OCR ...")
        raw_text = ocr_pdf_to_text(pdf_path)

    print(f"  提取 {len(raw_text)} 字符")

    print("[2/4] 提取字段 ...")
    fields = extract_fields(raw_text)
    for k, v in fields.items():
        print(f"  {k}: {v}")

    print("[3/4] 规则分类 + AI 复核 ...")
    result = classify_by_rules(fields)
    biz_type = result.get("business_type", "unknown")
    reason = result.get("reason", "")
    print(f"  业务类型: {biz_type}")
    print(f"  原因: {reason}")
    review = review_by_llm(fields, result)
    if not review["agree"]:
        biz_type = "needs_review"
        print(f"  ⚠ AI 复核不通过: {review['reason']}")
    else:
        print(f"  AI 复核: 通过（{review['reason']}）")

    print("[4/4] 输出结果 ...")
    print("\n" + "=" * 50)
    print(format_output(biz_type, reason, fields))
    print("=" * 50)



def find_pic(img, confidence=0.9, timeout=60):
    """在屏幕上定位图片并点击; 超时抛异常而不是无限等待"""
    import time
    a = get_pyautogui()
    deadline = time.time() + timeout
    while True:
        pos = a.locateCenterOnScreen(img, confidence=confidence, grayscale=True)
        if pos is not None:
            a.click(pos)
            return
        if time.time() > deadline:
            raise TimeoutError(f"超时未找到图片: {img}")
        a.sleep(0.5)
def process_single_file(path):
    result = llm_ocr(path)
    if not result:
        return
    for data in result:
        biz = data["biz"]
        fields = data["fields"]
        if biz == "settlement_refund":
            refund(fields)
        elif biz == "external_allocation":
            external(fields)
        elif biz == "needs_review":
            print(f"  ⚠ 跳过自动录入，待人工复核: 流水号 {fields.get('交易流水号', '?')} "
                  f"金额 {fields.get('金额', '?')} 复核意见: {data.get('review', '')}")
            continue
        else:
            break


# =========================
# 6. GUI 界面
# =========================

def run_gui():
    import threading
    import tkinter as tk
    from tkinter import filedialog, messagebox

    root = tk.Tk()
    root.title("凭证批量处理")
    root.geometry("640x420")

    # 文件夹选择区
    top_frame = tk.Frame(root)
    top_frame.pack(fill="x", padx=10, pady=10)
    tk.Label(top_frame, text="文件夹路径:").pack(side="left")
    path_var = tk.StringVar()
    path_entry = tk.Entry(top_frame, textvariable=path_var)
    path_entry.pack(side="left", fill="x", expand=True, padx=5)

    def browse():
        folder = filedialog.askdirectory()
        if folder:
            path_var.set(folder)

    tk.Button(top_frame, text="浏览...", command=browse).pack(side="left")

    # 日志区
    log_text = tk.Text(root, state="disabled", wrap="word")
    log_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def log(msg):
        # 从工作线程切回主线程写日志
        def _append():
            log_text.config(state="normal")
            log_text.insert("end", msg + "\n")
            log_text.see("end")
            log_text.config(state="disabled")
        root.after(0, _append)

    start_btn = tk.Button(root, text="开始处理")
    start_btn.pack(pady=(0, 10))

    def worker(folder):
        try:
            log(f"开始遍历文件夹: {folder}")
            log("提示: 处理期间请勿操作鼠标和键盘。")
            files = sorted(os.listdir(folder))
            pdf_files = [f for f in files if f.lower().endswith(".pdf")]
            if not pdf_files:
                log("未找到 PDF 文件。")
                return
            for i, name in enumerate(pdf_files, 1):
                path = os.path.join(folder, name)
                log(f"[{i}/{len(pdf_files)}] 处理: {name}")
                try:
                    process_single_file(path)
                    log(f"  完成: {name}")
                except Exception as e:
                    log(f"  ❌ 处理失败: {name} -> {e}")
            log("全部处理完成。")
        finally:
            root.after(0, lambda: start_btn.config(state="normal"))

    def on_start():
        folder = path_var.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("错误", "请选择有效的文件夹路径")
            return
        start_btn.config(state="disabled")
        threading.Thread(target=worker, args=(folder,), daemon=True).start()

    start_btn.config(command=on_start)
    root.mainloop()


if __name__ == "__main__":
    # PyInstaller 在 Windows 上需要 freeze_support 防止子进程重复启动
    multiprocessing.freeze_support()

    try:
        if len(sys.argv) >= 2:
            # 命令行模式: python ocr_to_llm.py <PDF文件路径>
            main()
        else:
            run_gui()
    except Exception as e:
        # 在 --windowed 模式下捕获异常并弹窗提示，避免静默崩溃
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("程序启动失败", f"错误信息:\n{e}\n\n请检查依赖是否完整安装。")
            root.destroy()
        except Exception:
            pass
        sys.exit(1)


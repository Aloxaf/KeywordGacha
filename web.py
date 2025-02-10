import os
import uuid
import zipfile
import shutil
import asyncio
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# 导入项目原有模块及函数
from main import process_text, load_config
from model.NER import NER
from module.FileManager import FileManager
from module.LogHelper import LogHelper

app = FastAPI()

# 允许跨域（可选，用于本地调试）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局任务存储：job_id -> {"progress": list, "status": str, "result": str}
jobs = {}

@app.get("/", response_class=HTMLResponse)
def get_index():
    # 直接从同级目录读取 index.html
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), language: str = Form(...)):
    if not file.filename.endswith(".zip"):
        raise HTTPException(400, "必须上传ZIP文件")
    
    # 生成唯一任务ID，并准备任务目录（temp/{job_id}）
    job_id = str(uuid.uuid4())
    job_folder = os.path.join("temp", job_id)
    input_folder = os.path.join(job_folder, "input")
    output_folder = os.path.join(job_folder, "output")
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    # 保存上传的ZIP文件
    zip_path = os.path.join(job_folder, "upload.zip")
    with open(zip_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # 解压ZIP到 input 文件夹
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(input_folder)
    except Exception as e:
        raise HTTPException(400, f"ZIP解压失败: {e}")

    # 初始化任务状态
    jobs[job_id] = {"progress": [], "status": "pending", "result": None}

    # 开启后台任务处理（使用 asyncio.create_task 非阻塞执行）
    asyncio.create_task(process_job(job_id, job_folder, language, input_folder, output_folder))
    
    return {"job_id": job_id}

@app.get("/progress/{job_id}")
def get_progress(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return {"status": jobs[job_id]["status"], "progress": jobs[job_id]["progress"]}

@app.get("/download/{job_id}")
def download_result(job_id: str):
    if job_id not in jobs or jobs[job_id]["status"] != "completed" or not jobs[job_id]["result"]:
        raise HTTPException(404, "Result not available")
    return FileResponse(jobs[job_id]["result"], media_type="application/zip", filename="result.zip")

# 工具函数：将前端提交的语言字符串映射到 NER.Language 常量
def map_language(lang_str: str):
    lang_str = lang_str.lower()
    if lang_str == "zh":
        return NER.Language.ZH
    elif lang_str == "en":
        return NER.Language.EN
    elif lang_str == "ja":
        return NER.Language.JA
    elif lang_str == "ko":
        return NER.Language.KO
    else:
        return NER.Language.JA  # 默认选择日文

# 后台任务：处理上传的任务
async def process_job(job_id: str, job_folder: str, language: str, input_folder: str, output_folder: str):
    try:
        jobs[job_id]["status"] = "processing"
        # 猴子补丁：禁用 os.system("pause") 等调用（避免阻塞）
        original_os_system = os.system
        os.system = lambda cmd: None

        # 备份 LogHelper 原有方法，并替换成同时记录进度的版本
        original_info = LogHelper.info
        original_print = LogHelper.print
        original_warning = LogHelper.warning
        original_error = LogHelper.error

        def patched_info(msg, *args, **kwargs):
            jobs[job_id]["progress"].append(str(msg))
            original_info(msg, *args, **kwargs)
        def patched_print(msg="", *args, **kwargs):
            jobs[job_id]["progress"].append(str(msg))
            original_print(msg, *args, **kwargs)
        def patched_warning(msg, *args, **kwargs):
            jobs[job_id]["progress"].append("WARNING: " + str(msg))
            original_warning(msg, *args, **kwargs)
        def patched_error(msg, *args, **kwargs):
            jobs[job_id]["progress"].append("ERROR: " + str(msg))
            original_error(msg, *args, **kwargs)

        LogHelper.info = patched_info
        LogHelper.print = patched_print
        LogHelper.warning = patched_warning
        LogHelper.error = patched_error

        # 加载配置和核心对象
        llm, ner, file_manager, config, version = load_config()

        # 为 FileManager 设置任务专用的输入输出目录
        file_manager.input_dir_override = input_folder
        file_manager.output_dir_override = output_folder

        from types import MethodType

        # 重写 load_lines_from_input_file 方法：读取 job 专属input目录
        def load_lines_override(self, language: int) -> list:
            import os
            paths = []
            for root, _, files in os.walk(self.input_dir_override):
                for f in files:
                    if f.endswith((".txt", ".csv", ".json", ".xlsx")):
                        paths.append(os.path.join(root, f))
            input_lines = []
            for path in paths:
                input_lines.extend(self.read_file(path))
            names, nicknames = {}, {}
            if os.path.isfile(os.path.join(self.input_dir_override, "Actors.json")):
                names, nicknames = self.load_names(os.path.join(self.input_dir_override, "Actors.json"))
            elif os.path.isfile(os.path.join(os.path.dirname(self.input_dir_override), "Actors.json")):
                names, nicknames = self.load_names(os.path.join(os.path.dirname(self.input_dir_override), "Actors.json"))
            input_lines_filtered = []
            from module.Normalizer import Normalizer
            from module.TextHelper import TextHelper
            from model.NER import NER
            for line in input_lines:
                line = self.cleanup(line, language, names, nicknames)
                if len(line) == 0:
                    continue
                if language == NER.Language.ZH and not TextHelper.has_any_cjk(line):
                    continue
                if language == NER.Language.EN and not TextHelper.has_any_latin(line):
                    continue
                if language == NER.Language.JA and not TextHelper.has_any_japanese(line):
                    continue
                if language == NER.Language.KO and not TextHelper.has_any_korean(line):
                    continue
                line = Normalizer.normalize(line, merge_space=True)
                input_lines_filtered.append(line)
            LogHelper.info(f"已读取到文本 {len(input_lines)} 行，其中有效文本 {len(input_lines_filtered)} 行 ...")
            return input_lines_filtered

        # 重写 write_result_to_file 方法：结果写入 job 专用的 output目录
        def write_result_override(self, words: list, language: int) -> None:
            import os
            os.makedirs(self.output_dir_override, exist_ok=True)
            file_name = "result"
            for group in {word.group for word in words}:
                words_by_type = [word for word in words if word.group == group]
                if not words_by_type:
                    continue
                self.write_words_log_to_file(words_by_type, os.path.join(self.output_dir_override, f"{file_name}_{group}_日志.txt"), language)
                self.write_words_list_to_file(words_by_type, os.path.join(self.output_dir_override, f"{file_name}_{group}_列表.json"), language)
                self.write_ainiee_dict_to_file(words_by_type, os.path.join(self.output_dir_override, f"{file_name}_{group}_ainiee.json"), language)
                self.write_galtransl_dict_to_file(words_by_type, os.path.join(self.output_dir_override, f"{file_name}_{group}_galtransl.txt"), language)

        file_manager.load_lines_from_input_file = MethodType(load_lines_override, file_manager)
        file_manager.write_result_to_file = MethodType(write_result_override, file_manager)

        # 将语言参数映射为 NER.Language 常量
        lang_constant = map_language(language)

        # 调用原有的处理函数（process_text）
        await process_text(llm, ner, file_manager, config, lang_constant)

        # 处理完成后，将 output 文件夹打包成ZIP文件
        result_zip = os.path.join(job_folder, "result.zip")
        with zipfile.ZipFile(result_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(output_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, output_folder)
                    zipf.write(file_path, arcname)
        
        jobs[job_id]["result"] = result_zip
        jobs[job_id]["status"] = "completed"
    except Exception as e:
        jobs[job_id]["progress"].append("处理过程中发生异常: " + str(e))
        jobs[job_id]["status"] = "failed"
    finally:
        # 恢复被修改的函数
        os.system = original_os_system
        LogHelper.info = original_info
        LogHelper.print = original_print
        LogHelper.warning = original_warning
        LogHelper.error = original_error

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
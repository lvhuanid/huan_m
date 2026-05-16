import json
import sys
import os
import click
from datetime import datetime

# ---------- 配置文件路径 ----------
def get_quiz_dir():
    quiz_dir = os.path.expanduser("~/.quiz")
    os.makedirs(quiz_dir, exist_ok=True)
    return quiz_dir

def get_error_file():
    return os.path.join(get_quiz_dir(), "errors.json")

def get_progress_file():
    return os.path.join(get_quiz_dir(), "progress.json")

def get_data_path(filename):
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, filename)

# ---------- 错误记录（含错误输入） ----------
def add_error_word(session_index_ref, en, zh, wrong_input):
    """
    实时添加错误单词，并记录此次错误的输入。
    session_index_ref: [index_or_None] 可变列表
    """
    error_file = get_error_file()
    try:
        if os.path.exists(error_file):
            with open(error_file, "r", encoding="utf-8") as f:
                sessions = json.load(f)
        else:
            sessions = []
    except:
        sessions = []

    # 首次出错才创建 session
    if session_index_ref[0] is None:
        new_session = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "words": []
        }
        sessions.append(new_session)
        session_index_ref[0] = len(sessions) - 1

    session_idx = session_index_ref[0]
    if session_idx < 0 or session_idx >= len(sessions):
        return

    words = sessions[session_idx]["words"]
    # 查找该单词是否已记录
    existing = next((w for w in words if w["en"] == en), None)
    if existing:
        # 追加错误输入（避免重复）
        if wrong_input not in existing.get("wrong", []):
            if "wrong" not in existing:
                existing["wrong"] = []
            existing["wrong"].append(wrong_input)
    else:
        # 新建错误单词记录
        words.append({"en": en, "zh": zh, "wrong": [wrong_input]})

    with open(error_file, "w", encoding="utf-8") as f:
        json.dump(sessions, f, ensure_ascii=False, indent=2)

# ---------- 进度管理 ----------
def load_progress():
    prog_file = get_progress_file()
    try:
        if os.path.exists(prog_file):
            with open(prog_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            idx = data.get("index", 0)
            if isinstance(idx, int) and idx >= 0:
                return idx
    except:
        pass
    return 0

def save_progress(index):
    with open(get_progress_file(), "w", encoding="utf-8") as f:
        json.dump({"index": index}, f)

# ---------- 命令组 ----------
@click.group()
def cli():
    """单词测验工具 - 学习、复习、进度记忆"""
    pass

# ---------- learn 子命令 ----------
@cli.command()
@click.option("-f", "--file", default=None, type=click.Path(dir_okay=False),
              help="自定义单词 JSON 文件（默认内置 a.json）")
@click.option("--start", type=int, default=None, help="起始索引（0 为第一个）")
@click.option("--word", type=str, default=None, help="从指定英文单词开始")
def learn(file, start, word):
    """根据中文释义输入英文，支持断点续测"""
    if file is None:
        data_file = get_data_path("a.json")
    else:
        data_file = file
        if not os.path.isfile(data_file):
            raise click.BadParameter(f"文件不存在: {data_file}")

    with open(data_file, "r", encoding="utf-8") as f:
        words = json.load(f)

    if not words:
        click.echo("没有找到任何单词，退出。")
        return

    total = len(words)

    # 起始位置
    if word is not None:
        found_idx = None
        for i, (en, _) in enumerate(words):
            if en.strip().lower() == word.strip().lower():
                found_idx = i
                break
        if found_idx is None:
            click.echo(f"错误：单词列表中未找到 '{word}'")
            return
        start_idx = found_idx
    elif start is not None:
        start_idx = start
        if start_idx < 0 or start_idx >= total:
            click.echo(f"错误：起始索引 {start_idx} 超出范围 (0~{total-1})")
            return
    else:
        start_idx = load_progress()
        if start_idx >= total:
            click.echo("已完成全部单词，从头开始。")
            start_idx = 0
        elif start_idx > 0:
            click.echo(f"从上次结束位置继续（第 {start_idx+1} 个）。")

    click.echo(f"共 {total} 个单词，从第 {start_idx+1} 个开始，按 Ctrl+C 退出。\n")

    session_index = [None]   # 延迟创建错误记录
    error_set = set()

    for idx in range(start_idx, total):
        en, zh = words[idx]
        zh_clean = zh.strip().replace("\r", "")
        click.echo(zh_clean)

        while True:
            user_input = click.prompt("请输入英文", type=str).strip()
            if user_input.lower() == en.strip().lower():
                click.echo("✓ 正确")
                save_progress(idx + 1)
                break
            else:
                click.echo(en.strip())
                # 记录错误单词和本次输入
                if en.strip() not in error_set:
                    error_set.add(en.strip())
                add_error_word(session_index, en.strip(), zh_clean, user_input)

    if error_set:
        click.echo("\n⚠️  本次答错的单词：")
        for en, zh in [(e, z) for e, z in
                       [(w[0].strip(), w[1].strip().replace("\r", "")) for w in words]
                       if e in error_set]:
            click.echo(f"  {en}  {zh}")
        click.echo("（错误及输入记录已实时保存到 errors.json）")
    else:
        click.echo("\n🎉 全部正确！")

    if start_idx + (total - start_idx) >= total:
        save_progress(0)
        click.echo("已完成全部单词，进度已重置。")
    click.echo("学习结束。")

# ---------- review 子命令 ----------
@cli.command()
@click.option("-t", "--time", "session_time", default=None, type=str,
              help="指定要复习的错误记录时间（格式 YYYY-MM-DD HH:MM:SS），默认最新")
def review(session_time):
    """复习之前答错的单词（从 errors.json 读取）"""
    error_file = get_error_file()
    if not os.path.exists(error_file):
        click.echo("错误记录文件不存在，请先使用 learn 学习。")
        return

    try:
        with open(error_file, "r", encoding="utf-8") as f:
            sessions = json.load(f)
    except:
        click.echo("错误记录文件损坏。")
        return

    # 过滤空记录（兼容旧版）
    sessions = [s for s in sessions if s.get("words")]

    if not sessions:
        click.echo("暂无有效的错误记录。")
        return

    # 选择目标 session
    target_session = None
    if session_time is not None:
        for s in sessions:
            if s.get("time", "") == session_time:
                target_session = s
                break
        if target_session is None:
            click.echo(f"未找到时间为 '{session_time}' 的错误记录。")
            return
    else:
        target_session = sessions[-1]

    words = target_session.get("words", [])
    if not words:
        click.echo("该记录中没有错误单词。")
        return

    click.echo(f"开始复习 {len(words)} 个错题（记录时间：{target_session['time']}）：\n")

    for w in words:
        zh = w["zh"].strip().replace("\r", "")
        en = w["en"].strip()
        click.echo(zh)

        while True:
            user_input = click.prompt("请输入英文", type=str).strip()
            if user_input.lower() == en.lower():
                click.echo("✓ 正确")
                break
            else:
                click.echo(en)

    click.echo("\n🎉 错题复习完成！")

# ---------- 入口 ----------
if __name__ == "__main__":
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\n已退出。")
        sys.exit(0)
    # pyinstaller --onefile --add-data "a.json:." quiz.py
    # sudo cp dist/quiz /usr/local/bin/
    # sudo chmod +x /usr/local/bin/quiz
    # quiz

    # quiz --word removal
    # --start
    # quiz --file /path/to/your/words.json
    # cat ~/.quiz/errors.json

#     quiz learn                    # 默认从进度继续
# quiz learn --start 0          # 从头开始
# quiz learn --word remove      # 从 "remove" 开始
# quiz learn -f mywords.json    # 用外部文件

# quiz review                   # 复习最新一次错误记录
# quiz review -t "2026-05-16 10:01:16"   # 复习指定时间的错误记录
import json
import sys
import os
import click

def get_data_path(filename):
    """获取数据文件的路径，兼容 PyInstaller 打包后的 _MEIPASS 环境"""
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, filename)

@click.command()
@click.option(
    "-f", "--file",
    default=None,  # 改为 None，不再让 click 验证路径
    type=click.Path(dir_okay=False),
    help="自定义单词 JSON 文件路径（默认使用内置 a.json）"
)
def cli(file):
    """
    简单单词测验：根据中文释义输入英文单词。
    \b
    JSON 格式要求：
    [
        ["英文", "中文释义"],
        ...
    ]
    """
    # 决定最终数据文件路径
    if file is None:
        # 使用内置数据
        data_file = get_data_path("a.json")
    else:
        data_file = file
        # 自行验证文件是否存在
        if not os.path.isfile(data_file):
            raise click.BadParameter(f"文件不存在: {data_file}")

    with open(data_file, "r", encoding="utf-8") as f:
        words = json.load(f)

    if not words:
        click.echo("没有找到任何单词，退出。")
        return

    click.echo(f"共 {len(words)} 个单词，按 Ctrl+C 可随时退出。\n")

    for en, zh in words:
        zh_clean = zh.strip().replace("\r", "")
        click.echo(zh_clean)

        while True:
            user_input = click.prompt("请输入英文", type=str).strip()
            if user_input.lower() == en.strip().lower():
                click.echo("✓ 正确")
                break
            else:
                click.echo(en.strip())

    click.echo("\n🎉 所有单词已完成！")

if __name__ == "__main__":
    cli()

    # pyinstaller --onefile --add-data "a.json:." quiz.py
    # sudo cp dist/quiz /usr/local/bin/
    # sudo chmod +x /usr/local/bin/quiz
    # quiz
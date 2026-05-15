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
    default="a.json",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="包含单词数据的 JSON 文件（默认 a.json）"
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
    if file == "a.json":
        data_file = get_data_path("a.json")
    else:
        data_file = file

    with open(data_file, "r", encoding="utf-8") as f:
        words = json.load(f)

    if not words:
        click.echo("没有找到任何单词，退出。")
        return

    click.echo(f"共 {len(words)} 个单词，按 Ctrl+C 可随时退出。\n")

    for en, zh in words:
        zh_clean = zh.strip().replace("\r", "")
        click.echo(zh_clean)

        # 循环直到输入正确
        while True:
            user_input = click.prompt("请输入英文", type=str).strip()
            if user_input.lower() == en.strip().lower():
                click.echo("✓ 正确")
                break
            else:
                # 错误时显示正确答案，然后继续要求输入
                click.echo(en.strip())

    click.echo("\n🎉 所有单词已完成！")

if __name__ == "__main__":
    cli()
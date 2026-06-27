import json
import sys
from typing import Dict, List, Any, Optional

class Node:
    def __init__(self, name: str, definition: Dict[str, Any], parent: Optional['Node'] = None):
        self.name = name
        self.yangType = definition.get('yangType', '')
        self.ypath = definition.get('ypath', [])
        self.namespace = definition.get('namespace', '')
        self.description = definition.get('description', '')
        self.key = definition.get('key', '')
        self.type = definition.get('type', '')
        self.config = definition.get('config', 'true')
        self.union_value = definition.get('union-value', [])
        self.enum = definition.get('enum', {})
        self.parent = parent
        self.children: Dict[str, Node] = {}

    def add_child(self, child: 'Node'):
        self.children[child.name] = child
        child.parent = self

def build_tree(json_obj: Dict) -> Node:
    root_name = next(iter(json_obj))
    return _build_node(root_name, json_obj[root_name], parent=None)

def _build_node(name: str, obj: Dict, parent: Optional[Node]) -> Node:
    if not isinstance(obj, dict):
        # 兜底非字典类型
        return Node(name, {}, parent)

    if 'yangType' in obj and obj.get('yangType') == 'leaf':
        node = Node(name, obj, parent)
        node.type = obj.get('type', '')
        node.config = obj.get('config', 'true')
        node.description = obj.get('description', '')
        node.union_value = obj.get('union-value', [])
        node.enum = obj.get('enum', {})
        return node

    definition = obj.get('definition', {})
    node = Node(name, definition, parent)
    
    for key, value in obj.items():
        if key == 'definition' or not isinstance(value, dict):
            continue
        child = _build_node(key, value, parent=node)
        node.add_child(child)
        
    if node.yangType == 'leaf':
        node.type = obj.get('type', definition.get('type', ''))
        node.config = obj.get('config', definition.get('config', 'true'))
        node.description = obj.get('description', definition.get('description', ''))
        node.union_value = obj.get('union-value', definition.get('union-value', []))
        node.enum = obj.get('enum', definition.get('enum', {}))
    return node

def get_full_xpath(node: Node) -> str:
    """获取标准的 XPath"""
    parts = []
    current = node
    path_nodes = []
    while current:
        path_nodes.insert(0, current)
        current = current.parent
    
    for n in path_nodes:
        # 如果 ypath 存在取最后一段，否则取节点名
        segment = n.ypath[-1] if n.ypath else n.name
        # 移除可能存在的 openconfig 前缀（根据具体 CLI 规范而定，这里保留核心名字）
        parts.append(segment)
    return '/' + '/'.join(parts)

def generate_rpc_commands(node: Node) -> List[Dict]:
    """专门为 RPC 节点解析并生成命令的函数"""
    entries = []
    if node.yangType != 'RPC':
        return entries

    parent_cmd = node.name.replace('_', '-').lower()
    xpath = get_full_xpath(node)
    
    args = []
    yang_nodes = {}

    # 获取 input 节点下的输入参数
    input_node = node.children.get('input')
    if input_node:
        for child_name, child in input_node.children.items():
            if child.yangType == 'leaf':
                desc = child.description.split('.')[0].strip() if child.description else f"Target {child_name}"
                args_item = {
                    "name": child_name,
                    "mode": "CLI_CMD_ARGUMENT",  # RPC 输入参数一般设为必填或根据需求调整
                    "help": desc
                }
                if child.union_value:
                    args_item["union-value"] = transform_union_value(child.union_value)
                if child.type == 'enumeration' and child.enum:
                    args_item["enum"] = transform_enum(child.enum)
                
                args.append(args_item)
                # 映射到 input 路径下
                yang_nodes[child_name] = f"input/{child_name}"

    # 如果有需要，在这里可以同样解析 output 节点
    # output_node = node.children.get('output')

    entry = {
        "parent": parent_cmd,
        "command": "action",  # RPC 通常对应 action 或 execute 动作
        "callback_name": "cmd_rpc_execute",
        "help_text": node.description.strip() if node.description else f"Execute {parent_cmd}",
        "action": "execute",
        "xpath": xpath,
        "args": args,
        "yang_nodes": yang_nodes
    }
    entries.append(entry)
    return entries

def transform_enum(raw_enum: Dict) -> Dict:
    transformed = {}
    for k, v in raw_enum.items():
        transformed[k] = {
            "value": k.split(":")[-1],
            "description": v.get("description", "").strip() if isinstance(v, dict) else ""
        }
    return transformed

def transform_union_value(union_list: List) -> List:
    transformed_list = []
    for item in union_list:
        new_item = dict(item)
        if item.get("type") == "enumeration" and "enum" in item:
            new_item["enum"] = transform_enum(item["enum"])
        transformed_list.append(new_item)
    return transformed_list

def main():
    if len(sys.argv) != 2:
        sys.exit(1)

    with open(sys.argv[1], 'r', encoding='utf-8') as f:
        data = json.load(f)

    if set(data.keys()) == {'rpc'}:
        data = data['rpc']

    all_commands = []
    for top_name, top_obj in data.items():
        root = build_tree({top_name: top_obj})
        
        # 判定如果是 RPC 节点，走全新的 RPC 专属解析逻辑
        if root.yangType == 'RPC':
            commands = generate_rpc_commands(root)
        else:
            # 兼容非 RPC 的旧逻辑（如果需要的话，这里保留原 generate_commands 逻辑）
            commands = [] 
            
        all_commands.extend(commands)

    print(json.dumps(all_commands, indent=4, ensure_ascii=False))

if __name__ == '__main__':
    main()

    # uv run yang_to_set_rpc.py "test_ input_rpc.json" > wwwrpc.json 2>&1; echo "Exit code: $?"
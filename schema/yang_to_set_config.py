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

def is_list(node: Node) -> bool:
    return node.yangType == 'list'

def has_config_leaves(node: Node) -> bool:
    config_node = node.children.get('config')
    if not config_node:
        return False
    return any(child.yangType == 'leaf' for child in config_node.children.values())

def get_full_xpath_with_keys(node: Node) -> str:
    parts = []
    current = node
    path_nodes = []
    while current:
        path_nodes.insert(0, current)
        current = current.parent
    
    for idx, n in enumerate(path_nodes):
        segment = n.ypath[-1] if n.ypath else n.name
        if is_list(n) and n.key:
            keys = [k for k in n.key.split() if k]
            key_exprs = [f"{k}='%s'" for k in keys]
            segment = f"{segment}[{ ' and '.join(key_exprs) }]"
        parts.append(segment)
    return '/' + '/'.join(parts)

def generate_commands(node: Node) -> List[Dict]:
    commands = []
    def walk(n: Node):
        if is_list(n) and has_config_leaves(n):
            commands.append(n)
        for child in n.children.values():
            walk(child)
    walk(node)

    entries = []
    for target in commands:
        entries.append(build_entry(target))
    return entries

# 保留你修正后的 transform_enum 逻辑
def transform_enum(enum_dict: dict) -> dict:
    if not enum_dict:
        return {}
    new_enum = {}
    for full_key, details in enum_dict.items():
        short_key = full_key.split(':')[-1] if ':' in full_key else full_key
        new_details = dict(details) if isinstance(details, dict) else {"description": str(details)}
        new_details["value"] = full_key  
        new_enum[short_key] = new_details
    return new_enum

# 新增：补全缺失的 transform_union_value 函数
def transform_union_value(union_list: List[Dict]) -> List[Dict]:
    if not union_list:
        return []
    new_union = []
    for item in union_list:
        new_item = dict(item)
        # 如果 union 成员中包含 enumeration，对它的 enum 字典也进行转换
        if item.get("type") == "enumeration" and "enum" in item:
            new_item["enum"] = transform_enum(item["enum"])
        new_union.append(new_item)
    return new_union

def parse_help_text(child: Node) -> str:
    if child.union_value:
        enum_values = []
        for item in child.union_value:
            if item.get("type") == "enumeration" and "enum" in item:
                for enum_key in item["enum"].keys():
                    enum_values.append(enum_key.split(":")[-1])
        if enum_values:
            return f"{child.name.title()}({'/'.join(enum_values)})"
            
    if child.type == 'enumeration' and child.enum:
        enum_values = [k.split(":")[-1] for k in child.enum.keys()]
        return f"{child.name.title()}({'/'.join(enum_values)})"
        
    return child.description.split('.')[0].strip() if child.description else f"Configure {child.name}"

def build_entry(target: Node) -> Dict:
    parent_cmd = target.name.replace('_', '-').lower()
    xpath = get_full_xpath_with_keys(target)
    
    args = []
    yang_nodes = {}
    
    list_keys = [k for k in target.key.split() if k] if target.key else []
    config_node = target.children.get('config')
    
    if config_node:
        for k in list_keys:
            key_leaf = config_node.children.get(k)
            desc = key_leaf.description.split('.')[0] if key_leaf and key_leaf.description else f"{k.title()} key"
            args_item = {
                "name": k,
                "mode": "CLI_CMD_ARGUMENT",
                "help": desc.strip()
            }
            if key_leaf:
                if key_leaf.union_value:
                    args_item["union-value"] = transform_union_value(key_leaf.union_value)
                if key_leaf.type == 'enumeration' and key_leaf.enum:
                    args_item["enum"] = transform_enum(key_leaf.enum)
            args.append(args_item)
            yang_nodes[k] = f"config/{k}"

        for child in config_node.children.values():
            if child.yangType == 'leaf' and child.name not in list_keys:
                desc = parse_help_text(child)
                args_item = {
                    "name": child.name,
                    "mode": "CLI_CMD_OPTIONAL_ARGUMENT",
                    "help": desc
                }
                if child.union_value:
                    args_item["union-value"] = transform_union_value(child.union_value)
                if child.type == 'enumeration' and child.enum:
                    args_item["enum"] = transform_enum(child.enum)
                args.append(args_item)
                yang_nodes[child.name] = f"config/{child.name}"

    return {
        "parent": parent_cmd,
        "command": "create",
        "callback_name": "cmd_config_data",
        "help_text": f"Create or modify {parent_cmd}",
        "action": "set",
        "xpath": xpath,
        "args": args,
        "yang_nodes": yang_nodes
    }

def main():
    if len(sys.argv) != 2:
        sys.exit(1)

    with open(sys.argv[1], 'r', encoding='utf-8') as f:
        data = json.load(f)

    if set(data.keys()) == {'schema'}:
        data = data['schema']

    all_commands = []
    for top_name, top_obj in data.items():
        root = build_tree({top_name: top_obj})
        # print(top_name, top_obj) # 如果输出嫌多可以注释掉
        commands = generate_commands(root)
        all_commands.extend(commands)

    print(json.dumps(all_commands, indent=4))

if __name__ == '__main__':
    main()
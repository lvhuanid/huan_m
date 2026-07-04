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
        self.default = definition.get('default', [])
        self.units = definition.get('units', [])
        self.type = definition.get('type', [])
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
        node.default = obj.get('default', [])
        node.units = obj.get('units', [])
        node.type = obj.get('type', [])
        node.enum = obj.get('enum', {})
        if parent:
            parent.add_child(node)
        return node

    definition = obj.get('definition', {})
    node = Node(name, definition, parent)
    if parent:
        parent.add_child(node)
        
    for key, value in obj.items():
        if key == 'definition' or not isinstance(value, dict):
            continue
        _build_node(key, value, parent=node)
        
    if node.yangType == 'leaf':
        node.type = obj.get('type', definition.get('type', ''))
        node.config = obj.get('config', definition.get('config', 'true'))
        node.description = obj.get('description', definition.get('description', ''))
        node.union_value = obj.get('union-value', definition.get('union-value', []))
        node.enum = obj.get('enum', definition.get('enum', {}))
    return node

def is_list(node: Node) -> bool:
    return node.yangType == 'list'

def find_all_config_nodes(node: Node) -> List[Node]:
    configs = []
    if node.name == 'config':
        if any(child.yangType == 'leaf' for child in node.children.values()):
            configs.append(node)
    for child in node.children.values():
        configs.extend(find_all_config_nodes(child))
    return configs

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
    entries = []
    
    def walk(n: Node):
        if is_list(n):
            config_subnodes = find_all_config_nodes(n)
            if config_subnodes:
                for cfg in config_subnodes:
                    entries.append(build_entry_for_config(n, cfg))
        for child in n.children.values():
            walk(child)
            
    walk(node)
    return entries

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

def transform_union_value(union_list: List[Dict]) -> List[Dict]:
    if not union_list:
        return []
    new_union = []
    for item in union_list:
        new_item = dict(item)
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

def get_relative_yang_path(ancestor: Node, descendant: Node) -> str:
    parts = []
    curr = descendant
    while curr and curr != ancestor:
        parts.insert(0, curr.name)
        curr = curr.parent
    return '/'.join(parts)

class NoLineBreakList(list):
    pass

class CustomJSONEncoder(json.JSONEncoder):
    def encode(self, obj):
        if isinstance(obj, NoLineBreakList):
            return '[' + ', '.join(json.dumps(el) for el in obj) + ']'
        return super().encode(obj)

    def iterencode(self, obj, _one_shot=False):
        if isinstance(obj, dict):
            fields = []
            for k, v in obj.items():
                if isinstance(v, NoLineBreakList):
                    encoded_v = '[' + ', '.join(json.dumps(el) for el in v) + ']'
                    fields.append(f"{json.dumps(k)}: {encoded_v}")
                else:
                    fields.append(f"{json.dumps(k)}: {self.encode(v)}")
            return '{\n' + ',\n'.join(fields) + '\n}'
        return super().iterencode(obj, _one_shot)

def process_argument_enum(args_item: dict, node_enum: dict, node_name: str):
    if node_enum:
        args_item["enum"] = transform_enum(node_enum)
        completor_name = node_name.replace('-', '_')
        args_item["completor"] = f"{completor_name}_completor"
        enum_keys = [k.split(":")[-1] for k in node_enum.keys()]
        args_item["completor_data"] = NoLineBreakList(enum_keys)

def build_entry_for_config(list_node: Node, config_node: Node) -> Dict:
    parent_cmd = list_node.name.replace('_', '-').lower()
    container_node = config_node.parent
    if container_node and container_node != list_node:
        parent_cmd = f"{parent_cmd}-{container_node.name.replace('_', '-').lower()}"
        
    xpath = get_full_xpath_with_keys(container_node if container_node else list_node)
    
    args = []
    yang_nodes = {}
    
    list_keys = [k for k in list_node.key.split() if k] if list_node.key else []
    
    list_config = list_node.children.get('config')
    for k in list_keys:
        key_leaf = None
        if list_config:
            key_leaf = list_config.children.get(k)
        if not key_leaf:
            key_leaf = list_node.children.get(k)
            
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
                process_argument_enum(args_item, key_leaf.enum, key_leaf.name)
        args.append(args_item)
        
        if list_config and k in list_config.children:
            yang_nodes[k] = f"config/{k}"
        else:
            yang_nodes[k] = k

    prefix_path = get_relative_yang_path(list_node, config_node)

    for child in config_node.children.values():
        if child.yangType == 'leaf' and child.name not in list_keys:
            desc = parse_help_text(child)
            args_item = {
                "name": child.name,
                "mode": "CLI_CMD_OPTIONAL_ARGUMENT",
                "help": desc,
                "default": child.default if child.default else None,
                "type": child.type if child.type else None,
                "units": child.units if child.units else None
            }
            if child.union_value:
                args_item["union-value"] = transform_union_value(child.union_value)
            if child.type == 'enumeration' and child.enum:
                process_argument_enum(args_item, child.enum, child.name)
            args.append(args_item)
            yang_nodes[child.name] = f"{prefix_path}/{child.name}"

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
        commands = generate_commands(root)
        all_commands.extend(commands)

    print(json.dumps(all_commands, indent=4, cls=CustomJSONEncoder))

if __name__ == '__main__':
    main()
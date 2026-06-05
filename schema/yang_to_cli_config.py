#!/usr/bin/env python3
"""
YANG Schema to CLI Commands Config Generator
Parses YANG schema JSON (tree format) and outputs CLI command entries
conforming to the cli_commands_config.json contract.
"""

import json
import sys
import re
from typing import Dict, List, Any, Optional, Tuple

# ──────────────────────────── Node class ────────────────────────────
class Node:
    def __init__(self, name: str, definition: Dict[str, Any], parent: Optional['Node'] = None):
        self.name = name
        self.yangType = definition.get('yangType', '')
        self.ypath = definition.get('ypath', [])
        self.namespace = definition.get('namespace', '')
        self.description = definition.get('description', '')
        self.key = definition.get('key', '')
        self.type = definition.get('type', '')
        self.units = definition.get('units', '')
        self.config = definition.get('config', 'true')  # default true if missing
        self.enum = definition.get('enum', None)  # for enumerations
        self.union_value = definition.get('union-value', None)
        self.mandatory = definition.get('mandatory', 'false')
        self.parent = parent
        self.children: Dict[str, Node] = {}

    def add_child(self, child: 'Node'):
        self.children[child.name] = child
        child.parent = self

# ─────────────────────── Tree builder ─────────────────────────────
def build_tree(json_obj: Dict) -> Node:
    """Recursively build node tree from the JSON schema representation."""
    root_name = next(iter(json_obj))
    return _build_node(root_name, json_obj[root_name], parent=None)

def _build_node(name: str, obj: Dict, parent: Optional[Node]) -> Node:
    # Flat leaf format: properties at top level, no 'definition' wrapper, no dict children
    if 'yangType' in obj and obj.get('yangType') == 'leaf':
        node = Node(name, obj, parent)
        node.type = obj.get('type', '')
        node.units = obj.get('units', '')
        node.config = obj.get('config', 'true')
        node.description = obj.get('description', '')
        node.enum = obj.get('enum', None)
        node.union_value = obj.get('union-value', None)
        return node

    definition = obj.get('definition', {})
    node = Node(name, definition, parent)
    # Process children (skip 'definition' and any non-dict values)
    for key, value in obj.items():
        if key == 'definition':
            continue
        if not isinstance(value, dict):
            continue
        child = _build_node(key, value, parent=node)
        node.add_child(child)
    # For leaf nodes declared via definition wrapper
    if node.yangType == 'leaf':
        node.type = obj.get('type', definition.get('type', ''))
        node.units = obj.get('units', definition.get('units', ''))
        node.config = obj.get('config', definition.get('config', 'true'))
        node.description = obj.get('description', definition.get('description', ''))
        node.enum = obj.get('enum', definition.get('enum', None))
        node.union_value = obj.get('union-value', definition.get('union-value', None))
    return node

# ─────────────────────── Helper functions ─────────────────────────
def has_state_data(node: Node) -> bool:
    """Check if a list/container has a 'state' child with at least one config:false leaf or container with leaves."""
    state_node = node.children.get('state')
    if not state_node:
        # Some YANG models put state leaves directly under list without a 'state' container?
        # Not seen in examples; we rely on explicit 'state' container.
        return False
    # Count leaves inside state (recursively check for leaves)
    leaves = _collect_state_leaves(state_node)
    return len(leaves) > 0

def _collect_state_leaves(state_node: Node) -> List[Node]:
    """Return all leaf nodes under state that have config:false."""
    result = []
    for child in state_node.children.values():
        if child.yangType == 'leaf' and child.config == 'false':
            result.append(child)
        elif child.yangType == 'container':
            # recursively collect from container
            result.extend(_collect_state_leaves(child))
    return result

def is_leaf(node: Node) -> bool:
    return node.yangType == 'leaf'

def is_container(node: Node) -> bool:
    return node.yangType == 'container'

def is_list(node: Node) -> bool:
    return node.yangType == 'list'

def get_full_xpath(node: Node) -> str:
    """Build absolute XPath from ypath array, e.g. /openconfig-system:system/cpus/cpu"""
    parts = []
    for p in node.ypath:
        # If the part already contains a prefix (e.g. 'openconfig-system:system'), use as is.
        parts.append(p)
    return '/' + '/'.join(parts)

def get_root_layer_name(node: Node) -> str:
    """Return the first ypath element without namespace prefix."""
    first = node.ypath[0]
    if ':' in first:
        return first.split(':')[1]
    return first

def get_child_relative_xpath(node: Node) -> str:
    """Relative path from root to the target node (excluding root element)."""
    # ypath[1:] joined by '/'
    return '/'.join(node.ypath[1:])

def get_alignment_and_width(leaf: Node) -> Tuple[str, int]:
    """Determine alignment and width based on YANG type."""
    t = leaf.type
    # Check if type is numeric
    numeric_types = ['uint8', 'uint16', 'uint32', 'uint64',
                     'int8', 'int16', 'int32', 'int64',
                     'decimal64', 'timeticks64', 'dscp']
    if t in numeric_types:
        align = 'right'
        width = 12
    else:
        align = 'left'
        width = 20
    # Adjust width based on common patterns
    name = leaf.name.lower()
    if 'address' in name:
        width = 40
    elif 'port' in name:
        width = 10
    elif 'id' in name or 'index' in name:
        width = 12
    elif 'protocol' in name or 'encoding' in name:
        width = 20
    elif 'severity' in name or 'type' in name:
        width = 15
    elif 'description' in name or 'text' in name:
        width = 30
    elif 'interval' in name or 'time' in name:
        width = 20
    return align, width

def make_label(leaf: Node, prefix: str = '') -> str:
    """Generate human-readable column header."""
    raw = leaf.name.replace('-', ' ').replace('_', ' ')
    if prefix:
        raw = f"{prefix} {raw}"
    return raw.title()

def add_parent_key_fields(target: Node, fields: List[Dict], ypath: List[str], root_node: Node) -> None:
    """
    For a deeply nested list target, add fields referencing parent list keys.
    Example: for sensor-paths/sensor-path, add sensor-group-id from grandparent.
    """
    # Find ancestor lists between root container and target
    # ypath indices: 0 is root, target index = len(ypath)-1
    target_index = len(ypath) - 1
    # Start from index 1 (skip root) up to target_index-1
    current = root_node
    for i, part in enumerate(ypath):
        if i == 0:
            continue
        # Navigate to the node
        current = current.children[part] if part in current.children else None
        if current is None:
            break
        if i < target_index and is_list(current):
            # This is a parent list, add its key as a field
            key = current.key
            if key:
                # Construct relative path back to parent's state/key
                depth_diff = target_index - i
                back = '../' * depth_diff
                rel_path = f"{back}state/{key}"
                # Determine alignment/width
                state_node = current.children.get('state')
                key_node = state_node.children.get(key) if state_node else None
                align, width = ('left', 25) if key_node is None else get_alignment_and_width(key_node)
                fields.insert(0, {
                    "label": key.replace('-', ' ').replace('_', ' ').title(),
                    "relative_path": rel_path,
                    "width": width,
                    "align": align
                })

# ─────────────────────── Main generator ──────────────────────────
def generate_commands(node: Node) -> List[Dict]:
    """Walk the tree and generate command entries for all suitable targets."""
    commands = []

    def walk(n: Node):
        if (is_list(n) or is_container(n)) and has_state_data(n):
            # Check if this node should be a command target.
            # Exclude containers that are just grouping (like 'state' itself)
            if n.name != 'state' and n.name != 'config':
                commands.append(n)
        # Recurse into children
        for child in n.children.values():
            walk(child)

    walk(node)

    # Now generate entry for each target
    entries = []
    root_name = get_root_layer_name(node)
    for target in commands:
        entry = build_entry(target, root_name)
        entries.append(entry)
    return entries

def build_entry(target: Node, root_name: str) -> Dict:
    ypath = target.ypath
    xpath = get_full_xpath(target)
    child_rel_xpath = get_child_relative_xpath(target)
    command = target.name.replace('_', '-').lower()  # e.g., persistent-subscription
    table_title = target.description.split('.')[0] if target.description else target.name.replace('-', ' ').title()

    fields = []

    # Collect state leaves
    state_node = target.children.get('state')
    if state_node:
        for child in state_node.children.values():
            if child.yangType == 'leaf' and child.config == 'false':
                align, width = get_alignment_and_width(child)
                unit = child.units if child.units else None
                fields.append({
                    "label": make_label(child),
                    "relative_path": f"state/{child.name}",
                    "width": width,
                    "align": align,
                    "unit": unit
                })
            elif child.yangType == 'container':
                # Flatten container: include all leaves inside
                container = child
                for leaf in container.children.values():
                    if leaf.yangType == 'leaf' and leaf.config == 'false':
                        align, width = get_alignment_and_width(leaf)
                        unit = leaf.units if leaf.units else None
                        # For common pattern like cpu/usage, show 'instant' as main metric
                        # We include all leaves, but may produce a wide table.
                        fields.append({
                            "label": make_label(leaf, prefix=container.name.replace('-', ' ').title()),
                            "relative_path": f"state/{container.name}/{leaf.name}",
                            "width": width,
                            "align": align,
                            "unit": unit
                        })

    # Add parent list key references if target is nested
    add_parent_key_fields(target, fields, ypath, target)

    # If target is a container (not list) and has no key, set child_relative_xpath to the last segment
    # For single container like 'memory', child_relative_xpath should be just 'memory'
    # (already correctly set by get_child_relative_xpath)

    return {
        "parent": "display",
        "command": command,
        "callback_name": "cmd_show_data",
        "xpath": xpath,
        "help_text": target.description.split('.')[0] if target.description else f"Show {command} information",
        "root_layer_name": root_name,
        "child_relative_xpath": child_rel_xpath,
        "table_title": table_title,
        "table_line": 140,
        "fields": fields
    }

# ─────────────────────────── Main ──────────────────────────────────
def main():
    if len(sys.argv) != 2:
        print("Usage: python yang_to_cli_config.py <input_yang_schema.json>")
        sys.exit(1)

    with open(sys.argv[1], 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Unwrap outer "schema" wrapper if present (e.g. {"schema": {modules}})
    if set(data.keys()) == {'schema'}:
        data = data['schema']

    # If the top-level object contains multiple modules (e.g. {"system":..., "telemetry-system":...})
    all_commands = []
    for top_name, top_obj in data.items():
        root = build_tree({top_name: top_obj})
        commands = generate_commands(root)
        all_commands.extend(commands)

    print(json.dumps(all_commands, indent=4))

if __name__ == '__main__':
    main()

    # uv run yang_to_cli_config.py "test_ input_schema.json" > cli_commands_config.json 2>&1; echo "Exit code: $?"
import json


def convert_and_add_list(input_path, output_path):
    # 1. 读取原始 JSON 文件
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2. 定位到需要操作的核心层级
    target = data["used-service-port-type-preconf"]
    old_enum = target["enum"]

    new_enum = {}
    enum_list = []  # 用于存放转为数组的值

    # 3. 遍历转换
    for full_key, inner_val in old_enum.items():
        # 提取冒号后面的部分作为新 Key
        extracted_key = full_key.split(":")[-1] if ":" in full_key else full_key

        # 步骤 B: 去掉开头的 "PT_"
        # 如果字符串是以 "PT_" 开头，则将其替换为空（仅替换开头的第一次）
        if extracted_key.startswith("PT_"):
            new_key = extracted_key.replace("PT_", "", 1)
        else:
            new_key = extracted_key

        # 构建新的 enum 内部字典结构
        new_enum[new_key] = {
            "value": full_key,
            "description": inner_val.get("description", ""),
        }

        # 将短键名追加到数组中
        enum_list.append(new_key)

    # 4. 将更新后的数据写回结构中
    target["enum"] = new_enum
    target["enum-list"] = enum_list  # 新增的数组字段

    # 5. 保存到新的 JSON 文件
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"转换成功！新文件已保存至: {output_path}")


# ========================================================
# 测试运行（为了方便您直接复制运行，这里写一个创建测试文件的逻辑）
# ========================================================
if __name__ == "__main__":
    # 创建一个测试用的输入文件 input.json
    test_data = {
        "used-service-port-type-preconf": {
            "yangType": "leaf",
            "namespace": "openconfig-platform-transceiver",
            "type": "enumeration",
            "enum": {
              "openconfig-transport-types:PT_2M": {
                "description": "Ethernet 2Mbps"
              },
              "openconfig-transport-types:PT_100M": {
                "description": "Ethernet 100Mbps"
              },
              "openconfig-transport-types:PT_1000M": {
                "description": "Ethernet 1000Mbps"
              },
              "openconfig-transport-types:PT_10GBASE": {
                "description": "Ethernet 100GBASE"
              },
              "openconfig-transport-types:PT_100GBASE": {
                "description": "Ethernet 100GBASE"
              },
              "openconfig-transport-types:PT_200G": {
                "description": "Ethernet 200G"
              },
              "openconfig-transport-types:PT_400G": {
                "description": "Ethernet 400G"
              },
              "openconfig-transport-types:PT_600G": {
                "description": "Ethernet 600G"
              },
              "openconfig-transport-types:PT_STM64": {
                "description": "STM-64"
              },
              "openconfig-transport-types:PT_FC8G": {
                "description": "FC8G"
              },
              "openconfig-transport-types:PT_FC10G": {
                "description": "FC10G"
              },
              "openconfig-transport-types:PT_FC16G": {
                "description": "FC16G"
              },
              "openconfig-transport-types:PT_FC32G": {
                "description": "FC32G"
              },
              "openconfig-transport-types:PT_OTU2": {
                "description": "OTU2"
              },
              "openconfig-transport-types:PT_OTU4": {
                "description": "OTU4"
              },
              "openconfig-transport-types:PT_OTUC2": {
                "description": "OTUC2"
              },
              "openconfig-transport-types:PT_OTUC4": {
                "description": "OTUC4"
              },
              "openconfig-transport-types:PT_OTUC8": {
                "description": "OTUC8"
              },
              "openconfig-transport-types:PT_OTUC12": {
                "description": "OTUC12"
              }
            },
            "description": "Indicates the service type of optical transceiver used on this\nport. It supports configuration on client ports and line ports.\nParameters include form-factor-preconf and ethernet-pmd-preconf\nare obsoleted. ",
            "config": "false"
          }
    }
    with open("input.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2)

    # 执行转换函数
    convert_and_add_list("input.json", "output.json")
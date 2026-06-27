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
        # new_enum[new_key] = {
        #     "value": full_key,
        #     "description": inner_val.get("description", ""),
        # }
        new_enum[full_key] = {
            "value": new_key,
            "description": inner_val.get("description", ""),
        }

        # 将短键名追加到数组中
        enum_list.append(full_key)

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
               "openconfig-ospf-types:DOWN": {
                            "description": "The initial state of a neighbor, indicating that no recent\ninformation has been received from the neighbor."
                        },
                        "openconfig-ospf-types:ATTEMPT": {
                            "description": "Utilised for neighbors that are attached to NBMA networks, it\nindicates that no information has been recently received from\nthe neighbor but that Hello packets should be directly sent\nto that neighbor."
                        },
                        "openconfig-ospf-types:INIT": {
                            "description": "Indicates that a Hello packet has been received from the\nneighbor but bi-directional communication has not yet been\nestablished. That is to say that the local Router ID does\nnot appear in the list of neighbors in the remote system's\nHello packet."
                        },
                        "openconfig-ospf-types:TWO_WAY": {
                            "description": "Communication between the local and remote system is\nbi-directional such that the local system's Router ID is listed\nin the received remote system's Hello packet."
                        },
                        "openconfig-ospf-types:EXSTART": {
                            "description": "An adjacency with the remote system is being formed. The local\nsystem is currently transmitting empty database description\npackets in order to establish the master/slave relationship for\nthe adjacency."
                        },
                        "openconfig-ospf-types:EXCHANGE": {
                            "description": "The local and remote systems are currently exchanging database\ndescription packets in order to determine which elements of\ntheir local LSDBs are out of date."
                        },
                        "openconfig-ospf-types:LOADING": {
                            "description": "The local system is sending Link State Request packets to the\nremote system in order to receive the more recently LSAs that\nwere discovered during the Exchange phase of the procedure\nestablishing the adjacency."
                        },
                        "openconfig-ospf-types:FULL": {
                            "description": "The neighboring routers are fully adjacent such that both\nLSDBs are synchronized. The adjacency will appear in Router and\nNetwork LSAs"
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
import json
import os
import re

def extract_image_number(image_value):
    """从 image 字段中提取编号。"""
    if isinstance(image_value, str):
        match = re.search(r'_([0-9]+)(?:\.jpg|\.png|\.jpeg|\.gif|\.bmp|\.webp)?$', image_value)
        if match:
            return int(match.group(1))
    elif isinstance(image_value, list) and len(image_value) == 2:
        # 对于列表，我们主要关心第二项的编号，因为它将是下一个数据项的起点
        second_img = image_value[1]
        match = re.search(r'_([0-9]+)(?:\.jpg|\.png|\.jpeg|\.gif|\.bmp|\.webp)?$', second_img)
        if match:
            return int(match.group(1))
    return None

def get_previous_actions(conv_item):
    """从 conversations 中提取 previous actions 信息。"""
    for conv in conv_item.get("conversations", []):
        if conv.get("from") == "human":
            value = conv.get("value", "")
            # 查找 Previous actions 部分
            match = re.search(r'Previous actions:\s*(.*?)(?:\n|$)', value, re.DOTALL)
            if match:
                prev_actions_text = match.group(1).strip()
                if prev_actions_text.lower() == "none":
                     return "none"
                elif "step" in prev_actions_text.lower():
                     return "has_steps"
            # 如果没找到 "Previous actions:"，但内容中没有 "Step"，可能隐含是第一步
            # 但 "Previous actions: None" 是最明确的标志
    return "unknown" # 如果没有找到 human 条目或没有匹配，标记为未知

def split_trajectories(input_file_path, output_dir):
    """
    读取 JSON 文件（假设内容是一个列表），按轨迹分割并保存到指定目录。

    Args:
        input_file_path: 输入 JSON 文件的路径（内容应为一个 JSON 数组）。
        output_dir: 输出目录的路径。
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading from {input_file_path}...")
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            # 尝试将整个文件内容解析为一个 JSON 对象（预期是一个列表）
            all_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse {input_file_path} as a single JSON object. Error: {e}")
        print("The script expects the file to contain a JSON array of data items.")
        return
    except FileNotFoundError:
        print(f"Error: File {input_file_path} not found.")
        return

    if not isinstance(all_data, list):
        print(f"Error: The content of {input_file_path} is not a JSON array. Got {type(all_data)}.")
        return

    trajectories = []  # 存储所有找到的轨迹
    current_trajectory = [] # 当前正在构建的轨迹
    last_image_num = -1
    last_image_prefix = None
    trajectory_counter = 0 # 用于生成文件名

    print(f"Processing {len(all_data)} items...")

    for idx, data_item in enumerate(all_data):
        # 确保 data_item 是一个字典
        if not isinstance(data_item, dict):
             print(f"Warning: Item at index {idx} is not a dictionary, skipping.")
             continue

        # 1. 检查 Previous Actions (最可靠的信号)
        prev_actions_status = get_previous_actions(data_item)
        is_new_trajectory_start = (prev_actions_status == "none")

        # 2. 检查 Image 信息 (辅助判断，处理列表情况)
        current_image_val = data_item.get("image")
        current_image_num = extract_image_number(current_image_val)

        # 提取前缀用于连续性检查 (处理列表情况)
        current_image_prefix = None
        if isinstance(current_image_val, str):
            prefix_match = re.match(r'^(.+?)_[0-9]+(?:\.[^.]*)?$', current_image_val)
            if prefix_match:
                current_image_prefix = prefix_match.group(1)
        elif isinstance(current_image_val, list) and len(current_image_val) == 2:
            # 检查列表第一项是否延续了上一个
            first_img = current_image_val[0]
            first_prefix_match = re.match(r'^(.+?)_[0-9]+(?:\.[^.]*)?$', first_img)
            if first_prefix_match:
                first_img_prefix = first_prefix_match.group(1)
                # 如果列表第一项的前缀和上一个前缀不同，或者编号不连续，可能意味着新轨迹
                if last_image_prefix is not None and last_image_num is not None:
                    expected_first_num = last_image_num + 1
                    first_img_num = extract_image_number(first_img)
                    if (first_img_prefix != last_image_prefix or first_img_num != expected_first_num):
                        print(f"Image discontinuity (list start) at index {idx}: {last_image_prefix}_{last_image_num} -> {first_img_prefix}_{first_img_num}. Assuming new trajectory.")
                        is_new_trajectory_start = True # 强制认为是新轨迹开始
                # 列表第二项的前缀和编号用于下一次迭代
                second_img = current_image_val[1]
                second_prefix_match = re.match(r'^(.+?)_[0-9]+(?:\.[^.]*)?$', second_img)
                if second_prefix_match:
                    current_image_prefix = second_prefix_match.group(1)
                # current_image_num 对于列表，我们之前定义为第二项的编号
                # current_image_num = extract_image_number(second_img) # 这句在上面已经赋值
        else: # image 既不是字符串也不是列表，或者列表长度不对
             print(f"Warning: Item at index {idx} has unexpected 'image' format: {type(current_image_val)}. Skipping.")
             continue


        # 如果 previous actions 是 "none"，则明确是新轨迹开始
        if is_new_trajectory_start:
             # 保存上一条轨迹（如果存在）
             if current_trajectory:
                 trajectories.append(current_trajectory)
                 print(f"Found trajectory {len(trajectories)} ending at index {idx-1} (Previous Actions: None).")
             # 开始新轨迹
             current_trajectory = [data_item]
             # 重置上一个图片信息，基于当前项
             if isinstance(current_image_val, str):
                 last_image_num = current_image_num
                 last_image_prefix = current_image_prefix
             elif isinstance(current_image_val, list) and len(current_image_val) == 2:
                 # 如果当前项是列表，下一项应接列表的第二项
                 second_img_num = extract_image_number(current_image_val[1])
                 second_img_prefix = None
                 second_prefix_match = re.match(r'^(.+?)_[0-9]+(?:\.[^.]*)?$', current_image_val[1])
                 if second_prefix_match:
                     second_img_prefix = second_prefix_match.group(1)
                 last_image_num = second_img_num
                 last_image_prefix = second_img_prefix
             continue # 处理完新轨迹起点，继续下一行

        # 如果 previous actions 不是 "none"
        if current_trajectory: # 如果当前轨迹非空
            # 检查编号是否连续 (在 previous actions 不为 none 时)
            # 这主要用于检测 previous actions 信息不明确或错误的情况
            expected_next_num = last_image_num + 1
            if current_image_num is not None and last_image_prefix is not None:
                if current_image_num != expected_next_num or current_image_prefix != last_image_prefix:
                    # 编号不连续或前缀不同，且 previous actions 不是 none
                    # 这可能表示 previous actions 信息有误，或者确实是两个独立的步骤序列被混在一起
                    # 但根据您的描述，previous actions 应该是可靠的
                    # 如果 previous actions 不是 none，我们倾向于认为是连续的，除非 image 跳跃非常大
                    # 让我们打印一个警告，但仍然将其加入当前轨迹，除非跳跃非常不合理
                    print(f"Warning: Potential discontinuity at index {idx} (Prev Actions: {prev_actions_status}): {last_image_prefix}_{last_image_num} -> {current_image_prefix}_{current_image_num}. Adding to current trajectory anyway.")
                    # 如果跳跃过大，可以考虑强制开启新轨迹，但这里先保持连续
                    # if abs(current_image_num - expected_next_num) > 10: # 例如，跳跃超过10个编号
                    #     print(f"Large jump detected, assuming new trajectory.")
                    #     trajectories.append(current_trajectory)
                    #     current_trajectory = [data_item]
                    #     last_image_num = current_image_num
                    #     last_image_prefix = current_image_prefix
                    #     continue

            # 如果连续性检查通过，或者虽然有警告但决定保持连续，则添加到当前轨迹
            current_trajectory.append(data_item)
            # 更新 last_image_num 和 last_image_prefix
            # 如果当前是列表，下一项应该接列表的第二项
            if isinstance(current_image_val, list) and len(current_image_val) == 2:
                last_image_num = extract_image_number(current_image_val[1])
                last_img_prefix_match = re.match(r'^(.+?)_[0-9]+(?:\.[^.]*)?$', current_image_val[1])
                if last_img_prefix_match:
                    last_image_prefix = last_img_prefix_match.group(1)
            else: # 是字符串
                last_image_num = current_image_num
                last_image_prefix = current_image_prefix

        else: # 当前轨迹为空，且 previous actions 不是 none (理论上不应发生，因为新轨迹应由 prev_actions=none 触发)
             # 这种情况可能意味着文件开头不是以 "Previous actions: None" 开始
             # 我们可以假设第一个数据项是一个新轨迹的开始
             if not trajectories: # 这是第一个数据项
                 print(f"Starting first trajectory at index {idx} (no previous actions check).")
                 current_trajectory = [data_item]
                 if isinstance(current_image_val, str):
                     last_image_num = current_image_num
                     last_image_prefix = current_image_prefix
                 elif isinstance(current_image_val, list) and len(current_image_val) == 2:
                     last_image_num = extract_image_number(current_image_val[1])
                     last_img_prefix_match = re.match(r'^(.+?)_[0-9]+(?:\.[^.]*)?$', current_image_val[1])
                     if last_img_prefix_match:
                         last_image_prefix = last_img_prefix_match.group(1)
             else:
                 # 如果之前已经有轨迹，但 current_trajectory 为空，且当前项 prev_actions 不是 none
                 print(f"Warning: Item at index {idx} seems to start a new trajectory implicitly (Prev Actions: {prev_actions_status}).")
                 current_trajectory = [data_item]
                 if isinstance(current_image_val, str):
                     last_image_num = current_image_num
                     last_image_prefix = current_image_prefix
                 elif isinstance(current_image_val, list) and len(current_image_val) == 2:
                     last_image_num = extract_image_number(current_image_val[1])
                     last_img_prefix_match = re.match(r'^(.+?)_[0-9]+(?:\.[^.]*)?$', current_image_val[1])
                     if last_img_prefix_match:
                         last_image_prefix = last_img_prefix_match.group(1)

    # 添加最后一个轨迹
    if current_trajectory:
        trajectories.append(current_trajectory)
        print(f"Found trajectory {len(trajectories)} ending at the last index.")

    print(f"Total trajectories found: {len(trajectories)}")

    # 保存轨迹到单独的文件
    for i, traj in enumerate(trajectories):
        output_filename = os.path.join(output_dir, f"trajectory_{i+1:04d}.json")
        with open(output_filename, 'w', encoding='utf-8') as out_f:
            # 写入整个 traj 数组作为一个 JSON 对象
            json.dump(traj, out_f, ensure_ascii=False, indent=2)
        print(f"Saved trajectory {i+1} ({len(traj)} items) to {output_filename}")


if __name__ == "__main__":
    
    selected_datasets = ['coat.json',
                         'android_control.json'
                         ]
    for dataset in selected_datasets:
        print(f"Processing dataset: {dataset}")
        input_file_path = "/mnt/home/user14/archive/zhangyaoyin/datasets/aguvis-stage2/" + dataset
        output_dir = "/mnt/home/user14/archive/zhangyaoyin/datasets/aguvis-stage2/processed_dataset/" + dataset.split('.')[0]
        split_trajectories(input_file_path, output_dir)
        print(f"Trajectory splitting complete.\n\n")
# import os
# import pandas as pd
# from collections import defaultdict

# # 加载文档类别信息
# csv_path = '../data/12_RGB_FullyLabeled_640/coco/class12_RGB_all_L.csv'
# df_classes = pd.read_csv(csv_path)
# doc_info = {row['Species_ID']: {'name': row['Species_Class'], 'doc_count': row['Labels']} 
#             for _, row in df_classes.iterrows()}

# # 标签目录路径
# train_label_dir = '../data/12_RGB_FullyLabeled_640/coco/train/labels'
# val_label_dir = '../data/12_RGB_FullyLabeled_640/coco/val/labels'

# def analyze_labels(label_dir):
#     """统计标签文件中的类别分布（0-based ID）"""
#     label_counts = defaultdict(int)
#     for filename in os.listdir(label_dir):
#         if not filename.endswith('.txt'):
#             continue
#         with open(os.path.join(label_dir, filename), 'r') as f:
#             for line in f:
#                 if line.strip():
#                     try:
#                         class_id = int(line.split()[0])
#                         label_counts[class_id] += 1
#                     except (ValueError, IndexError):
#                         continue
#     return dict(sorted(label_counts.items()))

# # 统计并合并训练集和验证集
# train_counts = analyze_labels(train_label_dir)
# val_counts = analyze_labels(val_label_dir)
# combined_counts = defaultdict(int)
# for label_id in set(train_counts) | set(val_counts):
#     combined_counts[label_id] = train_counts.get(label_id, 0) + val_counts.get(label_id, 0)

# # 生成匹配报告
# mismatches = []
# match_results = []

# for doc_id in sorted(doc_info.keys()):
#     label_id = doc_id - 1  # 文档ID转标签ID
#     actual_total = combined_counts.get(label_id, 0)
#     expected_total = doc_info[doc_id]['doc_count']
    
#     match_status = "匹配" if actual_total == expected_total else "不匹配"
#     if actual_total != expected_total:
#         mismatches.append((doc_id, expected_total, actual_total))
    
#     match_results.append({
#         '文档ID': doc_id,
#         '标签ID': label_id,
#         '类别名称': doc_info[doc_id]['name'],
#         '文档记录数': expected_total,
#         '实际统计数': actual_total,
#         '状态': match_status,
#         '训练集样本': train_counts.get(label_id, 0),
#         '验证集样本': val_counts.get(label_id, 0)
#     })

# # 输出匹配报告
# print("===== ID匹配验证报告 =====")
# print(f"总类别数（文档）: {len(doc_info)}")
# print(f"实际统计类别数: {len(combined_counts)}")
# print(f"匹配率: {(len(match_results)-len(mismatches))/len(match_results)*100:.1f}%\n")

# # 打印详细匹配表格
# match_df = pd.DataFrame(match_results)
# print(match_df.to_string(index=False))

# # 输出不匹配详情
# if mismatches:
#     print("\n===== 不匹配详情 =====")
#     for doc_id, expected, actual in sorted(mismatches):
#         print(f"文档ID {doc_id}({doc_info[doc_id]['name']}): 文档记录={expected} ≠ 实际统计={actual} (差{expected-actual:+d})")

# # 特殊检查类别36（标签ID 35）
# label_id_36 = 35
# doc_id_36 = 36
# print("\n===== 特殊检查 =====")
# print(f"类别 {doc_id_36}({doc_info[doc_id_36]['name']}):")
# print(f"  文档记录数: {doc_info[doc_id_36]['doc_count']}")
# print(f"  实际统计数: {combined_counts.get(label_id_36, 0)}")
# print(f"  训练集样本: {train_counts.get(label_id_36, 0)}")
# print(f"  验证集样本: {val_counts.get(label_id_36, 0)}")
# if val_counts.get(label_id_36, 0) == 0:
#     print("  ⚠️ 验证集样本为0")

# # 样本分布分析
# print("\n===== 样本分布分析 =====")
# total_actual = sum(combined_counts.values())
# total_doc = sum(info['doc_count'] for info in doc_info.values())
# print(f"文档总样本数: {total_doc}")
# print(f"实际总样本数: {total_actual}")
# print(f"差异: {total_doc - total_actual:+d}")

# min_samples = min(combined_counts.values())
# max_samples = max(combined_counts.values())
# print(f"\n样本数范围: {min_samples}-{max_samples}")
# print(f"最不平衡比例: {max_samples/min_samples:.1f}:1")
# print("样本最少的5个类别:")
# for label_id, count in sorted(combined_counts.items(), key=lambda x: x[1])[:5]:
#     print(f"  {doc_info[label_id+1]['name']}(ID:{label_id+1}): {count}样本")

import os
import pandas as pd
from collections import defaultdict
from difflib import get_close_matches

# 加载文档类别信息
csv_path = '../data/12_RGB_FullyLabeled_640/coco/class12_RGB_all_L.csv'
df_classes = pd.read_csv(csv_path)
doc_info = {row['Species_ID']: {'name': row['Species_Class'], 'doc_count': row['Labels']} 
            for _, row in df_classes.iterrows()}

# 标签目录路径
train_label_dir = '../data/12_RGB_FullyLabeled_640/coco/train/labels'
val_label_dir = '../data/12_RGB_FullyLabeled_640/coco/val/labels'

def analyze_labels(label_dir):
    """统计标签文件中的类别分布"""
    label_counts = defaultdict(int)
    for filename in os.listdir(label_dir):
        if not filename.endswith('.txt'):
            continue
        with open(os.path.join(label_dir, filename), 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        class_id = int(line.split()[0])
                        label_counts[class_id] += 1
                    except (ValueError, IndexError):
                        continue
    return dict(sorted(label_counts.items()))

# 统计并合并训练集和验证集
train_counts = analyze_labels(train_label_dir)
val_counts = analyze_labels(val_label_dir)
combined_counts = {label_id: train_counts.get(label_id, 0) + val_counts.get(label_id, 0)
                  for label_id in set(train_counts) | set(val_counts)}

# 构建数量映射关系用于匹配
doc_counts = {doc_id: info['doc_count'] for doc_id, info in doc_info.items()}
label_to_doc = {}

# 第一阶段：精确数量匹配
remaining_labels = set(combined_counts.items())
remaining_docs = set(doc_counts.items())

for (label_id, label_count), (doc_id, doc_count) in zip(
    sorted(remaining_labels, key=lambda x: -x[1]),
    sorted(remaining_docs, key=lambda x: -x[1])
):
    if label_count == doc_count:
        label_to_doc[label_id] = doc_id
        remaining_labels.remove((label_id, label_count))
        remaining_docs.remove((doc_id, doc_count))

# 第二阶段：模糊匹配剩余项
for label_id, label_count in sorted(remaining_labels, key=lambda x: -x[1]):
    closest = get_close_matches(str(label_count), 
                              [str(c) for _, c in remaining_docs], 
                              n=1, cutoff=0.8)
    if closest:
        matched_doc_id = next(doc_id for doc_id, count in remaining_docs 
                            if str(count) == closest[0])
        label_to_doc[label_id] = matched_doc_id
        remaining_docs.remove((matched_doc_id, int(closest[0])))

# 生成匹配报告
match_results = []
unmatched_labels = []
unmatched_docs = []

for label_id in sorted(combined_counts.keys()):
    doc_id = label_to_doc.get(label_id)
    if doc_id:
        match_status = (
            "完全匹配" if combined_counts[label_id] == doc_info[doc_id]['doc_count']
            else f"差异{combined_counts[label_id]-doc_info[doc_id]['doc_count']:+d}"
        )
        match_results.append({
            '标签ID': label_id,
            '文档ID': doc_id,
            '类别名称': doc_info[doc_id]['name'],
            '标签总数': combined_counts[label_id],
            '文档总数': doc_info[doc_id]['doc_count'],
            '匹配状态': match_status,
            '训练集': train_counts.get(label_id, 0),
            '验证集': val_counts.get(label_id, 0)
        })
    else:
        unmatched_labels.append(label_id)

for doc_id in doc_info:
    if doc_id not in label_to_doc.values():
        unmatched_docs.append(doc_id)

# 输出匹配报告
print("===== 基于数量的ID匹配报告 =====")
print(f"成功匹配: {len(match_results)}/{len(combined_counts)}个标签类别")
print(f"未匹配标签ID: {sorted(unmatched_labels)}")
print(f"未匹配文档ID: {sorted(unmatched_docs)}\n")

# 打印匹配详情
match_df = pd.DataFrame(match_results)
print(match_df.to_string(index=False))

# 特殊检查
print("\n===== 特殊检查 =====")
print("1. 验证集样本为0的类别:")
zero_val_classes = [
    (label_id, label_to_doc[label_id], doc_info[label_to_doc[label_id]]['name'])
    for label_id in combined_counts 
    if label_id in label_to_doc and val_counts.get(label_id, 0) == 0
]
for label_id, doc_id, name in zero_val_classes:
    print(f"  标签ID {label_id} -> 文档ID {doc_id}({name}): 验证集0样本")

print("\n2. 数量差异最大的5个类别:")
top_mismatches = sorted(
    [r for r in match_results if not r['匹配状态'].startswith('完全')],
    key=lambda x: abs(x['标签总数'] - x['文档总数']),
    reverse=True)[:5]
for item in top_mismatches:
    print(f"  标签ID {item['标签ID']} -> 文档ID {item['文档ID']}({item['类别名称']}): "
          f"标签数={item['标签总数']}, 文档数={item['文档总数']}")

# 保存匹配关系
pd.DataFrame([{'label_id': k, 'doc_id': v} for k, v in label_to_doc.items()])\
  .to_csv('id_mapping.csv', index=False)
print("\n匹配关系已保存到 id_mapping.csv")
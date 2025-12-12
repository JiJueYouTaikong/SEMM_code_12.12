import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from datetime import datetime

# 获取当前时间
now = datetime.now()

# 按所需格式转换为字符串
formatted_time = now.strftime("%Y-%m-%d-%H-%M-%S")


def load_predictions():
    """加载所有预测数据，并确保'Ours'排在最后"""
    methods = {

        "SUE-GB": "./可视化/测试集TNN/Pred-SUE-GB.npy",
        "SSM": "./可视化/测试集TNN/Pred_SSM_SUE_MSA_6.15_7.80.npy",
        # "DeepGravity": "./可视化/测试集TNN/Pred_DeepGravity_6.88.npy",
        "DeepGravity": "./可视化/测试集TNN/Pred_DLNa.npy",
        "GPT2": "./可视化/测试集TNN/Pred_GPT2.npy",

        "Ours": "./可视化/测试集TNN/Pred_RED-5.7127.npy"
    }
    predictions = {}
    for name, path in methods.items():
        if os.path.exists(path):
            predictions[name] = np.load(path)
        else:
            print(f"警告：找不到文件 {path}，跳过该方法。")

    # 确保 Ours 始终在最后
    sorted_predictions = {}
    for key in predictions:
        if key != "Ours":
            sorted_predictions[key] = predictions[key]
    if "Ours" in predictions:
        sorted_predictions["Ours"] = predictions["Ours"]

    return sorted_predictions


def load_real_data():
    try:
        return np.load('./可视化/测试集TNN/真实值.npy')
    except FileNotFoundError:
        print("真实值文件未找到，请检查路径。")
        return None

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

def plot_heatmaps(predictions_dict, real_od_t, fontsize=16):
    method_names = list(predictions_dict.keys())

    # === 所有时间步累计 ===
    all_real_od = np.sum(real_od_t, axis=0)  # shape: [origin, dest]
    all_pred_od_list = [np.sum(pred, axis=0) for pred in predictions_dict.values()]

    # === 自定义颜色映射 ===
    boundaries = [0, 10, 100, 500, 1000, np.inf]
    # 蓝色系
    cmap = ListedColormap(["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b", "#081d58"])

    # cmap = ListedColormap(["#f2eae7","#edd3d3", "#e8b3bc", "#db6882", "#c16277", "#875861"])

    norm = BoundaryNorm(boundaries, cmap.N, extend='max')

    # === 子图布局 ===
    total_plots = len(all_pred_od_list) + 1  # +1 for real OD
    rows = (total_plots + 2) // 3
    cols = min(3, total_plots)

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6.5 * rows))
    axes = axes.flatten()

    # === 绘制预测图 ===
    for i, (name, pred_od) in enumerate(zip(method_names, all_pred_od_list)):
        ax = axes[i]
        sns.heatmap(pred_od, cmap=cmap, norm=norm, cbar=False, ax=ax)
        ax.set_xlabel("Destination", fontsize=12)
        ax.set_ylabel("Origin", fontsize=12)
        ax.text(0.5, -0.14, f"({chr(97 + i)}) {name}", transform=ax.transAxes,
                ha='center', va='top', fontsize=fontsize)

    # === 绘制真实图 ===
    ax_real = axes[len(all_pred_od_list)]
    sns.heatmap(all_real_od, cmap=cmap, norm=norm, cbar=False, ax=ax_real)
    ax_real.set_xlabel("Destination", fontsize=12)
    ax_real.set_ylabel("Origin", fontsize=12)
    ax_real.text(0.5, -0.14, f"({chr(97 + len(all_pred_od_list))}) True OD", transform=ax_real.transAxes,
                 ha='center', va='top', fontsize=fontsize)

    # === 隐藏多余子图 ===
    for i in range(total_plots, len(axes)):
        axes[i].axis('off')

    # === 添加图注（legend） ===
    legend_labels = ['0–10', '10–100', '100–500', '500–1000', '1000+']
    legend_colors = cmap.colors
    patches = [mpatches.Patch(color=legend_colors[i], label=legend_labels[i]) for i in range(len(legend_labels))]
    fig.legend(handles=patches, loc='upper center', ncol=len(patches), fontsize=fontsize, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f'./可视化/结果图/ODE_1-{formatted_time}.png', format='png', dpi=300, bbox_inches='tight')



# def plot_heatmaps(predictions_dict, real_od_t, fontsize=16):
#     method_names = list(predictions_dict.keys())
#
#     # 查找真实值总和最大的时间步
#     od_sums = np.sum(real_od_t, axis=(1, 2))
#     t_max = np.argmax(od_sums)
#     t_max = 20
#     print(f"Max OD timestep: {t_max}")
#
#     # 取该时间步的真实 OD 和预测 OD
#     all_real_od = real_od_t[t_max]
#     all_pred_od_list = [pred[t_max] for pred in predictions_dict.values()]
#
#     vmin = 0
#     vmax = max([pred.max() for pred in all_pred_od_list] + [all_real_od.max()])
#
#     # 子图数量和布局
#     total_plots = len(all_pred_od_list) + 1  # 包括真实 OD
#     rows = (total_plots + 2) // 3
#     cols = min(3, total_plots)
#
#     # 创建子图
#     fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5.5 * rows))
#     axes = axes.flatten()
#     color = "Reds"  # coolwarm Blues
#     # 绘制预测图（在前面）
#     for i, (name, pred_od) in enumerate(zip(method_names, all_pred_od_list)):
#         ax = axes[i]
#         sns.heatmap(pred_od, cmap=color, cbar=True, vmin=vmin, vmax=vmax, ax=ax)
#         ax.set_xlabel("Destination", fontsize=12)
#         ax.set_ylabel("Origin", fontsize=12)
#         ax.text(0.5, -0.14, f"({chr(97 + i)}) {name}", transform=ax.transAxes,
#                 ha='center', va='top', fontsize=fontsize)
#
#     # 绘制真实 OD 图（在最后）
#     ax_real = axes[len(all_pred_od_list)]
#     sns.heatmap(all_real_od, cmap=color, cbar=True, vmin=vmin, vmax=vmax, ax=ax_real)
#     ax_real.set_xlabel("Destination", fontsize=12)
#     ax_real.set_ylabel("Origin", fontsize=12)
#     ax_real.text(0.5, -0.14, f"({chr(97 + len(all_pred_od_list))}) True OD", transform=ax_real.transAxes,
#                  ha='center', va='top', fontsize=fontsize)
#
#     # 隐藏多余子图
#     for i in range(total_plots, len(axes)):
#         axes[i].axis('off')
#
#     plt.tight_layout()
#     # plt.savefig("./可视化/结果图/ODE_1-{formatted_time}.png")
#     fig.savefig(f'./可视化/结果图/ODE_1-{formatted_time}.png', format='png', dpi=300, bbox_inches='tight')


# def plot_heatmaps(predictions_dict, real_od_t):
#     method_names = list(predictions_dict.keys())
#     all_pred_od_list = [np.mean(predictions_dict[name], axis=0) for name in method_names]
#     all_real_od = np.mean(real_od_t, axis=0)
#
#     vmin = min([pred.min() for pred in all_pred_od_list] + [all_real_od.min()])
#     vmax = max([pred.max() for pred in all_pred_od_list] + [all_real_od.max()])
#
#     fig, axes = plt.subplots(1, len(all_pred_od_list) + 1, figsize=(6 * (len(all_pred_od_list) + 1), 6))
#     sns.heatmap(all_real_od, cmap="Blues", cbar=True, vmin=vmin, vmax=vmax, ax=axes[0])
#     axes[0].set_title("Average True OD Matrix", fontsize=18)
#
#     for i, (name, pred_od) in enumerate(zip(method_names, all_pred_od_list)):
#         sns.heatmap(pred_od, cmap="Blues", cbar=True, vmin=vmin, vmax=vmax, ax=axes[i + 1])
#         axes[i + 1].set_title(f"Predicted ({name})", fontsize=18)
#
#     plt.tight_layout()
#     plt.savefig("./可视化/结果图/ODE_1-{formatted_time}.png")

def plot_heatmaps_sub(predictions_dict, real_od_t, fontsize=16):
    method_names = list(predictions_dict.keys())

    # 查找真实值总和最大的时间步
    od_sums = np.sum(real_od_t, axis=(1, 2))
    t_max = np.argmax(od_sums)
    t_max = 20
    print(f"Max OD timestep: {t_max}")

    # 取该时间步的真实 OD 和预测 OD
    all_real_od = real_od_t[t_max]
    all_pred_od_list = [pred[t_max] for pred in predictions_dict.values()]

    # ==== 新增：选取显著的 80x80 子网格 ====
    # 以真实OD为基准，找到行和列的重要度（流量和）
    row_sum = np.sum(all_real_od, axis=1)
    col_sum = np.sum(all_real_od, axis=0)

    # 找到流量和最大的前 80 个行和列索引
    top_rows = np.argsort(row_sum)[-50:]
    top_cols = np.argsort(col_sum)[-50:]

    # 排序保证热力图更可读（可选）
    top_rows = np.sort(top_rows)
    top_cols = np.sort(top_cols)

    # 根据选中的行列索引裁剪所有 OD
    all_real_od_sub = all_real_od[np.ix_(top_rows, top_cols)]
    all_pred_od_list_sub = [
        pred_od[np.ix_(top_rows, top_cols)] for pred_od in all_pred_od_list
    ]

    vmin = 0
    vmax = max([pred.max() for pred in all_pred_od_list_sub] + [all_real_od_sub.max()])

    # 子图数量和布局
    total_plots = len(all_pred_od_list_sub) + 1  # 包括真实 OD
    rows = (total_plots + 2) // 3
    cols = min(3, total_plots)

    # 创建子图
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5.5 * rows))
    axes = axes.flatten()
    color = "Reds"  # 可以换成 "coolwarm" 等

    # 绘制预测热力图
    for i, (name, pred_od) in enumerate(zip(method_names, all_pred_od_list_sub)):
        ax = axes[i]
        sns.heatmap(pred_od, cmap=color, cbar=True, vmin=vmin, vmax=vmax, ax=ax)
        ax.set_xlabel("Destination", fontsize=12)
        ax.set_ylabel("Origin", fontsize=12)
        ax.text(0.5, -0.14, f"({chr(97 + i)}) {name}", transform=ax.transAxes,
                ha='center', va='top', fontsize=fontsize)

    # 绘制真实 OD 热力图
    ax_real = axes[len(all_pred_od_list_sub)]
    sns.heatmap(all_real_od_sub, cmap=color, cbar=True, vmin=vmin, vmax=vmax, ax=ax_real)
    ax_real.set_xlabel("Destination", fontsize=12)
    ax_real.set_ylabel("Origin", fontsize=12)
    ax_real.text(0.5, -0.14, f"({chr(97 + len(all_pred_od_list_sub))}) True OD", transform=ax_real.transAxes,
                 ha='center', va='top', fontsize=fontsize)

    # 隐藏多余子图
    for i in range(total_plots, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    fig.savefig(f'./可视化/结果图/ODE_2-{formatted_time}.png', format='png', dpi=300, bbox_inches='tight')


def plot_scatter(predictions_dict, real_od_t, fontsize=16):
    method_names = list(predictions_dict.keys())
    all_real_flat = real_od_t.flatten()
    all_pred_flat_list = [pred.flatten() for pred in predictions_dict.values()]
    max_val = max(np.max(all_real_flat), *[np.max(pred) for pred in all_pred_flat_list])
    lims = [0, max_val]

    # 计算子图的行列数，每行显示2个子图
    total_plots = len(all_pred_flat_list)
    rows = (total_plots + 1) // 3
    cols = min(3, total_plots)

    # 创建子图
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 8 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    # 设置颜色列表
    base_colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']

    for i, (ax, name, pred_flat) in enumerate(zip(axes, method_names, all_pred_flat_list)):
        # 特殊处理 'ours' 方法为黑色
        color = 'black' if 'ours' in name.lower() else base_colors[i % len(base_colors)]
        ax.scatter(all_real_flat, pred_flat, alpha=1, color=color)
        ax.plot(lims, lims, color=color, linestyle='-', linewidth=2)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("True OD", fontsize=12)
        ax.set_ylabel("Predicted OD", fontsize=12)
        ax.text(0.5, -0.08, f"({chr(97 + i)}) {name}", transform=ax.transAxes,
                ha='center', va='top', fontsize=fontsize)

        # 去除背景色、网格，并加边框
        ax.set_facecolor('white')
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)
        ax.tick_params(colors='black')

    # 隐藏多余子图
    for i in range(total_plots, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    # 设置子图间距
    # plt.subplots_adjust(hspace=0.3)  # 设置行间距，数值可调

    fig.savefig(f'./可视化/结果图/ODE_4-{formatted_time}.png', format='png', dpi=300, bbox_inches='tight')
    # fig.savefig(f'./可视化/结果图/ODE_4.pdf', format='pdf', dpi=300, bbox_inches='tight')


# # *绘制散点+拟合线*
# import numpy as np
# import matplotlib.pyplot as plt
#
# def plot_scatter(predictions_dict, real_od_t):
#     method_names = list(predictions_dict.keys())
#     all_real_flat = real_od_t.flatten()
#     all_pred_flat_list = [pred.flatten() for pred in predictions_dict.values()]
#     max_val = max(np.max(all_real_flat), *[np.max(pred) for pred in all_pred_flat_list])
#     lims = [0, max_val]
#
#     fig, axes = plt.subplots(1, len(all_pred_flat_list), figsize=(8 * len(all_pred_flat_list), 8))
#     if len(all_pred_flat_list) == 1:
#         axes = [axes]
#
#     for ax, name, pred_flat in zip(axes, method_names, all_pred_flat_list):
#         ax.scatter(all_real_flat, pred_flat, alpha=0.5)
#         ax.plot(lims, lims, 'r-', linewidth=2)
#
#         # 计算拟合线
#         coefficients = np.polyfit(all_real_flat, pred_flat, 1)  # 一次多项式拟合（直线）
#         poly = np.poly1d(coefficients)
#         x_fit = np.linspace(lims[0], lims[1], 100)
#         y_fit = poly(x_fit)
#         ax.plot(x_fit, y_fit, 'g--', linewidth=2, label=f'Fit: y = {coefficients[0]:.4f}x + {coefficients[1]:.4f}')
#
#         ax.set_xlim(lims)
#         ax.set_ylim(lims)
#         ax.set_title(f"Real vs Predicted ({name})", fontsize=18)
#         ax.set_xlabel("True OD", fontsize=14)
#         ax.set_ylabel("Predicted OD", fontsize=14)
#         ax.legend()
#
#     plt.tight_layout()
#     plt.savefig(f'./可视化/结果图/ODE_4-{formatted_time}.png')


if __name__ == "__main__":
    predictions = load_predictions()
    real_values = load_real_data()

    if predictions and real_values is not None:
        plot_heatmaps(predictions, real_values) ## 1
        # plot_heatmaps_sub(predictions, real_values)  ## 2
        # plot_scatter(predictions, real_values)  ## 4

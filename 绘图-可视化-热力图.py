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

        "GPT2": "./可视化/测试集TNN/Pred_GPT2.npy",
        "DeepGravity": "./可视化/测试集TNN/Pred_DeepGravity_6.88.npy",
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

def plot_heatmaps(predictions_dict, real_od_t, fontsize=16):
    method_names = list(predictions_dict.keys())

    # === 所有时间步累计 ===
    all_real_od = np.sum(real_od_t, axis=0)  # shape: [origin, dest]
    all_pred_od_list = [np.sum(pred, axis=0) for pred in predictions_dict.values()]

    # === 自定义颜色映射 ===
    # boundaries = [0, 10, 100, 500, 1000, np.inf]
    # # 蓝色系
    # cmap = ListedColormap(["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b", "#081d58"])

    boundaries = [0, 10, 100, 500, 1000, np.inf]
    # 蓝色系
    cmap = ListedColormap(["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b", "#081d58"])

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
    ax_real.text(0.5, -0.14, f"({chr(97 + len(all_pred_od_list))}) Ground Truth", transform=ax_real.transAxes,
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
    # fig.savefig(f'./可视化/结果图/ODE_1-{formatted_time}.png', format='png', dpi=300, bbox_inches='tight')
    fig.savefig(f'./可视化/结果图/ODE_1-{formatted_time}.pdf', format='pdf', dpi=300, bbox_inches='tight')



from sklearn.metrics import r2_score

def plot_scatter(predictions_dict, real_od_t, fontsize=18):
    # 1. 过滤掉 SUE-GB 方法
    predictions_dict = {k: v for k, v in predictions_dict.items() if k != "SUE-GB"}

    method_names = list(predictions_dict.keys())
    all_real_flat = real_od_t.flatten()

    # 2. Flatten所有预测值
    all_pred_flat_list = []
    for name in method_names:
        pred_flat = predictions_dict[name].flatten().copy()

        # 3. 若是 Ours，则对真实值最大的6个位置加100
        # if name.lower() == "ours":
        #     top6_indices = np.argsort(all_real_flat)[-6:]
        #     pred_flat[top6_indices] += 300
        all_pred_flat_list.append(pred_flat)

    # 4. 设置坐标轴范围
    max_val = max(np.max(all_real_flat), *[np.max(pred) for pred in all_pred_flat_list])
    lims = [0, max_val]

    # 5. 准备绘图：2x2子图
    rows, cols = 2, 2
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 8 * rows))
    axes = axes.flatten()

    # 不同颜色列表
    line_colors = ['#008585', '#d68c45', '#194a7a', 'black']
    line_colors = ['#d68c45', '#194a7a', '#008585', 'black']


    for i, (ax, name, pred_flat) in enumerate(zip(axes, method_names, all_pred_flat_list)):
        color = line_colors[i % len(line_colors)]
        ax.scatter(all_real_flat, pred_flat, alpha=1, color=color, s=9)
        ax.plot(lims, lims, color=color, linestyle='-', linewidth=3)

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("True OD", fontsize=15)
        ax.set_ylabel("Predicted OD", fontsize=15)

        # 子图标题
        ax.text(0.5, -0.08, f"({chr(97 + i)}) {name}", transform=ax.transAxes,
                ha='center', va='top', fontsize=fontsize)

        # 计算 R²（注意：此处仍用原始真实值）
        # r2 = r2_score(all_real_flat, pred_flat)
        # ax.text(0.02, 0.98, f"$R^2$ = {r2:.2f}", transform=ax.transAxes,
        #         ha='left', va='top', fontsize=fontsize, color='black')

        # 美化子图边框
        ax.set_facecolor('white')
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)
        ax.tick_params(colors='black')

    # 6. 隐藏空白子图（如有）
    for j in range(len(method_names), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    fig.savefig(f'./可视化/结果图/ODE_4-{formatted_time}.png', format='png', dpi=300, bbox_inches='tight')
    # fig.savefig(f'./可视化/结果图/ODE_4-{formatted_time}.pdf', format='pdf', bbox_inches='tight')




if __name__ == "__main__":
    predictions = load_predictions()
    real_values = load_real_data()

    if predictions and real_values is not None:
        # plot_heatmaps(predictions, real_values) ## 1
        plot_scatter(predictions, real_values)  ## 4

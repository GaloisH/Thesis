import matplotlib.pyplot as plt
import json
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示为方块的问题

def plot(histogram, title="Histogram", xlabel="Value", ylabel="Frequency",stats_dir=r'config\dataset0610_stats.json'):
    """
    绘制总体素与分支数前30的柱状图
    """
    with open(stats_dir, "r") as f:
        stats_dict = json.load(f)
    sorted_stats = sorted(stats_dict.items(), key=lambda x: x[1]["total_voxels"], reverse=True)[:30]
    indices = [item[0] for item in sorted_stats]
    total_voxels = [item[1]["total_voxels"] for item in sorted_stats]
    branches = [item[1]["branches"] for item in sorted_stats]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('总体素数', color=color)
    ax1.bar(indices, total_voxels, color=color, alpha=0.6, label='Total Voxels')
    ax1.tick_params(axis='y', labelcolor=color)
    plt.xticks(rotation=90)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('连通分支数', color=color)
    ax2.plot(indices, branches, color=color, marker='o', label='Branches')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(title)
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot(histogram=None, title="体素数前30的case", xlabel="Case ID", ylabel="Frequency", stats_dir=r'config\dataset0610_stats.json')

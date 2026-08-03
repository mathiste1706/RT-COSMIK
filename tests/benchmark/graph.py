import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FLOATING_BASE_NAMES = [
    'FF_X',
    'FF_Y',
    'FF_Z',
    'FF_quatx',
    'FF_quaty',
    'FF_quatz',
    'FF_quatw',
]


def parse_rmse_csv(csv_path: str):
    """Parses an RMSE CSV file and reads solver parameters from header comments."""
    metadata = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#'):
                match = re.match(r'#\s*([^:]+):\s*(.*)', line.strip())
                if match:
                    metadata[match.group(1).strip()] = match.group(2).strip()
            else:
                break

    df = pd.read_csv(csv_path, comment='#')

    stem = Path(csv_path).stem
    solver = metadata.get('Solver', '')
    nlp = metadata.get('nlp_solver_type', '')
    N = metadata.get('N', '')

    label = f"{stem} ({solver} [{nlp}] N={N})" if solver else stem

    body_df = df[~df['joint'].isin(FLOATING_BASE_NAMES)].copy()
    body_df['rmse'] = pd.to_numeric(body_df['rmse'], errors='coerce')
    body_df['rmse_deg'] = np.degrees(body_df['rmse'])

    return body_df[['joint', 'rmse_deg']], label


def load_all_data(csv_files: list):
    """Loads all CSV files and combines them into a single unified DataFrame."""
    joint_dfs = []
    labels = []

    for file in csv_files:
        p = Path(file)
        if not p.exists():
            print(f"[WARN] File not found: {file}, skipping...")
            continue

        df_j, label = parse_rmse_csv(file)

        base_label = label
        counter = 1
        while label in labels:
            label = f"{base_label}_{counter}"
            counter += 1

        labels.append(label)
        joint_dfs.append(df_j.rename(columns={'rmse_deg': label}).set_index('joint'))

    if not joint_dfs:
        return None, []

    df_joints = pd.concat(joint_dfs, axis=1).reset_index()
    return df_joints, labels


def plot_per_joint_summary(df_joints: pd.DataFrame, labels: list):
    """Generates the overall consolidated horizontal bar chart."""
    fig, ax = plt.subplots(figsize=(12, 14))
    n_joints = len(df_joints)
    y = np.arange(n_joints)
    n_runs = len(labels)
    bar_height = 0.8 / n_runs
    cmap = plt.get_cmap('tab10')

    for i, col in enumerate(labels):
        offset = (i - (n_runs - 1) / 2) * bar_height
        y_vals = df_joints[col].fillna(0).values.astype(float)
        ax.barh(y + offset, y_vals, height=bar_height, label=col, color=cmap(i % 10))

    ax.set_yticks(y)
    ax.set_yticklabels(df_joints['joint'], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('RMSE (Degrees)', fontsize=11, fontweight='bold')
    ax.set_title('Per-Joint RMSE Comparison (Consolidated Summary)', fontsize=12, fontweight='bold', pad=15)
    ax.legend(loc='lower right', frameon=True, facecolor='white')
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('per_joint_rmse_degrees.png', dpi=300)
    print("[INFO] Saved summary chart: per_joint_rmse_degrees.png")
    
    plt.show()
    plt.close()


def plot_all_joints_grid(df_joints: pd.DataFrame, labels: list):
    """Plots a grid with narrow subplot frames, flush bars, and tight legend spacing."""
    joints = df_joints['joint'].values
    n_joints = len(joints)
    n_runs = len(labels)

    cols = 6
    rows = int(np.ceil(n_joints / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(12, 16))
    axes = axes.flatten()
    cmap = plt.get_cmap('tab10')

    x_indices = np.arange(n_runs)
    run_colors = [cmap(i % 10) for i in range(n_runs)]

    for idx, joint_name in enumerate(joints):
        ax = axes[idx]
        vals = df_joints.loc[df_joints['joint'] == joint_name, labels].values.flatten().astype(float)

        bars = ax.bar(
            x_indices,
            vals,
            color=run_colors,
            width=1.0,
            edgecolor='black',
            linewidth=0.5
        )

        ax.set_title(joint_name, fontsize=9, fontweight='bold')
        ax.set_xticks([])
        ax.grid(True, linestyle=':', alpha=0.6, axis='y')
        ax.set_ylabel('Deg', fontsize=7)

        # --- Adaptive Y-Axis Limits ---
        val_min = np.nanmin(vals) if len(vals) > 0 else 0.0
        val_max = np.nanmax(vals) if len(vals) > 0 else 1.0
        rng = val_max - val_min

        if rng < 0.3 * val_max and val_min > 0.5:
            y_min = max(0.0, val_min - max(0.3, rng * 1.5))
            y_max = val_max + max(0.6, rng * 2.5)
        else:
            y_min = 0.0
            y_max = val_max * 1.45 if val_max > 0 else 1.0

        ax.set_ylim(y_min, y_max)
        ax.set_xlim(-0.5, n_runs - 0.5)

        for bar in bars:
            height = bar.get_height()
            if height >= 0:
                ax.annotate(
                    f'{height:.2f}°',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha='center',
                    va='bottom',
                    rotation=90,
                    fontsize=15,
                    fontweight='bold'
                )

    for idx in range(n_joints, len(axes)):
        fig.delaxes(axes[idx])

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=run_colors[i], edgecolor='black')
        for i in range(n_runs)
    ]

    plt.suptitle('Individual Joint RMSE Breakdown Across Solver Runs (Degrees)', fontsize=14, fontweight='bold', y=0.995)
    
    # Tight bottom spacing (8%) to remove excess whitespace
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])

    fig.legend(
        legend_handles,
        labels,
        loc='lower center',
        ncol=min(2, n_runs),
        fontsize=10,
        frameon=True,
        facecolor='white',
        bbox_to_anchor=(0.5, 0.005)
    )

    plt.savefig('all_joints_grid.png', dpi=300, bbox_inches='tight')
    print("[INFO] Saved 6x6 grid plot: all_joints_grid.png")
    
    plt.show()
    plt.close()


def save_individual_joint_plots(df_joints: pd.DataFrame, labels: list, output_dir: str = "individual_plots"):
    """Saves a separate PNG for every joint with narrow frames and flush bars."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    joints = df_joints['joint'].values
    n_runs = len(labels)
    cmap = plt.get_cmap('tab10')

    x_indices = np.arange(n_runs)
    run_colors = [cmap(i % 10) for i in range(n_runs)]

    for joint_name in joints:
        fig, ax = plt.subplots(figsize=(4.5, 5))
        vals = df_joints.loc[df_joints['joint'] == joint_name, labels].values.flatten().astype(float)

        bars = ax.bar(
            x_indices,
            vals,
            color=run_colors,
            width=1.0,
            edgecolor='black',
            linewidth=0.6
        )

        val_min = np.nanmin(vals) if len(vals) > 0 else 0.0
        val_max = np.nanmax(vals) if len(vals) > 0 else 1.0
        rng = val_max - val_min

        if rng < 0.3 * val_max and val_min > 0.5:
            y_min = max(0.0, val_min - max(0.3, rng * 1.5))
            y_max = val_max + max(0.8, rng * 3.0)
        else:
            y_min = 0.0
            y_max = val_max * 1.35 if val_max > 0 else 1.0

        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f'{height:.2f}°',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center',
                va='bottom',
                rotation=90,
                fontsize=9,
                fontweight='bold'
            )

        ax.set_xticks([])
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(-0.5, n_runs - 0.5)

        ax.set_ylabel('RMSE (Degrees)', fontsize=11, fontweight='bold')
        ax.set_title(f'Joint: {joint_name}', fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5, axis='y')

        handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor=run_colors[i], edgecolor='black')
            for i in range(n_runs)
        ]
        ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1))

        plt.tight_layout()
        filename = out_path / f"{joint_name}.png"
        plt.savefig(filename, dpi=200, bbox_inches='tight')
        plt.close()

    print(f"[INFO] Saved {len(joints)} individual joint plots into directory: '{output_dir}/'")


if __name__ == '__main__':
    plt.style.use(
        'seaborn-v0_8-whitegrid'
        if 'seaborn-v0_8-whitegrid' in plt.style.available
        else 'default'
    )

    # csv_files = [
    #     "csv/ipopt_10.csv",
    #     "csv/ipoqt_3.csv",
    #     "csv/acados_3_E-4_5_SQP_Full_NO-BT.csv",
    #     "csv/acados_3_E-4_5_SQP_Partial_BT.csv",
    #     "csv/acados_3_E-4_5_SQP_Partial_NO-BT.csv",
    #     "csv/acados_3_E-4_10_SQP_Partial_BT.csv",
    #     "csv/acados_10_E-4_10_SQP_Partial_BT.csv",
    #     "csv/acados_3_E-4_5_RTI_Partial_BT.csv",
    #     "csv/acados_3_E-4_5_RTI_Full_BT.csv",
    #     "csv/acados_3_E-4_5_RTI_Partial_NO-BT.csv",
    #     "csv/acados_3_E-4_5_RTI_Full_NO-BT.csv",
    #     "csv/acados_3_E-4_10_RTI_Full_NO-BT.csv",
    #     "csv/acados_3_E-4_5_RTI_Partial_BT_1.csv",
    #     "csv/acados_3_E-4_5_SQP_Partial_BT_1.csv",
    #     "csv/acados_3_E-4_4_RTI_Full_NO-BT.csv",
    #     "csv/acados_3_E-4_3_RTI_Full_NO-BT.csv",
    #     "csv/acados_3_E-4_2_SQP_Full_NO-BT.csv",
    #     "csv/acados_3_E-3_10_SQP_Full_NO-BT.csv",
    #     "csv/acados_3_E-3_5_SQP_Full_NO-BT.csv",
    #     "csv/acados_3_E-3_4_SQP_Full_NO-BT.csv",
    #     "csv/acados_3_E-3_5_RTI_Full_NO-BT.csv"
    # ]

    csv_files = [
        "csv/overhead_M/acados_2_E-2_5_RTI_Full_NO-BT.csv",
        "csv/overhead_M/acados_2_E-2_5_RTI_Full_NO-BT_4_cam.csv",
    ]

    df_joints, labels = load_all_data(csv_files)

    if df_joints is not None:
        plot_per_joint_summary(df_joints, labels)
        plot_all_joints_grid(df_joints, labels)
        save_individual_joint_plots(df_joints, labels, output_dir="individual_plots")
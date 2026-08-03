import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd

HEADER_MAP = {
    'Freeflyer_X[m]': 'FF_X',
    'Freeflyer_Y[m]': 'FF_Y',
    'Freeflyer_Z[m]': 'FF_Z',
    'Freeflyer_quaternion_X': 'FF_quatx',
    'Freeflyer_quaternion_Y': 'FF_quaty',
    'Freeflyer_quaternion_Z': 'FF_quatz',
    'Freeflyer_quaternion_W': 'FF_quatw',
    'Left_Hip_Flexion_Extension[rad]': 'Lhip_flex_ext',
    'Left_Hip_Abduction_Adduction[rad]': 'Lhip_abd_add',
    'Left_Hip_Internal_External_Rotation[rad]': 'Lhip_int_ext_rot',
    'Left_Knee_Flexion_Extension[rad]': 'Lknee_flex_ext',
    'Left_Ankle_Plantarflexion_Dorsiflexion[rad]': 'Lankle_flex_ext',
    'Left_Ankle_Inversion_Eversion[rad]': 'Lankle_abd_add',
    'Lumbar_Flexion_Extension[rad]': 'Lumbar_flex_ext',
    'Lumbar_Lateral_Bending[rad]': 'Lumbar_lateral_flex',
    'Thoracic_Flexion_Extension[rad]': 'Thoracic_flex_ext',
    'Thoracic_Lateral_Bending[rad]': 'Thoracic_lateral_flex',
    'Thoracic_Internal_External_Rotation[rad]': 'Thoracic_rot_int_ext',
    'Left_Clavicle_Elevation_Depression[rad]': 'Lcalvicule_x',
    'Left_Shoulder_Flexion_Extension[rad]': 'Lshoulder_flex_ext',
    'Left_Shoulder_Abduction_Adduction[rad]': 'Lshoulder_abd_add',
    'Left_Shoulder_Internal_External_Rotation[rad]': 'Lshoulder_int_ext_rot',
    'Left_Elbow_Flexion_Extension[rad]': 'Lelbow_flex_ext',
    'Left_Elbow_Pronation_Supination[rad]': 'Lelbow_pron_supi',
    'Left_Wrist_Flexion_Extension[rad]': 'Lwrist_flex_ext',
    'Left_Wrist_Radial_Ulnar_Deviation[rad]': 'Lwrist_x',
    'Cervical_Flexion_Extension[rad]': 'Cervical_flex_ext',
    'Cervical_Lateral_Bending[rad]': 'Cervical_lat_bend',
    'Cervical_Internal_External_Rotation[rad]': 'Cervical_int_ext_rot',
    'Right_Clavicle_Elevation_Depression[rad]': 'rcalvicule_x',
    'Right_Shoulder_Flexion_Extension[rad]': 'Rshoulder_flex_ext',
    'Right_Shoulder_Abduction_Adduction[rad]': 'Rshoulder_abd_add',
    'Right_Shoulder_Internal_External_Rotation[rad]': 'Rshoulder_int_ext_rot',
    'Right_Elbow_Flexion_Extension[rad]': 'Relbow_flex_ext',
    'Right_Elbow_Pronation_Supination[rad]': 'Relbow_pron_supi',
    'Right_Wrist_Flexion_Extension[rad]': 'Rwrist_flex_ext',
    'Right_Wrist_Radial_Ulnar_Deviation[rad]': 'Rwrist_x',
    'Right_Hip_Flexion_Extension[rad]': 'Rhip_flex_ext',
    'Right_Hip_Abduction_Adduction[rad]': 'Rhip_abd_add',
    'Right_Hip_Internal_External_Rotation[rad]': 'Rhip_int_ext_rot',
    'Right_Knee_Flexion_Extension[rad]': 'Rknee_flex_ext',
    'Right_Ankle_Plantarflexion_Dorsiflexion[rad]': 'Rankle_flex_ext',
    'Right_Ankle_Inversion_Eversion[rad]': 'Rankle_abd_add',
}

STANDARD_SHORT_NAMES = list(HEADER_MAP.values())
FLOATING_BASE_NAMES = [
    'FF_X',
    'FF_Y',
    'FF_Z',
    'FF_quatx',
    'FF_quaty',
    'FF_quatz',
    'FF_quatw',
]


def extract_metadata_from_csv(csv_path: str) -> dict:
    """Extracts all # metadata comments from the top of joint_angles_pre.csv."""
    metadata = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#'):
                match = re.match(r'#\s*([^:]+):\s*(.*)', line.strip())
                if match:
                    metadata[match.group(1).strip()] = match.group(2).strip()
            else:
                break
    return metadata


def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    for col in ['frame', 'index', 'Unnamed: 0']:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df.rename(columns=HEADER_MAP)


def compute_rmse(pred_csv: str, gt_csv: str, output_csv: str = None):
    # Extract metadata recorded in joint_angles_pre.csv
    metadata = extract_metadata_from_csv(pred_csv)

    df_pred = normalize_headers(pd.read_csv(pred_csv, comment='#'))
    df_gt = normalize_headers(pd.read_csv(gt_csv, comment='#'))

    common_cols = [c for c in df_pred.columns if c in df_gt.columns]
    if (
        len(common_cols) < 43
        and len(df_pred.columns) == len(df_gt.columns) == 43
    ):
        df_pred.columns = STANDARD_SHORT_NAMES
        df_gt.columns = STANDARD_SHORT_NAMES
        common_cols = STANDARD_SHORT_NAMES

    min_len = min(len(df_pred), len(df_gt))
    pred_vals = df_pred[common_cols].iloc[:min_len].to_numpy()
    gt_vals = df_gt[common_cols].iloc[:min_len].to_numpy()

    diff = pred_vals - gt_vals
    per_joint_rmse = np.sqrt(np.mean(diff**2, axis=0))

    # Calculate Body Joint RMSE (excl. FF_X, FF_Y, FF_Z, FF_quat*)
    joint_cols = [c for c in common_cols if c not in FLOATING_BASE_NAMES]
    j_idx = [common_cols.index(c) for c in joint_cols]
    body_joint_rmse_rad = np.sqrt(np.mean(diff[:, j_idx] ** 2))
    body_joint_rmse_deg = np.degrees(body_joint_rmse_rad)

    results_df = pd.DataFrame({'joint': common_cols, 'rmse': per_joint_rmse})

    if output_csv:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            # Transfer all metadata into output rmse.csv
            for k, v in metadata.items():
                f.write(f'# {k}: {v}\n')
            f.write(f'# body_joint_rmse_deg: {body_joint_rmse_deg:.4f}\n')

        results_df.to_csv(output_path, mode='a', index=False)
        print(f'RMSE CSV with metadata saved to: {output_path.resolve()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', type=str, required=True)
    parser.add_argument('--gt', type=str, required=True)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    compute_rmse(args.pred, args.gt, args.output)
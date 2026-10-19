import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GroupShuffleSplit

DATASET_PATH = r"C:\Users\ilmar\Downloads\archive"  # ajusta esto a tu carpeta real

records = []
for patient_id in os.listdir(DATASET_PATH):
    patient_path = os.path.join(DATASET_PATH, patient_id)
    if not os.path.isdir(patient_path):
        continue
    for label in ['0', '1']:
        label_path = os.path.join(patient_path, label)
        if not os.path.isdir(label_path):
            continue
        for fname in os.listdir(label_path):
            if fname.endswith('.png'):
                records.append({
                    'filepath': os.path.join(label_path, fname),
                    'label': label,
                    'patient_id': patient_id
                })

df_full = pd.DataFrame(records)

N_PACIENTES = 120  # el mismo numero que usaste para el modelo oficial
rng = np.random.default_rng(seed=42)
pacientes_todos = np.sort(df_full['patient_id'].unique())
pacientes_muestra = rng.choice(pacientes_todos, size=min(N_PACIENTES, len(pacientes_todos)), replace=False)
df = df_full[df_full['patient_id'].isin(pacientes_muestra)].reset_index(drop=True)

gss1 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_val_idx, test_idx = next(gss1.split(df, groups=df['patient_id']))
df_test = df.iloc[test_idx].reset_index(drop=True)

test_patient_ids = sorted(df_test['patient_id'].unique())
print(f"Pacientes en test: {len(test_patient_ids)}")
print(test_patient_ids)
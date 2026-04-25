import pandas as pd
import numpy as np
import os

# Create synthetic student dataset matching the expected schema
np.random.seed(42)
n_samples = 395

df = pd.DataFrame({
    'school': np.random.choice(['GP', 'MS'], n_samples),
    'sex': np.random.choice(['F', 'M'], n_samples),
    'age': np.random.randint(15, 23, n_samples),
    'Medu': np.random.randint(0, 5, n_samples),
    'Fedu': np.random.randint(0, 5, n_samples),
    'Mjob': np.random.choice(['teacher', 'health', 'services', 'at_home', 'other'], n_samples),
    'Fjob': np.random.choice(['teacher', 'health', 'services', 'at_home', 'other'], n_samples),
    'reason': np.random.choice(['home', 'reputation', 'course', 'other'], n_samples),
    'guardian': np.random.choice(['mother', 'father', 'other'], n_samples),
    'traveltime': np.random.randint(1, 5, n_samples),
    'studytime': np.random.randint(1, 5, n_samples),
    'failures': np.random.randint(0, 5, n_samples),
    'schoolsup': np.random.choice(['yes', 'no'], n_samples),
    'famsup': np.random.choice(['yes', 'no'], n_samples),
    'paid': np.random.choice(['yes', 'no'], n_samples),
    'activities': np.random.choice(['yes', 'no'], n_samples),
    'nursery': np.random.choice(['yes', 'no'], n_samples),
    'higher': np.random.choice(['yes', 'no'], n_samples),
    'internet': np.random.choice(['yes', 'no'], n_samples),
    'romantic': np.random.choice(['yes', 'no'], n_samples),
    'famrel': np.random.randint(1, 6, n_samples),
    'freetime': np.random.randint(1, 6, n_samples),
    'goout': np.random.randint(1, 6, n_samples),
    'Dalc': np.random.randint(1, 6, n_samples),
    'Walc': np.random.randint(1, 6, n_samples),
    'health': np.random.randint(1, 6, n_samples),
    'absences': np.random.randint(0, 30, n_samples),
    'G1': np.random.randint(5, 20, n_samples),
    'G2': np.random.randint(5, 20, n_samples),
    'G3': np.random.randint(0, 20, n_samples),
})

# Save with semicolon separator (as expected by the script)
os.makedirs('Data', exist_ok=True)
df.to_csv('Data/student-mat.csv', index=False, sep=';')
print(f'✓ Created student dataset: {len(df)} rows, {len(df.columns)} columns')
print(f'✓ Saved to Data/student-mat.csv')

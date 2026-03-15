import os
import shutil
import random

source = "data/raw/shenzhen"

h1 = "data/federated/hospital_1"
h2 = "data/federated/hospital_2"
h3 = "data/federated/hospital_3"

files = [f for f in os.listdir(source) if f.endswith(".png")]
random.shuffle(files)

n = len(files)

split1 = int(0.33 * n)
split2 = int(0.66 * n)

for f in files[:split1]:
    shutil.copy(os.path.join(source, f), os.path.join(h1, f))

for f in files[split1:split2]:
    shutil.copy(os.path.join(source, f), os.path.join(h2, f))

for f in files[split2:]:
    shutil.copy(os.path.join(source, f), os.path.join(h3, f))

print("Dataset split complete.")
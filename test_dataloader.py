from data_loading.sound_loader import soundsamples
from options import Options

# Parse default options (you can override defaults by modifying sys.argv or the parser)
args = Options().parse()

# Instantiate the dataset
dataset = soundsamples(args)

# Print dataset length
print("Dataset length:", len(dataset))

# Fetch a sample
sample = dataset[0]

# Print sample details
print("Sample type:", type(sample))
for i, item in enumerate(sample):
    if hasattr(item, 'shape'):
        print(f"Sample[{i}] shape: {item.shape}")
    else:
        print(f"Sample[{i}]: {item}")

# Optionally, iterate through a few samples
for idx in range(3):
    sample = dataset[idx]
    print(f"Sample {idx} loaded.")
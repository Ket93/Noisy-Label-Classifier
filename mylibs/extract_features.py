import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# config
BASE_DIR = Path(__file__).parent
IMAGE_DIR = BASE_DIR.parent / 'images'
CLASSES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
BATCH_SIZE = 64
VAL_FRAC = 0.15
SEED = 42

# load images with labels
class FlowersRawDataset(Dataset):

    def __init__(self, image_dir, classes, transform=None):
        self.transform = transform
        self.samples = []

        for label, cls in enumerate(classes):
            folder = Path(image_dir) / cls
            for path in sorted(folder.glob('*.jpg')):
                self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label, str(path)


# extract features
def extract_dino_features(batch_size: int = BATCH_SIZE):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    model.eval().to(device)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset = FlowersRawDataset(IMAGE_DIR, CLASSES, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=0, pin_memory=(device.type == 'cuda'))

    features_list, labels_list, paths_list = [], [], []

    with torch.no_grad():
        for i, (imgs, lbls, paths) in enumerate(loader):
            imgs = imgs.to(device)
            feats = model(imgs)
            features_list.append(feats.cpu().numpy())
            labels_list.append(lbls.numpy())
            paths_list.extend(paths)

    features = np.concatenate(features_list)
    labels = np.concatenate(labels_list)
    paths = np.array(paths_list)


    # get fixed validation set
    rng = np.random.default_rng(SEED)
    val_indices = []
    for cls_idx in range(len(CLASSES)):
        cls_mask = np.where(labels == cls_idx)[0]
        n_val = max(1, int(len(cls_mask) * VAL_FRAC))
        chosen = rng.choice(cls_mask, size=n_val, replace=False)
        val_indices.extend(chosen.tolist())
    val_indices = np.array(sorted(val_indices))

    # store data
    np.save(BASE_DIR / 'features.npy', features)
    np.save(BASE_DIR / 'labels.npy', labels)
    np.save(BASE_DIR / 'image_paths.npy', paths)
    np.save(BASE_DIR / 'val_indices.npy', val_indices)

    class_counts = {CLASSES[i]: int((labels == i).sum()) for i in range(len(CLASSES))}
    print(f"\nClass distribution: {class_counts}")

    return features, labels, paths, val_indices


if __name__ == '__main__':
    extract_dino_features()

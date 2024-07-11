import glob
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.io import read_image

class JapArtDataset(Dataset):

    def __init__(self, args, transform=None):
        self.img_dir = args.augment if args.augment else 'data/jap-art/'
        self.dim = args.dim

        self.img_names = []
        self.labels = []
        self.labels_names = [
            'yokai',
            'early',
            'edo',
            'golden',
            'pre-meiji',
            'meiji',
            'modern',
        ]
        self.labels_map = dict(zip(self.labels_names, range(len(self.labels_names))))

        for f in glob.glob(self.img_dir + "*/*.jpg"):
            string_label = f.split('/')[-2]
            label = self.labels_map[string_label]
            self.img_names.append(f)
            self.labels.append(label)

        self.transform = transform if transform else v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(self.dim),
            v2.Normalize([0.5], [0.5]),
            v2.Lambda(lambda x: torch.clamp(x, 0, 1)),
        ])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        label = torch.tensor(self.labels[idx], dtype=torch.int64)
        img = read_image(self.img_names[idx])

        assert img.shape[0] <= 3
        if img.shape[0] < 3:
            img = torch.cat((img, img, img), dim=0)
            
        img = self.transform(img)

        return img, label
    
class FlickerFacesDataset(Dataset):

    def __init__(self, args, transform=None):
        self.img_dir = args.augment if args.augment else 'data/ff/real_faces_128'
        self.dim = args.dim
        self.transform = transform

        self.img_names = []
        self.labels = []
        self.labels_names = [
            'face',
        ]
        self.labels_map = dict(zip(self.labels_names, range(len(self.labels_names))))

        for f in glob.glob(self.img_dir + "/*.png"):
            label = 0
            self.img_names.append(f)
            self.labels.append(label)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        label = torch.tensor(self.labels[idx], dtype=torch.int64)
        img = read_image(self.img_names[idx])

        assert img.shape[0] <= 3

        if self.transform:
            img = self.transform(img)
        else:            
            c, _, _ = img.size()
            if c < 3:
                img = torch.cat((img, img, img), dim=0)
            
            transform = v2.Compose([
                v2.ToDtype(torch.float32, scale=True),
            ])
            img = transform(img)

        return img, label
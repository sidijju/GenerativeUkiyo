import glob
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.io import read_image

class TransformDataset(Dataset):

    def __init__(self):
        super().__init__()

    def set_transform(self, transform):
        self.transform = transform

    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        label = torch.tensor(self.labels[idx], dtype=torch.int64)
        img = read_image(self.img_names[idx])

        assert img.shape[0] <= 3
        
        c, _, _ = img.size()
        if c < 3:
            img = torch.cat((img, img, img), dim=0)
        
        img = self.transform(img)

        return img, label

class JapArtDataset(TransformDataset):

    def __init__(self, args, transform=None):
        self.img_dir = 'data/jap-art/'
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

        if not transform:
            transform = v2.Compose([
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize(self.dim),
                v2.RandomHorizontalFlip(p=0.5) if args.augment else v2.Identity()
            ])
        self.set_transform(transform)
    
class FlickerFacesDataset(TransformDataset):

    def __init__(self, args, transform=None):
        self.img_dir = 'data/ff/real_faces_128'
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

        if not transform:
            transform = v2.Compose([
                v2.ToDtype(torch.float32, scale=True),
            ])

        self.set_transform(transform)

    

    
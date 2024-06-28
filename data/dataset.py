import glob
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.io import read_image

class JapArtDataset(Dataset):

    def __init__(self, args, transform=None):
        self.img_dir = args.augment if args.augment else 'data/jap-art/'
        print(self.img_dir)
        self.dim = args.dim
        self.transform = transform

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
            print(f)
            string_label = f.split('/')[-2]
            label = self.labels_map[string_label]
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
            # resize so maximum dimension is 720
            c, w, h = img.size()
            pad = None
            if h > w:
                w = int(w * 720 / h)
                if w % 2 != 0:
                    w += 1
                h = 720

                pad = [0, (720 - w)//2]
            elif w > h:
                h = int(h * 720/w)
                if h % 2 != 0:
                    h += 1
                w = 720
                pad = [(720 - h)//2, 0]
            else:
                pad = 0

            if c < 3:
                img = torch.cat((img, img, img), dim=0)
            
            transform = v2.Compose([
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize(size=(w, h)),
                v2.Pad(pad, fill=1),
                v2.Resize(self.dim),
            ])
            img = transform(img)
            img = torch.clamp(img, 0, 1)

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
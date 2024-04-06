import glob
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.io import read_image

class JapArtDataset(Dataset):

    def __init__(self, args, transform=None):
        if args.augment:
            self.img_dir = args.new_dir
        else:
            self.img_dir = 'jap-art/'
        self.img_dim = args.img_dim
        self.transform = transform

        self.img_names = []
        self.labels = []
        self.labels_names = [
            'yokai',
            'okumura-masanobu',
            'hishikawa-moronobu',
            'torii-kiyomasu-ii',
            'nishikawa-sukenobu',
            'torii-kiyonobu-ii',
            'torii-kiyomasu-i',
            'nishimura-shigenaga',
            'torii-kiyonobu-i',
            'torii-kiyotada-i',
            'okumura-toshinobu',
            'katsukawa-shunsho',
            'suzuki-harunobu',
            'isoda-koryusai',
            'katsukawa-shunko',
            'katsukawa-shunei',
            'ippitsusai-buncho',
            'torii-kiyomitsu',
            'kitao-shigemasa',
            'utagawa-toyoharu',
            'ishikawa-toyonobu',
            'torii-kiyotsune',
            'torii-kiyohiro',
            'katsukawa-shunsen',
            'kitagawa-utamaro',
            'torii-kiyonaga',
            'toshusai-sharaku',
            'hosoda-eishi',
            'katsukawa-shuncho',
            'kubo-shunman',
            'eishosai-choki',
            'kitao-masayoshi',
            'katsukawa-shunzan',
            'chokosai-eisho',
            'utagawa-kunimasa',
            'ichirakutei-eisui',
            'utagawa-kunisada',
            'utagawa-hiroshige',
            'utagawa-kuniyoshi',
            'utagawa-toyokuni-i',
            'katsushika-hokusai',
            'utagawa-kunisada-ii',
            'utagawa-hiroshige-ii',
            'keisai-eisen',
            'utagawa-hirosada',
            'kikugawa-eizan',
            'totoya-hokkei',
            'utagawa-kuniyasu',
            'utagawa-kuniteru',
            'utagawa-kunikazu',
            'utagawa-yoshitsuya',
            'yashima-gakutei',
            'utagawa-yoshikazu',
            'utagawa-toyoshige',
            'utagawa-toyohiro',
            'utagawa-hirokage',
            'kano-shugen-sadanobu',
            'shunkosai-hokushu',
            'yoshifuji',
            'katsukawa-shuntei',
            'yanagawa-shigenobu',
            'gigado-ashiyuki',
            'hasegawa-sadanobu-i',
            'toyokawa-yoshikuni',
            'teisai-hokuba',
            'shotei-hokuju',
            'torii-kiyomine',
            'toyohara-kunichika',
            'tsukioka-yoshitoshi',
            'toyohara-chikanobu',
            'ochiai-yoshiiku',
            'tsukioka-kogyo',
            'kobayashi-kiyochika',
            'utagawa-yoshitora',
            'mizuno-toshikata',
            'utagawa-yoshitaki',
            'utagawa-kunisada-iii',
            'ogata-gekko',
            'utagawa-hiroshige-iii',
            'morikawa-chikashige',
            'utagawa-sadahide',
            'kawanabe-kyosai',
            'utagawa-toyosai',
            'tomioka-eisen',
            'inoue-yasuji',
            'takeuchi-keishu',
            'adachi-ginko',
            'miyagawa-shuntei',
            'watanabe-nobukazu',
            'ikkei',
            'migita-toshihide',
            'shibata-zeshin',
            'utagawa-kuniaki',
            'imao-keinen',
            'kono-bairei',
            'kajita-hanko',
            'utagawa-kunimasa-iii',
            'torii-kiyosada',
            'watanabe-shotei',
            'utagawa-fusatane',
            'utagawa-kunitoshi',
            'seiko',
            'utagawa-yoshitoyo',
            'kawase-hasui',
            'tsuchiya-koitsu',
            'kasamatsu-shiro',
            'yoshida-hiroshi',
            'ito-shinsui',
            'ohara-koson',
            'fujishima-takeji',
            'takahashi-hiroaki',
            'hashiguchi-goyo',
            'wada-sanzo',
            'torii-kotondo',
            'shotei-takahashi',
            'koho',
            'yoshida-toshi',
            'shoson-ohara',
            'hasegawa-sadanobu-iii',
            'kaburagi-kiyokata',
            'onchi-koshiro',
            'natori-shunsen',
            'yamamoto-shoun',
            'maekawa-senpan',
            'henmi-takashi',
            'takehisa-yumeji',
            'okiie',
            'oda-kazuma',
            'asano-takeji',
            'tokuriki-tomikichiro',
            'nishijima-katsuyuki',
            'kawano-kaoru',
            'tom-kristensen',
            'maeda-masao',
            'ikeda-shuzo',
            'bakufu-ohno',
            'kotozuka-eiichi',
            'paul-binnie',
            'morozumi-osamu',
            'mitsuaki-sora',
            'okamoto-ryusei',
            'azechi-umetaro',
            'kitaoka-fumio',
            'hagiwara-hideo',
            'maki-haku',
            'inagaki-tomoo',
            'mabuchi-toru',
            'watanabe-sadao',
            'kusaka-kenji',
            'shibata-shinsai',
            'asai-kiyoshi',
        ]
        self.labels_map = dict(zip(self.labels_names, range(len(self.labels_names))))

        for f in glob.glob(self.img_dir + "*/*.jpg"):
            img_name = f.split('/')[-1]
            string_label = f.split('/')[-2]
            label = self.labels_map[string_label]
            self.img_names.append(f)
            self.labels.append(label)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        label = torch.tensor(self.labels[idx], dtype=torch.int64)
        img = read_image(self.img_names[idx])

        if img.shape[0] > 3:
            print(self.img_names[idx])
            print(img.shape)

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
                v2.Resize(self.img_dim),
            ])
            img = transform(img)
            img = torch.clamp(img, 0, 1)

        return img, label
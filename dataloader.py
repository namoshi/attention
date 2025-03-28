import glob
import os
import random
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils.common import show_img




class DatasetLoader(Dataset):
    def __init__(self, root, phase, transform=None):
        super().__init__()
        self.transform = transform

        self.image_paths = []
        self.image_labels = []
        self.class_name = os.listdir(os.path.join(root, phase))
        self.class_name.sort()
        for i, x in enumerate(self.class_name):
            temp = sorted(glob.glob(os.path.join(root, phase, x, "*")))
            self.image_labels.extend([i] * len(temp))
            self.image_paths.extend(temp)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
            
        return {"image": image, "label": self.image_labels[index]}

    def __len__(self):
        return len(self.image_paths)
    


def get_val_dataloader(dataset_path, img_size=224, batch_size=128):
    
    def worker_init_fn(worker_id, seed=1):
        random.seed(worker_id+seed)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = DatasetLoader(dataset_path, 'val', transform)
    
    print(f'Load dataset, Num of dataset: {len(dataset)}')
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=8,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )

    # 画像を12枚表示したい場合
    # show_img(save_path='/homes/ypark/code/dataset/TransformerAnalysis/fig/ex.png', dataloader=data_loader)
    
    return data_loader

 
    

def get_dataloader(dataset_path, img_size=32, batch_size=128):

    ## データ拡張の設定 (自由に変更)
    # augmentation_list = ['rcrop', 'hflip', 'ra', 'cutout']
    augmentation_list = []
    print(f'Apply augmentation ... {augmentation_list}')

    additional_transform_list = []
    for augment in augmentation_list:
        # 画像の一部をランダムに切り出し
        if augment == 'rcrop':
            additional_transform_list.append(
                transforms.RandomResizedCrop(size=img_size, scale=(0.5, 1.00), ratio=(1.0, 1.0))
            )
        # 水平反転
        elif augment == 'hflip':
            additional_transform_list.append(
                transforms.RandomHorizontalFlip(p=0.5)
            )
        # 画像の一部を隠す
        elif augment == 'cutout':
            additional_transform_list.append(
                transforms.RandomApply(
                    [Cutout(n_holes=1, img_size=img_size, patch_size=0.5)],
                    p=0.5
                )
            )
        # RandAugment
        elif augment == 'ra':  # RandAugment
            additional_transform_list.append(
                transforms.RandAugment(num_ops=2, magnitude=9)
            )
        # Color変換
        elif augment == 'cjitter':
            additional_transform_list.append(
                transforms.ColorJitter(brightness=0.3,contrast=0.3,saturation=0.3,hue=0.3)
            )
        # グレー化
        elif augment == 'gray':
            additional_transform_list.append(
                transforms.RandomGrayscale(p=0.1)
            )
        # 上下反転
        elif augment == 'vflip':
            additional_transform_list.append(
                transforms.RandomVerticalFlip(p=1.0)
            )

    # trainにはデータ拡張を適用
    train_transform = transforms.Compose(
        [transforms.Resize((img_size, img_size))]
        + additional_transform_list
        + [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    # val, testにはデータ拡張はなし
    test_transform = transforms.Compose(
        [transforms.Resize((img_size, img_size))]
        + [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )

    # データセットを読み込む
    train_dataset = DatasetLoader(dataset_path, 'train', train_transform)
    val_dataset = DatasetLoader(dataset_path, 'val', test_transform)
    test_dataset = DatasetLoader(dataset_path, 'test', test_transform)
    
    # バッチごとに取り出せるようにDataLoaderに登録
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=False,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=False,
        drop_last=False,
    )

    print(f'Load dataset, Num of dataset ... Train : {len(train_dataset)}  Val : {len(val_dataset)}  Test : {len(test_dataset)}')

    return train_loader, val_loader, test_loader

 


        

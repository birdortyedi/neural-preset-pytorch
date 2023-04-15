import random
import glob
from pathlib import Path
from PIL import Image
from pillow_lut import load_cube_file   
from torch.utils.data import Dataset


class MSCOCO(Dataset):
    def __init__(self, root, transform=None, is_train=True) -> None:
        super().__init__()
        self.imgs = list((Path(root) / "images").iterdir())
        self.luts = [str(p) for p in glob.glob(str(Path(root) / "LUTS") + "/*/*.cube")]
        self.transform = transform
        self.is_train = is_train
    
    def __getitem__(self, index):
        lut1 = load_cube_file(random.choice(self.luts))
        lut2 = load_cube_file(random.choice(self.luts))
        gt = Image.open(self.imgs[index]).convert("RGB")
        img1 = gt.filter(lut1)
        img2 = gt.filter(lut2)
        if self.transform:
            gt, img1, img2 = self.transform(gt, img1, img2)
        
        return gt, img1, img2
    
    def __len__(self):
        return len(self.imgs)
    

if __name__ == "__main__":
    from transforms import *
    dataset = MSCOCO(
        "../../../Downloads/MSCOCO", 
        Compose([
            RandomCropThreeInstances((256, 256)),
            RandomHorizontalFlipThreeInstances(),
            ToTensor()
        ])
    )
    print(len(dataset))
    num = 0
    min_shape = None
    for i in range(len(dataset)):
        gt, img1, img2 = dataset.__getitem__(i)
        if min(gt.shape[1:]) < 256:
            num += 1
            print(f"{num}/{i}")
    print(f"{num}/{len(dataset)}")
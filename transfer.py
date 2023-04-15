import fire
import os
import torch
import cv2
import numpy as np
from typing import Union, Tuple
from pathlib import Path
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

from dncm import DNCM, Encoder


to_tensor = ToTensor()
to_pil = ToPILImage()


def build_model(
    ckpt: Union[str, Path],
    k: int = 16,
    sz: int = 256,
):
    nDNCM = DNCM(k)
    sDNCM = DNCM(k)
    encoder = Encoder(sz, k)
    checkpoints = torch.load(ckpt)
    nDNCM.load_state_dict(checkpoints["nDCNM"])
    sDNCM.load_state_dict(checkpoints["sDCNM"])
    encoder.load_state_dict(checkpoints["encoder"])
    nDNCM = nDNCM.cuda()
    sDNCM = sDNCM.cuda()
    encoder = encoder.cuda()
    return nDNCM, sDNCM, encoder


def patchify(
    arr: np.ndarray,
    stride: int = 1
):
    for i in range(0, arr.shape[0], stride):
        for j in range(0, arr.shape[1], stride):
            patch = arr[i:i+stride, j:j+stride]
            yield torch.tensor(patch).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.


def unpatchify(
    patch_arr: np.ndarray,
    output_shape: Tuple[int, int, int],
    stride: int = 1
):
    output = np.zeros(output_shape, dtype=np.uint8)
    step = 0
    for i in range(0, output.shape[0], stride):
        for j in range(0, output.shape[1], stride):
            output[i:i+stride, j:j+stride] = patch_arr[step]
            step += 1
    return output
    

def postprocess(
    T: torch.Tensor
):
    return (T.clamp(min=0., max=1.).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.).astype(np.uint8)


def run(
    src_image: Union[str, Path],
    tgt_image: Union[str, Path],
    ckpt: Union[str, Path],
    k: int = 16,
    sz: int = 256,
    patch_sz: int = 512,
    output_dir: Union[str, Path] = "outputs"
):
    src_image = Path(src_image)
    tgt_image = Path(tgt_image)
    os.makedirs(str(output_dir), exist_ok=True)
    nDNCM, sDNCM, encoder = build_model(ckpt=ckpt, k=k, sz=sz)
    src = to_tensor(Image.open(src_image).convert("RGB")).float().unsqueeze(0).cuda()
    tgt = to_tensor(Image.open(tgt_image).convert("RGB")).float().unsqueeze(0).cuda()
    _, c, h, w = tgt.shape
    tgt_ = np.zeros((h + patch_sz - (h % patch_sz), w + patch_sz - (w % patch_sz), c))
    tgt_[:h, :w] = tgt.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.
    out_patches = list()
    with torch.no_grad():
        _, r_j = encoder(src)
        d_i, _ = encoder(tgt)
        for patch in patchify(tgt_, patch_sz):
            out_patch = sDNCM(nDNCM(patch, d_i), r_j)
            out_patches.append(postprocess(out_patch))
    output = unpatchify(np.array(out_patches), tgt_.shape, patch_sz)[:h, :w, ::-1]
    cv2.imwrite(str(Path(output_dir) / f"{src_image.stem}_{tgt_image.stem}.png"), output)
    


if __name__ == "__main__":
    fire.Fire(run)
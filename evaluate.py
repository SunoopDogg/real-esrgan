import argparse
import csv
import cv2
import glob
import os
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim

from realesrgan import RealESRGANer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lq_dir', type=str, required=True, help='LQ image directory')
    parser.add_argument('--gt_dir', type=str, required=True, help='GT image directory')
    parser.add_argument('--output_dir', type=str, required=True, help='SR output directory')
    parser.add_argument('--csv_path', type=str, required=True, help='CSV output path')
    parser.add_argument('--scale', type=int, required=True, choices=[2, 4], help='Upscale factor (2 or 4)')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU device ID')
    parser.add_argument('--tile', type=int, default=0, help='Tile size, 0 for no tile')
    args = parser.parse_args()

    # Select model based on scale
    if args.scale == 2:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        model_name = 'RealESRGAN_x2plus'
        file_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'
        netscale = 2
    else:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        model_name = 'RealESRGAN_x4plus'
        file_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
        netscale = 4

    # Load model weights
    root_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(root_dir, 'weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        model_path = load_file_from_url(
            url=file_url, model_dir=os.path.join(root_dir, 'weights'), progress=True, file_name=None)

    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        model=model,
        tile=args.tile,
        tile_pad=10,
        pre_pad=0,
        half=True,
        gpu_id=args.gpu_id)

    os.makedirs(args.output_dir, exist_ok=True)

    lq_files = sorted(glob.glob(os.path.join(args.lq_dir, '*')))
    results = []

    for idx, lq_path in enumerate(lq_files):
        filename = os.path.basename(lq_path)
        gt_path = os.path.join(args.gt_dir, filename)

        if not os.path.isfile(gt_path):
            print(f'[SKIP] GT not found: {gt_path}')
            continue

        img_lq = cv2.imread(lq_path, cv2.IMREAD_COLOR)
        img_gt = cv2.imread(gt_path, cv2.IMREAD_COLOR)

        try:
            sr_img, _ = upsampler.enhance(img_lq, outscale=args.scale)
        except RuntimeError as e:
            print(f'[ERROR] {filename}: {e}')
            continue

        # Ensure same size as GT
        h, w = img_gt.shape[:2]
        if sr_img.shape[0] != h or sr_img.shape[1] != w:
            sr_img = cv2.resize(sr_img, (w, h), interpolation=cv2.INTER_LANCZOS4)

        # Save SR image
        save_path = os.path.join(args.output_dir, filename)
        cv2.imwrite(save_path, sr_img)

        # Calculate metrics (Y channel, crop_border based on scale)
        psnr = calculate_psnr(sr_img, img_gt, crop_border=args.scale, test_y_channel=True)
        ssim = calculate_ssim(sr_img, img_gt, crop_border=args.scale, test_y_channel=True)

        results.append({'filename': filename, 'psnr': psnr, 'ssim': ssim})
        print(f'[{idx+1}/{len(lq_files)}] {filename} - PSNR: {psnr:.4f}, SSIM: {ssim:.4f}')

    # Write CSV
    with open(args.csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'psnr', 'ssim'])
        writer.writeheader()
        writer.writerows(results)
        # Write average row
        if results:
            avg_psnr = np.mean([r['psnr'] for r in results])
            avg_ssim = np.mean([r['ssim'] for r in results])
            writer.writerow({'filename': 'AVERAGE', 'psnr': avg_psnr, 'ssim': avg_ssim})

    if results:
        avg_psnr = np.mean([r['psnr'] for r in results])
        avg_ssim = np.mean([r['ssim'] for r in results])
        print(f'\n=== Results ({len(results)} images) ===')
        print(f'Average PSNR: {avg_psnr:.4f}')
        print(f'Average SSIM: {avg_ssim:.4f}')
        print(f'CSV saved to: {args.csv_path}')


if __name__ == '__main__':
    main()

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import numpy as np
import cv2

def gamma_correction(img, gamma=2.2):
    """对输入图像进行 Gamma 校正"""
    return np.power(img, 1.0 / gamma)

def compute_psnr(img1, img2):
    """计算两个图像的 PSNR（峰值信噪比）"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')  # 图像完全相同
    max_pixel = 1.0  # EXR 文件通常是浮点格式
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def read_exr(path, channel=3,gamma=2.2):
    """
    path: exr文件的路径
    channel: 需要保存的通道数
    """
    # print("read exr from path: " + path)
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    # 把读取到的BGR转换为RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    image = image[:, :, :channel]
    return gamma_correction(image, gamma)

def compare_psnr(image1_path, image2_path, apply_gamma=True, gamma=2.2):
    """计算两张 EXR 图像的 PSNR（支持 Gamma 校正）"""
    img1 =  read_exr(image1_path)
    img2 = read_exr(image2_path)

    if img1 is None or img2 is None:
        print("图像加载失败")
        return None

    # 确保两个图像形状一致
    if img1.shape != img2.shape:
        print(f"图像尺寸不匹配: {img1.shape} vs {img2.shape}")
        return None

    psnr = compute_psnr(img1, img2)
    print(f"PSNR ({image1_path} vs {image2_path}, Gamma={gamma}): {psnr:.2f} dB")
    return psnr

# 示例：指定两张 EXR 图像的路径
image1_path = "./MedievalDocksGenColor.1218.mob.exr"
image2_path = "./MedievalDocksPreTonemapHDRColor.0609.exr"

# 计算 PSNR（默认应用 Gamma 校正）
compare_psnr(image1_path, image2_path, apply_gamma=True, gamma=2.2)

import numpy as np
import imageio.v3 as imageio

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

def load_exr_image(file_path, apply_gamma=True, gamma=2.2):
    """读取 EXR 文件并返回浮点数数组（可选 Gamma 校正）"""
    try:
        img = imageio.imread(file_path, format="exr").astype(np.float32)
        if apply_gamma:
            img = gamma_correction(img, gamma)
        return img
    except Exception as e:
        print(f"无法读取 {file_path}: {e}")
        return None

def compare_psnr(image1_path, image2_path, apply_gamma=True, gamma=2.2):
    """计算两张 EXR 图像的 PSNR（支持 Gamma 校正）"""
    img1 = load_exr_image(image1_path, apply_gamma, gamma)
    img2 = load_exr_image(image2_path, apply_gamma, gamma)

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
image1_path = "./MedievalDocksGenColor.1218.v2.exr"
image2_path = "./MedievalDocksPreTonemapHDRColor.0609.exr"

# 计算 PSNR（默认应用 Gamma 校正）
compare_psnr(image1_path, image2_path, apply_gamma=True, gamma=2.2)

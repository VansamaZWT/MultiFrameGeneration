import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


from opengl_util import *
from data_io import *


import numpy as np
import time
import cv2
import numpy as np
import math
from data_io import read_exr, write_exr, read_matrix
from mul_gen_I import mul_gen_main,mul_gen_I_init

def main(root="E:/workspace/zwtdataset/", sub_paths=["Bunker/train1-30fps-combine"], mode="mg_I", debug=False):
    if mode == "mg_I":
        programs = []
        shader_sources = ["gen_mv_1.comp","gen_mv_2.comp","inpaint.comp","warp_withDepth.comp","blending.comp"]
        programs = mul_gen_I_init(shader_sources)
        save_path_ = "./save/mg_I"

    else:
        raise ValueError(f"Invalid mode: {mode}")

    

    # 获取数据元信息
    input_paths = []
    label_paths = []
    scene_names = []
    index_ranges = []
    for sub_path in sub_paths:
        input_path = os.path.join(root, sub_path)
        input_paths.append(input_path)
        scene_names.append(sub_path.split('/')[0])
        # input_path下所有文件名都是xxx.index.exr的格式，找到index的范围
        buffer_names = sorted(os.listdir(input_path))
        index_start = int(buffer_names[0].split('.')[1])
        index_end = int(buffer_names[-1].split('.')[1])
        
        label_path = input_path.replace("3", "6")
        # 确认label_path下的文件的index范围能够与input_path对齐
        buffer_names = sorted(os.listdir(label_path))
        index_start_label = int(buffer_names[0].split('.')[1]) // 2
        index_end_label = int(buffer_names[-1].split('.')[1]) // 2
        label_paths.append(label_path)
        
        index_ranges.append([max(index_start, index_start_label), min(index_end, index_end_label)])
    
    for scene_name, input_path, index_range, label_path in zip(scene_names, input_paths, index_ranges, label_paths):
        save_path = os.path.join(save_path_, scene_name)
        save_path = os.path.join(save_path, label_path.split('/')[-1])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        times = 0
        for i in range( index_range[0] + 1, index_range[1] ):
            label_index_0 = 4 * i + 1
            times+=mul_gen_main(label_index_0, label_path,input_path, save_path, scene_name, programs,debug=False)
            
            print(label_path + ": " + str(label_index_0) + " " + str(label_index_0+1)+" " + str(label_index_0+2))

            # 如果处于debug模式，只循环3次
            if debug:
                if i >= index_range[0] + 9:
                    break
        print(f"计算着色器总运行时间（不包含I/O）: {times/range( index_range[0] + 1, index_range[1] ):.3f} ms")


if __name__ == "__main__":
    #init_egl(1280, 720)

    root = "/disk/zjw/data/zwtdataset/"
    sub_paths = ["MedievalDocks/test1-30fps"]
    main(root, sub_paths, mode="mg_I", debug=False)
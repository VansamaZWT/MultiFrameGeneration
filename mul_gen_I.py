import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import math
import cv2
from OpenGL.GL import *
from OpenGL.GLU import *
from data_io import read_exr, write_exr, read_matrix ,read_gen_matrix
from opengl_util import init_egl,create_compute_shader, create_compute_program, create_texture, read_texture 

def mul_gen_I(mv_0,mv_1,depth_0,depth_1,color_0,color_1,world_pos_0,world_pos_1,stencil_0,stencil_1,vp_matrix_0,vp_matrix_1,vp_matrix_gens,programs):
    height, width = color_1.shape[0], color_1.shape[1]
    alpha = 0.25

    # Step1:generate mix MV
    # 使用传入的ndarray中的数据创建纹理
    #mv_0:0->-1,mv_1:1->0
    #mv1:1->a mv2:0->a
    warp_mv1s = []
    warp_mv2s = []
    warp_mv1_texs = []
    warp_mv2_texs = []
    warp_depths = []
    # 创建多个查询对象，每个着色器调用需要两个查询
    num_shaders = 6*(round(1/alpha) - 1)  # 例如有3次计算着色器调用
    queries_start = glGenQueries(num_shaders)
    queries_end = glGenQueries(num_shaders)
    timer_counter = 0
    for i in range(round(1/alpha) - 1):
        mv1_tex = create_texture(mv_1, width, height)
        mv0_tex = create_texture(mv_0, width, height)
        depth_1_tex = create_texture(depth_1, width, height)
        depth_0_tex = create_texture(depth_0, width, height)
        color_1_tex = create_texture(color_1, width, height)
        stencil_0_tex = create_texture(stencil_1, width, height)
        stencil_1_tex = create_texture(stencil_1, width, height)
        world_pos_0_tex = create_texture(world_pos_0, width, height)
        world_pos_1_tex = create_texture(world_pos_1, width, height)
    
        # 创建存放结果的纹理
        warp_mv1_tex = create_texture(None, width, height)
        warp_mv2_tex = create_texture(None, width, height)
        warp_depth_1_tex = create_texture(None, width, height, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT)
        warp_depth_2_tex = create_texture(None, width, height, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT)


        glUseProgram(programs[0])
        # 将纹理绑定到着色器
        in_textures = [mv0_tex, mv1_tex, depth_1_tex, color_1_tex,stencil_0_tex,stencil_1_tex,world_pos_0_tex,world_pos_1_tex]
        glBindTextures(0, len(in_textures), in_textures)
        out_textures = [warp_mv1_tex,warp_mv2_tex, warp_depth_1_tex]
        glBindImageTextures(0, len(out_textures), out_textures)
        # 绑定uniform矩阵
        vp_loc = glGetUniformLocation(programs[0], "vp_matrix_0")
        if vp_loc >= 0:
            glUniformMatrix4fv(vp_loc, 1, GL_TRUE, vp_matrix_0)
        vp_loc = glGetUniformLocation(programs[0], "vp_matrix_1")
        if vp_loc >= 0:
            glUniformMatrix4fv(vp_loc, 1, GL_TRUE, vp_matrix_1)
        vp_loc = glGetUniformLocation(programs[0], "vp_matrix_gen")
        if vp_loc >= 0:
            glUniformMatrix4fv(vp_loc, 1, GL_TRUE, vp_matrix_gens[i])
        a_loc = glGetUniformLocation(programs[0], "alpha")
        if a_loc >= 0:
            glUniform1f(a_loc, alpha*(i+1))

        num_groups_x = math.ceil(width / 8)
        num_groups_y = math.ceil(height / 8)


        glQueryCounter(queries_start[timer_counter], GL_TIME_ELAPSED)
        

        glDispatchCompute(num_groups_x, num_groups_y, 1)
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

        glQueryCounter(queries_end[timer_counter], GL_TIME_ELAPSED)
        timer_counter+=1

        glUseProgram(programs[1])
        # 将纹理绑定到着色器
        in_textures = [mv0_tex, mv1_tex, depth_0_tex, color_1_tex,stencil_0_tex,stencil_1_tex,world_pos_0_tex,world_pos_1_tex]
        glBindTextures(0, len(in_textures), in_textures)
        out_textures = [warp_mv1_tex,warp_mv2_tex, warp_depth_2_tex]
        glBindImageTextures(0, len(out_textures), out_textures)
        # 绑定uniform矩阵
        vp_loc = glGetUniformLocation(programs[1], "vp_matrix_0")
        if vp_loc >= 0:
            glUniformMatrix4fv(vp_loc, 1, GL_TRUE, vp_matrix_0)
        vp_loc = glGetUniformLocation(programs[1], "vp_matrix_1")
        if vp_loc >= 0:
            glUniformMatrix4fv(vp_loc, 1, GL_TRUE, vp_matrix_1)
        vp_loc = glGetUniformLocation(programs[1], "vp_matrix_gen")
        if vp_loc >= 0:
            glUniformMatrix4fv(vp_loc, 1, GL_TRUE, vp_matrix_gens[i])
        a_loc = glGetUniformLocation(programs[1], "alpha")
        if a_loc >= 0:
            glUniform1f(a_loc, alpha*(i+1))

        num_groups_x = math.ceil(width / 8)
        num_groups_y = math.ceil(height / 8)


        glQueryCounter(queries_start[timer_counter], GL_TIME_ELAPSED)

        glDispatchCompute(num_groups_x, num_groups_y, 1)
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

        glQueryCounter(queries_end[timer_counter], GL_TIME_ELAPSED)
        timer_counter+=1

        warp_mv1_0 = read_texture(warp_mv1_tex, width, height)
        warp_mv2_0 = read_texture(warp_mv2_tex, width, height)
        warp_depth_0 = read_texture(warp_depth_1_tex, width, height, GL_RED_INTEGER, GL_UNSIGNED_INT)
        warp_depth_0 = ((2147483647 - np.expand_dims(warp_depth_0, axis=-1)) / 65535).astype(np.float32)
        warp_mv1_0 = np.reshape(warp_mv1_0, (height, width, 4))
        warp_mv2_0 = np.reshape(warp_mv2_0, (height, width, 4))
        warp_depth_0 = np.reshape(warp_depth_0, (height, width, 1))
        warp_mv1s.append(warp_mv1_0)
        warp_mv2s.append(warp_mv2_0)
        warp_depths.append(warp_depth_0)
        warp_mv1_texs.append(warp_mv1_tex)
        warp_mv2_texs.append(warp_mv2_tex)

    # Step2 inpaint MV
    inpaint_mv1_texs = []
    inpaint_mv2_texs = []
    inpaint_mv1s = []
    inpaint_mv2s = []
    for i in range(round(1/alpha) - 1):
        # 创建存放结果的纹理
        inpaint_mv1_tex = create_texture(None, width, height)
        inpaint_mv2_tex = create_texture(None, width, height)

        glUseProgram(programs[2])
        # 将纹理绑定到着色器
        in_textures = [warp_mv1_texs[i]]
        glBindTextures(0, len(in_textures), in_textures)
        out_textures = [inpaint_mv1_tex]
        glBindImageTextures(0, len(out_textures), out_textures)

        num_groups_x = math.ceil(width / 8)
        num_groups_y = math.ceil(height / 8)

        glQueryCounter(queries_start[timer_counter], GL_TIME_ELAPSED)

        glDispatchCompute(num_groups_x, num_groups_y, 1)
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

        glQueryCounter(queries_end[timer_counter], GL_TIME_ELAPSED)
        timer_counter+=1

        inpaint_mv1 = read_texture(inpaint_mv1_tex, width, height)
        inpaint_mv1 = np.reshape(inpaint_mv1, (height, width, 4))
        inpaint_mv1s.append(inpaint_mv1)
        inpaint_mv1_texs.append(inpaint_mv1_tex)

        glUseProgram(programs[2])
        # 将纹理绑定到着色器
        in_textures = [warp_mv2_texs[i]]
        glBindTextures(0, len(in_textures), in_textures)
        out_textures = [inpaint_mv2_tex]
        glBindImageTextures(0, len(out_textures), out_textures)

        num_groups_x = math.ceil(width / 8)
        num_groups_y = math.ceil(height / 8)

        glQueryCounter(queries_start[timer_counter], GL_TIME_ELAPSED)

        glDispatchCompute(num_groups_x, num_groups_y, 1)
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

        glQueryCounter(queries_end[timer_counter], GL_TIME_ELAPSED)
        timer_counter+=1

        inpaint_mv2 = read_texture(inpaint_mv2_tex, width, height)
        inpaint_mv2 = np.reshape(inpaint_mv2, (height, width, 4))
        inpaint_mv2s.append(inpaint_mv2)
        inpaint_mv2_texs.append(inpaint_mv2_tex)

    warp_color_1_texs = []
    warp_color_2_texs = []
    warp_depth_1_texs = []
    warp_depth_2_texs = []
    warp_color_1s = []
    warp_color_2s = []
    mask_1_texs = []
    mask_2_texs = []

    #Step3:warp
    for i in range(round(1/alpha) - 1):
        warp_color_1_tex = create_texture(None, width, height)
        warp_color_2_tex = create_texture(None, width, height)
        warp_depth_1_tex = create_texture(None, width, height, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT)
        warp_depth_2_tex = create_texture(None, width, height, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT)
        mask_1_tex = create_texture(None, width, height)
        mask_2_tex = create_texture(None, width, height)
        
        depth_0_tex = create_texture(depth_0, width, height)
        depth_1_tex = create_texture(depth_1, width, height)
        color_0_tex = create_texture(color_0, width, height)
        color_1_tex = create_texture(color_1, width, height)


        glUseProgram(programs[3])
        # 将纹理绑定到着色器
        in_textures = [inpaint_mv1_texs[i],inpaint_mv2_texs[i],color_1_tex,color_0_tex,depth_1_tex,depth_0_tex]
        glBindTextures(0, len(in_textures), in_textures)
        out_textures = [warp_color_1_tex, warp_depth_1_tex, warp_color_2_tex, warp_depth_2_tex,mask_1_tex,mask_2_tex]
        glBindImageTextures(0, len(out_textures), out_textures)

        num_groups_x = math.ceil(width / 8)
        num_groups_y = math.ceil(height / 8)

        glQueryCounter(queries_start[timer_counter], GL_TIME_ELAPSED)

        glDispatchCompute(num_groups_x, num_groups_y, 1)
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

        glQueryCounter(queries_end[timer_counter], GL_TIME_ELAPSED)
        timer_counter+=1

        mask_1_texs.append(mask_1_tex)
        mask_2_texs.append(mask_2_tex)
        warp_color_1_texs.append(warp_color_1_tex)
        warp_color_2_texs.append(warp_color_2_tex)
        warp_color_1 = read_texture(warp_color_1_tex, width, height)
        warp_color_1 = np.reshape(warp_color_1, (height, width, 4))
        warp_color_1s.append(warp_color_1)
        warp_color_2 = read_texture(warp_color_2_tex, width, height)
        warp_color_2 = np.reshape(warp_color_2, (height, width, 4))
        warp_color_2s.append(warp_color_2)
        warp_depth_1_texs.append(warp_depth_1_tex)
        warp_depth_2_texs.append(warp_depth_2_tex)

    #Step4: blending
    gen_colors = []
    for i in range(round(1/alpha) - 1):
        gen_color_tex = create_texture(None, width, height)

        glUseProgram(programs[4])
        # 将纹理绑定到着色器
        in_textures = [warp_color_1_texs[i],warp_color_2_texs[i],warp_depth_1_texs[i],warp_depth_2_texs[i],mask_1_texs[i],mask_2_texs[i]]
        glBindTextures(0, len(in_textures), in_textures)
        out_textures = [gen_color_tex]
        glBindImageTextures(0, len(out_textures), out_textures)
        a_loc = glGetUniformLocation(programs[4], "alpha")
        if a_loc >= 0:
            glUniform1f(a_loc, alpha*(i+1))

        num_groups_x = math.ceil(width / 8)
        num_groups_y = math.ceil(height / 8)

        glQueryCounter(queries_start[timer_counter], GL_TIME_ELAPSED)

        glDispatchCompute(num_groups_x, num_groups_y, 1)
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

        glQueryCounter(queries_end[timer_counter], GL_TIME_ELAPSED)
        timer_counter+=1

        gen_color = read_texture(gen_color_tex, width, height)
        gen_color = np.reshape(gen_color, (height, width, 4))
        gen_colors.append(gen_color)

        # print("queries_end:", queries_end)
        # # 确保所有查询数据可用
    done = False
    while not done:
        done = all(glGetQueryObjectiv(q, GL_QUERY_RESULT_AVAILABLE) for q in queries_end)

        # 计算每个计算着色器的运行时间
    total_gpu_time_ns = 0
    for i in range(num_shaders):
        start_time = glGetQueryObjectuiv(queries_start[i], GL_QUERY_RESULT)
        end_time = glGetQueryObjectuiv(queries_end[i], GL_QUERY_RESULT)
        shader_time_ns = end_time - start_time
        total_gpu_time_ns += shader_time_ns

        # 转换为毫秒
    total_gpu_time_ms = total_gpu_time_ns

    return gen_colors,warp_color_1s,warp_color_2s,inpaint_mv1s,inpaint_mv2s,total_gpu_time_ms

def mul_gen_I_init(shader_sources):
    # 初始化opengl和创建着色器
    programs = []
    
    init_egl(1280, 720)
    for source in shader_sources:
        with open(f"shader/{source}", "r") as f:
            shader_source = f.read()
        shader = create_compute_shader(shader_source)
        program = create_compute_program(shader)
        programs.append(program)
    return programs

def mul_gen_main(label_index_0, label_path,seq_path, save_path, scene_name, programs,debug=False):
    """
    :param label_index: 需要预测的标签帧的索引
    :param label_path: 60fps帧的路径
    :param seq_path: 序列帧的路径
    :param save_path: 保存路径
    :param scene_name: 场景名称
    :param program: 着色器程序
    :start_index:预测的第一帧索引
    """
    martrix_path = "/disk/zwt/MultiGen/tmp/MedievalDocks"
    input_index = (label_index_0 - 1) // 4
    label_index = input_index*2 + 1
    print(scene_name)
    mv_0 = read_exr(os.path.join(seq_path, f"{scene_name}MotionVector.{str(input_index).zfill(4)}.exr"), channel=4)
    mv_1 = read_exr(os.path.join(seq_path, f"{scene_name}MotionVector.{str(input_index+1).zfill(4)}.exr"), channel=4)
    depth_0 = read_exr(os.path.join(label_path, f"{scene_name}WorldNormalAndSceneDepth.{str(label_index-1).zfill(4)}.exr"), channel=4)
    depth_1 = read_exr(os.path.join(label_path, f"{scene_name}WorldNormalAndSceneDepth.{str(label_index+1).zfill(4)}.exr"), channel=4)
    color_0 = read_exr(os.path.join(label_path, f"{scene_name}PreTonemapHDRColor.{str(label_index-1).zfill(4)}.exr"), channel=4)
    color_1 = read_exr(os.path.join(label_path, f"{scene_name}PreTonemapHDRColor.{str(label_index+1).zfill(4)}.exr"), channel=4)
    depth_0[...,0:3] = depth_0[...,3:4]
    depth_0[...,3] = 1
    depth_1[...,0:3] = depth_1[...,3:4]
    depth_1[...,3] = 1
    world_pos_0 = read_exr(os.path.join(label_path, f"{scene_name}WorldPosition.{str(label_index-1).zfill(4)}.exr"), channel=4)
    world_pos_1 = read_exr(os.path.join(label_path, f"{scene_name}WorldPosition.{str(label_index+1).zfill(4)}.exr"), channel=4)
    stencil_0 = read_exr(os.path.join(label_path, f"{scene_name}MyStencil.{str(label_index-1).zfill(4)}.exr"), channel=4)
    stencil_1 = read_exr(os.path.join(label_path, f"{scene_name}MyStencil.{str(label_index+1).zfill(4)}.exr"), channel=4)
    
    vp_matrix_0 = read_matrix(os.path.join(label_path, f"{scene_name}Matrix.{str(label_index-1).zfill(4)}.txt"))
    vp_matrix_1 = read_matrix(os.path.join(label_path, f"{scene_name}Matrix.{str(label_index+1).zfill(4)}.txt"))
    # vp_matrix_gen_1 = read_gen_matrix(os.path.join(martrix_path, f"{scene_name}Matrix.{str(label_index_0).zfill(4)}.txt"))
    # vp_matrix_gen_2 = read_gen_matrix(os.path.join(martrix_path, f"{scene_name}Matrix.{str(label_index_0+1).zfill(4)}.txt"))
    # vp_matrix_gen_3 = read_gen_matrix(os.path.join(martrix_path, f"{scene_name}Matrix.{str(label_index_0+2).zfill(4)}.txt"))
    vp_matrix_gen_1 = read_gen_matrix(os.path.join(martrix_path, f"{scene_name}Matrix.{str(label_index_0).zfill(4)}.txt"))
    vp_matrix_gen_2 = read_gen_matrix(os.path.join(martrix_path, f"{scene_name}Matrix.{str(label_index_0+1).zfill(4)}.txt"))
    vp_matrix_gen_3 = read_gen_matrix(os.path.join(martrix_path, f"{scene_name}Matrix.{str(label_index_0+2).zfill(4)}.txt"))
    vp_matrix_gens = [vp_matrix_gen_1,vp_matrix_gen_2,vp_matrix_gen_3]
    gens,w1,w2,mv1,mv2,time = mul_gen_I(mv_0,mv_1,depth_0,depth_1,color_0,color_1,world_pos_0,world_pos_1,stencil_0,stencil_1,vp_matrix_0,vp_matrix_1,vp_matrix_gens,programs)
    print(f"计算着色器总运行时间（不包含I/O）: {time:.3f} ms")
    # mv = test(mv_0)
    # write_exr(os.path.join(save_path, f"{scene_name}mv.{str(label_index-1).zfill(4)}.exr"),mv)
    write_exr(os.path.join(save_path, f"{scene_name}GenColor.{str(label_index_0).zfill(4)}.exr"),gens[0])
    write_exr(os.path.join(save_path, f"{scene_name}GenColor.{str(label_index_0+1).zfill(4)}.exr"),gens[1])
    write_exr(os.path.join(save_path, f"{scene_name}GenColor.{str(label_index_0+2).zfill(4)}.exr"),gens[2])
    # write_exr(os.path.join(save_path, f"{scene_name}GTMV1.{str(label_index-1).zfill(4)}.exr"),mv_1)
    write_exr(os.path.join(save_path, f"{scene_name}Warpcolor1.{str(label_index_0).zfill(4)}.exr"),w1[0])
    write_exr(os.path.join(save_path, f"{scene_name}Warpcolor1.{str(label_index_0+1).zfill(4)}.exr"),w1[1])
    write_exr(os.path.join(save_path, f"{scene_name}Warpcolor1.{str(label_index_0+2).zfill(4)}.exr"),w1[2])
    write_exr(os.path.join(save_path, f"{scene_name}Warpcolor2.{str(label_index_0).zfill(4)}.exr"),w2[0])
    write_exr(os.path.join(save_path, f"{scene_name}Warpcolor2.{str(label_index_0+1).zfill(4)}.exr"),w2[1])
    write_exr(os.path.join(save_path, f"{scene_name}Warpcolor2.{str(label_index_0+2).zfill(4)}.exr"),w2[2])

    write_exr(os.path.join(save_path, f"{scene_name}WarpMV1.{str(label_index_0).zfill(4)}.exr"),mv1[0])
    write_exr(os.path.join(save_path, f"{scene_name}WarpMV1.{str(label_index_0+1).zfill(4)}.exr"),mv1[1])
    write_exr(os.path.join(save_path, f"{scene_name}WarpMV1.{str(label_index_0+2).zfill(4)}.exr"),mv1[2])
    write_exr(os.path.join(save_path, f"{scene_name}WarpMV2.{str(label_index_0).zfill(4)}.exr"),mv2[0])
    write_exr(os.path.join(save_path, f"{scene_name}WarpMV2.{str(label_index_0+1).zfill(4)}.exr"),mv2[1])
    write_exr(os.path.join(save_path, f"{scene_name}WarpMV2.{str(label_index_0+2).zfill(4)}.exr"),mv2[2])

    return time
    
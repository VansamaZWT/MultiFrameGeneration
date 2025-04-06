import os
import numpy as np
import math
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from data_io import read_exr, write_exr, read_matrix
from opengl_util import create_compute_shader, create_compute_program, create_texture, create_window, read_texture

def create_texture_bgbuffer(width, height, mipmap_levels):
    """创建支持多级 Mipmap 的纹理"""
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    
    # 创建每一层的存储空间
    for level in range(mipmap_levels):
        level_width = max(1, width // (2 ** level))
        level_height = max(1, height // (2 ** level))
        glTexImage2D(GL_TEXTURE_2D, level, GL_RGBA32F, level_width, level_height, 0, GL_RGBA, GL_FLOAT, None)
    
    # 设置 Mipmap 参数
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glBindTexture(GL_TEXTURE_2D, 0)
    return texture

def update_background_buffer(compute_shader, background_buffer, width, height):
    """调用 Compute Shader 更新 background_buffer"""
    glUseProgram(compute_shader)

    # 绑定输入纹理和背景缓冲
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D)
    glUniform1i(glGetUniformLocation(compute_shader, "input_texture"), 0)

    glBindImageTexture(0, background_buffer, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA32F)
    glUniform1i(glGetUniformLocation(compute_shader, "background_buffer"), 0)

    # 调用 Compute Shader，工作组尺寸匹配初始纹理大小
    glDispatchCompute((width + 7) // 8, (height + 7) // 8, 1)
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

def gffe_bc_init():
    # 初始化opengl和创建着色器
    create_window(1280, 720, "gffe_bc")
    with open("shaders/gffe_bc.comp", "r") as f:
        shader_source = f.read()
    shader = create_compute_shader(shader_source)
    program = create_compute_program(shader)
    glUseProgram(program)
    return program

def main():
    # 初始化 OpenGL 上下文
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA)
    glutCreateWindow("Mipmap Background Buffer")
    
    # 初始化纹理和 Compute Shader
    width, height = 512, 512  # 示例分辨率
    mipmap_levels = int(np.log2(max(width, height))) + 1
    background_buffer = create_texture_bgbuffer(width, height, mipmap_levels)
    
    # 示例输入纹理（当前帧图片）
    #input_texture = create_texture(width, height, 1)  # 假设只有 1 层 Mipmap

    # 加载并编译 Compute Shader
    compute_shader_source = "./shader/test.comp"
    compute_shader = glCreateShader(GL_COMPUTE_SHADER)
    glShaderSource(compute_shader, compute_shader_source)
    glCompileShader(compute_shader)
    program = glCreateProgram()
    glAttachShader(program, compute_shader)
    glLinkProgram(program)

    # 调用更新函数
    update_background_buffer(program, background_buffer, width, height)

if __name__ == "__main__":
    main()

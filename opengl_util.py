import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.EGL import *
from ctypes import byref


def init_egl(width, height):
    # 初始化EGL显示连接
    egl_display = eglGetDisplay(EGL_DEFAULT_DISPLAY)
    if egl_display == EGL_NO_DISPLAY:
        raise Exception("无法获取EGL显示连接")
    
    # 初始化显示
    major, minor = EGLint(), EGLint()
    if not eglInitialize(egl_display, major, minor):
        raise Exception("无法初始化EGL")

    # 绑定OpenGL API
    if not eglBindAPI(EGL_OPENGL_API):
        raise Exception("无法绑定OpenGL API")
    
     # 配置EGL
    config_attribs = [
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_BLUE_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_RED_SIZE, 8,
        EGL_DEPTH_SIZE, 24,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_NONE
    ]
    config_attribs = (EGLint * len(config_attribs))(*config_attribs)
    
    num_configs = EGLint()
    egl_config = EGLConfig()
    if not eglChooseConfig(egl_display, config_attribs, byref(egl_config), 1, byref(num_configs)):
        raise Exception("无法选择EGL配置")
    
    # 创建离屏缓冲区
    pbuffer_attribs = [
        EGL_WIDTH, width,
        EGL_HEIGHT, height,
        EGL_NONE
    ]
    pbuffer_attribs = (EGLint * len(pbuffer_attribs))(*pbuffer_attribs)
    egl_surface = eglCreatePbufferSurface(egl_display, egl_config, pbuffer_attribs)
    if egl_surface == EGL_NO_SURFACE:
        raise Exception("无法创建EGL表面")
    
    # 创建OpenGL上下文
    context_attribs = [
        EGL_CONTEXT_MAJOR_VERSION, 4,
        EGL_CONTEXT_MINOR_VERSION, 3,
        EGL_NONE
    ]
    context_attribs = (EGLint * len(context_attribs))(*context_attribs)
    egl_context = eglCreateContext(egl_display, egl_config, EGL_NO_CONTEXT, context_attribs)
    if egl_context == EGL_NO_CONTEXT:
        raise Exception("无法创建EGL上下文")
    
    # 使上下文成为当前上下文
    if not eglMakeCurrent(egl_display, egl_surface, egl_surface, egl_context):
        raise Exception("无法设置当前EGL上下文")
    
    # 检查浮点数原子操作扩展是否可用
    # num_extensions = glGetIntegerv(GL_NUM_EXTENSIONS)
    # extensions = [glGetStringi(GL_EXTENSIONS, i).decode('utf-8') 
    #              for i in range(num_extensions)]
    
    # if "GL_EXT_shader_atomic_float" not in extensions:
    #     raise Exception("GL_EXT_shader_atomic_float 扩展不可用")
    # if "GL_EXT_shader_atomic_float2" not in extensions:
    #     raise Exception("GL_EXT_shader_atomic_float2 扩展不可用")
    
    return egl_display, egl_surface, egl_context


# 创建并编译计算着色器
def create_compute_shader(shader_source):
    shader = glCreateShader(GL_COMPUTE_SHADER)
    glShaderSource(shader, shader_source)
    glCompileShader(shader)
    compile_status = glGetShaderiv(shader, GL_COMPILE_STATUS)
    if not compile_status:
        info_log = glGetShaderInfoLog(shader)
        raise Exception(f"Compute shader compilation failed: {info_log}")
    return shader


# 创建并链接计算着色器程序
def create_compute_program(shader):
    program = glCreateProgram()
    glAttachShader(program, shader)
    glLinkProgram(program)
    link_status = glGetProgramiv(program, GL_LINK_STATUS)
    if not link_status:
        info_log = glGetProgramInfoLog(program)
        raise Exception(f"Compute shader program linking failed: {info_log}")
    return program
    

# 创建纹理
def create_texture(image, width, height, internal_format=GL_RGBA32F, format=GL_RGBA, type=GL_FLOAT):
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexStorage2D(GL_TEXTURE_2D, 1, internal_format, width, height)
    if image is not None:
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, format, type, image)
    glBindTexture(GL_TEXTURE_2D, 0)
    return texture


# 用glReadPixels读取纹理数据
def read_texture(texture, width, height, format=GL_RGBA, type=GL_FLOAT):
    framebuffer = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)
    image = glReadPixels(0, 0, width, height, format, type)
    glFinish()
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    glDeleteFramebuffers(1, [framebuffer])
    return image

def create_background_buffer_textures(images,width, height, num_levels=4):
    textures = glGenTextures(num_levels)  # 创建 4 个纹理对象
    current_width, current_height = width, height
    
    for i, texture in enumerate(textures):
        glBindTexture(GL_TEXTURE_2D, texture)  # 绑定当前纹理
        
        # 为纹理分配存储空间（每层大小逐渐减小）
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, current_width, current_height)
        
        # 设置纹理参数
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        if images is not None:
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, current_width,  current_height, GL_RGBA, GL_FLOAT, images[i])
        glBindTexture(GL_TEXTURE_2D, 0)
        # 更新下一层的分辨率 (每次减半)
        current_width = max(1, current_width // 2)
        current_height = max(1, current_height // 2)
    
    glBindTexture(GL_TEXTURE_2D, 0)  # 解绑纹理
    return textures  # 返回纹理对象 ID 列表

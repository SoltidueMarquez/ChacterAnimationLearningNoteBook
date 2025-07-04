import sys
import os
import numpy as np
import scipy.interpolate as interpolate
import scipy.ndimage.filters as filters

# 添加自定义模块路径
sys.path.append('./motion')

# 导入自定义动画处理模块
import BVH as BVH
import Animation as Animation
from Quaternions import Quaternions
from Pivots import Pivots
from Learning import RBF

""" 全局配置选项 """
rng = np.random.RandomState(1234)  # 固定随机种子以确保可重复性
to_meters = 5.6444  # 将BVH单位转换为米的比例因子
window = 60  # 用于输入/输出的时间窗口大小（前后各60帧）
njoints = 31  # 骨骼关节数量

""" 动画数据文件列表 """
# 包含多种地形上的运动数据：平地、上楼梯、新捕捉数据等
# 每个原始动画都有对应的镜像版本（用于数据增强）
data_terrain = [
    # 平地行走动画
    './data/animations/LocomotionFlat01_000.bvh',
    # ... (其他平地动画)
    
    # 平地动画的镜像版本
    './data/animations/LocomotionFlat01_000_mirror.bvh',
    # ... (其他镜像动画)
    
    # 上楼梯动画
    './data/animations/WalkingUpSteps01_000.bvh',
    # ... (其他上楼梯动画)
    
    # 上楼梯动画的镜像版本
    './data/animations/WalkingUpSteps01_000_mirror.bvh',
    # ... (其他镜像动画)
    
    # 新捕捉的动画数据
    './data/animations/NewCaptures01_000.bvh',
    # ... (其他新捕捉动画)
    
    # 新捕捉动画的镜像版本
    './data/animations/NewCaptures01_000_mirror.bvh',
    # ... (其他镜像动画)
]

""" 加载地形高度图数据 """
# 从预处理的.npz文件中加载地形块和坐标信息
patches_database = np.load('patches.npz')
patches = patches_database['X'].astype(np.float32)  # 地形高度图网格
patches_coord = patches_database['C'].astype(np.float32)  # 地形坐标信息

""" 数据处理函数 """

def process_data(anim, phase, gait, type='flat'):
    """
    处理动画数据，提取运动特征
    参数:
        anim: Animation对象，包含骨骼动画数据
        phase: 一维数组，步态相位信息（0-1）
        gait: 二维数组，步态参数向量
        type: 地形类型（'flat', 'rocky', 'jumpy', 'beam'）
    返回:
        Pc: 相位信息
        Xc: 输入特征向量
        Yc: 输出目标向量
    """
    
    """ 正向运动学计算 """
    global_xforms = Animation.transforms_global(anim)  # 计算全局变换矩阵
    global_positions = global_xforms[:,:,:3,3] / global_xforms[:,:,3:,3]  # 提取全局位置
    global_rotations = Quaternions.from_transforms(global_xforms)  # 转换为四元数
    
    """ 计算前进方向 """
    # 关节索引：肩膀和髋部
    sdr_l, sdr_r, hip_l, hip_r = 18, 25, 2, 7
    # 计算躯干横向向量（左肩到右肩 + 左髋到右髋）
    across = (
        (global_positions[:,sdr_l] - global_positions[:,sdr_r]) + 
        (global_positions[:,hip_l] - global_positions[:,hip_r]))
    across = across / np.sqrt((across**2).sum(axis=-1))[...,np.newaxis]  # 归一化
    
    """ 平滑前进方向 """
    direction_filterwidth = 20  # 高斯滤波窗口大小
    # 计算前进方向（横向向量与垂直向量的叉积）
    forward = filters.gaussian_filter1d(
        np.cross(across, np.array([[0,1,0]])), direction_filterwidth, axis=0, mode='nearest')    
    forward = forward / np.sqrt((forward**2).sum(axis=-1))[...,np.newaxis]  # 归一化

    # 计算根骨骼旋转，使前进方向对齐Z轴
    root_rotation = Quaternions.between(forward, 
        np.array([[0,0,1]]).repeat(len(forward), axis=0))[:,np.newaxis] 
    
    """ 转换到局部空间 """
    # 相对根骨骼的局部位置
    local_positions = global_positions.copy()
    local_positions[:,:,0] = local_positions[:,:,0] - local_positions[:,0:1,0]  # X轴归零
    local_positions[:,:,2] = local_positions[:,:,2] - local_positions[:,0:1,2]  # Z轴归零
    
    # 应用根骨骼旋转
    local_positions = root_rotation[:-1] * local_positions[:-1]  # 局部位置
    local_velocities = root_rotation[:-1] * (global_positions[1:] - global_positions[:-1])  # 局部速度
    local_rotations = abs((root_rotation[:-1] * global_rotations[:-1])).log()  # 局部旋转（对数四元数）
    
    # 根骨骼速度和旋转速度
    root_velocity = root_rotation[:-1] * (global_positions[1:,0:1] - global_positions[:-1,0:1])
    root_rvelocity = Pivots.from_quaternions(root_rotation[1:] * -root_rotation[:-1]).ps
    
    """ 脚部触地检测 """
    fid_l, fid_r = np.array([4,5]), np.array([9,10])  # 左右脚关节索引
    velfactor = np.array([0.02, 0.02])  # 速度阈值（判断触地）
    
    # 左脚移动距离计算
    feet_l_x = (global_positions[1:,fid_l,0] - global_positions[:-1,fid_l,0])**2
    feet_l_y = (global_positions[1:,fid_l,1] - global_positions[:-1,fid_l,1])**2
    feet_l_z = (global_positions[1:,fid_l,2] - global_positions[:-1,fid_l,2])**2
    feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor)).astype(np.float)  # 低于阈值则为触地
    
    # 右脚移动距离计算
    feet_r_x = (global_positions[1:,fid_r,0] - global_positions[:-1,fid_r,0])**2
    feet_r_y = (global_positions[1:,fid_r,1] - global_positions[:-1,fid_r,1])**2
    feet_r_z = (global_positions[1:,fid_r,2] - global_positions[:-1,fid_r,2])**2
    feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float)  # 低于阈值则为触地
    
    """ 相位变化计算 """
    dphase = phase[1:] - phase[:-1]  # 相位差分
    dphase[dphase < 0] = (1.0-phase[:-1]+phase[1:])[dphase < 0]  # 处理相位循环（1.0 -> 0.0）
    
    """ 调整下蹲步态参数 """
    if type == 'flat':  # 只在平地动画中调整
        crouch_low, crouch_high = 80, 130  # 下蹲高度范围
        head = 16  # 头部关节索引
        # 根据头部高度计算下蹲程度（0-1）
        gait[:-1,3] = 1 - np.clip((global_positions[:-1,head,1] - crouch_low) / (crouch_high - crouch_low), 0, 1)
        gait[-1,3] = gait[-2,3]  # 最后帧使用前一帧的值

    """ 滑动窗口处理 """
    Pc, Xc, Yc = [], [], []  # 初始化结果列表
    
    # 以步长1滑动窗口（中心点为i，前后各window帧）
    for i in range(window, len(anim)-window-1, 1):
        # 轨迹位置（相对当前帧，降采样10倍）
        rootposs = root_rotation[i:i+1,0] * (global_positions[i-window:i+window:10,0] - global_positions[i:i+1,0])
        # 轨迹方向（降采样10倍）
        rootdirs = root_rotation[i:i+1,0] * forward[i-window:i+window:10]    
        # 步态参数（降采样10倍）
        rootgait = gait[i-window:i+window:10]
        
        Pc.append(phase[i])  # 当前相位
        
        # 构建输入特征向量Xc
        Xc.append(np.hstack([
                rootposs[:,0].ravel(), rootposs[:,2].ravel(), # 轨迹位置 (X,Z)
                rootdirs[:,0].ravel(), rootdirs[:,2].ravel(), # 轨迹方向 (X,Z)
                rootgait[:,0].ravel(), rootgait[:,1].ravel(), # 步态参数
                rootgait[:,2].ravel(), rootgait[:,3].ravel(), 
                rootgait[:,4].ravel(), rootgait[:,5].ravel(), 
                local_positions[i-1].ravel(),  # 关节位置 (前一帧)
                local_velocities[i-1].ravel(), # 关节速度 (前一帧)
                ]))
        
        # 下一时刻轨迹位置和方向
        rootposs_next = root_rotation[i+1:i+2,0] * (global_positions[i+1:i+window+1:10,0] - global_positions[i+1:i+2,0])
        rootdirs_next = root_rotation[i+1:i+2,0] * forward[i+1:i+window+1:10]   
        
        # 构建输出目标向量Yc
        Yc.append(np.hstack([
                root_velocity[i,0,0].ravel(), # 根骨骼速度X
                root_velocity[i,0,2].ravel(), # 根骨骼速度Z
                root_rvelocity[i].ravel(),    # 根骨骼旋转速度
                dphase[i],                    # 相位变化
                np.concatenate([feet_l[i], feet_r[i]], axis=-1), # 脚部触地状态
                rootposs_next[:,0].ravel(), rootposs_next[:,2].ravel(), # 下一时刻轨迹位置
                rootdirs_next[:,0].ravel(), rootdirs_next[:,2].ravel(), # 下一时刻轨迹方向
                local_positions[i].ravel(),  # 关节位置 (当前帧)
                local_velocities[i].ravel(), # 关节速度 (当前帧)
                local_rotations[i].ravel()   # 关节旋转 (对数四元数)
                ]))
                                                
    return np.array(Pc), np.array(Xc), np.array(Yc)
    

""" 地形高度采样函数 """    

def patchfunc(P, Xp, hscale=3.937007874, vscale=3.0):
    """
    在地形高度图上采样高度值（双线性插值）
    参数:
        P: 地形高度图 (nsamples, height, width)
        Xp: 采样点坐标 (N, 2)
        hscale: 水平缩放因子（单位转换）
        vscale: 垂直缩放因子（高度增强）
    返回:
        插值后的高度值 (N, 1)
    """
    # 缩放坐标并居中
    Xp = Xp / hscale + np.array([P.shape[1]//2, P.shape[2]//2])
    
    # 计算双线性插值权重
    A = np.fmod(Xp, 1.0)  # 小数部分
    # 四个相邻网格点
    X0 = np.clip(np.floor(Xp).astype(np.int), 0, np.array([P.shape[1]-1, P.shape[2]-1])
    X1 = np.clip(np.ceil (Xp).astype(np.int), 0, np.array([P.shape[1]-1, P.shape[2]-1]))
    
    # 四角高度值
    H0 = P[:,X0[:,0],X0[:,1]]
    H1 = P[:,X0[:,0],X1[:,1]]
    H2 = P[:,X1[:,0],X0[:,1]]
    H3 = P[:,X1[:,0],X1[:,1]]
    
    # X方向插值
    HL = (1-A[:,0]) * H0 + (A[:,0]) * H2
    HR = (1-A[:,0]) * H1 + (A[:,0]) * H3
    
    # Y方向插值并缩放
    return (vscale * ((1-A[:,1]) * HL + (A[:,1]) * HR))[...,np.newaxis]
    

def process_heights(anim, nsamples=10, type='flat'):
    """
    处理地形高度信息
    参数:
        anim: Animation对象
        nsamples: 生成的地形样本数量
        type: 地形类型
    返回:
        root_terrains: 轨迹地形高度
        root_averages: 轨迹平均高度
    """
    """ 正向运动学计算 """
    global_xforms = Animation.transforms_global(anim)
    global_positions = global_xforms[:,:,:3,3] / global_xforms[:,:,3:,3]
    global_rotations = Quaternions.from_transforms(global_xforms)
    
    """ 计算前进方向（与process_data相同） """
    sdr_l, sdr_r, hip_l, hip_r = 18, 25, 2, 7
    across = (
        (global_positions[:,sdr_l] - global_positions[:,sdr_r]) + 
        (global_positions[:,hip_l] - global_positions[:,hip_r]))
    across = across / np.sqrt((across**2).sum(axis=-1))[...,np.newaxis]
    
    direction_filterwidth = 20
    forward = filters.gaussian_filter1d(
        np.cross(across, np.array([[0,1,0]])), direction_filterwidth, axis=0, mode='nearest')    
    forward = forward / np.sqrt((forward**2).sum(axis=-1))[...,np.newaxis]

    root_rotation = Quaternions.between(forward, 
        np.array([[0,0,1]]).repeat(len(forward), axis=0))[:,np.newaxis] 

    """ 脚部触地检测（与process_data相同） """
    fid_l, fid_r = np.array([4,5]), np.array([9,10])
    velfactor = np.array([0.02, 0.02])
    
    feet_l_x = (global_positions[1:,fid_l,0] - global_positions[:-1,fid_l,0])**2
    feet_l_y = (global_positions[1:,fid_l,1] - global_positions[:-1,fid_l,1])**2
    feet_l_z = (global_positions[1:,fid_l,2] - global_positions[:-1,fid_l,2])**2
    feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor))
    
    feet_r_x = (global_positions[1:,fid_r,0] - global_positions[:-1,fid_r,0])**2
    feet_r_y = (global_positions[1:,fid_r,1] - global_positions[:-1,fid_r,1])**2
    feet_r_z = (global_positions[1:,fid_r,2] - global_positions[:-1,fid_r,2])**2
    feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor))
    
    # 扩展最后帧
    feet_l = np.concatenate([feet_l, feet_l[-1:]], axis=0)
    feet_r = np.concatenate([feet_r, feet_r[-1:]], axis=0)
    
    """ 脚部高度偏移 """
    toe_h, heel_h = 4.0, 5.0  # 脚趾和脚跟的高度偏移
    
    """ 提取脚部触地点位置 """
    feet_down = np.concatenate([
        global_positions[feet_l[:,0],fid_l[0]] - np.array([0, heel_h, 0]),  # 左脚跟
        global_positions[feet_l[:,1],fid_l[1]] - np.array([0,  toe_h, 0]),  # 左脚趾
        global_positions[feet_r[:,0],fid_r[0]] - np.array([0, heel_h, 0]),  # 右脚跟
        global_positions[feet_r[:,1],fid_r[1]] - np.array([0,  toe_h, 0])   # 右脚趾
    ], axis=0)
    
    """ 提取脚部离地点位置 """
    feet_up = np.concatenate([
        global_positions[~feet_l[:,0],fid_l[0]] - np.array([0, heel_h, 0]),
        global_positions[~feet_l[:,1],fid_l[1]] - np.array([0,  toe_h, 0]),
        global_positions[~feet_r[:,0],fid_r[0]] - np.array([0, heel_h, 0]),
        global_positions[~feet_r[:,1],fid_r[1]] - np.array([0,  toe_h, 0])
    ], axis=0)
    
    """ 触地点统计分析 """
    feet_down_xz = np.concatenate([feet_down[:,0:1], feet_down[:,2:3]], axis=-1)  # XZ平面位置
    feet_down_xz_mean = feet_down_xz.mean(axis=0)  # 平均位置
    feet_down_y = feet_down[:,1:2]  # 高度
    feet_down_y_mean = feet_down_y.mean(axis=0)  # 平均高度
    feet_down_y_std  = feet_down_y.std(axis=0)   # 高度标准差
        
    """ 离地点位置 """
    feet_up_xz = np.concatenate([feet_up[:,0:1], feet_up[:,2:3]], axis=-1)
    feet_up_y = feet_up[:,1:2]
    
    if len(feet_down_xz) == 0:  # 没有触地点数据
        terr_func = lambda Xp: np.zeros_like(Xp)[:,:1][np.newaxis].repeat(nsamples, axis=0)
        
    elif type == 'flat':  # 平地形
        terr_func = lambda Xp: np.zeros_like(Xp)[:,:1][np.newaxis].repeat(nsamples, axis=0) + feet_down_y_mean
    
    else:  # 复杂地形
        """ 地形高度拟合 """
        # 触地点地形高度采样
        terr_down_y = patchfunc(patches, feet_down_xz - feet_down_xz_mean)
        terr_down_y_mean = terr_down_y.mean(axis=1)  # 每个地形样本的平均高度
        terr_down_y_std  = terr_down_y.std(axis=1)   # 标准差
        # 离地点地形高度采样
        terr_up_y = patchfunc(patches, feet_up_xz - feet_down_xz_mean)
        
        """ 地形拟合误差计算 """
        # 触地点拟合误差
        terr_down_err = 0.1 * ((
            (terr_down_y - terr_down_y_mean[:,np.newaxis]) -  # 地形相对高度
            (feet_down_y - feet_down_y_mean)[np.newaxis])**2)[...,0].mean(axis=1)  # 与实际高度差的MSE
        
        # 离地点拟合误差（只惩罚地形低于脚的情况）
        terr_up_err = (np.maximum(
            (terr_up_y - terr_down_y_mean[:,np.newaxis]) -  # 地形相对高度
            (feet_up_y - feet_down_y_mean)[np.newaxis], 0.0)**2)[...,0].mean(axis=1)
        
        """ 跳跃地形特殊处理 """
        if type == 'jumpy':
            terr_over_minh = 5.0  # 最小离地高度
            # 惩罚地形过高导致脚穿入地面的情况
            terr_over_err = (np.maximum(
                ((feet_up_y - feet_down_y_mean)[np.newaxis] - terr_over_minh) -
                (terr_up_y - terr_down_y_mean[:,np.newaxis]), 0.0)**2)[...,0].mean(axis=1)
        else:
            terr_over_err = 0.0
        
        """ 独木桥地形特殊处理 """
        if type == 'beam':
            beam_samples = 1  # 额外采样点数量
            beam_min_height = 40.0  # 最小高度差（模拟独木桥）

            # 角色位置
            beam_c = global_positions[:,0]
            beam_c_xz = np.concatenate([beam_c[:,0:1], beam_c[:,2:3]], axis=-1)
            beam_c_y = patchfunc(patches, beam_c_xz - feet_down_xz_mean)

            # 随机偏移位置（模拟角色周围点）
            beam_o = (
                beam_c.repeat(beam_samples, axis=0) + np.array([50, 0, 50]) * 
                rng.normal(size=(len(beam_c)*beam_samples, 3)))

            beam_o_xz = np.concatenate([beam_o[:,0:1], beam_o[:,2:3]], axis=-1)
            beam_o_y = patchfunc(patches, beam_o_xz - feet_down_xz_mean)

            # 计算远离角色的点
            beam_pdist = np.sqrt(((beam_o[:,np.newaxis] - beam_c[np.newaxis,:])**2).sum(axis=-1))
            beam_far = (beam_pdist > 15).all(axis=1)

            # 惩罚桥两侧点过高的情况
            terr_beam_err = (np.maximum(beam_o_y[:,beam_far] - 
                (beam_c_y.repeat(beam_samples, axis=1)[:,beam_far] - 
                 beam_min_height), 0.0)**2)[...,0].mean(axis=1)
        else:
            terr_beam_err = 0.0
        
        """ 综合地形拟合误差 """
        terr = terr_down_err + terr_up_err + terr_over_err + terr_beam_err
        
        """ 选择最佳拟合地形 """
        terr_ids = np.argsort(terr)[:nsamples]  # 误差最小的地形样本
        terr_patches = patches[terr_ids]  # 选择最佳地形
        # 基础地形函数
        terr_basic_func = lambda Xp: (
            (patchfunc(terr_patches, Xp - feet_down_xz_mean) -  # 采样地形高度
            terr_down_y_mean[terr_ids][:,np.newaxis]) + feet_down_y_mean)  # 调整到绝对高度
        
        """ 地形残差修正 """
        # 计算实际高度与基础地形的残差
        terr_residuals = feet_down_y - terr_basic_func(feet_down_xz)
        # 使用RBF（径向基函数）拟合残差
        terr_fine_func = [RBF(smooth=0.1, function='linear') for _ in range(nsamples)]
        for i in range(nsamples): 
            terr_fine_func[i].fit(feet_down_xz, terr_residuals[i])
        # 最终地形函数 = 基础地形 + 残差修正
        terr_func = lambda Xp: (terr_basic_func(Xp) + np.array([ff(Xp) for ff in terr_fine_func]))
        
        
    """ 计算轨迹地形高度 """
    root_offsets_c = global_positions[:,0]  # 根骨骼位置
    # 根骨骼右侧偏移点
    root_offsets_r = (-root_rotation[:,0] * np.array([[+25, 0, 0]])) + root_offsets_c
    # 根骨骼左侧偏移点
    root_offsets_l = (-root_rotation[:,0] * np.array([[-25, 0, 0]])) + root_offsets_c

    # 采样地形高度
    root_heights_c = terr_func(root_offsets_c[:,np.array([0,2])])[...,0]  # 中心点
    root_heights_r = terr_func(root_offsets_r[:,np.array([0,2])])[...,0]  # 右侧点
    root_heights_l = terr_func(root_offsets_l[:,np.array([0,2])])[...,0]  # 左侧点
    
    """ 提取窗口地形高度特征 """
    root_terrains = []
    root_averages = []
    for i in range(window, len(anim)-window, 1): 
        # 构建地形特征向量 [右侧, 中心, 左侧] * 时间窗口
        root_terrains.append(
            np.concatenate([
                root_heights_r[:,i-window:i+window:10],  # 右侧点高度（降采样）
                root_heights_c[:,i-window:i+window:10],  # 中心点高度
                root_heights_l[:,i-window:i+window:10]], # 左侧点高度
            axis=1))
        # 当前窗口中心点平均高度
        root_averages.append(root_heights_c[:,i-window:i+window:10].mean(axis=1))
     
    # 调整维度 [nsamples, nwindows, features]
    root_terrains = np.swapaxes(np.array(root_terrains), 0, 1)
    root_averages = np.swapaxes(np.array(root_averages), 0, 1)
    
    return root_terrains, root_averages

""" 主处理流程：提取相位、输入、输出 """

P, X, Y = [], [], []  # 全局存储列表
            
for data in data_terrain:  # 遍历所有动画文件
    
    print('处理动画片段: %s' % data)
    
    """ 确定地形类型 """
    # 根据文件名分配地形类型
    if   'LocomotionFlat12_000' in data: type = 'jumpy'  # 跳跃地形
    elif 'NewCaptures01_000'    in data: type = 'flat'   # 平地
    elif 'NewCaptures02_000'    in data: type = 'flat'
    elif 'NewCaptures03_000'    in data: type = 'jumpy'
    elif 'NewCaptures03_001'    in data: type = 'jumpy'
    elif 'NewCaptures03_002'    in data: type = 'jumpy'
    elif 'NewCaptures04_000'    in data: type = 'jumpy'
    elif 'WalkingUpSteps06_000' in data: type = 'beam'   # 独木桥
    elif 'WalkingUpSteps09_000' in data: type = 'flat'
    elif 'WalkingUpSteps10_000' in data: type = 'flat'
    elif 'WalkingUpSteps11_000' in data: type = 'flat'
    elif 'Flat' in data: type = 'flat'  # 所有包含"Flat"的文件
    else: type = 'rocky'  # 默认崎岖地形
    
    """ 加载BVH动画数据 """
    anim, names, _ = BVH.load(data)
    anim.offsets *= to_meters  # 缩放骨骼尺寸
    anim.positions *= to_meters  # 缩放位置
    anim = anim[::2]  # 降采样（帧率减半）

    """ 加载相位和步态数据 """
    phase = np.loadtxt(data.replace('.bvh', '.phase'))[::2]  # 降采样匹配动画
    gait = np.loadtxt(data.replace('.bvh', '.gait'))[::2]

    """ 合并步态参数 """
    # 原始步态参数: [站,走,慢跑,跑,左转,右转,蹲,爬]
    # 合并后: [站,走,跑(慢跑+跑),蹲(蹲+爬),左转,右转]
    gait = np.concatenate([
        gait[:,0:1],  # 站立
        gait[:,1:2],  # 行走
        gait[:,2:3] + gait[:,3:4],  # 跑步 = 慢跑+跑
        gait[:,4:5] + gait[:,6:7],  # 下蹲 = 蹲+爬
        gait[:,5:6],  # 左转
        gait[:,7:8]   # 右转
    ], axis=-1)

    """ 预处理动画数据 """
    Pc, Xc, Yc = process_data(anim, phase, gait, type=type)

    """ 加载脚步标记 """
    with open(data.replace('.bvh', '_footsteps.txt'), 'r') as f:
        footsteps = f.readlines()
    
    """ 按步态周期处理 """
    for li in range(len(footsteps)-1):  # 遍历每个步态周期
    
        curr, next = footsteps[li+0].split(' '), footsteps[li+1].split(' ')
        
        """ 跳过无效周期 """
        # 标记为*的周期（数据质量问题）
        if len(curr) == 3 and curr[2].strip().endswith('*'): continue
        if len(next) == 3 and next[2].strip().endswith('*'): continue
        # 边界检查
        if len(next) <  2: continue
        if int(curr[0])//2-window < 0: continue  # 起始帧太早
        if int(next[0])//2-window >= len(Xc): continue  # 结束帧太晚
        
        """ 地形高度拟合 """
        # 截取当前步态周期（前后扩展window帧）
        slc = slice(int(curr[0])//2-window, int(next[0])//2-window+1)
        # 处理地形高度
        H, Hmean = process_heights(anim[
            int(curr[0])//2-window:
            int(next[0])//2+window+1], type=type)

        # 每个地形样本生成单独的数据
        for h, hmean in zip(H, Hmean):
            
            Xh, Yh = Xc[slc].copy(), Yc[slc].copy()
            
            """ 调整高度信息 """
            # 输入特征中的位置偏移
            xo_s, xo_e = ((window*2)//10)*10+1, ((window*2)//10)*10+njoints*3+1
            # 输出目标中的位置偏移
            yo_s, yo_e = 8+(window//10)*4+1, 8+(window//10)*4+njoints*3+1
            # 减去平均高度（使高度相对化）
            Xh[:,xo_s:xo_e:3] -= hmean[...,np.newaxis]
            Yh[:,yo_s:yo_e:3] -= hmean[...,np.newaxis]
            # 添加地形特征到输入
            Xh = np.concatenate([Xh, h - hmean[...,np.newaxis]], axis=-1)
            
            """ 添加到全局数据集 """
            # 相位信息（添加起始0.0和结束1.0）
            P.append(np.hstack([0.0, Pc[slc][1:-1], 1.0]).astype(np.float32))
            X.append(Xh.astype(np.float32))  # 输入特征
            Y.append(Yh.astype(np.float32))  # 输出目标
  
""" 数据集统计信息 """
print('总片段数: %i' % len(X))
print('最短片段: %i 帧' % min(map(len,X)))
print('最长片段: %i 帧' % max(map(len,X)))
print('平均片段: %i 帧' % np.mean(list(map(len,X))))

""" 合并所有片段数据 """

print('合并片段数据...')
Xun = np.concatenate(X, axis=0)  # 输入特征
Yun = np.concatenate(Y, axis=0)  # 输出目标
Pun = np.concatenate(P, axis=0)  # 相位信息

print('数据维度:', Xun.shape, Yun.shape, Pun.shape)

""" 保存数据库 """
print('保存数据库...')
np.savez_compressed('database.npz', Xun=Xun, Yun=Yun, Pun=Pun)
print('处理完成!')
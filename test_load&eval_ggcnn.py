
import torch
from models.ggcnn2 import GGCNN2

import pyrealsense2 as rs
import numpy as np
import cv2
from ggcnn_helpers.timeit import TimeIt

import scipy.ndimage as ndimage

import time
def process_depth_image(depth, crop_size, out_size=300, return_mask=False, crop_y_offset=0):
    imh, imw = depth.shape

    with TimeIt('1'):
        # Crop.中心裁剪
        depth_crop = depth[(imh - crop_size) // 2 - crop_y_offset:(imh - crop_size) // 2 + crop_size - crop_y_offset,
                           (imw - crop_size) // 2:(imw - crop_size) // 2 + crop_size]
    # depth_nan_mask = np.isnan(depth_crop).astype(np.uint8)

    # Inpaint
    # OpenCV inpainting does weird things at the border.
    with TimeIt('2'):
        depth_crop = cv2.copyMakeBorder(depth_crop, 1, 1, 1, 1, cv2.BORDER_DEFAULT)  #边界填充1
        depth_nan_mask = np.isnan(depth_crop).astype(np.uint8) # 深度缺失的位置（值为NaN）返回True。uint8：True -> 1  得到深度掩码图

    with TimeIt('3'):  
        depth_crop[depth_nan_mask==1] = 0   #将深度图中对应的掩膜图为1的位置，深度设为0

    with TimeIt('4'):  #深度图像归一化处理
        # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
        depth_scale = np.abs(depth_crop).max()  #获取深度图像的比例因子（深度值的最大值），它用于将深度值映射到0到1之间或其他所需的范围
        depth_crop = depth_crop.astype(np.float32) / depth_scale  # Has to be float32, 64 not supported. 
                                                                 # 深度图像的数据类型转换为np.float32，以便进行浮点数的除法运算  / depth_scale：归一化处理

    with TimeIt('Inpainting'): #深度图像修复，修复图像中的缺失或损坏的区域
        depth_crop = cv2.inpaint(depth_crop, depth_nan_mask, 1, cv2.INPAINT_NS)

        # Back to original size and value range.
        depth_crop = depth_crop[1:-1, 1:-1]  #将depth_crop数组的边缘像素裁剪掉，cv2.copyMakeBorder函数时添加了1像素的边界
        depth_crop = depth_crop * depth_scale #深度值恢复到原始深度图像中的实际深度值

    # with TimeIt('5'):
    #     # Resize
    #     depth_crop = cv2.resize(depth_crop, (out_size, out_size), cv2.INTER_AREA)

    if return_mask:
        with TimeIt('6'):
            depth_nan_mask = depth_nan_mask[1:-1, 1:-1]  
            depth_nan_mask = cv2.resize(depth_nan_mask, (out_size, out_size), cv2.INPAINT_NS)
        return depth_crop, depth_nan_mask
    else:
        return depth_crop

#加载ggcnn模型到GPU
ggcnn_model = GGCNN2()
"""
    torch.load() 函数将读取包含模型权重的文件，并将它们以字典形式返回
    load_state_dict() 方法将加载的模型权重加载到 ggcnn 模型
"""
ggcnn_model.load_state_dict(torch.load('ggcnn2_weights_cornell/epoch_50_cornell_statedict.pt'))
device = torch.device("cuda:0")
ggcnn_model.eval().to("cuda:0")

#启动realsense，获取深度图
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)
while True:
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    #原始深度图可视化处理
    depth_origanal_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET) 
    #处理深度图
    depth, depth_nan_mask = process_depth_image(depth_image, crop_size=300, out_size=300, return_mask=True, crop_y_offset=0)
    #修改后的深度图可视化处理
    depth = np.clip((depth - depth.mean()), -1, 1)  # 值裁剪到范围 [-1, 1]，减去 depth 的平均值，对深度值进行中心化
    depth_modify_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
    depthT = torch.from_numpy(depth.reshape(1, 1, 300, 300).astype(np.float32)).to(device) #将要输入的深度图加载到GPU
    #模型推理
    with torch.no_grad():
        pred_out = ggcnn_model(depthT)
    #print("===========pred_out.shape==============")
    print(pred_out[0].cpu().numpy().shape)
    #推理结果处理
    points_out = pred_out[0].cpu().numpy().squeeze()
    #print("===========pred_out[0].shape==============")
    print(points_out.shape)
    cos_out = pred_out[1].cpu().numpy().squeeze()
    sin_out = pred_out[2].cpu().numpy().squeeze()
    ang_out = np.arctan2(sin_out, cos_out) / 2.0
    width_out = pred_out[3].cpu().numpy().squeeze() * 150.0
    # Filter the outputs.滤波
    filters=(10.0, 1.0, 1.0)
    if filters[0]:
        points_out = ndimage.filters.gaussian_filter(points_out, filters[0])  # 3.0
    if filters[1]:
        ang_out = ndimage.filters.gaussian_filter(ang_out, filters[1])
    if filters[2]:
        width_out = ndimage.filters.gaussian_filter(width_out, filters[2])
        
    points_out = np.clip(points_out, 0.0, 1.0-1e-3)

    max_pixel = np.array(np.unravel_index(np.argmax(points_out), points_out.shape))

    # print("===========points_out==============")
    # print(points_out)
    # print("===========ang_out=============")
    # print(ang_out)
    # print("============width_out============")
    # print(width_out)
    cv2.circle(depth_modify_colormap, (max_pixel[0], max_pixel[1]), 1, (0,0,255), 10)
    cv2.imshow('Origanel Depth Image', depth_origanal_colormap)
    cv2.imshow('Modify Depth Image', depth_modify_colormap)
    print("===========max_pixel==============")
    print(max_pixel)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

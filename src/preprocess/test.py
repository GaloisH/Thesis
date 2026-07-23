from PIL import Image
import os

def resize_and_pad(img_path, output_path, target_size=(400, 514), pad_color=(255, 255, 255)):
    """
    将图像等比例缩小，并在一张固定大小的背景图背景上居中显示。
    """
    with Image.open(img_path) as img:
        # 转换为 RGB 模式，避免带有 alpha 通道的图片出现背景问题
        img = img.convert("RGB")
        
        # thumbnail 会在原图上进行原比例缩放，使得宽、高都不会超过给定尺寸
        img.thumbnail(target_size, Image.Resampling.LANCZOS)
        
        # 创建全白色的目标背景图片
        new_img = Image.new("RGB", target_size, pad_color)
        
        # 计算粘贴的左上角坐标（使其居中）
        paste_x = (target_size[0] - img.width) // 2
        paste_y = (target_size[1] - img.height) // 2
        
        # 将缩放后的原图粘贴到白色背景的指定位置
        new_img.paste(img, (paste_x, paste_y))
        
        # 保存图像，保持较高的质量
        new_img.save(output_path, quality=95)
        print(f"处理成功，图片已保存到: {output_path}")

if __name__ == "__main__":
    # 使用示例：将此处替换为您真实的输入/输出文件路径
    input_file = "Passport_Photo.jpg"   
    output_file = "output.jpg" 
    
    if os.path.exists(input_file):
        resize_and_pad(input_file, output_file)
    else:
        print(f"找不到输入文件: {input_file}，请修改路径参数后再运行。")

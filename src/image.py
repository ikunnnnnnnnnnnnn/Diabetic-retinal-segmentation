from PIL import Image

def get_image_dimensions(image_path):
  """
  读取指定图像文件的尺寸。

  Args:
    image_path: 图像文件的完整路径。

  Returns:
    一个包含图像宽度和高度的元组 (width, height)。
    如果文件不存在或无法打开，则返回 None。
  """
  try:
    img = Image.open(image_path)
    width, height = img.size
    return width, height
  except FileNotFoundError:
    print(f"错误：文件 '{image_path}' 未找到。")
    return None
  except Exception as e:
    print(f"错误：无法打开或读取图像文件 '{image_path}'。错误信息：{e}")
    return None

if __name__ == "__main__":
  image_file = input("请输入图像文件的路径：")
  dimensions = get_image_dimensions(image_file)
  if dimensions:
    width, height = dimensions
    print(f"图像 '{image_file}' 的尺寸为：宽度 = {width} 像素，高度 = {height} 像素。")
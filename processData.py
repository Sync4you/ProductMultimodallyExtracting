# encoding: utf-8
# @Author: William·Woo
# @Time:2024-03-28-10:53 AM
# @File:processData.py

filepath = "data.csv"

with open(filepath, "r", encoding="utf-8") as f, open("train.csv", "w", encoding="utf-8") as g:
    g.write("posting_id,image,title\n")
    for line in f.readlines():
        d = line.replace('\n', '').split('\t')
        image_id = d[0]
        title = d[2]
        image_name = image_id + ".jpeg"
        g.write(image_id + "," + image_name + "," + title + "\n")
    f.close()
    g.close()


# with open("train.csv", "r", encoding="utf-8") as f, \
#         open("train1.csv", "w", encoding="utf-8") as g:
#     d = f.readline()
#     g.write("posting_id,image,title\n")
#     for line in f.readlines():
#         d = line.replace("\n", "").split(",")
#         image_id = d[0]
#         image_name = d[1]
#         title = d[2]
#         image_name = image_name.replace("jpg", "jpeg")
#         g.write(image_id + "," + image_name + "," + title + "\n")
#     f.close()
#     g.close()


from PIL import Image
import os


def webp_to_jpeg(input_path, output_path):
    # 打开webp图片
    webp_image = Image.open(input_path)

    # 将webp图片转换为jpeg格式
    webp_image.save(output_path, 'JPEG')


# 使用示例
img_dir = "./images/"

for i in os.listdir(img_dir):
    input_img = img_dir + i
    output_img = input_img
    try:
        img = Image.open(input_img)
        if img.format == 'WEBP':
            output_img = input_img
            webp_to_jpeg(input_path=input_img, output_path=output_img)
            print(f"{output_img} is done!")

        else:
            img.convert('RGB')
            img.save(output_img)
            webp_to_jpeg(input_path=input_img, output_path=output_img)
            print(f"{output_img} is done!")
    except:
        print(f"read img {input_img} err")







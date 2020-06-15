from dataset import AVAImages
import os
import numpy as np

index2score_mean = {}
index2score_var = {}
index2style = {}
data = AVAImages()

folder = os.path.exists("./select_images/")
if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
    os.makedirs("./select_images/")  # makedirs 创建文件时如果路径不存在会创建这个路径


with open('AVA_dataset/AVA_check.txt', "r") as f:
    lines = f.readlines()
    for line in lines:
        seg = line.split(" ")
        seg = list(map(int, seg))
        index2score_mean[seg[1]] = data.cal_mean(seg[2: 12])
        index2score_var[seg[1]] = data.cal_var(seg[2: 12], index2score_mean[seg[1]])
        mean = data.cal_mean(seg[2: 12])
        var = data.cal_var(seg[2: 12], mean)
        # if mean < 5.05 and mean > 4.95:
        #     os.system("cp AVA_dataset/images/{num}.jpg ./select_images".format(num=str(seg[1])))
        #     # os.system("mv ./select_images/" + str(seg[1]) + ".jpg " + "./select_images/mean_" + str(mean) + "_var_" + str(var) + ".jpg")

with open('AVA_dataset/style_image_lists/styles.txt', "r") as f:
    lines = f.readlines()
    for line in lines:
        seg = line.split(" ")
        index2style[seg[0]] = seg[1][:-1]


stylecount_low = 0
stylecount_high = 0
low_total = 0.0
high_total = 0.0

with open('AVA_dataset/style_image_lists/test.txt', "r") as f1:
    with open('AVA_dataset/style_image_lists/test_y.txt', "r") as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
        i = 0
        for line in lines1:
            seg = line.split(" ")
            mean = index2score_mean[int(seg[0])]
            var = index2score_var[int(seg[0])]
            if mean < 11:
                # low_total += 1
                style = lines2[i].split(" ")
                style = list(map(int, style))
                style_index = np.nonzero(style)
                style_index2list = [index2style[str(style_index[0][i] + 1)] for i in range(style_index[0].shape[0])]
                # stylecount_low += len(style_index2list)
                # if len(style_index2list) >= 3:
                os.system("cp AVA_dataset/images/{num}.jpg ./select_images".format(num=seg[0]))
                # os.system("mv ./select_images/{num}.jpg ./select_images/{num}_m_{mean}_v_{var}_s_{style}.jpg".
                #           format(num=seg[0], mean=str(mean), var=str(var), style="+".join(style_index2list)))
                # os.system("mv ./select_images/" + str(seg[0]) + ".jpg " + "./select_images/mean_" + str(mean) + "_var_" + str(var) + ".jpg")
            # else:
            #     high_total += 1
            #     style = lines2[i].split(" ")
            #     style = list(map(int, style))
            #     style_index = np.nonzero(style)
            #     style_index2list = [index2style[str(style_index[0][i] + 1)] for i in range(style_index[0].shape[0])]
            #     stylecount_high += len(style_index2list)
            i += 1
#
# print("style in low quality images = {count}".format(count=stylecount_low/low_total))
# print("style in high quality images = {count}".format(count=stylecount_high/high_total))
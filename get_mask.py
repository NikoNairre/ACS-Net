from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image, ImageDraw
import os
dataDir='../DataSets/coco2017'
dataType='val2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
# 初始化标注数据的 COCO api 
coco=COCO(annFile)

print(coco)


# # get all images containing given categories, select one at random
#catIds = coco.getCatIds(catNms=[])   #catNms=['dog']
imgIds = coco.getImgIds()
print(len(imgIds))
np.random.seed(1000)
np.random.shuffle(imgIds)
# imgIds = coco.getImgIds(imgIds = [324158])
# # loadImgs() 返回的是只有一个元素的列表, 使用[0]来访问这个元素
# # 列表中的这个元素又是字典类型, 关键字有: ["license", "file_name", 
# #  "coco_url", "height", "width", "date_captured", "id"]
#img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
imgs = coco.loadImgs(imgIds[:5000])
#print(imgs)

# # 加载并显示图片,可以使用两种方式: 1) 加载本地图片, 2) 在线加载远程图片
# # 1) 使用本地路径, 对应关键字 "file_name"
# # I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))  

# # 2) 使用 url, 对应关键字 "coco_url"
# for img in imgs:
#     img_name = img["coco_url"].split("/")[-1]
#     img_path = dataDir + "/" + dataType + "/" + img_name
#     I = Image.open(img_path)  
#     I.save("./dataSets/fore/" + img_name)


#     width, height = I.size
#     dpi = 80
#     # black_img = Image.fromarray(np.zeros((height, width), dtype=np.uint8))
#     black_img = Image.new('RGB', (width, height))
#     plt.figure(figsize=(width/dpi, height/dpi), dpi= dpi)    #create new img to avoid image overlay
#     plt.imshow(black_img, extent=[0, width, height, 0]); plt.axis('off')
#     annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
#     anns = coco.loadAnns(annIds)
#     coco.showAnns(anns)
#     plt.savefig('./dataSets/intermediate_mask/' + img_name[:len(img_name) - 4] + ".png", bbox_inches='tight', pad_inches=0, dpi = dpi)
#     plt.close()
#     #plt.show()
#     #print(anns)


intermediate_mask_path = os.path.join("./dataSets/intermediate_mask")
mid_masks = os.listdir(intermediate_mask_path)
print(mid_masks)
for mid_mask in mid_masks:

    img2 = cv2.imread("./dataSets/intermediate_mask/" + mid_mask, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    _, binary = cv2.threshold(img2, 1, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.bitwise_not(binary)        #convert to bitwise images(black and white)
    #img2.convert("RGB")
    # 读取黑白图片
    # image = binary.copy()

    # # 将图像从BGR颜色空间转换为HSV颜色空间
    # hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # # 定义白色的HSV范围
    # lower_white = np.array([0, 0, 200])
    # upper_white = np.array([180, 255, 255])

    # # 创建一个白色区域的掩码
    # mask = cv2.inRange(hsv, lower_white, upper_white)

    # # 创建一个红色的图像
    # red_image = np.zeros_like(image)
    # red_image[:, :] = [0, 0, 128]  # BGR颜色空间

    # # 使用掩码将白色区域替换为红色
    # result = cv2.bitwise_and(red_image, red_image, mask=mask)

    # # 显示原图和结果图像
    # cv2.imshow('Original Image', image)
    # cv2.imshow('Result Image', result)

    cv2.imwrite("./dataSets/mask_need_resize/" + mid_mask, binary)
    # # 等待按键，然后关闭窗口
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

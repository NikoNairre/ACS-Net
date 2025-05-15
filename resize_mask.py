from PIL import Image
import os

# mask_need_resize_path = os.path.join("./dataSets/mask_need_resize")
# To_resize_masks = os.listdir(mask_need_resize_path)
# print(To_resize_masks)

# for img_str in To_resize_masks:
#     mask_img = Image.open("./dataSets/mask_need_resize/" + img_str)
#     ori_img = Image.open("./dataSets/fore/" + img_str[:len(img_str) - 4] + ".jpg")
#     w1, h1 = ori_img.size
#     mask_img = mask_img.resize((w1, h1))
#     mask_img.save("./dataSets/mask/" + img_str)

# img1 = Image.open("./useless_imgs/000000000139.jpg")
# img2 = Image.open("./useless_imgs/000000000139.png")

# w1, h1 = img1.size
# w2, h2 = img2.size
# img2 = img2.resize((w1, h1))
# img2.save("./useless_imgs/res_resize.png")
# print(w1, h1)
# print(img2.size)

adain_img_dir = "/cver/yhqian/CIG/compare_res/Adain/fore"
our_img_dir = "/cver/yhqian/CIG/LCG_v51_2/input/final/fore"

adain_img = os.listdir(adain_img_dir)
our_img = os.listdir(our_img_dir)

for cimg in adain_img:
    if cimg in our_img:
        # 这里找到B列表中与A列表中cimg名字相同的项
        matched_img = os.path.join(our_img_dir, cimg)
        temp = Image.open(matched_img)

        adjust = Image.open(os.path.join(adain_img_dir, cimg))
        res = adjust.resize(temp.size)
        res.save("/cver/yhqian/CIG/LCG_v51_2/input/AdAIN/fore/" + cimg)


import os
import cv2

folder_path = "/home/shelei/data/CIKM_data/LMC_test_3_23_hss_csi_resize"
new_folder_path = "/home/shelei/data/CIKM_data/LMC_test_3_23_hss_csi_resize_new"
list_folder = os.listdir(folder_path)
list_folder.sort(key=lambda x:int(x[6:]))
for index, folder_name in enumerate(list_folder):
    print(folder_name)
    index += 1
    new_folder_name = "sample_" + str(index)
    print(new_folder_name)
    if not os.path.exists(os.path.join(new_folder_path,new_folder_name)):
        os.mkdir(os.path.join(new_folder_path,new_folder_name))
    fold_img_list = os.listdir(os.path.join(folder_path,folder_name))
    fold_img_list.sort(key=lambda x:int(x[6:-4]))
    for indexx, img in enumerate(fold_img_list):
        indexx += 1
        print(img)
        img_data = cv2.imread(os.path.join(folder_path,folder_name,img))
        new_img_name = "img_" + str(indexx) + ".png"
        new_img_path = os.path.join(new_folder_path, new_folder_name, new_img_name)
        cv2.imwrite(new_img_path,img_data)




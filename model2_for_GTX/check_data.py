from data import AVAImages

dataset = AVAImages()
dataset.check_data(filedir="AVA_dataset/style_image_lists/train.txt",
                   newfiledir="AVA_dataset/style_image_lists/train_check.txt",
                   imgdir="AVA_dataset/images/",
                   url_i=0)

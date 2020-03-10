from data import AVAImages

dataset = AVAImages()
dataset.check_data(filedir="AVA_dataset/style_imageAVA.txt",
                   newfiledir="AVA_dataset/AVA_check.txt",
                   imgdir="AVA_dataset/images/",
                   url_i=1)

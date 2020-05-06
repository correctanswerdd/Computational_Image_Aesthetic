from flyai.train_helper import upload_data, download, sava_train_model

upload_data("AVA_data_score_dis_style.zip", overwrite=True)
download("AVA_data_score_dis_style.zip", decompression=True)

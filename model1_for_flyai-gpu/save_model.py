from flyai.train_helper import upload_data, download, sava_train_model

sava_train_model(model_file="./model_MTCNN", dir_name="/model", overwrite=False)
download("model/my_model")

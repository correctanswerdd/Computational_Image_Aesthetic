from data import AVAImages

if __name__ == '__main__':
    dataset = AVAImages()
    dataset.create_train_set(batch_size=32,
                             if_write=False,
                             size=227,
                             read_dir='AVA_data_score_dis_style/',
                             train_set_dir='train_raw/')

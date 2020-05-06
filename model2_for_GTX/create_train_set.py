from data import AVAImages

if __name__ == '__main__':
    dataset = AVAImages()
    dataset.create_train_set(batch_size=4,
                             if_write=True,
                             size=227,
                             read_dir='AVA_data_score_dis/',
                             train_set_dir='train_raw/')

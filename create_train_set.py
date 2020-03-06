from data import AVAImages

if __name__ == '__main__':
    dataset = AVAImages()
    dataset.create_train_set(batch_size=100,
                             read_dir='AVA_data_score/',
                             train_set_dir='train_raw/')

from data import AVAImages

if __name__ == '__main__':
    # dataset1 = AVAImages()
    # dataset1.split_data('tag')
    # dataset2 = AVAImages()
    # dataset2.split_data('score')
    dataset = AVAImages()
    dataset.split_data(data_type='score_bi',
                       filedir="AVA_dataset/AVA_check.txt",
                       save_dir='AVA_data_score_bi/',
                       train_prob=0.8,
                       test_prob=0.1,
                       val_prob=0.1)
                      
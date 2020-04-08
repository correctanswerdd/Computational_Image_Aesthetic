from data import AVAImages

if __name__ == '__main__':
    dataset = AVAImages()
    """
    data type:
    * score
    * score_bi
    * score_dis
    * score_mean_var_style
    * score_dis_style
    * tag
    * style
    * score_and_style
    """
    dataset.split_data(data_type='score_dis',
                       filedir="AVA_dataset/AVA.txt",
                       save_dir='AVA_data_score_dis/',
                       train_prob=0.002,
                       test_prob=0.0004,
                       val_prob=0.0004)

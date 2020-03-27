from data import AVAImages

if __name__ == '__main__':
    dataset = AVAImages()
    dataset.create_setx_for_Th(read_dir='AVA_data_score_dis_style/')

from data import AVAImages

if __name__ == '__main__':
    dataset = AVAImages()
    dataset.read_data(read_dir='AVA_data_score_dis_style/', flag="test")
    dataset.read_data(read_dir='AVA_data_score_dis_style/', flag="val")
    dataset.read_data(read_dir='AVA_data_score_dis_style/', flag="train")
    print()


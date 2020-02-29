from data import AVAImages

if __name__ == '__main__':
    dataset1 = AVAImages()
    dataset1.split_data('tag')
    dataset2 = AVAImages()
    dataset2.split_data('score')
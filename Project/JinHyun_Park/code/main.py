# import torch
import os, argparse
import numpy as np
from Model import MyModel
from DataLoader import load_data, train_valid_split, load_testing_images

def configure():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train")
    parser.add_argument("--batch_size", default=128)
    parser.add_argument("--save_interval", default=1)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = configure()
    model = MyModel(config).cuda()

    if config.mode == 'train':
        x_train, y_train, x_test, y_test = load_data("../data")
        # change the above line 0.9 to 1.0 when full training!
        x_train, y_train, x_valid, y_valid = train_valid_split(x_train, y_train, 0.9)
        print("shape of x_train:", x_train.shape)
        print("shape of y_train:", y_train.shape)
        
        # training
        max_epoch = 1
        print("max epoch:", max_epoch)
        model.train(x_train, y_train, max_epoch)
        
        # validation
        print("shape of x_train", x_train.shape)
        print("shape of x_valid", x_valid.shape)
        model.evaluate(x_valid, y_valid, [100, 150])

    elif config.mode == 'test':
        _, _, x_test, y_test = load_data("../data")
        model.evaluate(x_test, y_test, [100, 150])

    elif config.mode == 'private':
        x_test = load_testing_images("../data")
        os.makedirs("../results", exist_ok=True)
        result = model.predict_prob(x_test)
        np.save(os.path.join("../results", 'predictions.npy'), result)


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import HyperParameters as HP
import Models
import Data
import Train
import Evaluate
import os


def main():
    generator = Models.Generator()
    discriminator = Models.Discriminator()

    if HP.load_model:
        generator.load()
        discriminator.load()

    train_dataset, test_dataset = Data.load_dataset()
    test_dataset = Data.separate_dataset(test_dataset)

    if HP.biased_train_data:
        train_dataset = Data.load_train_biased_data(HP.biased_data_sizes)

    fids = []
    for epoch in range(HP.epochs):
        print('iter', epoch)
        start = time.time()
        Train.train(generator.model, discriminator.model, train_dataset, epoch)
        print('saving...')
        generator.save()
        discriminator.save()
        generator.save_images(epoch)
        print('saved')
        if HP.evaluate_model and (epoch + 1) % HP.epoch_per_evaluate == 0:
            fid = Evaluate.get_multi_fid(generator.model, test_dataset)
            print('fid :', fid)
            fids.append(fid)
        print('time: ', time.time() - start)

    if not HP.evaluate_model:
        fid = Evaluate.get_multi_fid(generator.model, test_dataset)
        print('fid :', fid)
        fids.append(fid)

    if not os.path.exists(HP.folder_name):
        os.makedirs(HP.folder_name)

    Data.save_graph(fids)


main()

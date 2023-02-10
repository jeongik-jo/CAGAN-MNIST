import time
import HyperParameters as hp
import Models
import Dataset
import Train
import Evaluate
import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    generator = Models.Generator()
    discriminator = Models.Discriminator()

    if hp.load_model:
        generator.load()
        discriminator.load()

    train_dataset, test_datasets = Dataset.load_dataset()

    results = {}
    for epoch in range(hp.epochs):
        print('epoch', epoch)
        start = time.time()
        train_results = Train.train(generator.model, discriminator.model, train_dataset, epoch)
        print('saving...')
        generator.save()
        discriminator.save()

        generator.to_ema()
        generator.save_images(epoch)
        print('saved')
        print('time: ', time.time() - start, '\n')

        if hp.evaluate_model and (epoch + 1) % hp.epoch_per_evaluate == 0:
            print('evaluating...')
            start = time.time()
            evaluate_results = Evaluate.evaluate(generator.model, test_datasets)
            for key in train_results:
                try:
                    results[key].append(train_results[key])
                except KeyError:
                    results[key] = [train_results[key]]
            for key in evaluate_results:
                try:
                    results[key].append(evaluate_results[key])
                except KeyError:
                    results[key] = [evaluate_results[key]]

            print('evaluated')
            print('time: ', time.time() - start, '\n')
            if not os.path.exists('results/figures'):
                os.makedirs('results/figures')
            for key in results:
                np.savetxt('results/figures/%s.txt' % key, results[key], fmt='%f')
                plt.title(key)
                plt.xlabel('Epochs')
                plt.ylabel(key)
                plt.plot([(i + 1) * hp.epoch_per_evaluate for i in range(len(results[key]))], results[key])
                plt.savefig('results/figures/%s.png' % key)
                plt.clf()

        generator.to_train()


main()

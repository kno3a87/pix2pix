# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 16:22:56 2019
学校のでやるならgpuは1に！
@author: KNO3
"""

#!/usr/bin/env python

# python train_facade.py -g 0 -i ./facade/base --out result_facade --snapshot_interval 10000

from __future__ import print_function
import argparse
import os

import chainer
from chainer import training
from chainer.training import extensions
from chainer import serializers

from net3 import Discriminator
from net3 import Encoder
from net3 import Decoder
from updater3 import FacadeUpdater

from facade_dataset3 import FacadeDataset
from facade_dataset3 import FacadeTestDataset
from facade_visualizer3_train import out_image

def main():
    parser = argparse.ArgumentParser(description='chainer implementation of pix2pix')
    parser.add_argument('--batchsize', '-b', type=int, default=3,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=200,
                        help='Number of sweeps over the dataset to train')
    # default = -1はGPU使わん
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--dataset', '-i', default='./kyoushiTpPfjpg/',
                        help='Directory of image files.')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--snapshot_interval', type=int, default=100,
                        help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=100,
                        help='Interval of displaying log to console')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    enc = Encoder(in_ch=3)
    dec = Decoder(out_ch=3)
    dis = Discriminator(in_ch=3, out_ch=3)
    
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        enc.to_gpu()  # Copy the model to the GPU
        dec.to_gpu()
        dis.to_gpu()

    # Setup an optimizer
    def make_optimizer(model, alpha=0.0002, beta1=0.5):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.00001), 'hook_dec')
        return optimizer
    opt_enc = make_optimizer(enc)
    opt_dec = make_optimizer(dec)
    opt_dis = make_optimizer(dis)

    # データ入れるよ（引数args書くと関数呼び出したとき食い違うから消す）
    train_d = FacadeDataset()
    test_d = FacadeTestDataset()
    #train_iter = chainer.iterators.MultiprocessIterator(train_d, args.batchsize, n_processes=4)
    #test_iter = chainer.iterators.MultiprocessIterator(test_d, args.batchsize, n_processes=4)
    train_iter = chainer.iterators.SerialIterator(train_d, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test_d, args.batchsize)
    
    # Set up a trainer
    updater = FacadeUpdater(
        models=(enc, dec, dis),
        iterator={
            'main': train_iter,
            'test': test_iter},
        optimizer={
            'enc': opt_enc, 'dec': opt_dec, 
            'dis': opt_dis},
        device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    
    snapshot_interval = (args.snapshot_interval, 'iteration')
    display_interval = (args.display_interval, 'iteration')
    trainer.extend(extensions.snapshot(
        filename='snapshot_iter_{.updater.iteration}.npz'),
                   trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        enc, 'enc_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        dec, 'dec_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        dis, 'dis_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'enc/loss', 'dec/loss', 'dis/loss',
    ]), trigger=display_interval)
    # プログレスバーは1回ごと表示で
    trainer.extend(extensions.ProgressBar(update_interval=1))
    trainer.extend(
        out_image(
            updater, enc, dec,
            1, 1, args.seed, args.out),
        trigger=snapshot_interval)

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)
        
    #print("ここまでOK！")

    # Run the training
    trainer.run()

if __name__ == '__main__':
    main()

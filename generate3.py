# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 01:27:55 2019
予測させるよ
@author: KNO3
"""

from __future__ import print_function
import argparse
import os

import chainer
from chainer import training

from net3 import Discriminator
from net3 import Encoder
from net3 import Decoder
from updater3 import FacadeUpdater

from facade_dataset3 import FacadeDataset
from facade_visualizer3 import out_image

import shutil
# これで警告消える
chainer.config.train = False

def main():
    parser = argparse.ArgumentParser(description='chainer implementation of pix2pix')
    parser.add_argument('--gpu', '-g', type=int, default=1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--model', '-m', default='./resultTp100/snapshot_iter_2400.npz',
                        help='model snapshot')
    parser.add_argument('--input', '-i', default='sample.jpg',
                        help='input jpg')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))

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

    # generate_tmpってディレクトリが存在したら、消す
    if os.path.exists('generate_tmp'):
        shutil.rmtree('generate_tmp')

    # generate_tmpっての作るお
    os.mkdir('generate_tmp')
    # 上のargsにあるsample.jpgをgenerate_tmpにtmp.jpgとしてコピー
    shutil.copyfile(args.input,'generate_tmp/tmp.jpg')
    # コピーできてる
    test_d = FacadeDataset('generate_tmp/', 'generate_tmp/')
    # テストなのでイテレーターは1回
    test_iter = chainer.iterators.SerialIterator(test_d, 1)


    # Set up a trainer
    updater = FacadeUpdater(
        models=(enc, dec, dis),
        iterator={},
        optimizer={'enc': opt_enc, 'dec': opt_dec,'dis': opt_dis},
        device=args.gpu)
    trainer = training.Trainer(updater, (200, 'epoch'), out='generate/')
    chainer.serializers.load_npz(args.model, trainer)


    out_image(
    	updater, enc, dec,
        1, 1, args.seed, 'generateTp100/',args.input,True,test_iter)(trainer)


if __name__ == '__main__':
    main()

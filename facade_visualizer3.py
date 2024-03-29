# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 16:40:57 2019

@author: KNO3
"""

#!/usr/bin/env python

import os

import numpy as np
from PIL import Image

import chainer
import chainer.cuda
from chainer import Variable

def out_image(updater, enc, dec, rows, cols, seed, dst, outputimg ,generate_mode=False, test_iterator=None):
    #print("びじゅあライザー！")
    @chainer.training.make_extension()
    def make_image(trainer):
        np.random.seed(seed)
        n_images = rows * cols
        xp = enc.xp

        w_in = 115
        h_in = 149
        w_out = 115
        h_out = 149

        in_ch = 3
        out_ch = 3
        
        in_all = np.zeros((n_images, in_ch, w_in, h_in)).astype("i")
        gt_all = np.zeros((n_images, out_ch, w_out, h_out)).astype("f")
        gen_all = np.zeros((n_images, out_ch, w_out, h_out)).astype("f")
        
        for it in range(n_images):
            if test_iterator!=None:
                batch = test_iterator.next()
                batchsize = len(batch)
            else:
                batch = updater.get_iterator('test').next()
                batchsize = len(batch)
            print('bsize:' + str(batchsize))

            x_in = xp.zeros((batchsize, in_ch, w_in, h_in)).astype("f")
            t_out = xp.zeros((batchsize, out_ch, w_out, h_out)).astype("f")

            for i in range(batchsize):
                x_in[i,:] = xp.asarray(batch[i][0])
                t_out[i,:] = xp.asarray(batch[i][1])
            x_in = Variable(x_in)

            z = enc(x_in)
            x_out = dec(z)
            
            if generate_mode==False:
                in_all[it,:] = x_in.data.get()[0,:]
                gt_all[it,:] = t_out.get()[0,:]
                gen_all[it,:] = x_out.data.get()[0,:]
            else:
                gen_all[it,:] = x_out.data.get()[0,:]
        
        def save_image(x, name, mode=None):
            _, C, H, W = x.shape
            x = x.reshape((rows, cols, C, H, W))
            x = x.transpose(0, 3, 1, 4, 2)
            if C==1:
                x = x.reshape((rows*H, cols*W))
            else:
                x = x.reshape((rows*H, cols*W, C))

            preview_dir = '{}preview'.format(dst)
            preview_path = preview_dir +\
                '/'+outputimg.format(name, trainer.updater.iteration)
            if not os.path.exists(preview_dir):
                os.makedirs(preview_dir)
            
            Image.fromarray(x, mode=mode).convert('RGB').save(preview_path)
        
        x = np.asarray(np.clip(gen_all * 128 + 128, 0.0, 255.0), dtype=np.uint8)
        save_image(x, "gen")
        print('save generated image!')
       
        if generate_mode == False:
        #x = np.ones((n_images, 3, w_in, h_in)).astype(np.uint8)*255
        #x[:,0,:,:] = 0
        #for i in range(3):
        #    x[:,0,:,:] += np.uint8(15*i*in_all[:,i,:,:])
        #save_image(x, "in", mode='HSV')
            x = np.asarray(np.clip(in_all * 128 + 128, 0.0, 255.0), dtype=np.uint8)
            save_image(x, "in")

            x = np.asarray(np.clip(gt_all * 128+128, 0.0, 255.0), dtype=np.uint8)
            save_image(x, "gt")
        
    return make_image

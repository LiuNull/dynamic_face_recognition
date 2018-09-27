"""Helper for evaluation on the Labeled Faces in the Wild dataset 
"""

# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import facenet
import random

def evaluate(embeddings, actual_issame, nrof_folds=10):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    # start from 0, step=2
    embeddings1 = embeddings[0::2]
    # start from 1, step=2
    embeddings2 = embeddings[1::2]
    # embeddings1 is corresponding with embeddings2
    tpr, fpr, accuracy = facenet.calculate_roc(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), nrof_folds=nrof_folds)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = facenet.calculate_val(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, val, val_std, far

def get_paths(lfw_dir, pairs, file_ext,timestep_size):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 5: # the same people
            peoplepath=os.path.join(lfw_dir,pair[0])
            video1path = os.path.join(peoplepath, pair[1])
            video2path = os.path.join(peoplepath, pair[3])
            for i in range(timestep_size):
                path = os.path.join(video1path, pair[0] + '_' + pair[1] + '_%04d' % int(random.randint(1, int(pair[2]))) + '.' + file_ext)
                path_list.append(path)
            path = os.path.join(video2path, pair[0] + '_' + pair[3] + '_%04d' % int(random.randint(1, int(pair[4]))) + '.' + file_ext)
            # path = os.path.join(video2path, pair[3] + '_label' + '.' + file_ext)
            path_list.append(path)
            '''
            path1 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(random.randint(1, int(pair[1]))) + '.' + file_ext)
            path2 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(random.randint(1, int(pair[1]))) + '.' + file_ext)
            path3 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(random.randint(1, int(pair[1]))) + '.' + file_ext)
            path4 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(random.randint(1, int(pair[1]))) + '.' + file_ext)
            path5 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(random.randint(1, int(pair[1])))+'.'+file_ext)
            '''
            issame = True
        elif len(pair) == 6:
            '''
            path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(random.randint(1, int(pair[1]))) + '.' + file_ext)
            path1 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(random.randint(1, int(pair[1]))) + '.' + file_ext)
            path2 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(random.randint(1, int(pair[1]))) + '.' + file_ext)
            path3 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(random.randint(1, int(pair[1]))) + '.' + file_ext)
            path4 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(random.randint(1, int(pair[1]))) + '.' + file_ext)
            path5 = os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(random.randint(1, int(pair[3])))+'.'+file_ext)
            '''
            people1path = os.path.join(lfw_dir, pair[0])
            people2path = os.path.join(lfw_dir, pair[3])
            video1path = os.path.join(people1path, pair[1])
            video2path = os.path.join(people2path, pair[4])
            for i in range(timestep_size):
                path = os.path.join(video1path, pair[0] + '_' + pair[1] + '_%04d' % int(random.randint(1, int(pair[2]))) + '.' + file_ext)
                path_list.append(path)
            path = os.path.join(video2path, pair[3] + '_' + pair[4] + '_%04d' % int(random.randint(1, int(pair[5]))) + '.' + file_ext)
            # path = os.path.join(video2path, pair[4] + '_label' + '.' + file_ext)
            path_list.append(path)
            issame = False
        issame_list.append(issame)
        '''
        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
            path_list += (path0,path1,path2,path3,path4,path5)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
            print('pairs path:'+"\n"+path0+"\n"+path1+"\n"+path2+"\n"+path3+"\n"+path4+"\n"+path5)
    if nrof_skipped_pairs>0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)
        '''
    return path_list, issame_list


def get_video_paths(lfw_dir, pairs, file_ext, timestep_size):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 5:  # the same people
            peoplepath = os.path.join(lfw_dir, pair[0])
            video1path = os.path.join(peoplepath, pair[1])
            video2path = os.path.join(peoplepath, pair[3])
            for i in range(timestep_size):
                path = os.path.join(video1path, pair[1] + '_%04d' % int(random.randint(1, int(pair[2]))) + '.' + file_ext)
                path_list.append(path)
            for i in range(timestep_size):
                path = os.path.join(video2path, pair[3] + '_%04d' % int(random.randint(1, int(pair[4]))) + '.' + file_ext)
                path_list.append(path)
            issame = True
        elif len(pair) == 6:
            people1path = os.path.join(lfw_dir, pair[0])
            people2path = os.path.join(lfw_dir, pair[3])
            video1path = os.path.join(people1path, pair[1])
            video2path = os.path.join(people2path, pair[4])
            for i in range(timestep_size):
                path = os.path.join(video1path, pair[1] + '_%04d' % int(random.randint(1, int(pair[2]))) + '.' + file_ext)
                path_list.append(path)
            for i in range(timestep_size):
                path = os.path.join(video2path, pair[4] + '_%04d' % int(random.randint(1, int(pair[5]))) + '.' + file_ext)
                path_list.append(path)
            issame = False
        issame_list.append(issame)
        '''
        if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
            path_list += (path0, path1, path2, path3, path4, path5)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
            print(
                'pairs path:' + "\n" + path0 + "\n" + path1 + "\n" + path2 + "\n" + path3 + "\n" + path4 + "\n" + path5)
    if nrof_skipped_pairs > 0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)
        '''
    return path_list, issame_list


def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)
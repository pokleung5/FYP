#!/usr/bin/env python

# This implements the FastMap algorithm
# for mapping points where only the distance between them is known
# to N-dimension coordinates.
# The FastMap algorithm was published in:
#
# FastMap: a fast algorithm for indexing, data-mining and
# visualization of traditional and multimedia datasets
# by Christos Faloutsos and King-Ip Lin
# http://portal.acm.org/citation.cfm?id=223812

# This code made available under the BSD license,
# details at the bottom of the file
# Copyright (c) 2009, Gunnar Aastrand Grimnes

import math
import random

# need scipy as usual
import scipy
import numpy

class FastMap: 

    def __init__(self, dist, itr): 
        if dist.max()>1:
            dist/=dist.max()

        self.dist=dist
        self.DISTANCE_ITERATIONS = itr

    def _furthest(self, o): 
        mx=-1000000
        idx=-1
        for i in range(len(self.dist)): 
            d=self._dist(i,o, self.col)
            if d>mx: 
                mx=d
                idx=i

        return idx

    def _pickPivot(self):
        """Find the two most distant points"""
        o1=random.randint(0, len(self.dist)-1)
        o2=-1

        i=self.DISTANCE_ITERATIONS

        while i>0: 
            o=self._furthest(o1)
            if o==o2: break
            o2=o
            o=self._furthest(o2)
            if o==o1: break
            o1=o
            i-=1

        self.pivots[self.col]=(o1,o2)
        return (o1,o2)


    def _map(self, K): 
        if K==0: return 
    
        px,py=self._pickPivot()

        if self._dist(px,py,self.col)==0: 
            return 
        for i in range(len(self.dist)):
            self.res[i][self.col]=self._x(i, px,py)

        self.col+=1
        self._map(K-1)

    def _x(self,i,x,y):
        """Project the i'th point onto the line defined by x and y"""
        dix=self._dist(i,x,self.col)
        diy=self._dist(i,y,self.col)
        dxy=self._dist(x,y,self.col)
        return (dix + dxy - diy) / 2*math.sqrt(dxy)

    def _dist(self, x,y,k): 
        """Recursively compute the distance based on previous projections"""
        if k==0: return self.dist[x,y]**2
    
        rec=self._dist(x,y, k-1)
        resd=(self.res[x][k] - self.res[y][k])**2
        return rec-resd


    def map(self, K): 
        self.col=0
        self.res=numpy.zeros((len(self.dist),K))
        self.pivots=numpy.zeros((K,2),"i")
        self._map(K)
        return self.res

def fastmap(dist, K=2, itr=1):
    """dist is a NxN distance matrix
    returns coordinates for each N in K dimensions
    """
    
    return FastMap(dist, itr).map(K)
    
# Copyright (c) 2009, Gunnar Aastrand Grimnes
# All rights reserved.

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

#     * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#     * Neither the name of the <ORGANIZATION> nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


#%%

import torch
import utils

if __name__ == "__main__":
    torch.set_default_tensor_type('torch.DoubleTensor')

    pts = torch.tensor([[
            [0, 2],
            [2, 5],
            [6, 4],
            [8, 7],
            [9, 10]
        ]])
    
    dm = utils.get_distanceSq_matrix(pts) ** 0.5

    dm = torch.tensor([[
            [0, 3, 3],
            [3, 0.0, 3],
            [3, 3, 0]
        ]])

    dm = utils.minmax_norm(dm)[0]
    print(dm)
    
    dm = numpy.array(dm[0].data)
    rs = fastmap(dm, 2)
    rs = torch.tensor(rs)
    print(rs)

    rs_dm = utils.get_distanceSq_matrix(rs) ** 0.5
    rs_dm = utils.minmax_norm(rs_dm)[0]
    print(rs_dm)
    
    print(torch.sum(((rs_dm - torch.tensor(dm)) ** 2)))



# %%

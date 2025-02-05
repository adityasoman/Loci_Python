# Import Statements
import os
import topogenesis as tg
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import cycle
import skcriteria as sk
from skcriteria.agg import similarity  # here lives TOPSIS
from skcriteria.pipeline import mkpipe  # this function is for create pipelines
from skcriteria.preprocessing import invert_objectives, scalers


class DesirabilityLattice:
    def __init__(self,baselattice,performance_matrix:np.array):
        self.performance_matrix = performance_matrix
        self.baselattice = baselattice
    
    def topsis (self, objectives_array, weights_array, criteria_array):
        """
        TOPSIS method to calculate the desirability of each voxel
        """
        decision_matrix = sk.mkdm(
        self.performance_matrix,
        objectives = objectives_array,
        weights = weights_array,
        criteria = criteria_array)

        pipe = mkpipe(
        invert_objectives.NegateMinimize(),
        scalers.VectorScaler(target="matrix"),  # this scaler transform the matrix
        scalers.SumScaler(target="weights"),  # and this transform the weights
        similarity.TOPSIS(),
        )
        rank = pipe.evaluate(decision_matrix)
        mcdm_result= rank.e_.similarity
        mcdm_result_reshaped = rank.e_.similarity.reshape(self.baselattice.shape)
        mcdm_colored_voxels = mcdm_result_reshaped[self.baselattice]
        return mcdm_colored_voxels,mcdm_result_reshaped

#!/usr/bin/python
import numpy as np
from copy import copy
from matplotlib.mlab import PCA
from sklearn import manifold
from sklearn import decomposition
from sklearn import cluster
import scipy.spatial.distance as dist
from collections import defaultdict
# from scipy.spatial import distance
from PyQt4.QtCore import *
from PyQt4.QtGui import *
#from PyQt4.Qt import *
# explicitly imported "hidden imports" for pyinstaller
#from sklearn.utils import weight_vector, lgamma
from sklearn.metrics.pairwise import euclidean_distances, pairwise_distances

# Dinos solver
import cpca.solvers as solvers
import cpca.skpca as skpca
import cpca.kernel_gen as kernel_gen
import cpca.utils as utils

import cpca.kmeans as kmeans

try:
    from sklearn.utils.sparsetools import _graph_validation
    from sklearn.neighbors import typedefs
except:
    pass

emb_unrestricted_dims = ["PCA", "cPCA", "MLE"]

class PopupSlider(QDialog):
    def __init__(self, label_text, default=4, minimum=1, maximum=20):
        QWidget.__init__(self)
        self.slider_value = 1

        name_label = QLabel()
        name_label.setText(label_text)
        name_label.setAlignment(Qt.AlignCenter)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(minimum)
        self.slider.setMaximum(maximum)
        self.slider.setValue(default)

        self.value_label = QLabel()
        self.value_label.setText('%d' % (self.slider.value()))
        self.slider.valueChanged.connect(self.slider_changed)

        self.button = QPushButton('Ok', self)
        self.button.clicked.connect(self.handleButton)
        self.button.pressed.connect(self.handleButton)

        layout = QGridLayout(self)
        layout.addWidget(name_label      , 1, 1, 1, 4, Qt.AlignLeft)
        layout.addWidget(self.slider     , 2, 1, 2, 1, Qt.AlignLeft)
        layout.addWidget(self.value_label, 2, 2, 2, 2, Qt.AlignCenter)
        layout.addWidget(self.button     , 2, 4, 2, 4, Qt.AlignRight)

        self.setWindowTitle('Parameter choice')


    def slider_changed(self):
        val = self.slider.value()
        self.value_label.setText('%d' %val)
        self.slider_value = val
 

    def handleButton(self):
        self.hide()


class Embedding(object):
    def __init__(self, data, points, parent):
        self.data = data
        self.original_control_points = None
        self.original_control_point_indices = None
        self.control_points = None
        self.control_point_indices = None
        self.parent = parent
        self.X = np.array([])
        self.Y = np.array([])
        self.ml = []
        self.cl = []
        self.cluster_association = []
        self.has_ml_cl_constraints = False
        self.projection_matrix = np.zeros((2, len(self.data[0])))
        self.name = ''
        self.is_dynamic = False
        self.update_control_points(points)

    def get_embedding(self):
        pass

    def update_must_and_cannot_link(self, ml, cl):
        self.ml = ml
        self.cl = cl
        if (len(self.ml) > 0) or (len(self.cl) > 0):
            self.has_ml_cl_constraints = True
        else:
            self.has_ml_cl_constraints = False

    def augment_control_points(self, e):
        avg_median = np.average(abs(np.median(e, axis=0)))
        tmp_points = defaultdict(list)
        if len(self.cl) > 0:
            for pair in self.cl:
                if len(pair) == 2:
                    i, j = list(pair)
                    x1 = e[i]
                    x2 = e[j]
                    diff = x1 - x2
                    norm = np.linalg.norm(diff)
                    new_x1 = x1 + (diff/norm)*5*avg_median
                    new_x2 = x2 - (diff/norm)*5*avg_median
                    if i not in self.control_point_indices:
                        e[i] = new_x1
                        tmp_points[i] = new_x1
                    if j not in self.control_point_indices:
                        e[j] = new_x2
                        tmp_points[j] = new_x2
        if len(self.ml) > 0:
            for pair in self.ml:
                if len(pair) == 2:
                    i, j = list(pair)
                    x1 = e[i]
                    x2 = e[j]
                    diff = x1 - x2
                    new_x1 = x1 - 0.45*diff
                    new_x2 = x2 + 0.45*diff
                    if i not in self.control_point_indices:
                        e[i] = new_x1
                        tmp_points[i] = new_x1
                    if j not in self.control_point_indices:
                        e[j] = new_x2
                        tmp_points[j] = new_x2
        for k,v in tmp_points.items():
            self.control_point_indices.append(k)
            self.control_points.append(v)
        self.X = self.data[self.control_point_indices]
        self.Y = np.array(self.control_points)

    def update_control_points(self, points):
        self.control_point_indices = []
        self.control_points = []
        for i, coords in points.items():
            self.control_point_indices.append(i)
            self.control_points.append(coords)

        #print "hi"
        #print self.control_point_indices
        #print self.control_points
        self.X = self.data[self.control_point_indices]
        self.Y = np.array(self.control_points)

    def finished_relocating(self):
        pass

class PCA(Embedding):
    def __init__(self, data, control_points, parent, dim=2):
        super(PCA, self).__init__(data, control_points, parent)
        self.name = "PCA"
        self.projection_matrix = None

        self.dim = dim

        try:
            pca = decomposition.PCA(n_components=self.dim)
            pca.fit(data)
            self.projection_matrix = pca.components_
            self.embedding = np.array(pca.transform(data))
        except:
            msg = "It seems like the embedding algorithm did not converge with the given parameter setting"
            QMessageBox.about(parent, "Embedding error", msg) 
    

    def get_embedding(self):
        return self.embedding.T
    

    def update_control_points(self, points):
        pass





class LLE(Embedding):
    def __init__(self, data, control_points, parent):
        super(LLE, self).__init__(data, control_points, parent)
        self.name = "LLE"
        try:
            self.w = PopupSlider('Enter number of neighbors to consider (default is 4):')
            self.w.exec_()
            num = int(self.w.slider_value)
            if num == '':
                num = 4
            try:
                lle = manifold.LocallyLinearEmbedding(n_neighbors=int(num), out_dim=2)
            except:
                lle = manifold.LocallyLinearEmbedding(n_neighbors=int(num), n_components=2)
            lle.fit(data)
            self.embedding = np.array(lle.transform(data))
        except:
            msg = "It seems like the embedding algorithm did not converge with the given parameter setting"
            QMessageBox.about(parent, "Embedding error", msg)


    def get_embedding(self):
        return self.embedding.T
    

    def update_control_points(self, points):
        pass






class XY(Embedding):
    def __init__(self, data, control_points, parent):
        super(XY, self).__init__(data, control_points, parent)
        self.name = "XY"
        used_attributes = []
        for row in range(self.parent.series_list_model.rowCount()):
            model_index = self.parent.series_list_model.index(row, 0)
            checked = self.parent.series_list_model.data(model_index, Qt.CheckStateRole) == QVariant(Qt.Checked)
            if checked:
                if len(used_attributes) < 2:
                    name = str(self.parent.series_list_model.data(model_index).toString())
                    used_attributes.append(list(self.parent.data.attribute_names).index(name))
                    # print self.parent.data.attribute_names[used_attributes[-1]]
                else:
                    break

        self.embedding = np.array(self.parent.data.original_data.T[used_attributes].T)   


    def get_embedding(self):
        return self.embedding.T
    

    def update_control_points(self, points):
        pass







class ISO(Embedding):
    def __init__(self, data, control_points, parent):
        super(ISO, self).__init__(data, control_points, parent)
        self.name = "ISO"
        try:
            self.w = PopupSlider('Enter number of neighbors to consider (default is 4):')
            self.w.exec_()
            num = int(self.w.slider_value)
            if num == '':
                num = 4
            try:
                iso = manifold.Isomap(n_neighbors=int(num), out_dim=2)
            except:
                iso = manifold.Isomap(n_neighbors=int(num), n_components=2)
            iso.fit(data)
            self.embedding = np.array(iso.transform(data))   
        except:
            msg = "It seems like the embedding algorithm did not converge with the given parameter setting"
            QMessageBox.about(parent, "Embedding error", msg)


    def get_embedding(self):
        return self.embedding.T
    

    def update_control_points(self, points):
        pass








class tSNE(Embedding):
    def __init__(self, data, control_points, parent):
        super(tSNE, self).__init__(data, control_points, parent)
        self.name = "t-SNE"
        try:
            self.w = PopupSlider('Enter perplexity (default is 30):', default=30, minimum=1, maximum=100)
            self.w.exec_()
            num = int(self.w.slider_value)
            if num == '':
                num = 30
            m, ok = QInputDialog.getText(parent, 'Metric', 'Enter number of the desired metric:\n1) Euclidean (Default)\n2) Jaccard\n3) L1 norm')
            metric = 'euclidean'
            if m == '2':
                metric = 'jaccard'
            elif m == '3':
                metric = 'l1'            
            tsne = manifold.TSNE(n_components=2, random_state=0, perplexity=num, metric=metric)
            self.embedding = np.array(tsne.fit_transform(data))
        except:
            msg = "It seems like the embedding algorithm did not converge with the given parameter setting"
            QMessageBox.about(parent, "Embedding error", msg) 

    def get_embedding(self):
        return self.embedding.T
    

    def update_control_points(self, points):
        pass






class MDS(Embedding):
    def __init__(self, data, control_points, parent):
        super(MDS, self).__init__(data, control_points, parent)
        self.name = "MDS"
        metric, ok = QInputDialog.getText(parent, 'Metric', 'Please select a metric:\n\n1) L1\n2) Euclidean (Default)\n3) Cosine\n4) Mahalanobis')
        if metric == '1':
            m = 'l1'
        elif metric == '2':
            m = 'euclidean'
        elif metric == '3':
            m = 'cosine'
        elif metric == '4':
            m = 'mahalanobis'
        else:
            m = 'euclidean'
        parent.setWindowTitle('InVis: ' + parent.data.dataset_name + ' (MDS [%s])'%m)
        dists = pairwise_distances(data, metric=m)
        dists = (dists + dists.T)/2.0
        e, stress = manifold.mds.smacof(dists, n_components=2)
        self.embedding = e


    def get_embedding(self):
        return self.embedding.T
    

    def update_control_points(self, points):
        pass





class ICA(Embedding):
    def __init__(self, data, control_points, parent):
        super(ICA, self).__init__(data, control_points, parent)
        self.name = "ICA"
        try:
            ica = decomposition.FastICA(n_components=2)
            ica.fit(data)
            self.embedding = np.array(ica.transform(data))   
        except:
            msg = "It seems like the embedding algorithm did not converge with the given parameter setting"
            QMessageBox.about(parent, "Embedding error", msg)


    def get_embedding(self):
        return self.embedding.T
    

    def update_control_points(self, points):
        pass


        
        



class LSP(Embedding):
    def __init__(self, data, points, parent):
        super(LSP, self).__init__(data, points, parent)
        self.name = "LSP"
        self.is_dynamic = True 
    

    def get_embedding(self, X=[]):
        if X == []:
            X=self.data.T
        return np.dot(self.projection_matrix, X)
    

    def update_control_points(self, points):
        super(LSP, self).update_control_points(points)
        if len(self.Y) > 0:
            self.projection_matrix = np.dot(self.Y.T, np.linalg.pinv(self.X.T))
        else:
            self.projection_matrix =  np.zeros((2, len(self.data[0])))
        if self.has_ml_cl_constraints:
            self.augment_control_points(self.get_embedding().T)
            if len(self.Y) > 0:
                self.projection_matrix = np.dot(self.Y.T, np.linalg.pinv(self.X.T))
            else:
                self.projection_matrix =  np.zeros((2, len(self.data[0])))


        
        



class cPCA_dummy(Embedding):
    def __init__(self, data, points, parent):
        super(cPCA, self).__init__(data, points, parent)
        self.name = "cPCA"
        self.is_dynamic = True 
        self.control_point_indices = []
        self.old_control_point_indices = []
        self.finished_relocating()
    

    def get_embedding(self):
        if set(self.control_point_indices) != self.old_control_point_indices:
            self.finished_relocating()
        self.old_control_point_indices = set(self.control_point_indices)
        return np.dot(self.projection_matrix, self.data.T)


    def finished_relocating(self):
        if len(self.Y) > 0:
            self.projection_matrix = np.dot(self.Y.T, np.linalg.pinv(self.X.T))
        else:
            self.projection_matrix =  np.zeros((2, len(self.data[0])))

        
        



class cPCA(Embedding):
    def __init__(self, data, points, parent, dim=2):
        self.data = data
        self.control_points = []
        self.control_point_indices = []
        self.parent = parent
        self.X = None
        self.Y = np.array([])
        self.projection_matrix = np.zeros((2, len(self.data[0])))
        self.name = ''
        self.is_dynamic = False

        self.dim=dim

        self.ml = []
        self.cl = []
        self.has_ml_cl_constraints = False

        self.name = "cPCA"
        self.projection = np.zeros((2, len(data)))
        self.pca_projection = np.zeros((2, len(data)))
        self.is_dynamic = True 
        self.old_control_point_indices = []

        self.params = {'r' : 3.0, 'slv_mode' : 'secular', 'sigma' : None, 'epsilon' : 0.5, 'degree' : 1}
        self.params['const_nu'] = 5e+3
        self.params['orth_nu'] = 5e+3
        self.params['sigma'] = utils.median_pairwise_distances(data)
        gk = kernel_gen.gaussian_kernel()
        # gk = kernel_gen.polynomial_kernel()
        K = gk.compute_matrix(data, self.params)
        #print "GUASSIAN KERNEL"
        #print K
        #print np.shape(np.array(K))
        self.embedder = solvers.embedder(2.56e-16, 800, True)
        self.kernel_sys = self.embedder.kernel_sys(K)
        #print self.kernel_sys
        self.parent.status_text.setText("Done, calculating Gaussean kernel.")

        label_mask = np.array([0])
        self.quad_eig_sys = self.embedder.sph_cl_var_term_eig_sys(self.kernel_sys)
        self.quad_eig_sys_original = copy(self.quad_eig_sys)
        if len(self.control_point_indices) == 0:
            placement_mask = np.array([0])
        else:
            placement_mask = np.array(self.control_point_indices)
        self.const_mu = self.embedder.const_nu(self.params, placement_mask, self.kernel_sys)

        self.update_control_points({})
        self.finished_relocating()

        pca_dirs = self.embedder.soft_cp_mode_directions(self.quad_eig_sys, label_mask, np.ones((1,self.dim)), self.kernel_sys, self.params, 1e-20)
        self.pca_projection = self.kernel_sys[0].dot(pca_dirs)
        if len(points) == 0:
            pass
        else:
            self.add_adjusted_control_points(points)
            for i in range(len(self.control_point_indices)):
                self.quad_eig_sys = self.embedder.sph_cp_quad_term_eig_sys(self.kernel_sys, self.quad_eig_sys, self.control_point_indices[i], self.const_mu)
            pca_dirs = self.embedder.soft_cp_mode_directions(self.quad_eig_sys, self.control_point_indices, self.Y, self.kernel_sys, self.params, self.const_mu)
            self.pca_projection = self.kernel_sys[0].dot(pca_dirs)
        print self.control_points
        print self.control_point_indices
        print "PCA_Projection"
        print np.shape(np.array(self.pca_projection))
        print self.pca_projection

    def get_embedding(self, X=None):
        if set(self.control_point_indices) != self.old_control_point_indices:
            self.pca_projection = self.finished_relocating()
        self.old_control_point_indices = set(self.control_point_indices)
        return self.pca_projection.T


    def finished_relocating(self):
        print self.control_points
        print self.control_point_indices
        if len(self.control_point_indices) > 0:
            directions = self.embedder.soft_cp_mode_directions(self.quad_eig_sys, self.control_point_indices, self.Y, self.kernel_sys, self.params, self.const_mu)
            self.pca_projection = self.kernel_sys[0].dot(directions)
        print "PCA_Projection"
        print np.shape(np.array(self.pca_projection))
        print self.pca_projection
        return self.pca_projection


    def update_control_points(self, points):
        super(cPCA, self).update_control_points(points)
        #print "Len self.y from update_control_points"
        #print len(self.Y)
        #print "Shape self.y from update_control_points"
        #print np.shape(np.array(self.Y))
        #print "dfd"
        #print self.Y
        #print "hmm"
        if len(self.control_point_indices) > len(self.old_control_point_indices):
                selected_point = self.parent.selected_point
                if selected_point == None:
                    selected_point = (list(set(self.control_point_indices) - set(self.old_control_point_indices)))[0]
                self.quad_eig_sys = self.embedder.sph_cp_quad_term_eig_sys(self.kernel_sys, self.quad_eig_sys, selected_point, self.const_mu)
                directions = self.embedder.soft_cp_mode_directions(self.quad_eig_sys, self.control_point_indices, self.Y, self.kernel_sys, self.params, self.const_mu)
                self.pca_projection = self.kernel_sys[0].dot(directions)
        elif len(self.control_point_indices) < len(self.old_control_point_indices):
            self.quad_eig_sys = copy(self.quad_eig_sys_original)
            for i in range(len(self.control_point_indices)):
                self.quad_eig_sys = self.embedder.sph_cp_quad_term_eig_sys(self.kernel_sys, self.quad_eig_sys, self.control_point_indices[i], self.const_mu)
            if len(self.control_point_indices) == 0:
                pca_dirs = self.embedder.soft_cp_mode_directions(self.quad_eig_sys, np.array([0]), np.ones((1,self.dim)), self.kernel_sys, self.params, 1e-20)
                self.pca_projection = self.kernel_sys[0].dot(pca_dirs)
            else:
                pca_dirs = self.embedder.soft_cp_mode_directions(self.quad_eig_sys, self.control_point_indices, self.Y, self.kernel_sys, self.params, self.const_mu)
                self.pca_projection = self.kernel_sys[0].dot(pca_dirs)
        self.old_control_point_indices = set(self.control_point_indices)

        if self.has_ml_cl_constraints:
            self.augment_control_points(self.get_embedding().T)
            self.quad_eig_sys = copy(self.quad_eig_sys_original)
            for i in range(len(self.control_point_indices)):
                self.quad_eig_sys = self.embedder.sph_cp_quad_term_eig_sys(self.kernel_sys, self.quad_eig_sys, self.control_point_indices[i], self.const_mu)
            if len(self.control_point_indices) == 0:
                pca_dirs = self.embedder.soft_cp_mode_directions(self.quad_eig_sys, np.array([0]), np.ones((1,self.dim)), self.kernel_sys, self.params, 1e-20)
                self.pca_projection = self.kernel_sys[0].dot(pca_dirs)
            else:
                pca_dirs = self.embedder.soft_cp_mode_directions(self.quad_eig_sys, self.control_point_indices, self.Y, self.kernel_sys, self.params, self.const_mu)
                self.pca_projection = self.kernel_sys[0].dot(pca_dirs)
        #print "PCA_Projection"
        #print np.shape(np.array(self.pca_projection))
        #print self.pca_projection

    def add_adjusted_control_points(self, points):
        e = self.get_embedding().T
        print np.shape(e)
        for i, coords in points.items():
            if(self.dim > len(coords)):
                newPoint = e[i]
                newPoint[0] = coords[0] 
                newPoint[1] = coords[1]
                points[i] = newPoint
            elif(self.dim == len(coords)):
                pass
            else:
                points[i] = points[i][0:self.dim]
        
        self.update_control_points(points)

    

        
        

'''FOR MLE WE HAVE INTEGRATED THE ABILITY TO EASILY SWITCH BETWEEN DIFFERENT DIMENSIONS OF PROJECTIONS'''

class MLE(Embedding):
    def __init__(self, data, points, parent, dim=2):
        self.data = data
        self.dim = dim
        pca = decomposition.PCA(n_components=self.dim)
        pca.fit(self.data)

        print "POINTS"
        print points
        self.control_points = []
        self.control_point_indices = []
        self.old_control_point_indices = []
        self.parent = parent
        self.X = None
        self.Y = None
        self.projection_matrix = None
    
        self.ml = []
        self.cl = []
        self.has_ml_cl_constraints = False

        self.M_base = pca.components_ # init M with PCA[1,2]
        self.M = self.M_base
        self.Psi_base = np.cov(self.data.T)
        self.sigma = 0.1*abs(np.min(self.Psi_base))
        self.Psi = self.Psi_base
        self.update_M_matrix()
        self.update_Psi_matrix()
        self.name = "MLE"
        self.is_dynamic = True 
        self.probabilities = None

        self.update_control_points({})
        self.add_adjusted_control_points(points)

    def update_Psi_matrix(self):
        Y = self.data[self.control_point_indices].T
        W = np.array(self.control_points).T
        # print "M  :", self.M_base.shape
        # print "X_m:", Y.shape
        # print "Psi:", self.Psi_base.shape
        # print "Y_m:", W.shape
        if len(self.control_point_indices) == 0:
            self.Psi = self.Psi_base
        else:
            self.Psi = self.Psi_base - self.Psi_base.dot(Y).dot(np.linalg.pinv(Y.T.dot(self.Psi_base).dot(Y) + self.sigma*np.eye(len(Y[0])))).dot(Y.T).dot(self.Psi_base)


    def update_M_matrix(self):
        Y = self.data[self.control_point_indices].T
        W = np.array(self.control_points).T
        #W = np.resize(W, (len(self.M_base), len(Y[0])))
        if len(self.control_point_indices) == 0:
            self.M = self.M_base
        else:
            # print "M  :", self.M_base.shape
            # print "X_m:", Y.shape
            # print "Psi:", self.Psi.shape
            # print "Y_m:", W.shape
            #self.M = self.M_base + (W - self.M_base.dot(Y)).dot(np.linalg.pinv(Y.T.dot(self.Psi).dot(Y) + self.sigma*np.eye(len(Y[0])))).dot(Y.T).dot(self.Psi)
            #print np.shape(np.array(self.data))
            #print np.shape(np.array(self.M_base))
            #print np.shape(np.array(self.Psi_base))
            #print np.shape(np.array(Y))
            #print np.shape(np.array(W))
            print W
            #print np.shape(np.array(self.M_base))
            #print np.shape(np.array((W - self.M_base.dot(Y)).dot(np.linalg.pinv(Y.T.dot(self.Psi_base).dot(Y) + self.sigma*np.eye(len(Y[0])))).dot(Y.T).dot(self.Psi_base)))
            self.M = self.M_base + (W - self.M_base.dot(Y)).dot(np.linalg.pinv(Y.T.dot(self.Psi_base).dot(Y) + self.sigma*np.eye(len(Y[0])))).dot(Y.T).dot(self.Psi_base)


    def get_embedding(self, X=[]):
        if X == []:
            X=self.data.T
        self.projection_matrix = self.M
        print np.shape(self.projection_matrix)
        return self.M.dot(X)
    

    def update_control_points(self, points):
        super(MLE, self).update_control_points(points)
        print self.control_point_indices
        print self.old_control_point_indices
        print "POINTS"
        print points
        print self.Y
        if set(self.control_point_indices) == self.old_control_point_indices:
            self.update_M_matrix()
        else:
            self.update_M_matrix()
            self.update_Psi_matrix()
        self.old_control_point_indices = set(self.control_point_indices)
        if self.has_ml_cl_constraints:
            self.augment_control_points(self.get_embedding().T)
            self.update_M_matrix()
            self.update_Psi_matrix()

    ''' When shifting between different no of dimensions, we have to make sure all control points are correctly adjusted (difficult when
        when increasing the number of dims.
        This method is called from KMEANS after initialising MLE with no control points, and then readding the control points after an initial
        embedding is calculated'''

    def add_adjusted_control_points(self, points):
        e = self.get_embedding().T
        print np.shape(e)
        for i, coords in points.items():
            if(self.dim > len(coords)):
                newPoint = e[i]
                newPoint[0] = coords[0] 
                newPoint[1] = coords[1]
                points[i] = newPoint
            elif(self.dim == len(coords)):
                pass
            else:
                points[i] = points[i][0:self.dim]
        
        self.update_control_points(points)

'''
    This class can be used to add things on top of the projections such as a clsutering algorithm. It is essentially a wrapper around the embeddings/projections.
    It would have been probably better to design the projections at the top of the heirarchy (An embedding uses a clustering algorithm) but this made 
    it easier to integrate with the gui.
'''
class CLUSTER_OVERLAY(object):
    def __init__(self, embed, points, parent, dim=2, num=3, met="euclidean", emb="MLE", clus="KMEANS"):
        self.parent = parent
        self.data = embed.data
        self.points = points

        self.cluster_association = []
        self.cluster_centers = []
        self.cluster_centers_embedding = []

        self.embedding = embed
        self.dim=2
        self.is_dynamic = self.embedding.is_dynamic
        self.name = self.embedding.name

        self.get_embedding()
        self.generate_cluster(num, met, clus)

    def set_embedding_type(self, name, dim):
        self.name = name
        self.dim=dim
        if (self.name == "XY"):
            self.embedding = XY(self.data, self.points, self.parent)
            self.dim=2
        elif (self.name == "PCA"):
            self.embedding = PCA(self.data, self.points, self.parent, dim=self.dim)    
        elif (self.name == "LLE"):
            self.embedding = LLE(self.data, self.points, self.parent)
            self.dim=2
        elif (self.name == "ISO"):
            self.embedding = ISO(self.data, self.points, self.parent)
            self.dim=2
        elif (self.name == "MDS"):
            self.embedding = MDS(self.data, self.points, self.parent)
            self.dim=2
        elif (self.name == "ICA"):
            self.embedding = ICA(self.data, self.points, self.parent)
            self.dim=2
        elif (self.name == "t-SNE"):
            self.embedding = tSNE(self.data, self.points, self.parent)
            self.dim=2
        elif (self.name == "LSP"):
            self.embedding = LSP(self.data, self.points, self.parent)  
            self.dim=2 
        elif (self.name == "MLE"):
            self.embedding = MLE(self.data, self.points, self.parent, dim=self.dim)
        elif (self.name == "cPCA"):
            self.embedding = cPCA(self.data, self.points, self.parent, dim=self.dim)
        else:
            self.embedding = PCA(self.data, self.points, self.parent, dim=self.dim)
        self.is_dynamic = self.embedding.is_dynamic
        self.name = self.embedding.name
        
    def get_embedding(self):
        embed = self.embedding.get_embedding()
        return embed

    def set_data_and_cp(self, data, points):
        self.data = data
        self.points = points
        self.set_embedding_type(self.name,self.dim)
        self.generate_cluster(self.num, self.met, self.clus)

    def get_embedding_object(self):
        '''Have to make sure dimensions are reduced back to 2 dimensions'''
        if ((self.name in emb_unrestricted_dims) and (self.dim > 2)):
            self.set_embedding_type(self.name, 2)
            return self.embedding
        else:
            return self.embedding

    def update_must_and_cannot_link(self, ml, cl):
        self.embedding.update_must_and_cannot_link(ml,cl)

    def update_control_points(self, points):
        self.embedding.update_control_points(points)
        #embed = self.embedding.get_embedding()
        #self.run_kmeans(embed.T, self.num, self.met)

    def finished_relocating(self):
        relocate = self.embedding.finished_relocating()
        return relocate

    def generate_cluster(self, num, met, clus):
        embed = self.get_embedding()
        self.num = num
        self.met = met
        self.clus = clus
        if (self.clus == "KMEANS"):
            self.run_kmeans(embed.T, self.num, self.met)
        else:
            self.run_kmeans(embed.T, self.num, self.met) # DEFAULT CLUSTERING

    def run_kmeans(self, data, num, met):
        try:    
            km = kmeans.Kmeans(data, k=num, nsample=50, delta=.001, maxiter=100, verbose=0, metric=met)
            self.cluster_association = km.Xtocentre
            self.cluster_centers = km.centres
            self.cluster_centers_embedding = self.cluster_centers
        except:
            msg = "It seems like the embedding algorithm did not converge with the given parameter setting"
            QMessageBox.about(self.parent, "Embedding error", msg)
            
    
    def get_cluster_centers(self):
        return self.cluster_centers

    def get_cluster_centers_embedding(self):
        return self.cluster_centers_embedding.T

    def get_cluster_assocations(self):
        return self.cluster_association
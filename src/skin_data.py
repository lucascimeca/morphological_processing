import numpy as np
from math import floor
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import animation
from sklearn import neighbors, cluster, mixture
import sklearn.metrics as sm
from itertools import permutations
from matplotlib.colors import ListedColormap
import random

"""
Class handling the skin data
The data is stored in two forms, 'non_refined_data' and 'data'. 
'non_refined_data' is a NxM matrix where M is the number of data points (taxels + heat sensors) and N are the
time frames where the data was sampled
'data' is a MxTxS matrix where M are the number of modules used for sensing, T is the number of taxels per each module
and S are the number of time samples in the data.
"""
class SkinData:
    modules_number = None
    taxels_number = None
    data = None
    non_refined_data = None
    class_names = None

    def __init__(self, skin_data=None, taxels_number=10, resolution_reduction=None):
        if taxels_number is None: self.taxels_number = 10
        self.data = dict()
        self.non_refined_data = dict
        if skin_data is not None:
            self.load_data(skin_data)
            self._clean_data(resolution_reduction)

    def get_data(self, which_object=None):
        if which_object:
            return self.data[which_object]
        return self.data

    def get_non_refined_data(self, which_object=None):
        if which_object:
            return self.data[which_object]
        return self.non_refined_data

    def get_module(self, key, module_number):
        return self.data[key][module_number, :, :]

    def get_taxel(self, key, module_number, taxel_number):
        return self.data[key][module_number, taxel_number, :]

    def get_time_snapshot(self, key, time=None, mode='avg'):
        if mode=='best':
            return self.data[key][:, :, find_highest_pressure(self.data[key])].reshape(1, -1)
        if time is not None:
            if mode=='avg':
                return np.mean(self.data[key][:, :, time:time+10], axis=2).reshape(1, -1)
            return self.data[key][:, :, time]
        raise ValueError('need to specify a time')


    def load_data(self, skin_data):
        #load data
        if isinstance(skin_data, dict):
            self.non_refined_data = skin_data
        elif isinstance(skin_data, str):
            self.non_refined_data = {k : v for k,v in loadmat(skin_data).items()
                                     if isinstance(v, np.ndarray) and len(v.shape)==3}

        #reshape and save
        for key in self.non_refined_data.keys():
            self.non_refined_data[key] = self.non_refined_data[key].reshape(self.non_refined_data[key].shape[1],
                                                                            self.non_refined_data[key].shape[2]).T

    def _clean_data(self, resolution_reduction):
        #create array of indexes to clean the data of heat sensor and other non used values
        key = list(self.non_refined_data.keys())[0]
        # del_rows = [i for i in list(range(2, self.non_refined_data[key].shape[1], 1))
        #             if self.non_refined_data[key][
        #                 floor(self.non_refined_data[key].shape[0]/2), i
        #             ] > 40000]
        # del_rows = del_rows + [del_rows[i]-1 for i in list(range(1,len(del_rows),1))
        #                             if del_rows[i]-del_rows[i-1] > 11]

        del_rows = [0, 1, 2, 13, 24, 36, 47, 58, 59]
        for key in self.non_refined_data.keys():
            self.data[key] = np.delete(self.non_refined_data[key], [0, 1] + del_rows, axis=1)

            # this is only going to work for 6 modules and 60 taxels!
            if resolution_reduction is not None and resolution_reduction !=0:
                index_reduction_list = []
                for i in range(6):
                    index_reduction_list += random.sample(range(i*10, i*10+10), resolution_reduction)
                self.data[key] = np.delete(self.data[key], index_reduction_list, axis=1)

            # change 6 to self.data[key].shape[0]/taxel_number if number of modules not 6
            self.data[key] = self.data[key].T.reshape(6, -1, self.data[key].shape[0])
            self.data[key] = self.data[key]-np.mean(self.data[key][:,:,0:10], axis=2).reshape(
                self.data[key].shape[0],
                self.data[key].shape[1],
                1
            )




"""Function printing the skin data from a SkinData time_snapshot onto a blank canvas.
        Inputs: canvas = (MxN) numpy array
                skin_snapshot = (AxB) numpy array where A<M and B<N
        Output: (MxN) matrix of snapshot in canvas"""
def fill_canvas(canvas, skin_snapshot):
    mid = floor(canvas.shape[1]/2)
    # ---------------- LAYOUT FOR EMBEDDED SKIN SENSOR IN EXAGON ------------
    canvas[0, mid-2:mid+2] =                                                 [skin_snapshot[5, 3]] + [skin_snapshot[5, 2]] + [skin_snapshot[5, 1]] + [skin_snapshot[5, 0]]
    canvas[1, mid-3:mid+2] =                                      [skin_snapshot[4, 3]] + [skin_snapshot[5, 4]] + [skin_snapshot[5, 9]] + [skin_snapshot[5, 8]] + [skin_snapshot[3, 3]]
    canvas[2, mid-3:mid+3] =                         [skin_snapshot[4, 4]] + [skin_snapshot[4, 2]] + [skin_snapshot[5, 5]] + [skin_snapshot[5, 7]] + [skin_snapshot[3, 4]] + [skin_snapshot[3, 2]]
    canvas[3, mid-4:mid+3] =             [skin_snapshot[4, 5]] + [skin_snapshot[4, 9]] + [skin_snapshot[4, 1]] + [skin_snapshot[5, 6]] + [skin_snapshot[3, 5]] + [skin_snapshot[3, 9]] + [skin_snapshot[3, 1]]
    canvas[4, mid-4:mid+4] = [skin_snapshot[4, 6]] + [skin_snapshot[4, 7]] + [skin_snapshot[4, 8]] + [skin_snapshot[4, 0]] + [skin_snapshot[3, 6]] + [skin_snapshot[3, 7]] + [skin_snapshot[3, 8]] + [skin_snapshot[3, 0]]
    canvas[5, mid-4:mid+4] = [skin_snapshot[0, 0]] + [skin_snapshot[0, 8]] + [skin_snapshot[0, 7]] + [skin_snapshot[0, 6]] + [skin_snapshot[2, 3]] + [skin_snapshot[2, 2]] + [skin_snapshot[2, 1]] + [skin_snapshot[2, 0]]
    canvas[6, mid-4:mid+3] =             [skin_snapshot[0, 1]] + [skin_snapshot[0, 9]] + [skin_snapshot[0, 5]] + [skin_snapshot[1, 0]] + [skin_snapshot[2, 4]] + [skin_snapshot[2, 9]] + [skin_snapshot[2, 8]]
    canvas[7, mid-3:mid+3] =                         [skin_snapshot[0, 2]] + [skin_snapshot[0, 4]] + [skin_snapshot[1, 1]] + [skin_snapshot[1, 8]] + [skin_snapshot[2, 5]] + [skin_snapshot[2, 7]]
    canvas[8, mid-3:mid+2] =                                       [skin_snapshot[0, 3]] + [skin_snapshot[1, 2]] + [skin_snapshot[1, 9]] + [skin_snapshot[1, 7]] + [skin_snapshot[2, 6]]
    canvas[9, mid-2:mid+2] =                                                 [skin_snapshot[1, 3]] + [skin_snapshot[1, 4]] + [skin_snapshot[1, 5]] + [skin_snapshot[1, 6]]

    # ---------------- LAYOUT FOR EMBEDDED SKIN SENSOR IN EXPERIMENTS ------------
    # canvas[0, mid-2:mid+2] =                                                 [skin_snapshot[1, 0]] + [skin_snapshot[1, 8]] + [skin_snapshot[1, 7]] + [skin_snapshot[1, 6]]
    # canvas[1, mid-3:mid+2] =                                      [skin_snapshot[0, 3]] + [skin_snapshot[1, 1]] + [skin_snapshot[1, 9]] + [skin_snapshot[1, 5]] + [skin_snapshot[2, 7]]
    # canvas[2, mid-3:mid+3] =                         [skin_snapshot[0, 4]] + [skin_snapshot[0, 2]] + [skin_snapshot[1, 2]] + [skin_snapshot[1, 4]] + [skin_snapshot[2, 8]] + [skin_snapshot[2, 6]]
    # canvas[3, mid-4:mid+3] =             [skin_snapshot[0, 5]] + [skin_snapshot[0, 9]] + [skin_snapshot[0, 1]] + [skin_snapshot[1, 3]] + [skin_snapshot[2, 0]] + [skin_snapshot[2, 9]] + [skin_snapshot[2, 5]]
    # canvas[4, mid-4:mid+4] = [skin_snapshot[0, 6]] + [skin_snapshot[0, 7]] + [skin_snapshot[0, 8]] + [skin_snapshot[0, 0]] + [skin_snapshot[2, 1]] + [skin_snapshot[2, 2]] + [skin_snapshot[2, 3]] + [skin_snapshot[2, 4]]
    # canvas[5, mid-4:mid+4] = [skin_snapshot[4, 8]] + [skin_snapshot[4, 7]] + [skin_snapshot[4, 6]] + [skin_snapshot[4, 5]] + [skin_snapshot[5, 6]] + [skin_snapshot[5, 5]] + [skin_snapshot[5, 4]] + [skin_snapshot[5, 3]]
    # canvas[6, mid-4:mid+3] =             [skin_snapshot[4, 0]] + [skin_snapshot[4, 9]] + [skin_snapshot[4, 4]] + [skin_snapshot[3, 1]] + [skin_snapshot[5, 7]] + [skin_snapshot[5, 9]] + [skin_snapshot[5, 2]]
    # canvas[7, mid-3:mid+3] =                         [skin_snapshot[4, 1]] + [skin_snapshot[4, 3]] + [skin_snapshot[3, 2]] + [skin_snapshot[3, 0]] + [skin_snapshot[5, 8]] + [skin_snapshot[5, 1]]
    # canvas[8, mid-3:mid+2] =                                       [skin_snapshot[4, 2]] + [skin_snapshot[3, 3]] + [skin_snapshot[3, 9]] + [skin_snapshot[3, 8]] + [skin_snapshot[5, 0]]
    # canvas[9, mid-2:mid+2] =                                                 [skin_snapshot[3, 4]] + [skin_snapshot[3, 5]] + [skin_snapshot[3, 6]] + [skin_snapshot[3, 7]]
    return canvas




"""Function animating the skin data in time series, dependent on the canvas filling function.
        Inputs: skin_data = (MxNxD) numpy array of skin data
        Output: animation of skin_snapshots over time"""
def simulate_skin_data(skin_data, interpolation=None):
    fig = plt.figure()
    #we make the data positive to create better animations
    skin_data = skin_data + np.abs(np.min(skin_data, axis=2).reshape(skin_data.shape[0], skin_data.shape[1], 1))
    min = np.min(skin_data)
    skin_canvas = np.multiply(np.ones((10, 10)), min)
    skin_array = fill_canvas(skin_canvas, skin_data[:,:,0])
    if interpolation is not None:
        im = plt.imshow(skin_array, cmap='hot', interpolation=interpolation)
    else:
        im = plt.imshow(skin_array, cmap='hot')
    def animate(i):
        skin_array = fill_canvas(skin_canvas, skin_data[:,:,i])
        im.set_array(skin_array)
        return [im]
    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=np.arange(0, skin_data.shape[2]), interval=50, blit = True)
    plt.show()
    return anim


"""Function printing the time snapshot of the skin sensor onto a canvas.
        Inputs: time_snapshot = (MxN) numpy array 
        Output: (10x10) plot of the skin time snapshot onto a canvas """
def print_skin_data(time_snapshot):
    skin_canvas = np.ones((10, 10))* np.min(time_snapshot, axis=(0,1))
    skin_array = fill_canvas(skin_canvas, time_snapshot)
    return plt.imshow(skin_array, cmap='gray', interpolation='bicubic')


"""Function finding the index of the snapshot with the highest cumulative pressure recorded.
        Inputs: skin_data = (MxN) numpy array 
        Output: ind = index of the time snapshot with the highest cumulative amount of pressure"""
def find_highest_pressure(skin_data):
    snapshot_pressures = np.sum(skin_data, axis=(0,1))
    max_press = np.max(snapshot_pressures)
    ind = [i for i in range(snapshot_pressures.shape[0]) if snapshot_pressures[i]==max_press]
    return ind[0]


"""Function applying K nearest neighbors to a dataset and returning a 2d plot
        Inputs: skin_data = (MxN) numpy array 
        Output: ind = index of the time snapshot with the highest cumulative amount of pressure"""
def apply_Knn(X, classes, n_neighbors=5, weights='distance', mesh=True, show=True):

    # getting data ready for processing
    X = X[:,:2]
    h = np.mean(X[:, 0].max() + 1-X[:, 0].min() - 1)/30  # step size in the mesh
    if len(classes.shape) > 1 and classes.shape[0] > classes.shape[1]:
        classes = classes.T

    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, classes)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - (h*30), X[:, 0].max() + (h*30)
    y_min, y_max = X[:, 1].min() - (h*30), X[:, 1].max() + (h*30)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    if show:
        plt.figure()
        if mesh:
            # Put the result into a color plot
            plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=classes, cmap=cmap_bold,
                    edgecolor='k', s=20)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("3-Class classification (k = %i, weights = '%s')"
                  % (n_neighbors, weights))
        plt.show()
    return Z


"""Function applying Kmeans Clustering to a dataset
        Inputs: X: np array of data
                targets: targets of the data (doesn't need to be specified - unsupervised learning)t
                show: set to True for figures
        Output: targets for each element in X (i.e. which cluster they belong to)"""
def apply_KMC(X, targets=None, tactile_objects=None, tactile_classes=None, target_to_cls_dict=None, class_names=None, n_clusters=2, task_mode=False, show=True):
    # FOR MASH
    h = np.mean(X[:, 0].max() + 1 - X[:, 0].min() - 1) / 1000  # step size in the mesh -->
    # Create color maps
    colors = ['#c6f0ff', '#ffb2a8']
    cmap_light = ListedColormap(colors)
    cmap_bold = ListedColormap(['#0000FF'])
    x_min, x_max = X[:, 0].min() - (h*30), X[:, 0].max() + (h*30)
    y_min, y_max = X[:, 1].min() - (h*30), X[:, 1].max() + (h*30)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, np.mean(X[:, 0].max() + 1 - X[:, 0].min() - 1)/10),
                         np.arange(y_min, y_max, np.mean(X[:, 0].max() + 1 - X[:, 0].min() - 1)/10))
                                                            # devide by 1000 for paper results, but slow
    # FOR MASH

    km = cluster.KMeans(n_clusters=n_clusters, random_state=0)
    data_clusters = km.fit_predict(X[:, :2])
    p_data_clusters = data_clusters

    # FOR MASH
    Z = km.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    k_means_cluster_centers = km.cluster_centers_
    center_class_labels = km.predict(k_means_cluster_centers)

    if targets is not None:
        data_clusters = reorder_clusters(km, targets)
        if any(p_data_clusters != data_clusters):
            tmp_dict = {v: k for k, v in target_to_cls_dict.items()}
            for key in tmp_dict.keys():
                tmp_dict[key] = int(not(tmp_dict[key]))
            target_to_cls_dict = {v: k for k, v in tmp_dict.items()}

    fig = None
    if show:

        #  ------------ PLOT STUFF -----------
        # plot mash
        fig = plt.figure(figsize=(13, 8))
        ax = fig.add_subplot(111)
        ax.set_xlabel('$\\vec{p}_1$', fontsize=48)
        ax.set_ylabel('$\\vec{p}_2$', fontsize=48)
        ax.pcolormesh(xx, yy, Z, cmap=cmap_light)

        # plot data points
        ax.scatter(X[:, 0], X[:, 1], cmap=cmap_bold,
                        edgecolor='k', s=240)

        # draw Centroids and line between them
        ax.scatter(k_means_cluster_centers[:, 0], k_means_cluster_centers[:, 1], marker='+',
                c='#158400', linewidth=6, s=450)
        plt.plot(k_means_cluster_centers[:, 0], k_means_cluster_centers[:, 1], 'k--', linewidth=1)

        # draw line separating data
        axes = plt.axis()
        y_max = axes[3]
        y_min = axes[2]
        center_x = (k_means_cluster_centers[0, 0]+k_means_cluster_centers[1, 0])/2
        center_y = (k_means_cluster_centers[0, 1]+k_means_cluster_centers[1, 1])/2
        slope = -  (k_means_cluster_centers[1, 0]-k_means_cluster_centers[0, 0])/\
                   (k_means_cluster_centers[1, 1]-k_means_cluster_centers[0, 1])
        shift = center_y - slope * center_x
        p1_y = y_min
        p1_x = (p1_y - shift)/slope
        p2_y = y_max
        p2_x = (p2_y - shift)/slope
        plt.plot([p1_x, p2_x], [p1_y, p2_y], 'k-', linewidth=4)
        plt.xlim( [axes[0], axes[1]])
        plt.ylim( [axes[2], axes[3]])
        ax.text(center_x, center_y, "$l_{KMC}$",
                fontsize = 34)

        # plot classes names
        if task_mode:
            cluster_objects = [stringfy_set(target_to_cls_dict[label]) for label in center_class_labels]
            for i in range(len(cluster_objects)):
                for old_cls_name in tactile_classes:
                    cluster_objects[i] = cluster_objects[i].replace(old_cls_name, class_names[old_cls_name], 10)
        else:
            cluster_objects = tactile_objects.get_objects(center_class_labels)
            for i in range(len(cluster_objects)):
                for old_cls_name in tactile_classes:
                    cluster_objects[i] = cluster_objects[i].replace(old_cls_name, class_names[old_cls_name], 10)

        #  ----- ADD PLOT LABELS AND LEGENTS -------
        # plot class name
        if tactile_classes is not None:
            for i in range(X.shape[0]):
                mid_x = (max(X[:,0])+min(X[:,0]))/2
                mid_y = (max(X[:,1])+min(X[:,1]))/2
                h_align = 'left'
                v_align = 'bottom'
                if mid_x-X[i,0] <0:
                    h_align = 'right'
                if mid_y-X[i,1] <0:
                    v_align = 'top'
                if class_names:
                    ax.text(X[i, 0], X[i, 1], class_names[tactile_classes[i]],
                            horizontalalignment=h_align,
                            verticalalignment=v_align,
                            style='italic',
                            fontsize=32)
                else:
                    ax.text(X[i,0], X[i, 1], tactile_classes[i],
                        horizontalalignment=h_align,
                        verticalalignment=v_align,
                        style='italic',
                        fontsize = 32)

        # add legends for meshes
        patch1 = mpatches.Patch(color=colors[0], label=cluster_objects[0])
        patch2 = mpatches.Patch(color=colors[1], label=cluster_objects[1])
        ax.legend(handles=[patch1, patch2], fontsize=18)

        # add Centroids labels
        for cluster_center, cluster_object, i in zip(k_means_cluster_centers, cluster_objects, range(len(cluster_objects))):
            ax.text(cluster_center[0], cluster_center[1], "$C_"+str(i)+"$",
                    style='italic',
                    fontsize = 34)
        #  -----------------------------------

    return (data_clusters, fig)




"""Function applying Kmeans Clustering to a dataset
        Inputs: X: np array of data
                targets: targets of the data (doesn't need to be specified - unsupervised learning)
                show: set to True for figures
        Output: targets for each element in X (i.e. which cluster they belong to)"""
def apply_GMM(X, targets=None, tactile_objects=None, tactile_classes=None, target_to_cls_dict=None, n_clusters=2, task_mode=False, show=True):
    gmm = mixture.GaussianMixture(n_components=n_clusters, covariance_type='full')
    gmm.fit(X[:, :2])
    data_clusters = gmm.predict(X[:, :2])
    if targets is not None:
        data_clusters = reorder_clusters_GMM(targets, data_clusters)

    fig = None
    if show:
        # plot data points
        fig = plt.figure(figsize=(13, 8))
        ax = fig.add_subplot(111)
        plt.scatter(X[:, 0], X[:, 1], c=data_clusters,
                        edgecolor='k', s=20)
        if tactile_classes is not None:
            for i in range(X.shape[0]):
                ax.text(X[i,0], X[i, 1], stringfy_set(tactile_classes[i]),
                        style='italic',
                        bbox={'facecolor': 'blue', 'alpha': 0.1, 'pad': 1})
        # plot classes names
        gmm_cluster_centers = gmm.means_
        center_class_labels = gmm.predict(gmm_cluster_centers)
        if task_mode:
            cluster_objects = [target_to_cls_dict[label] for label in center_class_labels]
        else:
            cluster_objects = tactile_objects.get_objects(center_class_labels)

        for cluster_center, cluster_object in zip(gmm_cluster_centers, cluster_objects):
            ax.text(cluster_center[0], cluster_center[1], cluster_object,
                    style='italic',
                    bbox={'facecolor': 'red', 'alpha': 0.9, 'pad': 4})
        plt.show()

    return (data_clusters, fig)


"""Function renaming the clusters so to match the targets with the highest accuracy
        Inputs: km = KMeans class from sklearn library
                targets = np array of targets for the data
        Output: np array of outputs (same shape as targets)"""
def reorder_clusters(km, targets):
    lables = km.labels_
    perms = list(permutations(np.unique(lables)))
    accuracies = [sm.accuracy_score(targets, np.choose(km.labels_, perm_labels).astype(np.int64))
                  for perm_labels in perms]
    return np.choose(km.labels_, perms[np.argmax(accuracies)]).astype(np.int64)


"""Function renaming the clusters so to match the targets with the highest accuracy
        Inputs: km = KMeans class from sklearn library
                targets = np array of targets for the data
        Output: np array of outputs (same shape as targets)"""
def reorder_clusters_GMM(targets, labels):
    lables = labels
    perms = list(permutations(np.unique(lables)))
    accuracies = [sm.accuracy_score(targets, np.choose(labels, perm_labels).astype(np.int64))
                  for perm_labels in perms]
    return np.choose(labels, perms[np.argmax(accuracies)]).astype(np.int64)


"""Function computing some metrics for the clustering
        Inputs: outputs = outputs from the clustering algorithm
                targets = target outputs for the data
        Output: tuple (cm, accuracy, fms) where cm is the confusion
                matrix of the data, accuracy the accuracy of the predictions
                and fms the Fowlkes-Mallows score."""
def get_cluster_metrics(outputs, targets, tact_objects=None, target_to_cls_dict=None, task_mode=False, show=True):
    n = np.unique(outputs).shape[0]
    cm = sm.confusion_matrix(targets, outputs)
    accuracy = sm.accuracy_score(targets, outputs)
    fms = sm.fowlkes_mallows_score(targets, outputs)

    fig = None
    if show:
        fig_size = (10, 5)
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(1, 1, 1)
        plt.imshow(cm, interpolation='none', cmap='Blues')
        for (i, j), z in np.ndenumerate(cm):
            plt.text(j, i, z, ha='center', va='center', fontsize=24)
        plt.xlabel("kmeans labels", fontsize=16)
        plt.ylabel("true labels", fontsize=16)
        if tact_objects is not None:
            if task_mode:
                plt.xticks(range(n), [stringfy_set(target_to_cls_dict[i]) + "\nguess" for i in range(n)], size='medium')
                plt.yticks(range(n), [stringfy_set(target_to_cls_dict[i]) for i in range(n)], size='small')
            else:
                plt.xticks(range(n), [cls + "_guess" for cls in tact_objects.get_objects(list(range(n)))], size='medium')
                plt.yticks(range(n), tact_objects.get_objects(list(range(n))), size='medium')

        plt.draw()

    # print('The confusion Matrix for Task {} is:'.format(str([stringfy_set(target_to_cls_dict[i]) + "\nguess" for i in range(n)])))
    # print('The accuracy is: {0:.4f}'.format(accuracy))
    # print('The Fowlkes-Mallows scores is: {0:.4f}'.format(fms))
    return (fig, (cm, accuracy, fms))

def stringfy_set(set):
    out_str = "{ "
    for elem in set:
        out_str += str(elem) + ', '
    return out_str[:-2] + " }"
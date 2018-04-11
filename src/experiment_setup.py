from skin_data import (
    SkinData,
    apply_KMC,
    apply_GMM,
    get_cluster_metrics,
    stringfy_set
)
from tactile_objects import (
    TactileObjectsTask
)
import numpy as np
from sklearn import neighbors, cluster, decomposition
import matplotlib.pyplot as plt

"""
Class implementing the experiments for --PAPER--. 
experiment_data is a dictionary containing as key the key of the different experiments (eg 3mil, 6mil etc) and data
a tuple containing the data np array as first element and the targets np array as the second.

The matrices names need to be in the format 'classname_experimentname' eg. 'Cube_10mil' for class cube and experiment
10mil
"""
class ExperimentSetup:

    skin = None
    taskObj = None
    classes = list()
    experiments = set()
    trials = set()
    experiment_data = dict()
    target_to_cls_dict = dict()
    pca = None
    class_names = None

    def __init__(self, path_to_data="./../data/skin_experiments.mat", best_pressure_time=140, resolution_reduction=False):
        self.class_names = {
            'Cube': 'Cube',
            'Sphere': 'Sphere',
            'HSphere': 'Half-Cilinder',
            'HCube': 'Cuboid'
        }
        self.skin = SkinData(path_to_data, resolution_reduction=resolution_reduction)
        all_skin_data = self.skin.get_data()
        self.pca = decomposition.PCA(n_components=3)
        self.classes = np.sort(np.unique([obj.split('_')[0] for obj in all_skin_data.keys()]))
        self.experiments = np.unique([obj.split('_')[1] for obj in all_skin_data.keys()])
        self.trials = np.unique([obj.split('_')[2] for obj in all_skin_data.keys()])
        self.taskObj = TactileObjectsTask(self.classes)
        for exp in self.experiments:
            self.experiment_data[exp] = self._get_trial_snapshot(
                self.classes[0],
                exp,
                all_skin_data,
                best_pressure_time
            )
            for i in range(1, self.classes.shape[0]):
                self.experiment_data[exp] = np.concatenate([
                    self.experiment_data[exp],
                    self._get_trial_snapshot(self.classes[i], exp, all_skin_data, best_pressure_time)
                ], axis=0)


    def run_experiment(self, experiment_name, which_clustering='k_means', n_clusters=2, task=None, show=True, save=True):
        self.pca.fit(self.experiment_data[experiment_name])
        projected_data = self.pca.transform(self.experiment_data[experiment_name])
        target_dict = self._get_targets(self.taskObj.get_tasks())
        if task==None:
            for key in target_dict.keys():
                if which_clustering == 'k_means':
                    output_targets, cls_plt = apply_KMC(
                        projected_data,
                        targets = target_dict[key],
                        tactile_objects = self.taskObj,
                        tactile_classes = self.classes,
                        target_to_cls_dict = self.target_to_cls_dict[key],
                        class_names = self.class_names,
                        n_clusters = n_clusters,
                        task_mode=True,
                        show=show
                    )
                else:
                    output_targets, cls_plt = apply_GMM(
                        projected_data,
                        targets = target_dict[key],
                        tactile_objects = self.taskObj,
                        tactile_classes = self.classes,
                        target_to_cls_dict = self.target_to_cls_dict[key],
                        n_clusters = n_clusters,
                        task_mode=True,
                        show=show
                    )

                cm_fig, (_, acc, _) = get_cluster_metrics(
                    output_targets,
                    target_dict[key],
                    tact_objects = self.taskObj,
                    target_to_cls_dict = self.target_to_cls_dict[key],
                    task_mode=True,
                    show=show
                )
                if save:
                    # saving plots
                    task_name  = ""
                    for s in key:
                        task_name += stringfy_set(s)
                    cm_fig.tight_layout()
                    cm_fig.savefig('../manuscript/generated_figures/'+ experiment_name + '_' + 'cm_(' + task_name + ').pdf',
                    bbox_inches="tight")

                    cls_plt.tight_layout()
                    cls_plt.savefig('../manuscript/generated_figures/'+ experiment_name  + '_' +'clsplt_(' + task_name + ').pdf',
                    bbox_inches="tight")

        else:
            if which_clustering == 'k_means':
                output_targets, cls_plt = apply_KMC(
                    projected_data,
                    targets=target_dict[task],
                    tactile_objects=self.taskObj,
                    tactile_classes=self.classes,
                    target_to_cls_dict=self.target_to_cls_dict[task],
                    class_names = self.class_names,
                    n_clusters=n_clusters,
                    task_mode=True,
                    show=show
                )
            else:
                output_targets, cls_plt = apply_GMM(
                    projected_data,
                    targets=target_dict[task],
                    tactile_objects=self.taskObj,
                    tactile_classes=self.classes,
                    target_to_cls_dict=self.target_to_cls_dict[task],
                    n_clusters=n_clusters,
                    task_mode=True,
                    show=show
                )

            cm_fig, (_, acc, _) = get_cluster_metrics(
                output_targets,
                target_dict[task],
                tact_objects=self.taskObj,
                target_to_cls_dict=self.target_to_cls_dict[task],
                task_mode=True,
                show=show
            )
            if save:
                # saving plots
                task_name = ""
                for s in task:
                    task_name += stringfy_set(s)
                cm_fig.tight_layout()
                cm_fig.savefig(
                    '../manuscript/generated_figures/' + experiment_name + '_' + 'cm_(' + task_name + ').pdf',
                    bbox_inches="tight"
                )

                cls_plt.tight_layout()
                cls_plt.savefig(
                    '../manuscript/generated_figures/' + experiment_name + '_' + 'clsplt_(' + task_name + ').pdf',
                    bbox_inches="tight"
                )
            return acc
        plt.close('all')

    def _get_targets(self, tasks):
        targets = dict()
        for task in tasks:
            #generate targets for each task
            iter_tasks = iter(task)
            class1 = next(iter_tasks)
            self.target_to_cls_dict[task] = dict({1: class1, 0: next(iter_tasks)})
            targets[task] = np.multiply(
                [self.classes[i] in class1 for i in range(len(self.classes))],
                1)
        return targets

    def _get_trial_snapshot(self, cls, exp, all_skin_data, best_pressure_time):
        # first element
        trial = self.trials[0]
        cls_exp_trial_key = [key for key in all_skin_data.keys() if
                                 exp == key.split('_')[1] and
                                 cls == key.split('_')[0] and
                                 trial == key.split('_')[2]][0]
        experiment_data = self.skin.get_time_snapshot(
            cls_exp_trial_key, best_pressure_time
        ).reshape(1, -1)
        # the rest
        for j in range(1, self.trials.shape[0]):
            trial = self.trials[j]
            cls_exp_trial_key = [key for key in all_skin_data.keys() if
                                 exp == key.split('_')[1] and
                                 cls == key.split('_')[0] and
                                 trial == key.split('_')[2]][0]
            experiment_data = experiment_data + self.skin.get_time_snapshot(
                    cls_exp_trial_key, best_pressure_time
                ).reshape(1, -1)
        return experiment_data / self.trials.shape[0]

    def get_skin(self):
        return self.skin

    def get_experiment_data_cls_pos(self):
        return self.classes

    def get_experiment_data(self):
        return self.experiment_data

    def set_experiment_data(self, experiment_data):
        self.experiment_data = experiment_data
        return True

    def get_experiments(self):
        return self.experiments

    def get_tasks(self):
        return self.taskObj.get_tasks()

    def get_classes(self):
        return self.classes

    def get_taskObj(self):
        return self.taskObj

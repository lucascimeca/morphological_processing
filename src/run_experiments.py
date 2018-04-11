from experiment_setup import ExperimentSetup

task2 = frozenset({frozenset({'Sphere', 'HSphere', 'HCube'}), frozenset({'Cube'})})
task4 = frozenset({frozenset({'Sphere', 'HSphere'}), frozenset({'HCube', 'Cube'})})
task5 = frozenset({frozenset({'Sphere', 'Cube'}), frozenset({'HCube', 'HSphere'})})

exp_setup = ExperimentSetup("./../data/skin_experiments.mat", best_pressure_time=140)
# exp_setup.run_experiment(experiment_name='3mil', which_clustering='k_means', task=task4, show=True, save=True)
exp_setup.run_experiment(experiment_name='6mil', which_clustering='k_means', task=task2, show=True, save=True)
# exp_setup.run_experiment(experiment_name='10mil', which_clustering='k_means', task=task5, show=True, save=True)
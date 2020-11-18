import random
from itertools import combinations
import math
import numpy as np

"""class handling the objects sensed through the skin data
    The class will code the object names with classes (numerical), and assign
    a color to each. The assigments are randomly generated and held in dictionaries internally"""
class TactileObjects:
    objects = []
    colors = []
    class_pairings = dict()
    color_dict = dict()

    def __init__(self, objects=None, class_parings=None):
        if objects is not None:
            if class_parings is None:
                class_parings = list(range(len(objects)))
            self.add_objects(objects, class_parings)

    def add_object(self, name, paired_numer):
        self.objects += [name]
        self.class_pairings[name] = paired_numer
        self._generate_colors(name)

    def add_objects(self, names, paired_numbers):
        for i in range(len(names)):
            self.add_object(names[i], paired_numbers[i])

    def get_objects(self, classes=None):
        if classes is not None:
            objects = []
            # bad computationally! but we only expect few classes
            obj_pairing = {v: k for k, v in self.class_pairings.items()}
            for cls in classes:
                objects.append(obj_pairing[cls])
            return objects
        return self.objects

    def get_colors(self, objects=None):
        if objects is not None:
            colors = []
            for obj in objects:
                if isinstance(obj, str):
                    colors.append(self.color_dict[self.class_pairings[obj]])
                else:
                    colors.append(self.color_dict[obj])
            return colors
        return self.colors

    def get_class_pairings(self):
        return self.class_pairings

    def get_color_dict(self):
        return self.color_dict

    def _generate_colors(self, obj=None):
        if obj is None:
            for obj in self.objects:
                class_name = self.class_pairings[obj]
                if class_name not in self.color_dict.keys():
                    new_color = generate_new_color(self.colors, pastel_factor=0.9)
                    self.colors.append(new_color)
                    self.color_dict[class_name] = new_color
        else:
            class_name = self.class_pairings[obj]
            if class_name not in self.color_dict.keys():
                new_color = generate_new_color(self.colors, pastel_factor=0.9)
                self.colors.append(new_color)
                self.color_dict[class_name] = new_color


"""class handling the tasks as set up in the experiments for --NAME OF PAPER--
    A task is defined as a cluster containing one or more object.
    
    'class_pairings' atribute a numerical class to each task
    color_dict"""
class TactileObjectsTask:
    objects = []
    colors = []
    tasks = set()
    class_pairings = dict()
    color_dict = dict()

    def __init__(self, objects=None):
        if objects is not None:
            self.add_objects(objects)
            self._generate_colors()

    def add_objects(self, objects):
        self._generate_tasks(objects)

    def get_objects(self, classes=None):
        if classes is not None:
            objects = []
            # bad computationally! but we only expect few classes
            obj_pairing =  {v: k for k, v in self.class_pairings.items()}
            for i in range(len(classes)):
                objects.append(obj_pairing[classes[i]])
            return objects
        return self.objects

    def get_tasks(self):
        return self.tasks

    def get_colors(self, objects=None):
        if objects is not None:
            colors = []
            for obj in objects:
                if isinstance(obj, str):
                    colors.append(self.color_dict[self.class_pairings[obj]])
                else:
                    colors.append(self.color_dict[obj])
            return colors
        return self.colors

    def get_class_pairings(self):
        return self.class_pairings

    def get_color_dict(self):
        return self.color_dict

    def _generate_tasks(self, objects):
        n = len(objects)
        objects_set = set(objects)
        if n <=1:
            raise ValueError('The number of objects must be greater than 1!')
        # generate all possible task pairings
        for i in range(1, math.floor(n/2+1)):
            combs = set(combinations(objects, i))
            for comb in combs:
                self.tasks.add(frozenset([frozenset(comb), frozenset({elem for elem in objects_set if elem not in comb})]))
        for task in self.tasks:
            for obj_clst in task:
                if obj_clst not in self.objects:
                    self.objects = np.append(self.objects, obj_clst)
        # encode each object with a numerical class
        for cls, obj in zip(range(len(self.objects)), self.objects):
            self.class_pairings[obj] = cls


    def _generate_colors(self, obj=None):
        if obj is None:
            for obj_clst in self.objects:
                class_name = self.class_pairings[obj_clst]
                if class_name not in self.color_dict.keys():
                    new_color = generate_new_color(self.colors, pastel_factor=0.9)
                    self.colors.append(new_color)
                    self.color_dict[class_name] = new_color
        else:
            class_name = self.class_pairings[obj]
            if class_name not in self.color_dict.keys():
                new_color = generate_new_color(self.colors, pastel_factor=0.9)
                self.colors.append(new_color)
                self.color_dict[class_name] = new_color



def get_random_color(pastel_factor=0.5):
    return [(x + pastel_factor) / (1.0 + pastel_factor) for x in
            [random.uniform(0, 1.0) for i in [1, 2, 3]]]


def color_distance(c1, c2):
    return sum([abs(x[0] - x[1]) for x in zip(c1, c2)])


def generate_new_color(existing_colors, pastel_factor=0.5):
    max_distance = None
    best_color = None
    for i in range(0, 100):
        color = get_random_color(pastel_factor=pastel_factor)
        if not existing_colors:
            return color
        best_distance = min([color_distance(color, c) for c in existing_colors])
        if not max_distance or best_distance > max_distance:
            max_distance = best_distance
            best_color = color
    return best_color
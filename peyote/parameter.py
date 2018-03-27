#!/bin/python

import prior


class Parameter:
    instances = []

    def __init__(self, name, prior=None, default=None):
        self.name = name
        self.default = default
        self.prior = prior
        self.is_fixed = False
        Parameter.instances.append(self)

    def fix(self, value=None):
        self.is_fixed = True
        if value is not None: self.default = value
        self.prior = prior.deltaFunction(self.default)
        return None
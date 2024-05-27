# -*- coding: utf-8 -*-
#
# File: controllers/mycontroller.py
# Description: This file defines the MyController class, which controls the 
# flow of the application.

from services.service import MyService
from models.model import MyModel

class MyController:
    def __init__(self, view):
        self.view = view
        self.service = MyService()

    def run(self):
        data = self.service.get_data()
        model = MyModel(data)
        self.view.display(model.data)

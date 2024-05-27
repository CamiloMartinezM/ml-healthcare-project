# -*- coding: utf-8 -*-
#
# File: main.py
# Description: This is the main entry point of the application. It creates 
# instances of the view and controller, and starts the application.

from controllers.controller import MyController
from views.view import ConsoleView

def main():
    view = ConsoleView()
    controller = MyController(view)
    controller.run()
    
if __name__ == '__main__':
    main()

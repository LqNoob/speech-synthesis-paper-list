# -*- coding: utf-8 -*-

import os

def print_colored(message, color='reset', bold=False, **kwargs):
    color_dict = {
        'bold': '\033[1m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'red': '\033[91m',
        'blue': '\033[94m',
        'grey': '\033[90m',
        'white': '\033[97m',
        'reset': '\033[0m'
    }

    color_code = color_dict.get(color.lower(), color_dict['reset'])
    prefix = color_dict['bold'] if bold else ''
    print(f"{prefix}{color_code}{message}{color_dict['reset']}", **kwargs)

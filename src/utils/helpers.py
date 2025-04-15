import os


def prettify_category(s):
    s = s.replace('-', ' ')
    s = s.title()
    s = s.replace(' & ', ' & ')
    return s
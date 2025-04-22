import os


def prettify_category(s):
    s = s.replace('-', ' ')
    s = s.title()
    s = s.replace(' & ', ' & ')
    return s


def prettify_tool(s):
    s = s.replace('_', ' ')
    return s
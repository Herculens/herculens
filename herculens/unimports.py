# This file creates fake classes to throw errors to the user when they
# try to import classes that depend on packages that are not installed.
#
# Copyright (c) 2026, herculens developers and contributors


def unimport_class(class_name, required_module_name, import_error_message):
    def __init__(self, *args, **kwargs):
        raise ImportError(f"The {class_name} class could not be imported, "
                          f"likely because `{required_module_name}` is not installed."
                          f"\nOriginal error message: {import_error_message}")
    target_class = type(
        class_name, 
        (object,), 
        {'__init__': __init__},
    )
    return target_class

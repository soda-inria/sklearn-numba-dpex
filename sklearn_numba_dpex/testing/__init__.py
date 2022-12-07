from contextlib import contextmanager


@contextmanager
def override_attr_context(obj, **attrs):
    """Within this context manager values of existing attributes of an object obj are
    overriden. The initial values are restored when exiting the context.

    Trying to override attributes that don't exist will result in an AttributeError"""
    try:
        attrs_before = dict()
        for attr_name, attr_value in attrs.items():
            # raise AttributeError if obj does not have the attribute attr_name
            attrs_before[attr_name] = getattr(obj, attr_name)
            setattr(obj, attr_name, attr_value)

        yield

    finally:
        for attr_name, attr_value in attrs_before.items():
            setattr(obj, attr_name, attr_value)

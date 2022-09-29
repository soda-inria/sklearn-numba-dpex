from contextlib import contextmanager


@contextmanager
def override_attr_context(obj, **attrs):
    attrs_before = dict()
    for attr_name, attr_value in attrs.items():
        attrs_before[attr_name] = getattr(obj, attr_name)
        setattr(obj, attr_name, attr_value)

    yield

    for attr_name, attr_value in attrs_before.items():
        setattr(obj, attr_name, attr_value)

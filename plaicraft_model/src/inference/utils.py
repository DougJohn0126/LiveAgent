def get_register_fn(_CLASSES):
    def register_fn(cls=None, *, name=None):
        """A decorator for registering classes."""

        def _register(cls):
            if name is None:
                local_name = cls.__name__
            else:
                local_name = name
            if local_name in _CLASSES:
                raise ValueError(f"Already registered class with name: {local_name}")
            _CLASSES[local_name] = cls
            return cls

        if cls is None:
            return _register
        else:
            return _register(cls)

    return register_fn


class DotDict(dict):
    """
    a dictionary that supports dot notation 
    as well as dictionary access notation 
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value

    def __getstate__(self):
        # Return the dictionary representation of the DotDict for serialization
        return dict(self)

    def __setstate__(self, state):
        # Restore the DotDict from the serialized dictionary
        for key, value in state.items():
            if isinstance(value, dict):
                value = DotDict(value)
            self[key] = value
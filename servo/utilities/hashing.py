import hashlib


def get_hash(data):
    """md5 hash of Python data. This is limited to scalars that are convertible to string and container
    structures (list, dict) containing such scalars. Some data items are not distinguishable, if they have
    the same representation as a string, e.g., hash(b'None') == hash('None') == hash(None)"""
    hasher = hashlib.md5()
    dump_container(data, hasher.update)
    return hasher.hexdigest()


def dump_container(c, func):
    """stream the contents of a container as a string through a function
    in a repeatable order, suitable, e.g., for hashing
    """
    #
    if isinstance(c, dict):  # dict
        func("{".encode("utf-8"))
        for k in sorted(c):  # for all repeatable
            func("{}:".format(k).encode("utf-8"))
            dump_container(c[k], func)
            func(",".encode("utf-8"))
        func("}".encode("utf-8"))
    elif isinstance(c, list):  # list
        func("[".encode("utf-8"))
        for k in c:  # for all repeatable
            dump_container(k, func)
            func(",".encode("utf-8"))
        func("]".encode("utf-8"))
    else:  # everything else
        if isinstance(c, type(b"")):
            pass  # already a stream, keep as is
        elif isinstance(c, str):
            # encode to stream explicitly here to avoid implicit encoding to ascii
            c = c.encode("utf-8")
        else:
            c = str(c).encode("utf-8")  # convert to string (e.g., if integer)
        func(c)  # simple value, string or convertible-to-string

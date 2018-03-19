def _get_class(name, classes, label):
    if name not in classes:
        msg = '%s name %s is invalid. Valid options are:\n' % (label, name)
        for tmp_name in classes:
            msg += '   %s\n' % tmp_name
        raise ValueError(msg)
    else:
        return classes[name]

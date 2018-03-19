def get_rate_units(units, time_units):
    if units is not None and time_units is not None:
        return '%s/%s' % (units, time_units)
    elif units is not None:
        return units
    elif time_units is not None:
        return '1.0 / %s' % time_units
    else:
        return None

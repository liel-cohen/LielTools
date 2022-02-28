import time

# former: get_time_from
def get_time_from_start(start_time, sec_or_minute='minute'):
    """
    Returns time that passed since start time
    in seconds or minutes.

    :param start_time: start time (from time.time() )
    :param sec_or_minute: 'second' or 'minute'
    :return: float - time in seconds or minutes
    """
    if sec_or_minute== 'minute':
        units = 60.0
    elif sec_or_minute== 'second':
        units = 1.0
    else: raise Exception('unknown second_or_minute value')

    return (time.time() - start_time) / units

# former: get_str_time_from
def get_str_time_from_start(start_time, sec_or_minute='minute'):
    """
    Returns time that passed since start time, as a string,
    in seconds or minutes format.

    :param start_time: start time (from time.time() )
    :param sec_or_minute: 'second' or 'minute'
    :return: string - time in seconds or minutes
    """
    time_from = get_time_from_start(start_time, sec_or_minute=sec_or_minute)

    return ("%2.4f %ss" % (time_from, sec_or_minute))


def get_timedate_str_of_time(time_float, format='%m/%d/%Y %H:%M:%S'):
    str_time = time.strftime(format, time.localtime(time_float))
    return str_time
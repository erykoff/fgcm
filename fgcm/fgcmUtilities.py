from __future__ import print_function

def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

def resourceUsage(where):
    status = None
    result = {'peak':0, 'rss':0}
    try:
        status = open('/proc/self/status')
        for line in status:
            parts = line.split()
            key = parts[0][2:-1].lower()
            if key in result:
                result[key] = int(parts[1])/1000
        print('Memory usage at %s:  %d MB current and %d MB peak.'  %(where, result['rss'], result['peak']))
    except:
        print('Could not find self status.')
    return


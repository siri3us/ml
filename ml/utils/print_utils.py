# -*- coding: utf-8 -*-

def print_columns(*args):
    all_labels = []
    all_values = []
    v_length = 0
    m_length = 0
    for label, values in args:
        m_length = max(m_length, len(label))
        all_labels.append(label)
        all_values.append(values)
        v_length = max(v_length, max([len(str(v)) for v in values]))
    for label, values in args:
        s = []
        if m_length > 0:
            s.append(label.ljust(m_length) + ':')
        for v in values:
            s.append(str(v).ljust(v_length))
        print(' '.join(s))

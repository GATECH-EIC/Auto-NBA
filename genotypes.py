from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'k3_e1' ,
    'k3_e1_g2' ,
    'k3_e3' ,
    'k3_e6' ,
    'k5_e1' ,
    'k5_e1_g2',
    'k5_e3',
    'k5_e6',
    'skip'
]

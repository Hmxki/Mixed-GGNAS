from collections import namedtuple

Genotype = namedtuple('Genotype', 'down down_concat up up_concat')

CellLinkDownPos = [
    'avg_pool',
    'max_pool',
    'down_cweight',
    'down_dil_conv',
    'down_dep_conv',
    'down_conv'
]

CellLinkUpPos = [
    'up_cweight',
    'up_dep_conv',
    'up_conv',
    'up_dil_conv'
]

CellPos = [
    'identity',
    'none',
    'cweight',
    'dil_conv',
    'dep_conv',
    'shuffle_conv',
    'conv',
]

# DARTS_V1 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
# DARTS_V2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])

darts_unet = Genotype(down=[('max_pool', 1), ('down_conv', 0), ('max_pool', 1), ('down_conv', 0), ('dep_conv', 3), ('down_dep_conv', 0), ('down_dil_conv', 0), ('dep_conv', 2)], down_concat=range(2, 6),
                      up=[('shuffle_conv', 0), ('up_dil_conv', 1), ('conv', 0), ('up_dil_conv', 1), ('up_dil_conv', 1), ('dep_conv', 3), ('dep_conv', 2), ('shuffle_conv', 0)], up_concat=range(2, 6))
darts_cell_cvc = Genotype(down=[('avg_pool', 1), ('avg_pool', 0), ('avg_pool', 1), ('avg_pool', 0), ('down_conv', 0), ('avg_pool', 1), ('down_dep_conv', 1), ('avg_pool', 0)], down_concat=range(2, 6), up=[('dep_conv', 0), ('up_dil_conv', 1), ('dil_conv', 2), ('up_dep_conv', 1), ('dep_conv', 3), ('up_dep_conv', 1), ('up_dep_conv', 1), ('dil_conv', 2)], up_concat=range(2, 6))
darts_cell_busi = Genotype(down=[('down_cweight', 0), ('down_cweight', 1), ('down_dep_conv', 0), ('dil_conv', 2), ('down_cweight', 1), ('dil_conv', 2), ('dil_conv', 3), ('dil_conv', 4)], down_concat=range(2, 6),
                           up=[('dil_conv', 0), ('up_cweight', 1), ('dil_conv', 0), ('dil_conv', 2), ('dil_conv', 3), ('dil_conv', 2), ('dil_conv', 3), ('dil_conv', 4)], up_concat=range(2, 6))
darts_cell_camus = Genotype(down=[('down_dil_conv', 0), ('down_dep_conv', 1), ('identity', 2), ('down_dep_conv', 1), ('dil_conv', 2), ('avg_pool', 0), ('conv', 2), ('dep_conv', 3)], down_concat=range(2, 6),
                            up=[('up_dil_conv', 1), ('identity', 0), ('dep_conv', 0), ('up_dep_conv', 1), ('identity', 3), ('dil_conv', 2), ('conv', 2), ('dep_conv', 3)], up_concat=range(2, 6))
darts_cell_slo = Genotype(down=[('max_pool', 0), ('down_conv', 1), ('down_dep_conv', 0), ('dep_conv', 2), ('dep_conv', 2), ('identity', 3), ('down_conv', 0), ('shuffle_conv', 2)], down_concat=range(2, 6), up=[('dil_conv', 0), ('up_dil_conv', 1), ('up_dep_conv', 1), ('dep_conv', 2), ('dep_conv', 2), ('identity', 3), ('up_conv', 1), ('shuffle_conv', 2)], up_concat=range(2, 6))

darts_cell_idrid = Genotype(down=[('down_dil_conv', 0), ('down_dil_conv', 1), ('down_dil_conv', 0), ('down_dil_conv', 1), ('cweight', 2), ('down_dil_conv', 1), ('down_dep_conv', 1), ('dil_conv', 4)], down_concat=range(2, 6), up=[('cweight', 0), ('up_cweight', 1), ('up_cweight', 1), ('identity', 0), ('cweight', 2), ('identity', 0), ('dil_conv', 0), ('dil_conv', 4)], up_concat=range(2, 6))

darts_cell_fives = Genotype(down=[('max_pool', 0), ('max_pool', 1), ('max_pool', 1), ('dep_conv', 2), ('conv', 2), ('max_pool', 1), ('identity', 3), ('dep_conv', 2)], down_concat=range(2, 6),
                            up=[('up_dep_conv', 1), ('shuffle_conv', 0), ('up_conv', 1), ('dep_conv', 2), ('dep_conv', 3), ('conv', 2), ('shuffle_conv', 0), ('dep_conv', 2)], up_concat=range(2, 6))
darts_cell_polyp =Genotype(down=[('down_cweight', 1), ('avg_pool', 0), ('down_cweight', 0), ('dil_conv', 2), ('down_dil_conv', 1), ('conv', 3), ('down_conv', 1), ('dep_conv', 2)], down_concat=range(2, 6),
                           up=[('up_dil_conv', 1), ('shuffle_conv', 0), ('dil_conv', 2), ('dil_conv', 0), ('conv', 3), ('up_cweight', 1), ('dep_conv', 2), ('dil_conv', 0)], up_concat=range(2, 6))

nasunet = Genotype(down=[('max_pool', 1), ('down_dil_conv', 0), ('cweight', 2), ('down_cweight', 0), ('down_dil_conv', 1), ('dil_conv', 3), ('dil_conv', 4), ('avg_pool', 0)], down_concat=range(2, 6),
                   up=[('up_dil_conv', 1), ('dep_conv', 0), ('up_cweight', 1), ('shuffle_conv', 0), ('up_dil_conv', 1), ('dil_conv', 0), ('up_dil_conv', 1), ('shuffle_conv', 0)], up_concat=range(2, 6))

nasunet_camus = Genotype(down=[('down_cweight', 1), ('down_dil_conv', 0), ('max_pool', 0), ('dil_conv', 2), ('dep_conv', 3), ('max_pool', 0), ('shuffle_conv', 4), ('dep_conv', 3)], down_concat=range(2, 6), up=[('up_conv', 1), ('dil_conv', 0), ('dil_conv', 2), ('up_dil_conv', 1), ('up_dil_conv', 1), ('dil_conv', 0), ('shuffle_conv', 4), ('dep_conv', 3)], up_concat=range(2, 6))

nasunet_polyp = Genotype(down=[('down_cweight', 1), ('down_dep_conv', 0), ('dil_conv', 2), ('avg_pool', 1), ('dil_conv', 2), ('dil_conv', 3), ('dil_conv', 2), ('dil_conv', 3)], down_concat=range(2, 6), up=[('up_dil_conv', 1), ('shuffle_conv', 0), ('up_dil_conv', 1), ('conv', 0), ('up_dil_conv', 1), ('dil_conv', 3), ('dil_conv', 3), ('up_dil_conv', 1)], up_concat=range(2, 6))

nasunet_idrid = Genotype(down=[('down_dep_conv', 1), ('down_cweight', 0), ('max_pool', 0), ('down_conv', 1), ('down_cweight', 0), ('dil_conv', 2), ('down_conv', 0), ('dil_conv', 2)], down_concat=range(2, 6), up=[('up_dep_conv', 1), ('identity', 0), ('up_dil_conv', 1), ('identity', 0), ('dil_conv', 2), ('dil_conv', 0), ('dil_conv', 2), ('dil_conv', 0)], up_concat=range(2, 6))
#DARTS = darts_unet
DARTS = darts_cell_polyp
nasunet_cell = nasunet



import numpy as np

from bmtk.builder.networks import NetworkBuilder


def get_coords(N, radius_min=0.0, radius_max=400.0):
    phi = 2.0 * np.pi * np.random.random([N])
    r = np.sqrt((radius_min ** 2 - radius_max ** 2) * np.random.random([N]) + radius_max ** 2)
    x = r * np.cos(phi)
    y = np.random.uniform(400.0, 500.0, size=N)
    z = r * np.sin(phi)
    return x, y, z


def exc_exc_rule(source, target, max_syns):
    """Connect rule for exc-->exc neurons, should return an integer 0 or greater"""
    if source['node_id'] == target['node_id']:
        # prevent a cell from synapsing with itself
        return 0

    # calculate the distance between tuning angles and use it to choose
    # number of connections using a binomial distribution.
    src_tuning = source['tuning_angle']
    trg_tuning = target['tuning_angle']
    tuning_dist = np.abs((src_tuning - trg_tuning + 180) % 360 - 180)
    probs = 1.0 - (np.max((tuning_dist, 10.0)) / 180.0)
    return np.random.binomial(n=max_syns, p=probs)


def others_conn_rule(source, target, max_syns, max_distance=300.0, sigma=60.0):
    if source['node_id'] == target['node_id']:
        return 0

    dist = np.sqrt((source['x'] - target['x']) ** 2 + (source['z'] - target['z']) ** 2)
    if dist > max_distance:
        return 0

    prob = np.exp(-(dist / sigma) ** 2)
    return np.random.binomial(n=max_syns, p=prob)


def build_l4(output_dir):
    l4 = NetworkBuilder('l4')
    x, y, z = get_coords(80)
    l4.add_nodes(
        N=80,
        model_type='biophysical',
        model_template='ctdb:Biophys1.hoc',
        model_processing='aibs_perisomatic',
        dynamics_params='Scnn1a_485510712_params.json',
        morphology='Scnn1a_485510712_morphology.swc',
        x=x, y=y, z=z,
        rotation_angle_xaxis=np.random.uniform(0.0, 2 * np.pi, size=80),
        rotation_angle_yaxis=np.random.uniform(0.0, 2 * np.pi, size=80),
        rotation_angle_zaxis=3.646878266,
        tuning_angle=np.linspace(start=0.0, stop=360.0, num=80, endpoint=False),
        model_name='Scnn1a',
        ei_type='e'
    )

    x, y, z = get_coords(80)
    l4.add_nodes(
        # Rorb excitatory cells
        N=80,
        model_type='biophysical',
        model_template='ctdb:Biophys1.hoc',
        dynamics_params='Rorb_486509958_params.json',
        morphology='Rorb_486509958_morphology.swc',
        model_processing='aibs_perisomatic',
        x=x, y=y, z=z,
        rotation_angle_xaxis=np.random.uniform(0.0, 2 * np.pi, size=80),
        rotation_angle_yaxis=np.random.uniform(0.0, 2 * np.pi, size=80),
        rotation_angle_zaxis=4.159763785,

        model_name='Rorb',
        ei_type='e',
        tuning_angle=np.linspace(start=0.0, stop=360.0, num=80, endpoint=False),
    )

    x, y, z = get_coords(80)
    l4.add_nodes(
        N=80,
        model_type='biophysical',
        model_template='ctdb:Biophys1.hoc',
        dynamics_params='Nr5a1_485507735_params.json',
        morphology='Nr5a1_485507735_morphology.swc',
        model_processing='aibs_perisomatic',
        x=x, y=y, z=z,
        rotation_angle_xaxis=np.random.uniform(0.0, 2 * np.pi, size=80),
        rotation_angle_yaxis=np.random.uniform(0.0, 2 * np.pi, size=80),
        rotation_angle_zaxis=4.159763785,

        model_name='Nr5a1',
        ei_type='e',
        tuning_angle=np.linspace(start=0.0, stop=360.0, num=80, endpoint=False),
    )

    x, y, z = get_coords(60)
    l4.add_nodes(
        N=60,
        model_type='biophysical',
        model_template='ctdb:Biophys1.hoc',
        dynamics_params='Pvalb_473862421_params.json',
        morphology='Pvalb_473862421_morphology.swc',
        model_processing='aibs_perisomatic',
        x=x, y=y, z=z,
        rotation_angle_xaxis=np.random.uniform(0.0, 2 * np.pi, size=60),
        rotation_angle_yaxis=np.random.uniform(0.0, 2 * np.pi, size=60),
        rotation_angle_zaxis=2.539551891,
        model_name='PValb',
        ei_type='i',
    )

    ## ADD EDGES
    conns = l4.add_edges(
        # filter for subpopulation or source and target nodes
        source=l4.nodes(ei_type='e'),
        target=l4.nodes(ei_type='e'),
        connection_rule=exc_exc_rule,
        connection_params={'max_syns': 5},
        syn_weight=3.0e-05,
        delay=2.0,
        dynamics_params='AMPA_ExcToExc.json',
        model_template='Exp2Syn',
    )
    conns.add_properties(
        ['afferent_section_id', 'afferent_section_pos', 'afferent_swc_id', 'afferent_swc_pos'],
        rule=rand_syn_locations,
        rule_params={
            'sections': ['basal', 'apical'],
            'distance_range': [30.0, 150.0],
            'morphology_dir': 'components/morphologies'
        },
        dtypes=[np.int, np.float, np.int, np.float]
    )

    ## Create e --> i connections
    conns = l4.add_edges(
        source=l4.nodes(ei_type='e'),
        target=l4.nodes(ei_type='i'),
        connection_rule=others_conn_rule,
        connection_params={'max_syns': 8},
        syn_weight=0.0006,
        delay=2.0,
        dynamics_params='AMPA_ExcToInh.json',
        model_template='Exp2Syn',
    )
    conns.add_properties(
        ['afferent_section_id', 'afferent_section_pos', 'afferent_swc_id', 'afferent_swc_pos'],
        rule=rand_syn_locations,
        rule_params={
            'sections': ['somatic', 'basal'],
            'distance_range': [0.0, 1.0e+20],
            'morphology_dir': 'components/morphologies'
        },
        dtypes=[np.int, np.float, np.int, np.float]
    )

    ## Create i --> e connections
    conns = l4.add_edges(
        source=l4.nodes(ei_type='i'),
        target=l4.nodes(ei_type='e'),
        connection_rule=others_conn_rule,
        connection_params={'max_syns': 4},
        syn_weight=0.0002,
        delay=2.0,
        dynamics_params='GABA_InhToExc.json',
        model_template='Exp2Syn',
    )
    conns.add_properties(
        ['afferent_section_id', 'afferent_section_pos', 'afferent_swc_id', 'afferent_swc_pos'],
        rule=rand_syn_locations,
        rule_params={
            'sections': ['somatic', 'basal', 'apical'],
            'distance_range': [0.0, 50.0],
            'morphology_dir': 'components/morphologies'
        },
        dtypes=[np.int, np.float, np.int, np.float]
    )

    ## Create i --> i connections
    conns = l4.add_edges(
        source=l4.nodes(ei_type='i'),
        target=l4.nodes(ei_type='e'),
        connection_rule=others_conn_rule,
        connection_params={'max_syns': 4},
        syn_weight=0.00015,
        delay=2.0,
        dynamics_params='GABA_InhToInh.json',
        model_template='Exp2Syn',
    )
    conns.add_properties(
        ['afferent_section_id', 'afferent_section_pos', 'afferent_swc_id', 'afferent_swc_pos'],
        rule=rand_syn_locations,
        rule_params={
            'sections': ['somatic', 'basal', 'apical'],
            'distance_range': [0.0, 1.0e+20],
            'morphology_dir': 'components/morphologies'
        },
        dtypes=[np.int, np.float, np.int, np.float]
    )

    l4.build()
    l4.save(output_dir=output_dir)
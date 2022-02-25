from bmtk.simulator import popnet

configure = popnet.config.from_json('config.simulation_popnet.json')
configure.build_env()

network = popnet.PopNetwork.from_config(configure)
sim = popnet.PopSimulator.from_config(configure, network)
sim.run()
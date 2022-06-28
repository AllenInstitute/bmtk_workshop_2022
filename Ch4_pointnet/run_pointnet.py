import os, sys
from bmtk.simulator import pointnet
from bmtk.analyzer.spike_trains import plot_raster


def run(config_file):
    configure = pointnet.Config.from_json(config_file)
    configure.build_env()

    graph = pointnet.PointNetwork.from_config(configure)
    sim = pointnet.PointSimulator.from_config(configure, graph)
    sim.run()


if __name__ == '__main__':
    if __file__ != sys.argv[-1]:
        config_path = sys.argv[-1]
    else:
        config_path = 'config.simulation.json'

    print(config_path)
    run(config_path)
    plot_raster(config_file=config_path, group_by='model_name', show=True)

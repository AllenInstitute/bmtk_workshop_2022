import os, sys
from datetime import datetime, timedelta

from bmtk.simulator import pointnet


def run(config_file):
    configure = pointnet.Config.from_json(config_file)
    configure.build_env()

    graph = pointnet.PointNetwork.from_config(configure)
    sim = pointnet.PointSimulator.from_config(configure, graph)
    sim.run()


if __name__ == '__main__':
    start = datetime.now()
    run('config.simulation_pointnet.json')
    end = datetime.now()
    print('build time:', timedelta(seconds=(end - start).total_seconds()))
"""
geospatial/swmm_runner.py

Assessment of Urban Flooding and Drainage Capacity using GIS and SWMM.

Thin wrapper around EPA SWMM (via the `pyswmm` bindings) to run an urban
drainage network model (.inp file), extract pipe/node capacity results,
and flag surcharged nodes for the flood-extent map
(geospatial/inundation_map.py).
"""

from dataclasses import dataclass
from pathlib import Path

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SurchargedNode:
    node_id: str
    max_depth_m: float
    invert_elevation_m: float
    surcharge_duration_hours: float
    is_flooded: bool


def run_swmm_simulation(inp_path: str, rainfall_scenario: str = "design_25yr") -> dict:
    """
    Run an EPA SWMM model and summarize network performance.

    inp_path : path to a SWMM .inp model (conduits, sub-catchments,
               junctions already built in PCSWMM/SWMM GUI from the GIS
               storm-drain network).
    rainfall_scenario : label only -- swap the RAINGAGE time series
                         block in the .inp before calling this, or
                         extend this function to programmatically patch it.

    Returns summary dict with surcharged nodes, peak flows per conduit,
    and total flood volume (runoff that could not be conveyed).
    """
    from pyswmm import Simulation, Nodes, Links

    inp_path = str(inp_path)
    surcharged: list[SurchargedNode] = []
    peak_flows: dict[str, float] = {}
    total_flood_volume_m3 = 0.0

    with Simulation(inp_path) as sim:
        nodes = Nodes(sim)
        links = Links(sim)
        node_max_depth = {n.nodeid: 0.0 for n in nodes}
        node_flood_hours = {n.nodeid: 0.0 for n in nodes}
        link_max_flow = {l.linkid: 0.0 for l in links}

        for step in sim:
            for node in nodes:
                node_max_depth[node.nodeid] = max(node_max_depth[node.nodeid], node.depth)
                if node.flooding > 0:
                    node_flood_hours[node.nodeid] += sim.step_advance() / 3600.0 if sim.step_advance() else 0
                    total_flood_volume_m3 += node.flooding * (sim.step_advance() or 1)
            for link in links:
                link_max_flow[link.linkid] = max(link_max_flow[link.linkid], link.flow)

        for node in nodes:
            if node_flood_hours[node.nodeid] > 0:
                surcharged.append(SurchargedNode(
                    node_id=node.nodeid,
                    max_depth_m=node_max_depth[node.nodeid],
                    invert_elevation_m=node.invert_elevation,
                    surcharge_duration_hours=node_flood_hours[node.nodeid],
                    is_flooded=True,
                ))
        peak_flows = link_max_flow

    logger.info("swmm run complete", extra={
        "scenario": rainfall_scenario, "n_surcharged_nodes": len(surcharged),
        "total_flood_volume_m3": total_flood_volume_m3,
    })

    return {
        "surcharged_nodes": surcharged,
        "peak_conduit_flows_cms": peak_flows,
        "total_flood_volume_m3": total_flood_volume_m3,
    }

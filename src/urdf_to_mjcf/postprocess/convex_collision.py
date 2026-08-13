"""Replace collision meshes with their convex hull."""

import argparse
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

from urdf_to_mjcf.core.utils import save_xml
from urdf_to_mjcf.postprocess.convex import (
    MeshInfo,
    MeshParts,
    export_convex_parts,
    load_collision_mesh,
    replace_collision_meshes,
)

logger = logging.getLogger(__name__)

LABEL = "convex hull replacement"


def process_single_mesh(mesh_info: MeshInfo) -> Optional[MeshParts]:
    """Save the convex hull of one mesh as a new mesh asset.

    Returns:
        The mesh name and its single hull part, or None if the mesh could not be
        processed.
    """
    loaded = load_collision_mesh(mesh_info)
    if loaded is None:
        return None
    mesh_file, mesh_data = loaded
    mesh_name, mesh_file_relative, _ = mesh_info

    logger.info(f"Processing mesh {mesh_name}: generating convex hull")

    try:
        convex_hull = mesh_data.convex_hull
    except Exception as e:
        logger.error(f"Failed to compute convex hull for mesh {mesh_name}: {e}")
        return None

    part_info = export_convex_parts(
        mesh_file,
        mesh_file_relative,
        "_convex",
        [(f"{mesh_file.stem}_convex", convex_hull)],
    )
    return (mesh_name, part_info) if part_info else None


def convex_collision_assets(mjcf_path: str | Path, root: ET.Element, max_processes: Optional[int] = None) -> None:
    """Replace every collision mesh geom with its convex hull.

    Args:
        mjcf_path: Path to the MJCF file.
        root: Root element of the MJCF file.
        max_processes: Process count. Defaults to the CPU core count minus 4.
    """
    replace_collision_meshes(mjcf_path, root, process_single_mesh, LABEL, max_processes)


def convex_collision(mjcf_path: str | Path, max_processes: Optional[int] = None) -> None:
    """Replace the collision meshes of an MJCF file with their convex hulls.

    Args:
        mjcf_path: Path to the MJCF file.
        max_processes: Process count. Defaults to the CPU core count minus 4.
    """
    tree = ET.parse(mjcf_path)
    convex_collision_assets(mjcf_path, tree.getroot(), max_processes)
    save_xml(mjcf_path, tree)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replace the collision meshes of an MJCF file with their convex hulls")
    parser.add_argument("mjcf_path", type=Path, help="Path to the MJCF file.")
    parser.add_argument("--processes", type=int, help="Process count. Defaults to the CPU core count minus 4.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    convex_collision(args.mjcf_path, args.processes)


if __name__ == "__main__":
    main()

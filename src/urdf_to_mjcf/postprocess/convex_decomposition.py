"""Replace collision meshes with their convex decomposition."""

import argparse
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import coacd
import trimesh

from urdf_to_mjcf.core.utils import save_xml
from urdf_to_mjcf.postprocess.convex import (
    MeshInfo,
    MeshParts,
    export_convex_parts,
    load_collision_mesh,
    replace_collision_meshes,
)

logger = logging.getLogger(__name__)

LABEL = "convex decomposition"


def process_single_mesh(mesh_info: MeshInfo) -> Optional[MeshParts]:
    """Decompose one mesh into convex parts.

    Returns:
        The mesh name and its parts, or None if the mesh was already convex or
        could not be processed.
    """
    loaded = load_collision_mesh(mesh_info)
    if loaded is None:
        return None
    mesh_file, mesh_data = loaded
    mesh_name, mesh_file_relative, _ = mesh_info

    logger.info(f"Processing mesh {mesh_name} for convex decomposition")

    try:
        parts = coacd.run_coacd(coacd.Mesh(mesh_data.vertices, mesh_data.faces))
    except Exception as e:
        logger.error(f"Failed to decompose mesh {mesh_name}: {e}")
        return None

    logger.info(f"Mesh {mesh_name} decomposed into {len(parts)} parts")
    if len(parts) <= 1:
        logger.info(f"Mesh {mesh_name} is already convex, skipping")
        return None

    part_info = export_convex_parts(
        mesh_file,
        mesh_file_relative,
        "_parts",
        [
            (f"{mesh_file.stem}_part{i + 1}", trimesh.Trimesh(vertices=vertices, faces=faces))
            for i, (vertices, faces) in enumerate(parts)
        ],
    )
    return (mesh_name, part_info) if part_info else None


def convex_decomposition_assets(mjcf_path: str | Path, root: ET.Element, max_processes: Optional[int] = None) -> None:
    """Replace every collision mesh geom with its convex decomposition.

    Args:
        mjcf_path: Path to the MJCF file.
        root: Root element of the MJCF file.
        max_processes: Process count. Defaults to the CPU core count minus 4.
    """
    replace_collision_meshes(mjcf_path, root, process_single_mesh, LABEL, max_processes)


def convex_decomposition(mjcf_path: str | Path, max_processes: Optional[int] = None) -> None:
    """Convex-decompose the collision meshes of an MJCF file in place.

    Args:
        mjcf_path: Path to the MJCF file.
        max_processes: Process count. Defaults to the CPU core count minus 4.
    """
    tree = ET.parse(mjcf_path)
    convex_decomposition_assets(mjcf_path, tree.getroot(), max_processes)
    save_xml(mjcf_path, tree)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replace the collision meshes of an MJCF file with their convex decomposition"
    )
    parser.add_argument("mjcf_path", type=Path, help="Path to the MJCF file.")
    parser.add_argument("--processes", type=int, help="Process count. Defaults to the CPU core count minus 4.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    convex_decomposition(args.mjcf_path, args.processes)


if __name__ == "__main__":
    main()

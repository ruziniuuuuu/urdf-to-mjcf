"""Mark meshes whose vertices are all coplanar with inertia="shell"."""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh

from urdf_to_mjcf.core.utils import save_xml

logger = logging.getLogger(__name__)


def read_mesh_vertices(file_path: Path) -> Optional[np.ndarray]:
    """Read the vertices of a mesh file.

    Args:
        file_path: Path to the mesh file.

    Returns:
        An (N, 3) vertex array, or None if the file could not be read.
    """
    if not file_path.exists():
        logger.warning(f"Mesh file does not exist: {file_path}")
        return None

    try:
        mesh_data = trimesh.load(file_path, force="mesh")

        if mesh_data is None:
            logger.warning(f"Could not load mesh file: {file_path}")
            return None

        if not isinstance(mesh_data, trimesh.Trimesh):
            logger.warning(f"Mesh file {file_path} is not a triangle mesh, skipping")
            return None

        vertices = mesh_data.vertices

        if vertices is None or len(vertices) == 0:
            logger.warning(f"Mesh file {file_path} has no vertices")
            return None

        logger.debug(f"Read {len(vertices)} vertices from {file_path}")
        return vertices

    except Exception as e:
        logger.warning(f"Failed to read mesh file {file_path}: {e}")
        return None


def check_coplanar(vertices: np.ndarray, tolerance: float = 1e-6) -> bool:
    """Return whether every vertex lies on a common plane.

    Args:
        vertices: An (N, 3) vertex array.
        tolerance: Maximum point-to-plane distance still counted as coplanar.

    Returns:
        True if all points are coplanar.
    """
    if len(vertices) < 4:
        # Three points or fewer always share a plane
        logger.debug(f"{len(vertices)} vertices < 4, treating as coplanar")
        return True

    # Define the plane from the first three non-collinear points
    p0 = vertices[0]

    # Second point: the first one that does not coincide with p0
    p1 = None
    for i in range(1, len(vertices)):
        if not np.allclose(vertices[i], p0, atol=tolerance):
            p1 = vertices[i]
            break

    if p1 is None:
        logger.debug("All vertices coincide, treating as coplanar")
        return True

    # Third point: the first one not collinear with p0-p1
    p2 = None
    v1 = p1 - p0
    for i in range(len(vertices)):
        if np.allclose(vertices[i], p0, atol=tolerance) or np.allclose(vertices[i], p1, atol=tolerance):
            continue

        v2 = vertices[i] - p0
        # Collinear means a near-zero cross product
        cross = np.cross(v1, v2)
        if np.linalg.norm(cross) > tolerance:
            p2 = vertices[i]
            break

    if p2 is None:
        logger.debug("All vertices are collinear, treating as coplanar")
        return True

    # Three non-collinear points give the plane normal
    v1 = p1 - p0
    v2 = p2 - p0
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)

    # d in the plane equation ax + by + cz + d = 0
    d = -np.dot(normal, p0)

    # Every remaining point has to sit on that plane
    for vertex in vertices:
        distance = abs(np.dot(normal, vertex) + d)
        if distance > tolerance:
            logger.debug(f"Vertex off the plane by {distance:.2e} > tolerance {tolerance:.2e}")
            return False

    logger.debug(f"All {len(vertices)} vertices lie on one plane")
    return True


def check_shell_meshes(mjcf_path: Path) -> None:
    """Set inertia="shell" on every mesh asset whose vertices are all coplanar.

    Args:
        mjcf_path: Path to the MJCF file.
    """
    logger.info(f"Checking {mjcf_path} for shell meshes...")

    try:
        tree = ET.parse(mjcf_path)
        root = tree.getroot()

        asset_elem = root.find("asset")
        if asset_elem is None:
            logger.info("No asset element found, skipping the shell check")
            return

        mjcf_dir = mjcf_path.parent

        # Mesh paths resolve against the compiler meshdir
        compiler_elem = root.find("compiler")
        mesh_dir = mjcf_dir
        if compiler_elem is not None:
            meshdir = compiler_elem.get("meshdir", ".")
            mesh_dir = mjcf_dir / meshdir

        modified = False

        for mesh_elem in asset_elem.findall("mesh"):
            mesh_name = mesh_elem.get("name", "")
            mesh_file = mesh_elem.get("file", "")

            if not mesh_file:
                logger.debug(f"Mesh {mesh_name} has no file attribute, skipping")
                continue

            # An explicit inertia already on the asset is left alone
            if mesh_elem.get("inertia") is not None:
                logger.debug(f"Mesh {mesh_name} already has an inertia attribute, skipping")
                continue

            mesh_path = mesh_dir / mesh_file

            vertices = read_mesh_vertices(mesh_path)
            if vertices is None:
                logger.debug(f"Could not read mesh file {mesh_file}, skipping")
                continue

            if check_coplanar(vertices):
                logger.info(f"Mesh {mesh_name} ({mesh_file}) is coplanar, setting inertia='shell'")
                mesh_elem.set("inertia", "shell")
                modified = True
            else:
                logger.debug(f"Mesh {mesh_name} ({mesh_file}) is not coplanar, keeping the default inertia")

        if modified:
            save_xml(mjcf_path, tree)
            logger.info(f"Updated shell mesh attributes in {mjcf_path}")
        else:
            logger.info("No mesh needed a shell inertia")

    except Exception as e:
        logger.error(f"Failed to check for shell meshes: {e}")


def main() -> None:
    """Command-line entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Mark coplanar meshes in an MJCF file as shell")
    parser.add_argument("mjcf_path", type=str, help="Path to the MJCF file.")
    parser.add_argument("--log-level", type=int, default=logging.INFO, help="The log level to use.")

    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    mjcf_path = Path(args.mjcf_path)
    check_shell_meshes(mjcf_path)


if __name__ == "__main__":
    main()

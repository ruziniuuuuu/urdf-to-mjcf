import argparse
import logging
import multiprocessing as mp
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import coacd
import trimesh

from urdf_to_mjcf.core.utils import save_xml

logger = logging.getLogger(__name__)


def process_single_mesh(mesh_info: tuple[str, str, Path]) -> Optional[tuple[str, list[tuple[str, str]]]]:
    """Decompose one mesh into convex parts.

    Args:
        mesh_info: A (mesh_name, mesh_file_path, mesh_dir) tuple.

    Returns:
        (mesh_name, [(part_name, part_file), ...]), or None if the mesh was
        already convex or could not be processed.
    """
    mesh_name, mesh_file_relative, mesh_dir = mesh_info

    mesh_file = None
    # Handle regular paths (relative to URDF file or absolute)
    if Path(mesh_file_relative).is_absolute():
        mesh_file = Path(mesh_file_relative)
    else:
        mesh_file = mesh_dir / mesh_file_relative

    if mesh_file is None or not mesh_file.exists():
        logger.warning(f"Mesh file {mesh_file_relative} does not exist at {mesh_file}.")
        return None

    logger.info(f"Processing mesh {mesh_name} for convex decomposition")

    try:
        mesh_data = trimesh.load(mesh_file, force="mesh")
    except Exception as e:
        logger.error(f"Failed to load mesh {mesh_file}: {e}")
        return None

    if not isinstance(mesh_data, trimesh.Trimesh):
        logger.warning(f"Mesh {mesh_file} is not a trimesh.Trimesh instance, skipping")
        return None

    try:
        mesh_coacd = coacd.Mesh(mesh_data.vertices, mesh_data.faces)
        parts = coacd.run_coacd(mesh_coacd)
        logger.info(f"Mesh {mesh_name} decomposed into {len(parts)} parts")

        # A single part means the mesh was already convex
        if len(parts) <= 1:
            logger.info(f"Mesh {mesh_name} is already convex (only {len(parts)} part), skipping")
            return None

    except Exception as e:
        logger.error(f"Failed to decompose mesh {mesh_name}: {e}")
        return None

    mesh_stem = mesh_file.stem
    parts_dir = mesh_file.parent / f"{mesh_stem}_parts"
    parts_dir.mkdir(exist_ok=True)

    part_info: list[tuple[str, str]] = []
    for i, part in enumerate(parts):
        try:
            convex_mesh = trimesh.Trimesh(vertices=part[0], faces=part[1])
            part_filename = f"{mesh_stem}_part{i + 1}.stl"
            part_file = parts_dir / part_filename
            convex_mesh.export(part_file)

            original_dir = str(Path(mesh_file_relative).parent)
            relative_part_path = f"{original_dir}/{mesh_stem}_parts/{part_filename}"
            part_name = f"{mesh_stem}_part{i + 1}"
            part_info.append((part_name, relative_part_path))

            logger.info(f"Saved part {i + 1} to {relative_part_path}")
        except Exception as e:
            logger.error(f"Failed to save part {i + 1} of mesh {mesh_name}: {e}")

    if part_info:
        return (mesh_name, part_info)
    else:
        return None


def convex_decomposition_assets(mjcf_path: str | Path, root: ET.Element, max_processes: Optional[int] = None) -> None:
    """Replace every collision mesh geom with its convex decomposition.

    Args:
        mjcf_path: Path to the MJCF file.
        root: Root element of the MJCF file.
        max_processes: Process count. Defaults to the CPU core count minus 4.
    """
    mjcf_path = Path(mjcf_path)
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.attrib["meshdir"] = "."

    mesh_dir = mjcf_path.parent / compiler.attrib["meshdir"]

    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    # Existing mesh assets, by name
    mesh_assets: dict[str, str] = {}
    for mesh in asset.findall("mesh"):
        mesh_assets[mesh.attrib["name"]] = mesh.attrib["file"]

    # Collision geoms referencing a mesh
    collision_geoms: list[tuple[ET.Element, ET.Element]] = []
    for body in root.findall(".//body"):
        for geom in body.findall("geom"):
            if geom.attrib.get("class") == "collision" and geom.attrib.get("type") == "mesh" and "mesh" in geom.attrib:
                collision_geoms.append((body, geom))

    unique_meshes: set[str] = set()
    for _, geom in collision_geoms:
        mesh_name = geom.attrib["mesh"]
        if mesh_name in mesh_assets:
            unique_meshes.add(mesh_name)
        else:
            logger.warning(f"Mesh {mesh_name} not found in assets.")

    if not unique_meshes:
        logger.info("No meshes to process for convex decomposition")
        return

    mesh_info_list: list[tuple[str, str, Path]] = []
    for mesh_name in unique_meshes:
        mesh_info_list.append((mesh_name, mesh_assets[mesh_name], mesh_dir))

    cpu_count = os.cpu_count() or 4
    if max_processes is None:
        max_processes = max(1, cpu_count - 4)
    else:
        max_processes = max(1, max_processes)
    actual_processes = min(max_processes, len(mesh_info_list))

    logger.info(f"Detected {cpu_count} CPU cores, using {actual_processes} processes for convex decomposition")

    new_mesh_parts: dict[str, list[tuple[str, str]]] = {}
    if actual_processes == 1:
        for mesh_info in mesh_info_list:
            result = process_single_mesh(mesh_info)
            if result:
                mesh_name, part_info = result
                new_mesh_parts[mesh_name] = part_info
    else:
        with mp.Pool(processes=actual_processes) as pool:
            results = pool.map(process_single_mesh, mesh_info_list)

            for result in results:
                if result:
                    mesh_name, part_info = result
                    new_mesh_parts[mesh_name] = part_info

    logger.info(f"Successfully processed {len(new_mesh_parts)} meshes with convex decomposition")

    for mesh_name, part_info in new_mesh_parts.items():
        for part_name, part_file in part_info:
            new_mesh = ET.SubElement(asset, "mesh")
            new_mesh.attrib["name"] = part_name
            new_mesh.attrib["file"] = part_file

    for body, geom in collision_geoms:
        geom_mesh_name = geom.attrib.get("mesh")
        if geom_mesh_name and geom_mesh_name in new_mesh_parts:
            body.remove(geom)

            for part_name, _ in new_mesh_parts[geom_mesh_name]:
                new_geom = ET.SubElement(body, "geom")
                for attr_name, attr_value in geom.attrib.items():
                    if attr_name != "mesh" and attr_name != "name":
                        new_geom.attrib[attr_name] = attr_value

                original_name = geom.attrib.get("name", f"{geom_mesh_name}_collision")
                new_geom.attrib["name"] = f"{original_name}_{part_name}"
                new_geom.attrib["mesh"] = part_name


def convex_decomposition(mjcf_path: str | Path, max_processes: Optional[int] = None) -> None:
    """Convex-decompose the collision meshes of an MJCF file in place.

    Args:
        mjcf_path: Path to the MJCF file.
        max_processes: Process count. Defaults to the CPU core count minus 4.
    """
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    convex_decomposition_assets(mjcf_path, root, max_processes)

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

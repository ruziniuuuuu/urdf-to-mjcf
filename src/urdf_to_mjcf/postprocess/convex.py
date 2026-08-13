"""Shared machinery for replacing collision meshes with convex geometry.

Both convex passes do the same thing to an MJCF file: take every collision geom
that references a mesh, turn that mesh into one or more convex meshes on disk,
and point the geom at the results. They differ only in how a single mesh becomes
convex parts, so that step is the worker function passed in here.
"""

import logging
import multiprocessing as mp
import os
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Optional

import trimesh

logger = logging.getLogger(__name__)

# A mesh asset to process: its name, its file path as written in the MJCF, and
# the directory that path resolves against.
MeshInfo = tuple[str, str, Path]

# The convex parts a mesh was replaced by: the original mesh name, then one
# (part name, part file path relative to meshdir) pair per part.
MeshParts = tuple[str, list[tuple[str, str]]]

# Turns one mesh into convex parts, or returns None if it could not be processed.
MeshWorker = Callable[[MeshInfo], Optional[MeshParts]]


def load_collision_mesh(mesh_info: MeshInfo) -> Optional[tuple[Path, trimesh.Trimesh]]:
    """Resolve and load the mesh a worker was handed.

    Returns:
        The resolved file path and the loaded mesh, or None if the file is
        missing or does not hold a single mesh.
    """
    _, mesh_file_relative, mesh_dir = mesh_info

    mesh_file = Path(mesh_file_relative)
    if not mesh_file.is_absolute():
        mesh_file = mesh_dir / mesh_file_relative

    if not mesh_file.exists():
        logger.warning(f"Mesh file {mesh_file_relative} does not exist at {mesh_file}.")
        return None

    try:
        mesh_data = trimesh.load(mesh_file, force="mesh")
    except Exception as e:
        logger.error(f"Failed to load mesh {mesh_file}: {e}")
        return None

    if not isinstance(mesh_data, trimesh.Trimesh):
        logger.warning(f"Mesh {mesh_file} is not a trimesh.Trimesh instance, skipping")
        return None

    return mesh_file, mesh_data


def export_convex_parts(
    mesh_file: Path,
    mesh_file_relative: str,
    dir_suffix: str,
    parts: Sequence[tuple[str, trimesh.Trimesh]],
) -> list[tuple[str, str]]:
    """Write convex parts as STL into their own directory beside the source mesh.

    Args:
        mesh_file: The source mesh on disk.
        mesh_file_relative: The source mesh path as written in the MJCF.
        dir_suffix: Appended to the source mesh stem to name the parts directory.
        parts: The part name and mesh of every part to write.

    Returns:
        A (part name, path relative to meshdir) pair per part that saved. The
        returned paths keep the directory layout of the source mesh so the MJCF
        can reference them unchanged.
    """
    part_dir = mesh_file.parent / f"{mesh_file.stem}{dir_suffix}"
    part_dir.mkdir(exist_ok=True)
    source_dir = Path(mesh_file_relative).parent

    part_info: list[tuple[str, str]] = []
    for part_name, part_mesh in parts:
        part_filename = f"{part_name}.stl"
        try:
            part_mesh.export(part_dir / part_filename)
        except Exception as e:
            logger.error(f"Failed to save part {part_name}: {e}")
            continue

        relative_part_path = f"{source_dir}/{part_dir.name}/{part_filename}"
        part_info.append((part_name, relative_part_path))
        logger.info(f"Saved part {part_name} to {relative_part_path}")

    return part_info


def _process_count(task_count: int, max_processes: Optional[int]) -> int:
    """Pick a worker count, leaving cores free for the rest of the machine."""
    cpu_count = os.cpu_count() or 4
    if max_processes is None:
        max_processes = cpu_count - 4
    return max(1, min(max_processes, task_count))


def replace_collision_meshes(
    mjcf_path: str | Path,
    root: ET.Element,
    worker: MeshWorker,
    label: str,
    max_processes: Optional[int] = None,
) -> None:
    """Replace every collision mesh geom with the convex parts the worker makes.

    Args:
        mjcf_path: Path to the MJCF file, used to resolve mesh paths.
        root: Root element of the MJCF file, modified in place.
        worker: Turns one mesh into convex parts. Runs in a subprocess, so it
            has to be a module-level function.
        label: What the worker produces, for log messages.
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

    mesh_assets = {mesh.attrib["name"]: mesh.attrib["file"] for mesh in asset.findall("mesh")}

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
        logger.info(f"No meshes to process for {label}")
        return

    mesh_info_list: list[MeshInfo] = [(name, mesh_assets[name], mesh_dir) for name in sorted(unique_meshes)]
    processes = _process_count(len(mesh_info_list), max_processes)
    logger.info(f"Using {processes} processes for {label}")

    if processes == 1:
        results: Sequence[Optional[MeshParts]] = [worker(mesh_info) for mesh_info in mesh_info_list]
    else:
        with mp.Pool(processes=processes) as pool:
            results = pool.map(worker, mesh_info_list)

    new_mesh_parts = {result[0]: result[1] for result in results if result}
    logger.info(f"Processed {len(new_mesh_parts)} meshes for {label}")

    # The original mesh asset stays: a visual geom may still reference it.
    for part_info in new_mesh_parts.values():
        for part_name, part_file in part_info:
            ET.SubElement(asset, "mesh", attrib={"name": part_name, "file": part_file})

    for body, geom in collision_geoms:
        geom_mesh_name = geom.attrib.get("mesh")
        if geom_mesh_name is None or geom_mesh_name not in new_mesh_parts:
            continue

        body.remove(geom)
        original_name = geom.attrib.get("name", f"{geom_mesh_name}_collision")
        inherited = {key: value for key, value in geom.attrib.items() if key not in ("mesh", "name")}

        for part_name, _ in new_mesh_parts[geom_mesh_name]:
            ET.SubElement(
                body,
                "geom",
                attrib={**inherited, "name": f"{original_name}_{part_name}", "mesh": part_name},
            )

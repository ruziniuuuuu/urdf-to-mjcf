"""Defines a post-processing function that updates the mesh of the Mujoco model.

This script updates the mesh of the MJCF file.
"""

import argparse
import logging
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TypedDict

import numpy as np
import pymeshlab
from scipy.spatial.transform import Rotation

from urdf_to_mjcf.core.materials import is_source_scoped_mtl_material
from urdf_to_mjcf.core.utils import save_xml

logger = logging.getLogger(__name__)

# Quadric edge collapse takes a target face count, and the vertex-to-face ratio
# only holds exactly for clean manifolds, so a pass can land just over budget.
# Re-measure and run another pass; two are enough in practice.
MAX_DECIMATION_PASSES = 5


class GeomMergeInfo(TypedDict):
    element: ET.Element
    mesh_name: str
    pos: np.ndarray
    quat: np.ndarray | None
    euler: np.ndarray | None
    material_attribs: list[tuple[str, str]]


class TransformedMesh(TypedDict):
    vertices: np.ndarray
    faces: np.ndarray


def simplify_mesh_assets(mjcf_path: str | Path, max_vertices: int) -> None:
    """Decimate every mesh asset that exceeds the vertex budget.

    Args:
        mjcf_path: The path to the MJCF file to process.
        max_vertices: The vertex budget a single mesh may not exceed.
    """
    mjcf_path = Path(mjcf_path)
    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.attrib["meshdir"] = "."

    dir_path = mjcf_path.parent / compiler.attrib["meshdir"]

    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    for mesh in asset.findall("mesh"):
        mesh_file = dir_path / mesh.attrib["file"]
        if not mesh_file.exists():
            logger.warning(f"<update mesh> Mesh file {mesh_file} does not exist.")
            continue

        if mesh_file.suffix.lower() not in [".obj", ".stl"]:
            logger.warning(f"Unsupported mesh format: {mesh_file.suffix}")
            continue

        if mesh.attrib["file"].startswith("./"):
            mesh.attrib["file"] = str(mesh_file.relative_to(dir_path))

        try:
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(str(mesh_file))

            vertex_count = len(ms.current_mesh().vertex_matrix())
            if vertex_count <= max_vertices:
                continue

            for _ in range(MAX_DECIMATION_PASSES):
                face_count = ms.current_mesh().face_number()
                target_faces = max(4, int(face_count * max_vertices / vertex_count))
                ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces)
                vertex_count = len(ms.current_mesh().vertex_matrix())
                if vertex_count <= max_vertices:
                    break
            else:
                logger.warning(
                    f"{mesh_file.name} still has {vertex_count} vertices after "
                    f"{MAX_DECIMATION_PASSES} decimation passes, budget is {max_vertices}"
                )

            ms.save_current_mesh(str(mesh_file))
            # Decimation drops the UVs the MTL described, and the MJCF gets its
            # materials from the asset section rather than from this sidecar.
            for mtl_file in (mesh_file.with_suffix(".mtl"), mesh_file.with_name(mesh_file.name + ".mtl")):
                mtl_file.unlink(missing_ok=True)

        except Exception as e:
            logger.error(f"Failed to simplify mesh file {mesh_file}: {e}")
            continue

    save_xml(mjcf_path, root)


def remove_unused_mesh(mjcf_path: str | Path) -> None:
    """Remove the unused mesh of the MJCF file.

    Args:
        mjcf_path: The path to the MJCF file to process.
    """
    mjcf_path = Path(mjcf_path)
    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    compiler = root.find("compiler")
    if compiler is None:
        logger.warning("No compiler element found in MJCF file")
        return

    meshdir = compiler.attrib.get("meshdir", ".")
    mesh_dir_path = mjcf_path.parent / meshdir

    # Meshes and materials referenced anywhere in the model
    used_meshes = set()
    used_materials = set()

    for geom in root.iter("geom"):
        mesh_name = geom.attrib.get("mesh")
        if mesh_name:
            used_meshes.add(mesh_name)

        material_name = geom.attrib.get("material")
        if material_name:
            used_materials.add(material_name)

    for site in root.iter("site"):
        material_name = site.attrib.get("material")
        if material_name:
            used_materials.add(material_name)

    for body in root.iter("body"):
        material_name = body.attrib.get("material")
        if material_name:
            used_materials.add(material_name)

    logger.info(f"Meshes in use: {used_meshes}")
    logger.info(f"Materials in use: {used_materials}")

    asset = root.find("asset")
    if asset is None:
        logger.warning("No asset element found in MJCF file")
        return

    meshes_to_remove = []
    mesh_files_to_remove: list[Path] = []

    for mesh in asset.findall("mesh"):
        mesh_name = mesh.attrib.get("name")
        mesh_file = mesh.attrib.get("file")

        if mesh_name not in used_meshes:
            logger.info(f"Unused mesh: {mesh_name} (file: {mesh_file})")
            meshes_to_remove.append(mesh)

            if mesh_file:
                full_mesh_path = mesh_dir_path / mesh_file
                if full_mesh_path.exists():
                    mesh_files_to_remove.append(full_mesh_path)

                # An OBJ takes its MTL sidecar with it
                if mesh_file.lower().endswith(".obj"):
                    mtl_file = full_mesh_path.with_suffix(".mtl")
                    if mtl_file.exists():
                        mesh_files_to_remove.append(mtl_file)

    materials_to_remove = []

    for material in asset.findall("material"):
        material_name = material.attrib.get("name")

        if material_name not in used_materials:
            logger.info(f"Unused material: {material_name}")
            materials_to_remove.append(material)

    for mesh in meshes_to_remove:
        asset.remove(mesh)
        logger.info(f"Removed mesh element: {mesh.attrib.get('name')}")

    for material in materials_to_remove:
        asset.remove(material)
        logger.info(f"Removed material element: {material.attrib.get('name')}")

    # Mesh files on disk that no asset points at are byproducts of earlier steps
    if mesh_dir_path.exists():
        referenced_files = set()
        for mesh in asset.findall("mesh"):
            mesh_file = mesh.attrib.get("file")
            if mesh_file:
                referenced_files.add(mesh_file)

        mesh_extensions = [".obj", ".stl", ".dae", ".mtl"]
        for mesh_file_path in mesh_dir_path.rglob("*"):
            if mesh_file_path.is_file() and mesh_file_path.suffix.lower() in mesh_extensions:
                relative_path = mesh_file_path.relative_to(mesh_dir_path)
                relative_path_str = str(relative_path).replace("\\", "/")

                if relative_path_str not in referenced_files:
                    logger.info(f"Unreferenced file: {relative_path_str}")
                    mesh_files_to_remove.append(mesh_file_path)

    # Keep files a surviving mesh asset still points at. Visual and collision
    # geoms can share one file, so dropping the collision asset must not delete
    # the file the visual asset still needs.
    retained_file_paths: set[Path] = set()
    if mesh_dir_path.exists():
        for mesh in asset.findall("mesh"):
            mesh_file = mesh.attrib.get("file")
            if mesh_file:
                try:
                    retained_file_paths.add((mesh_dir_path / mesh_file).resolve())
                except OSError:
                    continue

    deleted_files: list[Path] = []
    seen_targets: set[Path] = set()
    for file_path in mesh_files_to_remove:
        try:
            resolved = file_path.resolve()
        except OSError:
            resolved = file_path
        if resolved in retained_file_paths:
            logger.debug(f"Keeping still-referenced file: {file_path}")
            continue
        if resolved in seen_targets:
            continue
        seen_targets.add(resolved)
        try:
            if file_path.exists():
                file_path.unlink()
                deleted_files.append(file_path)
                logger.info(f"Deleted file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {e}")

    if meshes_to_remove or materials_to_remove:
        save_xml(mjcf_path, tree)
        logger.info(f"Updated MJCF file: {mjcf_path}")

    logger.info("Cleanup done:")
    logger.info(f"  mesh elements removed: {len(meshes_to_remove)}")
    logger.info(f"  material elements removed: {len(materials_to_remove)}")
    logger.info(f"  files removed: {len(deleted_files)}")

    if deleted_files:
        logger.info("Removed files:")
        for deleted_file in deleted_files:
            logger.info(f"  - {deleted_file}")


def merge_materials(mjcf_path: str | Path) -> None:
    """Merge materials with identical attributes in the MJCF file.

    Args:
        mjcf_path: The path to the MJCF file to process.
    """
    mjcf_path = Path(mjcf_path)
    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    asset = root.find("asset")
    if asset is None:
        logger.warning("No asset element found in MJCF file")
        return

    # Materials sharing an attribute signature collapse onto the first one seen
    material_signature_map = {}  # signature -> canonical material name
    material_name_to_signature = {}  # material name -> signature

    logger.info("Collecting material information...")
    for material in asset.findall("material"):
        material_name = material.attrib.get("name")
        if not material_name:
            continue

        # Source-scoped MTL materials keep their identity even when RGBA values match.
        attrib_items = sorted([(k, v) for k, v in material.attrib.items() if k != "name"])
        if is_source_scoped_mtl_material(material_name):
            signature = str([("name", material_name), *attrib_items])
        else:
            signature = str(attrib_items)

        logger.debug(f"Material '{material_name}' signature: {signature}")

        material_name_to_signature[material_name] = signature

        if signature not in material_signature_map:
            material_signature_map[signature] = material_name
            logger.debug("  -> first sighting of this signature, taking it as canonical")
        else:
            canonical_name = material_signature_map[signature]
            if material_name != canonical_name:
                logger.info(f"Duplicate material: '{material_name}' matches '{canonical_name}', merging")

    material_rename_map = {}  # old name -> canonical name
    for mat_name, signature in material_name_to_signature.items():
        canonical_name = material_signature_map[signature]
        if mat_name != canonical_name:
            material_rename_map[mat_name] = canonical_name

    logger.info(f"Scanned {len(material_name_to_signature)} materials, {len(material_rename_map)} of them duplicates")

    if material_rename_map:
        logger.info(f"Material renames ({len(material_rename_map)}):")
        for old_name, new_name in material_rename_map.items():
            logger.info(f"    {old_name} -> {new_name}")

        for geom in root.iter("geom"):
            material_name = geom.attrib.get("material")
            if material_name and material_name in material_rename_map:
                new_name = material_rename_map[material_name]
                geom.attrib["material"] = new_name
                logger.info(f"Retargeted geom material: {material_name} -> {new_name}")

        for site in root.iter("site"):
            material_name = site.attrib.get("material")
            if material_name and material_name in material_rename_map:
                new_name = material_rename_map[material_name]
                site.attrib["material"] = new_name
                logger.info(f"Retargeted site material: {material_name} -> {new_name}")

        for body in root.iter("body"):
            material_name = body.attrib.get("material")
            if material_name and material_name in material_rename_map:
                new_name = material_rename_map[material_name]
                body.attrib["material"] = new_name
                logger.info(f"Retargeted body material: {material_name} -> {new_name}")

        for material in list(asset.findall("material")):
            material_name = material.attrib.get("name")
            if material_name and material_name in material_rename_map:
                asset.remove(material)
                logger.info(f"Removed duplicate material element: {material_name}")

        save_xml(mjcf_path, tree)
        logger.info(f"Merged materials, updated {mjcf_path}")


def remove_empty_or_invalid_meshes(mjcf_path: str | Path) -> None:
    """Drop mesh assets that hold no vertices, along with the geoms using them.

    A mesh file that fails to load counts as unusable too: MuJoCo would refuse
    to compile the model, so the asset and its geoms go the same way.
    """
    mjcf_path = Path(mjcf_path)
    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    compiler = root.find("compiler")
    if compiler is None:
        logger.warning("No compiler element found in MJCF file")
        return

    meshdir = compiler.attrib.get("meshdir", ".")
    mesh_dir_path = mjcf_path.parent / meshdir

    asset = root.find("asset")
    if asset is None:
        logger.warning("No asset element found in MJCF file")
        return

    # Empty meshes, and the names geoms reference them by
    empty_mesh_elements: list[ET.Element] = []
    empty_mesh_names: set[str] = set()

    for mesh in list(asset.findall("mesh")):
        mesh_name = mesh.attrib.get("name")
        mesh_file_attr = mesh.attrib.get("file")
        if not mesh_file_attr:
            continue

        mesh_file_path = mesh_dir_path / mesh_file_attr
        if not mesh_file_path.exists():
            continue

        if mesh_file_path.suffix.lower() not in [".obj", ".stl"]:
            continue

        try:
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(str(mesh_file_path))
            vertices = ms.current_mesh().vertex_matrix()
            if len(vertices) == 0:
                logger.warning(
                    f"Empty mesh: name={mesh_name}, file={mesh_file_attr}. Dropping the asset and its geoms."
                )
                empty_mesh_elements.append(mesh)
                if mesh_name:
                    empty_mesh_names.add(mesh_name)
        except Exception as e:
            logger.error(f"Failed to read mesh file {mesh_file_path}: {e}")
            logger.warning(f"Unusable mesh: name={mesh_name}, file={mesh_file_attr}. Dropping the asset and its geoms.")
            empty_mesh_elements.append(mesh)
            if mesh_name:
                empty_mesh_names.add(mesh_name)

            continue

    if not empty_mesh_elements and not empty_mesh_names:
        return

    # Removing a geom needs its parent, so walk parents and their direct
    # children rather than iterating over geoms alone.
    if empty_mesh_names:
        for parent in root.iter():
            for child in list(parent):
                if child.tag == "geom":
                    ref_name = child.attrib.get("mesh")
                    if ref_name and ref_name in empty_mesh_names:
                        parent.remove(child)
                        logger.info(
                            f"Removed geom name={child.attrib.get('name')} from {parent.tag}: "
                            f"it referenced empty mesh '{ref_name}'"
                        )

    for mesh in empty_mesh_elements:
        try:
            mesh_name = mesh.attrib.get("name")
            asset.remove(mesh)
            logger.info(f"Removed empty mesh asset: {mesh_name}")
        except Exception as e:
            logger.error(f"Failed to remove mesh from asset: {e}")

    save_xml(mjcf_path, tree)


def remove_empty_mesh_dirs(mjcf_path: str | Path) -> None:
    """Recursively remove empty directories under the meshdir declared in MJCF.

    This will walk the meshdir bottom-up and remove any empty subdirectories.
    The meshdir root itself will not be removed.
    """
    mjcf_path = Path(mjcf_path)
    try:
        tree = ET.parse(mjcf_path)
        root = tree.getroot()
    except Exception:
        logger.debug("Could not parse the MJCF file for its meshdir; skipping empty directory cleanup")
        return

    compiler = root.find("compiler")
    if compiler is None:
        logger.debug("No compiler element found in MJCF file; skipping empty directory cleanup")
        return

    meshdir = compiler.attrib.get("meshdir", ".")
    mesh_dir_path = mjcf_path.parent / meshdir
    if not mesh_dir_path.exists():
        logger.debug(f"Mesh dir {mesh_dir_path} does not exist; skipping cleanup")
        return

    removed = []
    # Walk bottom-up so subdirectories are processed before their parents
    for dirpath, dirnames, filenames in os.walk(mesh_dir_path, topdown=False):
        p = Path(dirpath)
        # Don't remove the meshdir root itself
        if p == mesh_dir_path:
            continue
        try:
            # If directory is empty (no files and no subdirectories), remove it
            if not any(p.iterdir()):
                p.rmdir()
                removed.append(str(p))
        except Exception as e:
            logger.debug(f"Failed to remove directory {p}: {e}")

    if removed:
        logger.info(f"Removed {len(removed)} empty directories under {mesh_dir_path}")
        for d in removed:
            logger.info(f"  - {d}")


def _geom_rotation(geom_info: GeomMergeInfo) -> np.ndarray:
    """Build the 3x3 rotation matrix of a geom from its euler angles or quaternion.

    MuJoCo's default eulerseq is "xyz" in lowercase, meaning intrinsic rotations
    about moving axes, which is scipy's "XYZ". Angles are radians here because
    make_degrees runs after this postprocess step.
    """
    euler = geom_info["euler"]
    if euler is not None:
        return Rotation.from_euler("XYZ", euler).as_matrix()

    quat = geom_info["quat"]
    if quat is not None:
        w, x, y, z = quat
        return Rotation.from_quat([x, y, z, w]).as_matrix()

    return np.eye(3)


def merge_geoms_by_material(mjcf_path: str | Path) -> None:
    """Merge geoms sharing a material within a body, and their mesh files.

    Args:
        mjcf_path: The path to the MJCF file to process.
    """
    mjcf_path = Path(mjcf_path)
    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    compiler = root.find("compiler")
    if compiler is None:
        logger.warning("No compiler element found in MJCF file")
        return

    meshdir = compiler.attrib.get("meshdir", ".")
    mesh_dir_path = mjcf_path.parent / meshdir

    asset = root.find("asset")
    if asset is None:
        logger.warning("No asset element found in MJCF file")
        return

    # Mesh assets by name, resolved to a path on disk
    mesh_name_to_file: dict[str, Path] = {}
    for mesh in asset.findall("mesh"):
        mesh_name = mesh.attrib.get("name")
        mesh_file = mesh.attrib.get("file")
        if mesh_name and mesh_file:
            if mesh_file.startswith(meshdir):
                mesh_name_to_file[mesh_name] = mjcf_path.parent / mesh_file
            else:
                mesh_name_to_file[mesh_name] = mesh_dir_path / mesh_file

    merged_count = 0
    for body in root.iter("body"):
        body_name = body.attrib.get("name", "unnamed")

        # Visual mesh geoms in this body, grouped by material
        material_to_geoms: dict[str, list[GeomMergeInfo]] = {}

        for geom in list(body.findall("geom")):
            if geom.attrib.get("type") != "mesh":
                continue

            mesh_name = geom.attrib.get("mesh")
            mesh_class = geom.attrib.get("class", "")
            if not mesh_name or mesh_class == "collision":
                continue

            material_attribs: list[tuple[str, str]] = []
            for key in ["material", "rgba", "class"]:
                if key in geom.attrib:
                    material_attribs.append((key, geom.attrib[key]))

            material_signature = str(sorted(material_attribs))

            pos = geom.attrib.get("pos", "0 0 0")
            quat = geom.attrib.get("quat", "1 0 0 0")
            euler = geom.attrib.get("euler", None)

            if material_signature not in material_to_geoms:
                material_to_geoms[material_signature] = []

            material_to_geoms[material_signature].append(
                {
                    "element": geom,
                    "mesh_name": mesh_name,
                    "pos": np.array([float(x) for x in pos.split()]),
                    "quat": np.array([float(x) for x in quat.split()]) if euler is None else None,
                    "euler": np.array([float(x) for x in euler.split()]) if euler is not None else None,
                    "material_attribs": material_attribs,
                }
            )

        for material_sig, geom_list in material_to_geoms.items():
            if len(geom_list) <= 1:
                continue

            logger.info(f"Body '{body_name}': merging {len(geom_list)} geoms that share a material")

            try:
                # Bake each geom pose into its mesh vertices before combining
                all_transformed_meshes: list[TransformedMesh] = []

                for i, geom_info in enumerate(geom_list):
                    mesh_name = geom_info["mesh_name"]
                    mesh_path = mesh_name_to_file.get(mesh_name)

                    if mesh_path is None or not mesh_path.exists():
                        logger.warning(f"  Mesh file does not exist: {mesh_name}")
                        continue

                    temp_ms = pymeshlab.MeshSet()
                    temp_ms.load_new_mesh(str(mesh_path))

                    transform_matrix = np.eye(4)
                    transform_matrix[:3, :3] = _geom_rotation(geom_info)
                    transform_matrix[:3, 3] = geom_info["pos"]

                    vertices = temp_ms.current_mesh().vertex_matrix()
                    faces = temp_ms.current_mesh().face_matrix()

                    vertices_homo = np.hstack([vertices, np.ones((vertices.shape[0], 1))])
                    transformed_vertices = (transform_matrix @ vertices_homo.T).T[:, :3]

                    all_transformed_meshes.append({"vertices": transformed_vertices, "faces": faces})

                    logger.debug(f"  Mesh {i}: {len(transformed_vertices)} vertices, {len(faces)} faces")

                if len(all_transformed_meshes) == 0:
                    logger.warning(f"  Body '{body_name}': no mesh could be loaded, skipping the merge")
                    continue

                vertex_offset = 0
                all_vertices = []
                all_faces = []

                for mesh_data in all_transformed_meshes:
                    all_vertices.append(mesh_data["vertices"])
                    # Face indices shift by the vertices already added
                    adjusted_faces = mesh_data["faces"] + vertex_offset
                    all_faces.append(adjusted_faces)
                    vertex_offset += len(mesh_data["vertices"])

                combined_vertices = np.vstack(all_vertices)
                combined_faces = np.vstack(all_faces)

                logger.info(f"  Merged: {len(combined_vertices)} vertices, {len(combined_faces)} faces")

                merged_ms = pymeshlab.MeshSet()
                merged_mesh = pymeshlab.Mesh(combined_vertices, combined_faces)
                merged_ms.add_mesh(merged_mesh)

                # The merged mesh lands next to the first source mesh
                first_geom = geom_list[0]
                original_mesh_name = first_geom["mesh_name"]
                original_mesh_file = mesh_name_to_file.get(original_mesh_name)

                if original_mesh_file and original_mesh_file.exists():
                    target_dir = original_mesh_file.parent
                else:
                    target_dir = mesh_dir_path

                merged_mesh_name = f"{body_name}_merged_{original_mesh_name.split('_')[0]}"
                merged_mesh_file = target_dir / f"{merged_mesh_name}.obj"

                counter = 1
                while merged_mesh_file.exists():
                    merged_mesh_name = f"{body_name}_merged_{original_mesh_name.split('_')[0]}_{counter}"
                    merged_mesh_file = target_dir / f"{merged_mesh_name}.obj"
                    counter += 1

                merged_ms.save_current_mesh(str(merged_mesh_file))
                logger.info(f"  Saved merged mesh: {merged_mesh_file}")

                mesh_file_relative = str(merged_mesh_file.relative_to(mjcf_path.parent))

                new_mesh_element = ET.Element("mesh", name=merged_mesh_name, file=mesh_file_relative)
                asset.append(new_mesh_element)

                # The first geom takes over the merged mesh
                first_geom_element = geom_list[0]["element"]
                first_geom_element.attrib["mesh"] = merged_mesh_name

                material_attribs = geom_list[0]["material_attribs"]
                if material_attribs:
                    logger.info("  Kept material attributes:")
                    for key, value in material_attribs:
                        first_geom_element.attrib[key] = value
                        logger.info(f"    {key} = {value}")

                # The pose now lives in the vertices, so reset it
                first_geom_element.attrib["pos"] = "0 0 0"
                if "quat" in first_geom_element.attrib:
                    first_geom_element.attrib["quat"] = "1 0 0 0"
                if "euler" in first_geom_element.attrib:
                    del first_geom_element.attrib["euler"]

                for i in range(1, len(geom_list)):
                    body.remove(geom_list[i]["element"])

                merged_count += 1
                logger.info(f"  Merged {len(geom_list)} geoms into one")

            except Exception as e:
                logger.exception(f"  Failed to merge the geoms of body '{body_name}': {e}")
                continue

    if merged_count > 0:
        save_xml(mjcf_path, tree)
        logger.info(f"Merged {merged_count} geom groups")
    else:
        logger.info("No geoms needed merging")


def update_mesh(mjcf_path: str | Path, max_vertices: int = 1000000) -> None:
    """Update the mesh of the MJCF file.

    Args:
        mjcf_path: The path to the MJCF file to process.
    """
    remove_empty_or_invalid_meshes(mjcf_path)
    simplify_mesh_assets(mjcf_path, max_vertices)
    merge_materials(mjcf_path)
    merge_geoms_by_material(mjcf_path)
    remove_unused_mesh(mjcf_path)
    # Earlier steps leave directories behind once their meshes are gone
    try:
        remove_empty_mesh_dirs(mjcf_path)
    except Exception as e:
        logger.warning(f"Failed to clean up empty directories under meshdir: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Updates the mesh of the MJCF file.")
    parser.add_argument("mjcf_path", type=Path, help="Path to the MJCF file.")
    parser.add_argument("--max-vertices", type=int, default=200000, help="Maximum number of vertices in the mesh.")
    args = parser.parse_args()
    update_mesh(args.mjcf_path, args.max_vertices)


if __name__ == "__main__":
    main()

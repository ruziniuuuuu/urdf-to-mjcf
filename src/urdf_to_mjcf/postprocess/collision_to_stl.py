"""Convert collision meshes of a MuJoCo model to STL.

Every geom with class="collision" gets its mesh converted to STL, and the
matching asset entry and geom reference are retargeted to the new file.
"""

import argparse
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

from urdf_to_mjcf.core.utils import save_xml

logger = logging.getLogger(__name__)


def collision_to_stl(mjcf_path: str | Path) -> None:
    """Convert the collision meshes of an MJCF file to STL and update its assets.

    Args:
        mjcf_path: Path to the MJCF file to process.
    """
    mjcf_path = Path(mjcf_path)
    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    compiler = root.find("compiler")
    if compiler is None:
        raise ValueError("MJCF file does not contain a compiler element")

    mesh_dir_path = mjcf_path.parent / compiler.attrib["meshdir"]
    asset = root.find("asset")
    if asset is None:
        raise ValueError("MJCF file does not contain an asset element")

    asset_to_add: list[tuple[str, str]] = []
    for geom in root.iter("geom"):
        if geom.attrib.get("type") == "mesh":
            mesh_name = geom.attrib.get("mesh")
            class_name = geom.attrib.get("class")
            if class_name == "collision" and mesh_name is not None:
                for mesh in asset.findall("mesh"):
                    if mesh.attrib.get("name") == mesh_name:
                        mesh_file = mesh_dir_path / mesh.attrib["file"]
                        if not mesh_file.exists():
                            logger.error(f"Mesh file {mesh_file} does not exist.")
                            raise FileNotFoundError(f"Mesh file {mesh_file} does not exist.")

                        if Path(mesh_file).suffix.lower() != ".stl":
                            logger.info(f"Converting collision mesh {mesh_name} to STL.")
                            # An STL that is already there is reused as is
                            stl_file = mesh_file.with_suffix(".stl")
                            if not stl_file.exists():
                                try:
                                    import pymeshlab

                                    ms = pymeshlab.MeshSet()
                                    ms.load_new_mesh(str(mesh_file))
                                    ms.save_current_mesh(str(stl_file))
                                    logger.info(f"Converted {mesh_file} to {stl_file}")
                                except Exception as e:
                                    logger.error(f"Failed to convert {mesh_file} to STL: {e}")
                            geom.attrib["mesh"] = Path(mesh_name).with_suffix(".stl").name
                            logger.info(f"Geom {geom.attrib.get('name')} now uses mesh {stl_file.name}")
                            # Collected here and appended to asset after the walk
                            stl_rel_path = str(stl_file.relative_to(mesh_dir_path))
                            asset_to_add.append((geom.attrib["mesh"], stl_rel_path))

                            logger.info(f"Queued asset update: {mesh.attrib.get('name')} -> {mesh.attrib.get('file')}")
                        break

    for mesh_name, mesh_file_attr in asset_to_add:
        asset.append(ET.Element("mesh", name=mesh_name, file=mesh_file_attr))

    save_xml(mjcf_path, tree)


def main() -> None:
    parser = argparse.ArgumentParser(description="Updates the mesh of the MJCF file.")
    parser.add_argument("mjcf_path", type=Path, help="Path to the MJCF file.")
    args = parser.parse_args()
    collision_to_stl(args.mjcf_path)


if __name__ == "__main__":
    main()

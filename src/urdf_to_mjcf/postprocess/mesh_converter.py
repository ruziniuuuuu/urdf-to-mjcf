import logging
from pathlib import Path
from typing import Any

import collada
import trimesh

logger = logging.getLogger(__name__)


def dae2obj(dae_path: str | Path, obj_path: str | Path):
    """Convert DAE file to OBJ format with proper material handling.

    Args:
        dae_path: Path to input DAE file
        obj_path: Path to output OBJ file
    """
    # Convert inputs to Path objects for consistent handling
    dae_path = Path(dae_path)
    obj_path = Path(obj_path)

    try:
        logger.info(f"Converting DAE to OBJ: {dae_path} -> {obj_path}")

        # collada.Collada and trimesh.load expect string paths
        dae = collada.Collada(str(dae_path))
        id2dae_goem = {}
        for geom in dae.geometries:
            id2dae_goem[geom.id] = geom

        mesh_data = trimesh.load(str(dae_path))
        mtl_path = obj_path.with_suffix(".mtl")
        mtl_name = mtl_path.name  # Just the filename for export

        logger.info(f"Loaded DAE file with {len(id2dae_goem)} geometries")

    except Exception as e:
        logger.error(f"Failed to load DAE file {dae_path}: {e}")
        raise

    if isinstance(mesh_data, trimesh.Scene):
        # Collect the unique materials
        unique_materials = {}
        geom_to_material = {}

        for id, geom in mesh_data.geometry.items():
            # Take the material object and its name
            material_obj = id2dae_goem[id].primitives[0].material
            material_name = material_obj.id if hasattr(material_obj, "id") else str(material_obj)

            # Record the geometry-to-material mapping
            geom_to_material[id] = material_name

            if material_name not in unique_materials:
                unique_materials[material_name] = material_obj

            # Name the material after the DAE material
            geom.visual.material.name = material_name

        # Export first, then read back the material names trimesh assigned
        export_kwargs: dict[str, Any] = {"mtl_name": mtl_name}
        mesh_data.export(str(obj_path), **export_kwargs)

        # Recover trimesh's assignment from the exported files
        material_mapping = {}
        if mtl_path.exists():
            with open(str(mtl_path), "r") as f:
                mtl_content = f.read()

            # Find which material each geometry uses
            if obj_path.exists():
                with open(str(obj_path), "r") as f:
                    obj_content = f.read()

                # Parse geometry names and their materials out of the OBJ
                lines = obj_content.split("\n")
                current_geom = None
                for line in lines:
                    if line.startswith("o "):
                        current_geom = line[2:].strip()
                    elif line.startswith("usemtl ") and current_geom:
                        trimesh_material = line[7:].strip()
                        if current_geom in geom_to_material and trimesh_material not in material_mapping:
                            actual_material = geom_to_material[current_geom]
                            material_mapping[trimesh_material] = actual_material

            # Rename the materials in the MTL
            for old_name, new_name in material_mapping.items():
                mtl_content = mtl_content.replace(f"newmtl {old_name}", f"newmtl {new_name}")

            with open(str(mtl_path), "w") as f:
                f.write(mtl_content)

        # Retarget the material references in the OBJ
        if obj_path.exists():
            with open(str(obj_path), "r") as f:
                obj_content = f.read()

            for old_name, new_name in material_mapping.items():
                obj_content = obj_content.replace(f"usemtl {old_name}", f"usemtl {new_name}")

            with open(str(obj_path), "w") as f:
                f.write(obj_content)
    else:
        mesh_export_kwargs: dict[str, Any] = {"mtl_name": mtl_name}
        mesh_data.export(str(obj_path), **mesh_export_kwargs)

    logger.info(f"Successfully converted DAE to OBJ: {obj_path}")


def glb2obj(glb_path: str | Path, obj_path: str | Path) -> None:
    """Convert GLB/GLTF file to OBJ format using trimesh.

    Args:
        glb_path: Path to input GLB/GLTF file
        obj_path: Path to output OBJ file
    """
    glb_path = Path(glb_path)
    obj_path = Path(obj_path)

    logger.info(f"Converting GLB to OBJ: {glb_path} -> {obj_path}")

    try:
        mesh_data = trimesh.load(str(glb_path))
    except Exception as e:
        logger.error(f"Failed to load GLB file {glb_path}: {e}")
        raise

    export_kwargs: dict[str, Any] = {"mtl_name": obj_path.with_suffix(".mtl").name}
    mesh_data.export(str(obj_path), **export_kwargs)
    logger.info(f"Successfully converted GLB to OBJ: {obj_path}")

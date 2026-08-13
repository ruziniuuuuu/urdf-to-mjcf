# align_stp — asset preparation scripts

Standalone scripts for getting CAD exports into a shape the converter can use:
splitting STEP files, merging a URDF's meshes, aligning a high-resolution
appearance model onto the white model, and scaling, transforming or decimating
meshes.

## Support boundary

These scripts are **not part of the `urdf_to_mjcf` package**. Read that as:

- They are not installed by `pip install urdf-to-mjcf`, and they are not
  importable from it. Run them from a clone of this repository.
- They are outside the `src/urdf_to_mjcf tests` targets, so `make lint`,
  `make typecheck` and `make test` never look at them.
- Two of them need packages the project does not declare: `align_meshes.py`
  stops at import without `open3d`, and `split_stp.py` starts but refuses to
  read a STEP file without `pythonocc-core`. The rest run in the project
  environment as they are.
- They take a mesh and write a mesh. The conversion pipeline neither calls them
  nor knows they ran.

Install the extras you need for the scripts you use:

```shell
uv pip install open3d       # align_meshes.py, decimate_mesh.py
uv pip install pyfqmr       # decimate_mesh.py (optional backend)
uv pip install pythonocc-core  # split_stp.py — needs conda on most platforms
```

## The scripts

| Script | What it does | Extra packages |
| --- | --- | --- |
| `split_stp.py` | Converts a STEP file, or splits it into one file per part and reports the overall AABB. | `pythonocc-core` |
| `merge_urdf.py` | Merges every link's visual and collision meshes into `visual.obj`, `collision.obj` and `mapping.json`, in the URDF's own frames. | — |
| `align_meshes.py` | Registers a high-resolution OBJ onto a low-resolution STL/OBJ reference (FPFH + RANSAC, then multi-scale ICP and GICP), compensating mm↔m scale differences. | `open3d` |
| `assign_mesh_part.py` | Uses `mapping.json` to decide which part a standalone mesh belongs to, and rewrites it into that part's local frame. | — |
| `scale_mesh.py` | Scales a mesh uniformly by a factor, to a target bounding-box size, or to a target volume. | — |
| `transform_mesh.py` | Rotates and translates an OBJ at text level, keeping UVs, normals, materials and vertex colours. | — |
| `decimate_mesh.py` | Decimates an OBJ per material group against a face budget. | — (`pyfqmr`, `open3d` are fallback backends) |

## Workflow

1. Export a URDF plus STL meshes (the white model) from SolidWorks, and make the
   mesh paths in the URDF relative.

2. Merge the URDF's bodies into one model:

   ```shell
   uv run python align_stp/merge_urdf.py /path/to/robot.urdf
   ```

   `-o` picks the output directory; it defaults to
   `/path/to/robot.urdf/../merged_model`.

3. Export the full robot with its appearance as one OBJ plus MTL, or one DAE
   (`robot_id.obj`).

4. Optional: if that high-resolution model does not share the frame or scale of
   the merged white model, align it first.

   ```shell
   # One pair
   uv run python align_stp/align_meshes.py \
       --ref /path/to/white_model.STL \
       --src /path/to/high_res.obj \
       --out /path/to/aligned.obj

   # A whole directory, matched by filename
   uv run python align_stp/align_meshes.py \
       --ref-dir /path/to/stl_meshes \
       --src-dir /path/to/obj_meshes \
       --out-dir /path/to/aligned_meshes \
       --ref-ext .STL --src-ext .obj
   ```

   Scale differences of 10x or 100x (mm↔m and so on) are detected and
   compensated automatically. `--no-scale` turns that off, `--no-global` skips
   global registration when the two models already roughly line up, and
   `--voxel-size` / `--max-icp-dist` override the parameters that default to 0,
   meaning automatic.

5. Assign the appearance model to the parts of the merged model. Both
   `visual.obj` and `mapping.json` come from step 2.

   ```shell
   uv run python align_stp/assign_mesh_part.py /path/to/robot_id.obj \
       -g /path/to/merged_model/visual.obj \
       -m /path/to/merged_model/mapping.json
   ```

   `-o` picks the output directory; it defaults to
   `/path/to/robot_id.obj/../meshes_aligned`.

6. Point the visual meshes in `/path/to/robot.urdf` at the results in
   `meshes_aligned`.

7. Convert:

   ```shell
   urdf-to-mjcf /path/to/robot.urdf -o /path/to/mjcf/robot.xml
   ```

   Collision meshes are used as they are unless you ask for something else with
   `--collision-type decomposition` or `--collision-type convex_hull`.

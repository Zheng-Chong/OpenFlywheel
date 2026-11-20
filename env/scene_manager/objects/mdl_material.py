try:
    from isaacsim.replicator.behavior.utils.scene_utils import create_mdl_material
except ImportError:  # removed in Isaac Sim 6
    import omni.kit.commands
    import omni.usd
    from pxr import Sdf, UsdShade

    def create_mdl_material(mtl_url: str, mtl_name: str, mtl_path: str) -> UsdShade.Material:
        omni.kit.commands.execute("CreateMdlMaterialPrim", mtl_url=mtl_url, mtl_name=mtl_name, mtl_path=mtl_path)
        mtl_prim = omni.usd.get_context().get_stage().GetPrimAtPath(mtl_path)
        shader = UsdShade.Shader(omni.usd.get_shader_from_material(mtl_prim, get_prim=True))
        shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset)
        shader.CreateInput("project_uvw", Sdf.ValueTypeNames.Bool)
        shader.CreateInput("texture_scale", Sdf.ValueTypeNames.Float2)
        shader.CreateInput("texture_rotate", Sdf.ValueTypeNames.Float)
        return UsdShade.Material(mtl_prim)

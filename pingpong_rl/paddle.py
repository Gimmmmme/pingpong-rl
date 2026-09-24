"""Realistic fixed table-tennis paddles for all four bimanual gripper links.

Blade: elliptical 160 x 170 x 10 mm prism, face normal local X.  The 120 mm
wooden handle extends in the blade plane along local -Z, as on a real racket.
A narrow metal L bracket connects the handle to the H-gripper mounting point.
Blade collision uses the same explicit ellipse mesh with convex-hull cooking;
no nonuniform primitive scaling and no independent kinematic rigid bodies.
"""
from __future__ import annotations
from typing import Any
import math

PADDLE_OFFSET_V2=(.235,0.,0.)
BLADE_RADIUS=.080
BLADE_HALF_HEIGHT=.085
BLADE_THICKNESS=.010
HANDLE_LENGTH=.120
HANDLE_WIDTH=.030
HANDLE_THICKNESS=.018
HANDLE_CENTER_X=PADDLE_OFFSET_V2[0]
HANDLE_CENTER_Z=-.140
PADDLE_PRIMITIVE_NAMES=('BladeCollision','BladeWoodEdge','RubberRedFront','RubberBlackBack',
                      'HandleCollision','HandleWoodVisual','HandleMountCollision','HandleMountVisual',
                      'HandleClampCollision','HandleClampVisual')


def _ellipse_mesh(stage,path,center,thickness,width=.160,height=.170,segments=64):
    from pxr import Gf,UsdGeom
    mesh=UsdGeom.Mesh.Define(stage,path)
    pts=[]
    for x in (-thickness/2,thickness/2):
        for k in range(segments):
            a=2*math.pi*k/segments
            pts.append(Gf.Vec3f(center[0]+x,center[1]+width/2*math.cos(a),center[2]+height/2*math.sin(a)))
    counts=[segments,segments]+[4]*segments
    # Clockwise from outside on -X, counter-clockwise on +X.
    indices=list(reversed(range(segments)))+list(range(segments,2*segments))
    for k in range(segments):
        j=(k+1)%segments
        indices.extend([k,j,segments+j,segments+k])
    mesh.CreatePointsAttr(pts);mesh.CreateFaceVertexCountsAttr(counts);mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr('none');mesh.CreateDoubleSidedAttr(False)
    mesh.CreateExtentAttr([Gf.Vec3f(center[0]-thickness/2,center[1]-width/2,center[2]-height/2),
                           Gf.Vec3f(center[0]+thickness/2,center[1]+width/2,center[2]+height/2)])
    return mesh.GetPrim()


def _visual_material(stage,path,color,roughness=.55):
    from pxr import UsdShade,Sdf
    mat=UsdShade.Material.Define(stage,path)
    shader=UsdShade.Shader.Define(stage,path+'/Shader');shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(color)
    shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(0.)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
    return mat


def _bind_visual(prim,material):
    from pxr import UsdShade
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def _make_physics_material(stage,path,material):
    from pxr import UsdShade,UsdPhysics,PhysxSchema
    mat=UsdShade.Material.Define(stage,path)
    api=UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(float(getattr(material,'static_friction',.01)))
    api.CreateDynamicFrictionAttr(float(getattr(material,'dynamic_friction',.01)))
    api.CreateRestitutionAttr(float(getattr(material,'restitution',.88)))
    p=PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
    p.CreateRestitutionCombineModeAttr('max');p.CreateFrictionCombineModeAttr('min')
    return mat


def _collision(prim,physics_material,collision):
    from pxr import UsdPhysics,PhysxSchema,UsdShade
    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexHull')
    api=PhysxSchema.PhysxCollisionAPI.Apply(prim)
    # Sit the contact shell outside the visible blade so a stiff arm drive
    # stops on the table instead of drawing the mesh through it.
    api.CreateContactOffsetAttr(max(float(getattr(collision,'contact_offset',.001)),.006))
    api.CreateRestOffsetAttr(float(getattr(collision,'rest_offset',0.)))
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(physics_material,materialPurpose='physics')


def _cuboid(sim,path,size,translation,collision=None,material=None,color=(.08,.09,.10)):
    from isaaclab.utils.string import string_to_callable
    cfg=sim.CuboidCfg(size=size,collision_props=collision,physics_material=material,
                     visual_material=sim.PreviewSurfaceCfg(diffuse_color=color))
    fn=string_to_callable(cfg.func) if isinstance(cfg.func,str) else cfg.func
    fn(path,cfg,translation=translation)


def attach_paddle_to_link(stage:Any,link_path:str,sim:Any,collision:Any,material:Any,
                          *,offset=PADDLE_OFFSET_V2,prefix='PingPongPaddleV2')->dict[str,str]:
    """Add a complete racket before physics starts, directly under terminal link."""
    from pxr import UsdGeom
    # The terminal link is already a dynamic rigid body from the robot USD.
    # Keep collision on the explicit Blade/Handle meshes below; applying a
    # collision schema to the link Xform would also include its inherited
    # visual mesh as a dynamic triangle collider in PhysX.
    base=link_path+'/'+prefix
    UsdGeom.Xform.Define(stage,base)
    paths={name:base+'/'+name for name in PADDLE_PRIMITIVE_NAMES}
    mats=base+'/Materials';UsdGeom.Scope.Define(stage,mats)
    red=_visual_material(stage,mats+'/RedRubber',(.72,.018,.020),.62)
    black=_visual_material(stage,mats+'/BlackRubber',(.009,.010,.012),.66)
    wood=_visual_material(stage,mats+'/Wood',(.53,.27,.095),.7)
    phys=_make_physics_material(stage,mats+'/Physics',material)
    # Collider and the visible wooden edge are coincident in YZ, but only
    # BladeCollision participates in physics. Visual rubber has no collision.
    blade=_ellipse_mesh(stage,paths['BladeCollision'],offset,BLADE_THICKNESS)
    _collision(blade,phys,collision)
    UsdGeom.Imageable(blade).CreateVisibilityAttr('invisible')
    edge=_ellipse_mesh(stage,paths['BladeWoodEdge'],offset,.006)
    _bind_visual(edge,wood)
    front=(offset[0]+.004,offset[1],offset[2])
    back=(offset[0]-.004,offset[1],offset[2])
    _bind_visual(_ellipse_mesh(stage,paths['RubberRedFront'],front,.002,.158,.168),red)
    _bind_visual(_ellipse_mesh(stage,paths['RubberBlackBack'],back,.002,.158,.168),black)
    # The grip extends from the lower blade edge (-.08) to -.20, in-plane.
    handle=(offset[0],offset[1],offset[2]+HANDLE_CENTER_Z)
    wood_color=(.53,.27,.095)
    _cuboid(sim,paths['HandleCollision'],(HANDLE_THICKNESS,HANDLE_WIDTH,HANDLE_LENGTH),
            handle,collision,material,wood_color)
    # Wood skin slightly oversizes physical handle to avoid coincident faces.
    _cuboid(sim,paths['HandleWoodVisual'],(.0184,.0304,.1204),handle,color=wood_color)
    # The H gripper grips a short horizontal adaptor around x=.155.  A slim
    # metal L bracket drops from that adaptor to the wood grip and supports
    # the handle below the paddle; it does not intersect either striking face.
    metal=(.095,.105,.115)
    x0=.115;x1=offset[0]-.012;zgrip=offset[2]-.125
    _cuboid(sim,paths['HandleMountCollision'],(x1-x0,.018,.014),
            ((x0+x1)/2,offset[1],zgrip),collision,material,metal)
    _cuboid(sim,paths['HandleMountVisual'],(x1-x0+.0004,.0184,.0144),
            ((x0+x1)/2,offset[1],zgrip),color=metal)
    _cuboid(sim,paths['HandleClampCollision'],(.018,.018,.13),
            (.125,offset[1],offset[2]-.06),collision,material,metal)
    _cuboid(sim,paths['HandleClampVisual'],(.0184,.0184,.1304),
            (.125,offset[1],offset[2]-.06),color=metal)
    return paths


def attach_v2_all_links(stage:Any,sim:Any,collision:Any,material:Any,*,num_envs:int,
                        roots=('RobotLeft','RobotRight'),arms=('left','right'))->list[dict[str,str]]:
    from pxr import Usd,UsdPhysics
    out=[]
    for e in range(int(num_envs)):
        for root in roots:
            prim=stage.GetPrimAtPath(f'/World/envs/env_{e}/{root}')
            if not prim.IsValid():raise RuntimeError(f'robot root missing: {prim}')
            for arm in arms:
                name=f'{arm}_arm_link6'
                links=[p for p in Usd.PrimRange(prim) if p.GetName()==name and p.HasAPI(UsdPhysics.RigidBodyAPI)]
                if len(links)!=1:raise RuntimeError(f'expected one {name} under {root}, got {[str(p.GetPath()) for p in links]}')
                out.append(attach_paddle_to_link(stage,str(links[0].GetPath()),sim,collision,material))
    return out

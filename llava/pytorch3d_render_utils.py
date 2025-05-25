import torch
import torch.nn as nn
import numpy as np

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    OrthographicCameras,
    PointLights,
    RasterizationSettings,
    MeshRasterizer,
    HardPhongShader,
    MeshRenderer,
    SoftSilhouetteShader,
    TexturesUV,
    TexturesVertex,
    BlendParams)


class TexturedIUVRenderer(nn.Module):
    def __init__(self,
                 device='cuda',
                 img_wh=256,
                 blur_radius=0.0,
                 faces_per_pixel=1,
                 ):
        
        super().__init__()
        self.img_wh = img_wh

        raster_settings = RasterizationSettings(image_size=img_wh,
                                                blur_radius=blur_radius,
                                                faces_per_pixel=faces_per_pixel,)
        
        self.cameras = PerspectiveCameras()
        self.rasterizer = MeshRasterizer(cameras=self.cameras, raster_settings=raster_settings)  # Specify camera in forward pass
        self.iuv_shader = SoftSilhouetteShader()
        
        self.to(device)

    def to(self, device):
        self.rasterizer.to(device)
        self.iuv_shader.to(device)

    def forward(self, vertices, faces, cam_t=None, cameras=None, focal_length=5000):
        img_wh = self.img_wh
        img_center=((img_wh * 0.5, img_wh * 0.5),)
        cameras = PerspectiveCameras(device=vertices.device,
                                        focal_length=focal_length,
                                        principal_point=img_center,
                                        image_size=((img_wh, img_wh),),
                                        in_ndc=False)
        device=vertices.device

        if cam_t is not None:
            vertices = vertices + cam_t[:, None, :]
        
        vertices = vertices * torch.tensor([-1., -1., 1.], device=device).float()

        textures_iuv = TexturesVertex(verts_features=torch.ones_like(vertices))
        meshes_iuv = Meshes(verts=vertices, faces=faces, textures=textures_iuv)

        # Rasterize
        fragments = self.rasterizer(meshes_iuv, cameras=cameras)

        # Render RGB and IUV outputs
        iuv_image = self.iuv_shader(fragments, meshes_iuv)

        return iuv_image

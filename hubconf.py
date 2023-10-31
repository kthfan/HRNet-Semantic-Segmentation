"""File for accessing HRNet via PyTorch Hub https://pytorch.org/hub/

Usage:
    import torch
    model = torch.hub.load('AlexeyAB/PyTorch_YOLOv4:u5_preview', 'yolov4_pacsp_s', pretrained=True, channels=3, classes=80)
"""

dependencies = ['torch']
import torch
from lib.models.seg_hrnet import get_seg_model
from lib.models.seg_hrnet_ocr import get_seg_model as get_seg_model_ocr

from lib.models.hrnet import hrnet18, hrnet32, hrnet48, model_urls
from yacs.config import CfgNode as CN
from lib.config import config, update_config, MODEL_EXTRAS
from lib.config.hrnet_config import MODEL_CONFIGS


def hrnet48_cityscapes(pretrained=False, **kwargs):
	  """ # This docstring shows up in hub.help()
    HRNetW48 model pretrained on Cityscapes
    pretrained (bool): kwargs, load pretrained weights into the model
    """
	  cfg = config.clone()
	  cfg_net = MODEL_CONFIGS['hrnet48'].clone()

	  cfg_net.defrost()
	  cfg_net.set_new_allowed(True)
	  cfg_net.merge_from_other_cfg(MODEL_EXTRAS['seg_hrnet'].clone())
	  cfg_net.freeze()

	  cfg.MODEL.EXTRA = cfg_net

	  model = get_seg_model(cfg)
	  if pretrained:
	  	  state_dict = torch.hub.load_state_dict_from_url(model_urls['hrnet48_cityscapes'], progress=True)
	  	  model.load_state_dict(state_dict)
	  return model


def hrnet48_ocr_cityscapes(pretrained=False, **kwargs):
	""" # This docstring shows up in hub.help()
  HRNetW48 model pretrained on Cityscapes
  pretrained (bool): kwargs, load pretrained weights into the model
  """
	cfg = config.clone()
	cfg_net = MODEL_CONFIGS['hrnet48'].clone()

	cfg_net.defrost()
	cfg_net.set_new_allowed(True)
	cfg_net.merge_from_other_cfg(MODEL_EXTRAS['seg_hrnet'].clone())
	cfg_net.freeze()

	cfg.MODEL.EXTRA = cfg_net

	model = get_seg_model_ocr(cfg)
	if pretrained:
		state_dict = torch.hub.load_state_dict_from_url(model_urls['hrnet48_ocr_cityscapes'], progress=True)
		model.load_state_dict(state_dict)
	return model

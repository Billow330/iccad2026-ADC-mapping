# CPU stub for cuda_ext - GPU path is only triggered when inputs.is_cuda is True.
# On CPU, tensor_quant.py uses legacy_quant_func() which does not call these functions.

def fake_tensor_quant(*args, **kwargs):
    raise RuntimeError("cuda_ext.fake_tensor_quant requires a CUDA-enabled GPU.")

def fake_tensor_quant_with_axis(*args, **kwargs):
    raise RuntimeError("cuda_ext.fake_tensor_quant_with_axis requires a CUDA-enabled GPU.")

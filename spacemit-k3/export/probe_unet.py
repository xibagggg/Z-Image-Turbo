import os, time, warnings, torch
warnings.filterwarnings("ignore")
from diffusers import UNet2DConditionModel
u = UNet2DConditionModel.from_pretrained("build/sd-turbo", subfolder="unet", variant="fp16", torch_dtype=torch.float16).eval()
args=(torch.zeros(1,4,64,64,dtype=torch.float16), torch.tensor([1.0],dtype=torch.float16), torch.zeros(1,77,1024,dtype=torch.float16))
t0=time.time()
try:
    torch.onnx.export(u,args,"build/probe_dynamo.onnx",opset_version=17,dynamo=True,
                      input_names=["sample","timestep","encoder_hidden_states"],output_names=["out_sample"],
                      do_constant_folding=False)
    print("dynamo OK %.1fs  %.1f MB" % (time.time()-t0, os.path.getsize("build/probe_dynamo.onnx")/1e6))
    import onnx
    g=onnx.load("build/probe_dynamo.onnx")
    ops=sorted({n.op_type for n in g.graph.node})
    print("ops(%d): %s" % (len(ops), ops))
except Exception as e:
    print("dynamo FAILED %.1fs: %s: %s" % (time.time()-t0, type(e).__name__, str(e)[:400]))

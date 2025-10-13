import sys
import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
# from core.igev_stereo import IGEVStereo
from core_rt.rt_igev_stereo import IGEVStereo
from core.utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
import os
import skimage.io
import cv2
# from utils.frame_utils import readPFM
import onnxruntime as ort
from time import time

import openvino as ov
from openvino.tools.mo import convert_model
import nncf
from nncf import QuantizationPreset
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader

DEVICE = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# iters/test_mode 고정 래퍼 (ONNX/IR에서 kwargs 보존 안 되므로)
class ExportWrapper(torch.nn.Module):
    def __init__(self, net, iters=16):
        super().__init__()
        self.net = net
        self.iters = iters
    def forward(self, left, right):
        # 좌/우 입력은 pad가 적용된 텐서(NCHW, float32, 0~255)라고 가정
        return self.net(left, right, iters=self.iters, test_mode=True)

def load_image(imfile):
    img = Image.open(imfile)
    img = img.resize((640, 352))
    img = np.array(img).astype(np.uint8)[..., :3]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def custom_transform(img : Image.Image):
    img_f32 = np.array(img).astype(np.float32)
    img_transposed = (img_f32, (2, 0, 1))

class CustomImageDataset(Dataset):
    def __init__(self, root_path, ext_list=(".jpg", "*.png", "*.ppm")):
        if root_path == "im01":
            
            folders = ["demo-imgs/"]
            left_list = ['im0.', 'view1.']
            
            self.left_img_list = []
            self.right_img_list = []
            for folder in folders:
                for root, _, files in os.walk(folder):
                    for f in files:
                        if os.path.splitext(f)[1].lower() in [os.path.splitext(ext)[1].lower() for ext in ext_list]:
                            if "image_2" in root: # for kitti 2015
                                left_file_path = os.path.join(root, f)
                                right_file_path = left_file_path.replace("image_2", "image_3")
                                if os.path.exists(right_file_path):
                                    self.left_img_list.append(left_file_path)
                                    self.right_img_list.append(right_file_path)
                                    continue
                            else:
                                for left_name in left_list:    
                                    if left_name in f[0]:
                                        n4 = f.replace(left_name, "im4.")
                                        n1 = f.replace(left_name, "im1.")
                                        if os.path.exists(n4):
                                            self.left_img_list.append(os.path.join(root, f))
                                            self.right_img_list.append(os.path.join(root, n4))
                                            break
                                        elif os.path.exists(n1):
                                            self.left_img_list.append(os.path.join(root, f))
                                            self.right_img_list.append(os.path.join(root, n1))
                                            break
                                    elif left_name in f[1]:
                                        n5 = f.replace(left_name, "view5.")
                                        if os.path.exists(n5):
                                            self.left_img_list.append(os.path.join(root, f))
                                            self.right_img_list.append(os.path.join(root, n5))
                                            break
                                        
    def __len__(self):
        return len(self.left_img_list)
    
    def __getitem__(self, idx):
        l_path = self.left_img_list[idx]
        r_path = self.right_img_list[idx]
        left_img = load_image(l_path)
        right_img = load_image(r_path)
        padder = InputPadder(left_img.shape, divis_by=32)
        pad_left_img , pad_right_img =padder.pad(left_img, right_img)
        input_left_npimg = pad_left_img.cpu().numpy()
        input_right_npimg = pad_right_img.cpu().numpy()
        # return {"left":input_left_npimg, "right":input_right_npimg}
        return input_left_npimg, input_right_npimg
        
def demo(args):
    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))
    model = model.module
    model.to(DEVICE)
    model.eval()
    
    model_cpu = IGEVStereo(args)
    model_cpu.load_state_dict(model.state_dict(), strict=True)
    model_cpu.eval().to('cpu')

    wrapper = ExportWrapper(model_cpu, iters=args.valid_iters).eval()
    
    # for m in model_cpu.modules():
    #     if isinstance(m, torch.nn.InstanceNorm2d):
    #         print("before:", m.track_running_stats, ",", m.training)
    #         m.track_running_stats = True
    #         m.training = False

    output_directory = Path(args.output_directory)
    os.makedirs(output_directory, exist_ok=True)
    # output_directory.mkdir(exist_ok=True)

    with torch.no_grad():
        left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        print(f"Found {len(left_images)} images. Saving files to {output_directory}/")
        
        left_name = left_images[0]
        right_name = right_images[0]
        sample_image1 = load_image(left_name)
        sample_image2 = load_image(right_name)
        padder = InputPadder(sample_image1.shape, divis_by=32)
        pad_sample_image1, pad_sample_image2 = padder.pad(sample_image1, sample_image2)
        
        
        # ONNX CONVERTING
        dummy_input = (pad_sample_image1.cpu().float(), pad_sample_image2.cpu().float())
        
        dynamic_axes = {
            "left": {0:"N", 2:"H", 3:"W"},
            "right": {0:"N", 2:"H", 3:"W"},
            "pred_disp": {0:"N", 2:"H", 3:"W"},
        }

        if os.path.exists(args.output_onnx):
            print("ONNX model already exported.")
        else:
            torch.onnx.export(
                # model_cpu,
                wrapper,
                dummy_input,
                args.output_onnx,
                input_names=["left","right"], 
                output_names=["pred_disp"],
                # dynamic_axes=dynamic_axes,
                opset_version=16,
                do_constant_folding=True,
                training=torch.onnx.TrainingMode.EVAL
            )
            print(f"ONNX model exported to {args.output_onnx}")
        
        # OPENVINO CONVERTING
        if os.path.exists(args.output_ir) and os.path.exists(args.output_ir.replace(".xml", ".bin")):
            print("OPENVINO IR model already exported.")
        else:
            ov_model = convert_model(input_model = args.output_onnx, compress_to_fp16=False)
            ov_model.outputs[0].tensor.set_names({"pred_disp"})
            ov.serialize(ov_model, args.output_ir, args.output_ir.replace('.xml', '.bin'))
            
        # Openvino Quantize
        print("OPENVINO QUANT INT8")
        q_model_path = args.output_ir.replace('.xml','_quant.xml')
        # if os.path.exists(q_model_path) and os.path.exists(q_model_path.replace(".xml", ".bin")):
        if False:
            print("OPENVINO QUANT model already exported.")
        else:
            ie = ov.Core()
            ir_model = ie.read_model(args.output_ir, args.output_ir.replace('.xml', '.bin'))

            calib_dataset = CustomImageDataset(root_path="im01")
            calib_loader = DataLoader(calib_dataset, batch_size=1)
            def transform_fn(data_item):
                return data_item[0][0].numpy(), data_item[1][0].numpy()
            calibration_dataset = nncf.Dataset(calib_loader, transform_fn)
            q_model = nncf.quantize(
                    model=ir_model,
                    calibration_dataset = calibration_dataset,
                    preset=QuantizationPreset.MIXED,
                    #advanced_parameters =  AdvancedQuantizationParameters(
                    #     fast_bias_correction=True,
                    #     overflow_fix=True,
                    #     dump_quantization_statistics=True,
                    # )
            )
            ov.serialize(q_model, q_model_path, q_model_path.replace('.xml', '.bin'))
            
        # Inference TEST
        
        for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
            print("Filename : ", imfile1)
            
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            print("img shape : ", image1.shape)
            
            padder = InputPadder(image1.shape, divis_by=32)
            image1_pad, image2_pad = padder.pad(image1, image2)
            print("padded img shape:", image1_pad.shape)
            st = time()
            disp = model(image1_pad, image2_pad, iters=args.valid_iters, test_mode=True)
            ed = time()
            print("torch(GPU) inference time: ", ed -st , "[s]")
            
            st = time()
            disp = model_cpu(image1_pad.to("cpu"), image2_pad.to("cpu"), iters=args.valid_iters, test_mode=True)
            ed = time()
            print("torch(CPU) inference time: ", ed -st , "[s]")
            disp_unpad = padder.unpad(disp)
            
            file_stem = imfile1.split('/')[-2]
            filename = os.path.join(output_directory, f'{file_stem}.png')
            disp_res = disp_unpad.cpu().numpy().squeeze()
            plt.imsave(filename.replace(".png", "_plt.png"), disp_res.squeeze(), cmap='jet')
            
            if args.save_numpy:
                np.save(output_directory / f"{file_stem}.npy", disp.squeeze())

            disp_ui16 = np.round(disp_res * 256).astype(np.uint16)
            cv2.imwrite(filename.replace(".png", "_cv2.png"), cv2.applyColorMap(cv2.convertScaleAbs(disp_ui16.squeeze(), alpha=0.01),cv2.COLORMAP_JET), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
            
            print("ONNX RUNTIME")
            ort_session = ort.InferenceSession(args.output_onnx)
            input_l_np = image1_pad.cpu().numpy()
            input_r_np = image2_pad.cpu().numpy()
            print("numpy img shape: ", input_l_np.shape)
            
            onnx_inputs = {"left": input_l_np, "right": input_r_np}
            st = time()            
            onnx_outputs = ort_session.run(['pred_disp'], onnx_inputs)
            ed = time()
            res_onnx = onnx_outputs[0].squeeze()
            print("   * onnx forwarding time : " , ed - st , " [s]")
            plt.imsave(filename.replace(".png", "_plt_onnx.png"), res_onnx, cmap='jet')
            res_onnx_ui16 = np.round(res_onnx * 256).astype(np.uint16)
            cv2.imwrite(filename.replace(".png", "_cv2_onnx.png"), cv2.applyColorMap(cv2.convertScaleAbs(res_onnx_ui16.squeeze(), alpha=0.01), cv2.COLORMAP_JET), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
            
            print("OPENVINO IR RUNTIME")
            ie = ov.Core()
            ir_model = ie.read_model(args.output_ir, args.output_ir.replace('.xml', '.bin'))
            compiled = ie.compile_model(ir_model, "CPU")
            input_left = compiled.input(0)
            input_right = compiled.input(1)
            output = compiled.output(0)
            
            st = time()
            ir_res = compiled([input_l_np, input_r_np])[output].squeeze()
            ed = time()
            print("   * ir forwarding time : ", ed - st , " [s]")
            # print("save reuslt")
            plt.imsave(filename.replace(".png", "_plt_ir.png"), ir_res.squeeze(), cmap='jet')
            res_ir_ui16 = np.round(ir_res * 256).astype(np.uint16)
            cv2.imwrite(filename.replace(".png", "_cv2_ir.png"), cv2.applyColorMap(cv2.convertScaleAbs(res_ir_ui16.squeeze(), alpha=0.01), cv2.COLORMAP_JET), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
            
            
            
            print("OPENVINO IR Quant RUNTIME")
            ie = ov.Core()
            ir_model = ie.read_model(q_model_path, q_model_path.replace('.xml', '.bin'))
            compiled = ie.compile_model(ir_model, "CPU")
            input_left = compiled.input(0)
            input_right = compiled.input(1)
            output = compiled.output(0)
            
            st = time()
            ir_res = compiled([input_l_np, input_r_np])[output].squeeze()
            ed = time()
            print("   * ir forwarding time : ", ed - st , " [s]")
            # print("save reuslt")
            plt.imsave(filename.replace(".png", "_plt_ir_quant.png"), ir_res.squeeze(), cmap='jet')
            res_ir_ui16 = np.round(ir_res * 256).astype(np.uint16)
            cv2.imwrite(filename.replace(".png", "_cv2_ir_quant.png"), cv2.applyColorMap(cv2.convertScaleAbs(res_ir_ui16.squeeze(), alpha=0.01), cv2.COLORMAP_JET), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
            

            print("done")
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", 
                        # default='./pretrained_models/igev_plusplus/sceneflow.pth'
                        default='./pretrained_models/igev_rt/sceneflow.pth'
                        )
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="./demo-imgs/samples/**/im0.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="./demo-imgs/samples/**/im1.png")
    
    # parser.add_argument('--output_directory', help="directory to save output", default="demo_output/sceneflow")
    parser.add_argument('--output_directory', help="directory to save output", default="demo_output/rt_sceneflow")
    parser.add_argument("--output_onnx", help="path to save onnx", default="demo_output/rt_sceneflow/model.onnx")
    parser.add_argument("--output_ir", help="path to save openvino ir", default="demo_output/rt_sceneflow/openvino_ir.xml")
    
    parser.add_argument('--mixed_precision', action='store_true', default=False, help='use mixed precision')
    parser.add_argument('--precision_dtype', default='float32', choices=['float16', 'bfloat16', 'float32'], help='Choose precision type: float16 or bfloat16 or float32')
    parser.add_argument('--valid_iters', type=int, default=2, help='number of flow-field updates during forward pass')

    # Architecture choices
    # parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    # parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    # parser.add_argument('--max_disp', type=int, default=768, help="max disp range")
    
    parser.add_argument('--hidden_dim', nargs='+', type=int, default=96, help="hidden state and context dimensions")
    parser.add_argument('--n_gru_layers', type=int, default=1, help="number of hidden GRU levels")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")

    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    # parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently") #
    parser.add_argument('--slow_fast_gru', action='store_false', help="iterate the low-res GRUs more frequently") #
    
    # used IGEV++ only.
    parser.add_argument('--s_disp_range', type=int, default=48, help="max disp of small disparity-range geometry encoding volume")
    parser.add_argument('--m_disp_range', type=int, default=96, help="max disp of medium disparity-range geometry encoding volume")
    parser.add_argument('--l_disp_range', type=int, default=192, help="max disp of large disparity-range geometry encoding volume")
    parser.add_argument('--s_disp_interval', type=int, default=1, help="disp interval of small disparity-range geometry encoding volume")
    parser.add_argument('--m_disp_interval', type=int, default=2, help="disp interval of medium disparity-range geometry encoding volume")
    parser.add_argument('--l_disp_interval', type=int, default=4, help="disp interval of large disparity-range geometry encoding volume")
    
    args = parser.parse_args()

    demo(args)

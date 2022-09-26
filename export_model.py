from networks.InvDN_model_oneForward import InvNet,constructor

import os,math
import argparse
import paddle

#动转静导出模型，使用修改后的只需一次forward的InvDN_model_oneForward

parser = argparse.ArgumentParser(description="export model")
parser.add_argument("--save-inference-dir", type=str, default="./export", help='path of model for export')
parser.add_argument("--weights", type=str, default="./pretrained/model_best.pdparams", help='path of model checkpoint')

opt = parser.parse_args()

def main(opt):

    model = InvNet(channel_in=3, channel_out=3, subnet_constructor=constructor, block_num=[8, 8], down_num=2)
    ckpt = paddle.load(opt.weights)
    model.set_state_dict(ckpt['state_dict'])

    print('Loaded trained params of model successfully.')

    shape = [-1, 3, 256, 256]

    new_model = model

    new_model.eval()
    new_net = paddle.jit.to_static(
        new_model,
        input_spec=[paddle.static.InputSpec(shape=shape, dtype='float32')])
    save_path = os.path.join(opt.save_inference_dir, 'model')
    paddle.jit.save(new_net, save_path)


    print(f'Model is saved in {opt.save_inference_dir}.')


if __name__ == '__main__':
    main(opt)

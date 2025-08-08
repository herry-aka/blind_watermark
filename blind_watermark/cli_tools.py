from optparse import OptionParser
from .blind_watermark import WaterMark
import blind_watermark

usage1 = 'blind_watermark --embed --pwd 1234 image.jpg "watermark text" embed.png'
usage2 = 'blind_watermark --extract --pwd 1234 --wm_shape 111 embed.png'
optParser = OptionParser(usage=usage1 + '\n' + usage2)

optParser.add_option('--embed', dest='work_mode', action='store_const', const='embed'
                     , help='Embed watermark into images')
optParser.add_option('--extract', dest='work_mode', action='store_const', const='extract'
                     , help='Extract watermark from images')

optParser.add_option('-p', '--pwd', dest='password', help='password, like 1234')
optParser.add_option('--wm_shape', dest='wm_shape', help='Watermark shape, like 120')

(opts, args) = optParser.parse_args()


def main():
    bwm1 = WaterMark(password_img=int(opts.password))
    if opts.work_mode == 'embed':
        if not len(args) == 3:
            print('Error! Usage: ')
            print(usage1)
            return
        else:
            bwm1.read_img(args[0])
            bwm1.read_wm(args[1], mode='str')
            bwm1.embed(args[2])
            print('Embed succeed! to file ', args[2])
            print('Put down watermark size:', len(bwm1.wm_bit))

    if opts.work_mode == 'extract':
        if not len(args) == 1:
            print('Error! Usage: ')
            print(usage2)
            return

        else:
            wm_str = bwm1.extract(filename=args[0], wm_shape=int(opts.wm_shape), mode='str')
            print('Extract succeed! watermark is:')
            print(wm_str)


'''
python -m blind_watermark.cli_tools --embed --pwd 1234 examples/pic/ori_img.jpeg "watermark text" examples/output/embedded.png
python -m blind_watermark.cli_tools --extract --pwd 1234 --wm_shape 111 examples/output/embedded.png


cd examples
blind_watermark --embed --pwd 1234 examples/pic/ori_img.jpeg "watermark text" examples/output/embedded.png
blind_watermark --extract --pwd 1234 --wm_shape 111 examples/output/embedded.png
'''

'''
from optparse import OptionParser
from .blind_watermark import WaterMark
import cv2
usage1 = 'blind_watermark --embed --pwd 1234 image.jpg "watermark text" embed.png'
usage2 = 'blind_watermark --extract --pwd 1234 --wm_shape 111 embed.png'
usage3 = 'blind_watermark --embed --pwd 1234 --mode img 原图.jpg 水印图.png 输出.png'
usage4 = 'blind_watermark --extract --pwd 1234 --mode img --wm_shape 128,128 带水印.png'
#optParser = OptionParser(usage=usage1 + '\n' + usage2)
optParser = OptionParser(usage='\n'.join([usage1, usage2, usage3, usage4]))

optParser.add_option('--embed', dest='work_mode', action='store_const', const='embed'
                     , help='Embed watermark into images')
optParser.add_option('--extract', dest='work_mode', action='store_const', const='extract'
                     , help='Extract watermark from images')

optParser.add_option('-p', '--pwd', dest='password', help='password, like 1234')
optParser.add_option('--wm_shape', dest='wm_shape', help='Watermark shape, like 120')
optParser.add_option('--mode', dest='mode', default='str', 
                     help='Watermark mode: "str" (text) or "img" (image)')
(opts, args) = optParser.parse_args()


def main():
    # 初始化水印工具（同时传入图片和水印的密码）
    bwm1 = WaterMark(
        password_img=int(opts.password),
        password_wm=int(opts.password)
    )

    if opts.work_mode == 'embed':
        if opts.mode == 'str':
            # 文本水印：参数为 [原图路径, 水印文本, 输出路径]
            if len(args) != 3:
                print('错误！文本水印用法：')
                print(usage1)
                return
            bwm1.read_img(args[0])
            bwm1.read_wm(args[1], mode='str')
            bwm1.embed(args[2])
            print(f'文本水印嵌入成功！输出到 {args[2]}')
            print(f'请记录水印长度（提取时需要）：{len(bwm1.wm_bit)}')
        
        elif opts.mode == 'img':
            # 图片水印：参数为 [原图路径, 水印图路径, 输出路径]
            if len(args) != 3:
                print('错误！图片水印用法：')
                print(usage3)
                return
            bwm1.read_img(args[0])
            bwm1.read_wm(args[1], mode='img')  # 读取图片水印
            bwm1.embed(args[2])
            # 获取水印图尺寸（提取时需要）
            wm_img = cv2.imread(args[1], cv2.IMREAD_GRAYSCALE)
            wm_shape = f'{wm_img.shape[1]},{wm_img.shape[0]}'  # 宽,高
            print(f'图片水印嵌入成功！输出到 {args[2]}')
            print(f'请记录水印尺寸（提取时需要）：{wm_shape}')

    elif opts.work_mode == 'extract':
        if opts.mode == 'str':
            # 提取文本水印：需要水印长度
            if len(args) != 1 or not opts.wm_shape:
                print('错误！文本水印提取用法：')
                print(usage2)
                return
            wm_str = bwm1.extract(
                filename=args[0],
                wm_shape=int(opts.wm_shape),
                mode='str'
            )
            print('文本水印提取成功：')
            print(wm_str)
        
        elif opts.mode == 'img':
            # 提取图片水印：需要水印尺寸（宽,高）
            if len(args) != 1 or not opts.wm_shape:
                print('错误！图片水印提取用法：')
                print(usage4)
                return
            # 解析水印尺寸为元组（宽,高）
            wm_shape = tuple(map(int, opts.wm_shape.split(',')))
            bwm1.extract(
                filename=args[0],
                wm_shape=wm_shape,
                mode='img',
                out_wm_name='embedded1.png'  # 提取后保存的文件名
            )
            print(f'图片水印提取成功！保存为:')



python -m blind_watermark.cli_tools --embed --pwd 1234 examples/pic/ori_img.jpeg "watermark text" examples/output/embedded.png
python -m blind_watermark.cli_tools --extract --pwd 1234 --wm_shape 111 examples/output/embedded.png


cd examples
blind_watermark --embed --pwd 1234 examples/pic/ori_img.jpeg "watermark text" examples/output/embedded.png
blind_watermark --extract --pwd 1234 --wm_shape 111 examples/output/embedded.png
'''

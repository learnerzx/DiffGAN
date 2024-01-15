import re
import xlsxwriter
txt_url = r"E:\code\code_backup_vvt+diffusion\OUTPUT_PATH\IXI_results\Gaussian2D_8x_33.21_0.9082_diffusion_VVT\新建文本文档.txt"
sava_url = r'C:\Users\zx\Desktop\data\MICCAI\Gaussian2D_8x_33.21_0.9082_diffusion_VVT.xlsx'
# 读取txt 并提取指定字符
def readTXT():
    with open(txt_url, 'r', encoding='utf-8') as f:
        content = f.read()  # 读取txt文件内容 保存到content中

    result1 = re.findall(r"\sZF_PSNR:(.*).*", content)  # 提取指定字符  表示提取 time: ~ ms之间的字符
    result2 = re.findall(r"\sZF_SSIM:(.*).*", content)  # 提取指定字符 提取TPS:~  之后的字符 包含了*号 暂时没有学到好的匹配方法
    l2 = len(result2)
    for i in range(l2):
        result2[i] = result2[i].replace("*", "")  # *号 暂时没有好的方法 去除*号
    return result1, result2
def saveExcel(str1, str2):
    # todo 创建excel文件
    xl = xlsxwriter.Workbook(sava_url)
    # todo 添加sheet
    sheet = xl.add_worksheet('sheet1')
    k = -4
    j = 0
    # todo 往单元格cell添加数据,索引写入
    for i in range(len(str1)):
        if i % 200 == 0:			# 数据50一组
            k = k + 4			# 每一组数据后 整体表格右移4格
            j = 0				# 每一组数据后 重新从0行 开始存入
        sheet.write_string(j, k, str1[i])	# j 表示行  k表示 列
        sheet.write_string(j, k + 1, str2[i]) #(excel从左上角开始算0行0列)
        j = j + 1				# 行+1 准备写入下一行
    # # todo 设置单元格宽度大小
    # sheet.set_column('A:B', 30)
    # todo 关闭文件
    xl.close()

str1, str2 = readTXT()
print('PSNR:',str1)
print('SSIM:',str2)
saveExcel(str1, str2)

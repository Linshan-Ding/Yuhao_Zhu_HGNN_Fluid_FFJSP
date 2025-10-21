def truncate_and_rewrite(input_file, output_file, max_length):
    """
    读取文本文件，截断每行到指定长度，并写入新文件

    :param input_file: 输入文件路径
    :param output_file: 输出文件路径
    :param max_length: 每行保留的最大元素数量
    """
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            # 分割行内容为元素列表
            elements = line.strip().split()

            # 截断到指定长度
            truncated = elements[:max_length]

            # 重新组合为字符串并写入
            f_out.write(' '.join(truncated) + '\n')


# 示例使用
if __name__ == "__main__":
    input_file = 'num2000_lam0.05_change0__3.txt'  # 输入文件路径
    output_file = 'test_num2000_lam0.05_change0__3.txt'  # 输出文件路径
    max_length = 30  # 每行保留的最大元素数量

    truncate_and_rewrite(input_file, output_file, max_length)
    print(f"文件已处理并保存到 {output_file}")
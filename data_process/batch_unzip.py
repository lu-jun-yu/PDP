"""
批量解压 ZIP 文件到指定路径
支持 Windows 系统
"""

import os
import zipfile
import argparse
import struct
from pathlib import Path


def decode_zip_filename(zip_file_path, member_info, encodings=['cp936', 'gbk', 'gb2312', 'utf-8', 'cp437']):
    """
    从 ZIP 文件原始字节中读取并解码文件名
    
    Args:
        zip_file_path: ZIP 文件路径
        member_info: ZipInfo 对象
        encodings: 要尝试的编码列表
    
    Returns:
        解码后的文件名（str）
    """
    try:
        # 方法1: 尝试从原始 ZIP 文件读取文件名字节
        with open(zip_file_path, 'rb') as f:
            # ZIP 文件格式：本地文件头
            # 查找对应的本地文件头
            f.seek(member_info.header_offset)
            # 读取本地文件头签名（4字节）
            signature = f.read(4)
            if signature != b'PK\x03\x04':
                # 如果找不到，使用备用方法
                raise ValueError("Invalid local file header")
            
            # 读取版本（2字节）
            version = struct.unpack('<H', f.read(2))[0]
            # 读取通用位标志（2字节）
            flags = struct.unpack('<H', f.read(2))[0]
            # 检查是否使用 UTF-8 编码（位 11）
            use_utf8 = (flags & 0x800) != 0
            
            # 读取压缩方法（2字节）
            f.read(2)
            # 读取修改时间（2+2字节）
            f.read(4)
            # 读取 CRC32（4字节）
            f.read(4)
            # 读取压缩大小（4字节）
            f.read(4)
            # 读取未压缩大小（4字节）
            f.read(4)
            # 读取文件名长度（2字节）
            filename_len = struct.unpack('<H', f.read(2))[0]
            # 读取扩展字段长度（2字节）
            extra_len = struct.unpack('<H', f.read(2))[0]
            # 读取原始文件名字节
            filename_bytes = f.read(filename_len)
            
            # 如果 ZIP 文件明确标记使用 UTF-8，优先使用 UTF-8
            if use_utf8:
                try:
                    return filename_bytes.decode('utf-8')
                except (UnicodeDecodeError, UnicodeError):
                    pass
            
            # 尝试使用各种编码解码（优先使用中文编码）
            for encoding in encodings:
                try:
                    decoded = filename_bytes.decode(encoding)
                    # 验证解码结果是否合理
                    if decoded and all(ord(c) < 0x10000 for c in decoded):
                        # 如果包含中文字符，很可能是正确的
                        if any('\u4e00' <= c <= '\u9fff' for c in decoded):
                            return decoded
                except (UnicodeDecodeError, UnicodeError):
                    continue
            
            # 如果都失败，使用错误处理（默认使用 CP936）
            return filename_bytes.decode('cp936', errors='replace')
            
    except Exception:
        # 方法2: 如果读取原始字节失败，使用 ZipInfo 的 filename 并尝试重新编码
        original_name = member_info.filename
        
        # 尝试重新编码解码（适用于已经被错误解码的情况）
        for encoding in encodings:
            try:
                # 使用 latin1 编码保持原始字节，再用目标编码解码
                name_bytes = original_name.encode('latin1')
                decoded = name_bytes.decode(encoding)
                # 简单验证：如果解码后包含中文字符，可能是正确的
                if any('\u4e00' <= c <= '\u9fff' for c in decoded):
                    return decoded
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
        
        # 如果都失败，返回原始名称
        return original_name


def batch_unzip(zip_dir, output_dir=None, pattern="*.zip", remove_zip=False):
    """
    批量解压 ZIP 文件
    
    Args:
        zip_dir: ZIP 文件所在目录
        output_dir: 解压目标目录，如果为 None 则解压到 ZIP 文件所在目录
        pattern: 文件匹配模式，默认为 "*.zip"
        remove_zip: 解压后是否删除原 ZIP 文件
    """
    zip_dir = Path(zip_dir)
    
    if not zip_dir.exists():
        print(f"❌ 错误: 目录不存在: {zip_dir}")
        return
    
    # 查找所有 ZIP 文件
    zip_files = list(zip_dir.glob(pattern))
    
    if not zip_files:
        print(f"⚠️  未找到匹配 {pattern} 的文件")
        return
    
    print(f"📦 找到 {len(zip_files)} 个 ZIP 文件")
    print("=" * 60)
    
    success_count = 0
    fail_count = 0
    skip_count = 0  # 跳过的文件数量
    
    for zip_file in zip_files:
        try:
            print(f"\n📂 正在解压: {zip_file.name}")
            
            # 确定解压目标目录
            if output_dir:
                extract_to = Path(output_dir)
            else:
                # 解压到 ZIP 文件所在目录，使用 ZIP 文件名（不含扩展名）作为子目录
                extract_to = zip_dir / zip_file.stem
            
            # 创建目标目录
            extract_to.mkdir(parents=True, exist_ok=True)
            
            # 解压文件（处理中文编码问题）
            # 统一使用手动解码，确保文件名正确
            zip_ref = zipfile.ZipFile(zip_file, 'r')
            
            try:
                # 获取 ZIP 文件中的所有文件列表
                file_list = zip_ref.namelist()
                print(f"   包含 {len(file_list)} 个文件")
                
                # 统计当前 ZIP 文件的解压情况
                zip_success = 0
                zip_skip = 0
                
                # 手动解压每个文件，处理编码问题
                for member in zip_ref.infolist():
                    try:
                        # 从原始 ZIP 文件读取并解码文件名
                        decoded_name = decode_zip_filename(zip_file, member)
                        
                        # 更新成员信息中的文件名
                        member.filename = decoded_name
                        
                        # 确保目标路径在提取目录内（安全措施）
                        target_path = extract_to / decoded_name
                        # 防止路径遍历攻击
                        target_path = target_path.resolve()
                        extract_to_resolved = extract_to.resolve()
                        if not str(target_path).startswith(str(extract_to_resolved)):
                            print(f"   ⚠️  跳过不安全路径: {decoded_name}")
                            continue
                        
                        # 检查目标文件是否已存在
                        if target_path.exists() and target_path.is_file():
                            zip_skip += 1
                            skip_count += 1
                            continue  # 跳过已存在的文件
                        
                        # 创建父目录
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # 解压文件
                        with zip_ref.open(member) as source:
                            with open(target_path, 'wb') as target:
                                target.write(source.read())
                        
                        zip_success += 1
                                
                    except Exception as e:
                        print(f"   ⚠️  解压文件失败: {member.filename} - {e}")
                        continue
                
                # 显示当前 ZIP 文件的解压统计
                if zip_skip > 0:
                    print(f"   📊 解压: {zip_success} 个, 跳过: {zip_skip} 个已存在文件")
            finally:
                zip_ref.close()
            
            print(f"   ✅ 已解压到: {extract_to}")
            
            # 如果需要，删除原 ZIP 文件
            if remove_zip:
                zip_file.unlink()
                print(f"   🗑️  已删除原文件: {zip_file.name}")
            
            success_count += 1
            
        except zipfile.BadZipFile:
            print(f"   ❌ 错误: {zip_file.name} 不是有效的 ZIP 文件")
            fail_count += 1
        except Exception as e:
            print(f"   ❌ 解压失败: {e}")
            fail_count += 1
    
    print("\n" + "=" * 60)
    print(f"📊 解压完成: 成功 {success_count} 个 ZIP 文件, 失败 {fail_count} 个")
    if skip_count > 0:
        print(f"⏭️  跳过 {skip_count} 个已存在的文件")


def main():
    parser = argparse.ArgumentParser(
        description="批量解压 ZIP 文件到指定路径",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 解压当前目录下所有 ZIP 文件到各自同名文件夹
  python batch_unzip.py -d ./data/pku_fabao
  
  # 解压到指定目录
  python batch_unzip.py -d ./data/pku_fabao -o ./data/pku_fabao/extracted
  
  # 解压后删除原 ZIP 文件
  python batch_unzip.py -d ./data/pku_fabao --remove-zip
        """
    )
    
    parser.add_argument(
        '-d', '--dir',
        type=str,
        required=True,
        help='ZIP 文件所在目录'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='解压目标目录（默认：解压到 ZIP 文件所在目录的同名子文件夹）'
    )
    
    parser.add_argument(
        '-p', '--pattern',
        type=str,
        default='*.zip',
        help='文件匹配模式（默认: *.zip）'
    )
    
    parser.add_argument(
        '--remove-zip',
        action='store_true',
        help='解压后删除原 ZIP 文件'
    )
    
    args = parser.parse_args()
    
    batch_unzip(
        zip_dir=args.dir,
        output_dir=args.output,
        pattern=args.pattern,
        remove_zip=args.remove_zip
    )


if __name__ == "__main__":
    main()

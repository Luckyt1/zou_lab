import sys
import json

def process_file(filename):
    print(f"正在处理文件: {filename}")
    with open(filename, 'r') as f:
        data = json.load(f)
        if(len(data["Frames"][0])!=70):
            print("文件已被处理过了")
            exit()
        for i,actions in enumerate(data["Frames"]):
            data["Frames"][i] = actions[14:29]+actions[29+14:] #去除手臂
        return data

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        processed = process_file(file_path)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(processed, f, indent=4)
    else:
        print("请提供文件名！用法: python script.py <文件名>")